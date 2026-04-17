"""Download audio from YouTube, split vocals vs accompaniment, transcribe vocals with Basic Pitch.

Accompaniment (no_vocals) is used for coarse key estimation to suggest tonic/mode in the API,
and optionally librosa beat tracking (see ACCOMONTAGE_NO_BEAT_TRACK) to nudge tempo / phrase cuts.
Chord progression and texture are still produced by Chorderator from the melody MIDI; the
cleaner vocal stem reduces bleed from backing instruments into the estimated lead line.
"""
from __future__ import annotations

import importlib.util
import io
import logging
import os
import re
import shutil
import subprocess
import sys
from urllib.parse import urlparse

import numpy as np

_YOUTUBE_HOST = re.compile(
    r"^(www\.)?(youtube\.com|youtu\.be|m\.youtube\.com|music\.youtube\.com)$",
    re.I,
)


def is_allowed_youtube_url(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    u = url.strip()
    if len(u) > 500:
        return False
    try:
        p = urlparse(u)
    except Exception:
        return False
    if p.scheme not in ("http", "https"):
        return False
    host = (p.hostname or "").lower()
    return bool(_YOUTUBE_HOST.match(host))


def _ffmpeg_location_dir() -> str | None:
    """Directory containing ffmpeg/ffprobe for yt-dlp --ffmpeg-location."""
    loc = (os.environ.get("FFMPEG_LOCATION") or "").strip()
    if loc:
        if os.path.isdir(loc):
            return os.path.abspath(loc)
        if os.path.isfile(loc) and "ffmpeg" in os.path.basename(loc).lower():
            return os.path.dirname(os.path.abspath(loc))
    w = shutil.which("ffmpeg")
    if w:
        return os.path.dirname(os.path.abspath(w))
    for cand in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if os.path.isfile(cand):
            return os.path.dirname(cand)
    return None


def _js_runtime_args() -> list[str]:
    """yt-dlp 2025+ prefers a JS runtime for YouTube; pass explicit path when found."""
    return []


def _enriched_path_env() -> dict[str, str]:
    env = os.environ.copy()
    parts = [p for p in env.get("PATH", "").split(os.pathsep) if p]
    for extra in ("/opt/homebrew/bin", "/usr/local/bin"):
        if extra not in parts:
            parts.insert(0, extra)
    env["PATH"] = os.pathsep.join(parts)
    return env


def _run(cmd: list[str], cwd: str | None = None, timeout: int = 600) -> None:
    r = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_enriched_path_env(),
    )
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()[:800]
        raise RuntimeError(err or f"command failed: {' '.join(cmd[:6])} ...")


# Krumhansl–Kessler weights (C = index 0), for key estimation from instrumental stem
_KK_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float64
)
_KK_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float64
)
_TONIC_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _beat_track_no_vocals(wav_path: str, max_seconds: float = 240.0) -> dict | None:
    """
    Beat-track the instrumental stem for tempo / phrase nudging in melody_analyze.
    Disabled when ACCOMONTAGE_NO_BEAT_TRACK is 1/true/yes.
    """
    if os.environ.get("ACCOMONTAGE_NO_BEAT_TRACK", "").lower() in ("1", "true", "yes"):
        return None
    try:
        import librosa
    except ImportError:
        return None
    try:
        y, sr = librosa.load(wav_path, mono=True, sr=22050, duration=max_seconds)
    except Exception as e:
        logging.warning("beat_track load failed: %s", e)
        return None
    if y.size < sr * 2:
        return None
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    except Exception as e:
        logging.warning("beat_track beat_track failed: %s", e)
        return None
    tarr = np.asarray(tempo, dtype=np.float64).ravel()
    bpm = float(tarr[0]) if tarr.size else 120.0
    if not (56.0 < bpm < 200.0):
        return None
    beat_times = librosa.frames_to_time(beats, sr=sr)
    bt = np.asarray(beat_times, dtype=np.float64).ravel()
    if bt.size < 8:
        return None
    if bt.size > 2000:
        bt = bt[:2000]
    return {"beat_bpm": bpm, "beat_times_sec": bt.tolist()}


def _estimate_key_from_wav(wav_path: str, max_seconds: float = 120.0) -> tuple[str | None, str | None]:
    """Return (tonic, mode) with mode in {'maj','min'} using chroma + KK profiles, or (None, None)."""
    try:
        import librosa
    except ImportError:
        return None, None

    try:
        y, sr = librosa.load(wav_path, mono=True, sr=22050, duration=max_seconds)
    except Exception:
        return None, None
    if y.size < sr * 2:
        return None, None

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    ch = np.maximum(chroma.mean(axis=1), 0.0)
    s = float(ch.sum())
    if s < 1e-6:
        return None, None
    ch = ch / s

    def best_for(profile: np.ndarray) -> tuple[int, float]:
        p = profile / (np.linalg.norm(profile) + 1e-9)
        best_i, best_c = 0, -1.0
        for shift in range(12):
            rolled = np.roll(p, shift)
            c = float(np.dot(ch, rolled))
            if c > best_c:
                best_c, best_i = c, shift
        return best_i, best_c

    maj_i, maj_c = best_for(_KK_MAJOR)
    min_i, min_c = best_for(_KK_MINOR)
    if maj_c >= min_c:
        return _TONIC_SHARP[maj_i], "maj"
    return _TONIC_SHARP[min_i], "min"


def _find_demucs_vocals_pair(sep_root: str) -> tuple[str, str]:
    for root, _dirs, files in os.walk(sep_root):
        if "vocals.wav" in files and "no_vocals.wav" in files:
            v = os.path.join(root, "vocals.wav")
            nv = os.path.join(root, "no_vocals.wav")
            return v, nv
    raise RuntimeError(
        "Demucs finished but vocals.wav / no_vocals.wav were not found under "
        f"{sep_root!r}. Check Demucs version and --two-stems=vocals support."
    )


def _trim_wav_for_basic_pitch(src: str, work_dir: str, max_seconds: int) -> str:
    """Optionally shorten audio before Basic Pitch (env ACCOMONTAGE_BP_MAX_SECONDS, e.g. 240)."""
    if max_seconds <= 0:
        return src
    out = os.path.join(work_dir, "vocal_for_bp_trimmed.wav")
    ff = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ff,
        "-y",
        "-i",
        src,
        "-t",
        str(max_seconds),
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        out,
    ]
    try:
        _run(cmd, timeout=min(600, max_seconds + 120))
    except Exception as e:
        logging.warning("ACCOMONTAGE_BP_MAX_SECONDS trim failed, using full file: %s", e)
        return src
    return out if os.path.isfile(out) else src


def _run_demucs_vocals_stem(wav_path: str, out_root: str, timeout: int = 1800) -> tuple[str, str]:
    """Run Demucs htdemucs two-stem (vocals / no_vocals). Returns paths to each wav."""
    os.makedirs(out_root, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems",
        "vocals",
        "-n",
        "htdemucs",
        "-o",
        out_root,
        wav_path,
    ]
    _run(cmd, timeout=timeout)
    return _find_demucs_vocals_pair(out_root)


def youtube_url_to_midi_bytes(url: str, work_dir: str, use_vocal_only: bool = True) -> tuple[bytes, dict]:
    """
    Download audio with yt-dlp, isolate vocals with Demucs, run Basic Pitch on vocals.
    Returns (midi_bytes, hints). hints may include suggested_tonic and suggested_mode from
    the instrumental stem (Krumhansl-style chroma). If Demucs fails, falls back to the full
    mix for Basic Pitch and returns empty hints.
    work_dir: existing directory for temp files (session folder).
    """
    if not is_allowed_youtube_url(url):
        raise ValueError("Only http(s) YouTube / YouTube Music URLs are allowed.")

    os.makedirs(work_dir, exist_ok=True)
    audio_pattern = os.path.join(work_dir, "yt_audio.%(ext)s")

    ff_dir = _ffmpeg_location_dir()
    if not ff_dir:
        raise RuntimeError(
            "ffmpeg/ffprobe not found. Install FFmpeg (e.g. brew install ffmpeg) "
            "or set FFMPEG_LOCATION to the directory containing ffmpeg and ffprobe."
        )

    ytdlp_cmd = ["yt-dlp", "--no-playlist", "--ffmpeg-location", ff_dir]
    ytdlp_cmd.extend(_js_runtime_args())
    ytdlp_cmd.extend(
        [
            "-f",
            "bestaudio/best",
            "--extractor-args",
            "youtube:player_client=android,web",
            "-x",
            "--audio-format",
            "wav",
            "--audio-quality",
            "0",
            "-o",
            audio_pattern,
            url,
        ]
    )

    try:
        _run(ytdlp_cmd, timeout=600)
    except FileNotFoundError as e:
        raise RuntimeError(
            "yt-dlp is not installed. Install with: pip install yt-dlp"
        ) from e

    wav_path = os.path.join(work_dir, "yt_audio.wav")
    if not os.path.isfile(wav_path):
        # yt-dlp may keep original container
        for name in os.listdir(work_dir):
            if name.startswith("yt_audio.") and name.endswith((".wav", ".m4a", ".webm", ".opus", ".mp3")):
                wav_path = os.path.join(work_dir, name)
                break
        else:
            raise RuntimeError("Download finished but no audio file was found.")

    logging.info("youtube_melody: download complete, wav=%s", wav_path)

    hints: dict = {"melody_source": "full_mix"}
    melody_wav = wav_path
    force_skip_demucs = os.environ.get("ACCOMONTAGE_YOUTUBE_NO_DEMUCS", "").lower() in (
        "1",
        "true",
        "yes",
    )
    skip_demucs = force_skip_demucs or (not bool(use_vocal_only))
    if not bool(use_vocal_only):
        hints["melody_source"] = "full_mix_user_choice"
    if not skip_demucs and importlib.util.find_spec("demucs") is not None:
        sep_root = os.path.join(work_dir, "demucs_stems")
        try:
            logging.info("youtube_melody: starting Demucs (can take many minutes on CPU)…")
            vocals_path, no_vocals_path = _run_demucs_vocals_stem(wav_path, sep_root)
            melody_wav = vocals_path
            hints["melody_source"] = "vocal_stem"
            logging.info("youtube_melody: Demucs finished, vocals=%s", melody_wav)
            st, sm = _estimate_key_from_wav(no_vocals_path)
            if st and sm:
                hints["suggested_tonic"] = st
                hints["suggested_mode"] = sm
            beat_info = _beat_track_no_vocals(no_vocals_path)
            if beat_info:
                hints.update(beat_info)
        except Exception as e:
            logging.warning(
                "Vocal / instrumental separation failed; using full mix for melody (%s)", e
            )
            melody_wav = wav_path
            hints["melody_source"] = "full_mix_fallback"
            hints["melody_source_reason"] = str(e)[:200]
    elif not skip_demucs:
        hints["melody_source"] = "full_mix_no_demucs"
        logging.warning(
            "demucs is not installed (pip install demucs); using full mix for Basic Pitch"
        )

    # If we did not obtain beat hints from no_vocals, try coarse beat tracking on the current source.
    if hints.get("beat_bpm") is None:
        beat_info = _beat_track_no_vocals(melody_wav)
        if beat_info:
            hints.update(beat_info)

    try:
        max_bp = int(os.environ.get("ACCOMONTAGE_BP_MAX_SECONDS", "0") or 0)
    except ValueError:
        max_bp = 0
    if max_bp > 0:
        logging.info(
            "youtube_melody: trimming to first %s s for Basic Pitch (ACCOMONTAGE_BP_MAX_SECONDS)",
            max_bp,
        )
        melody_wav = _trim_wav_for_basic_pitch(melody_wav, work_dir, max_bp)

    from basic_pitch.inference import predict

    midi_tempo = 120.0
    try:
        bb = float(hints.get("beat_bpm")) if hints.get("beat_bpm") is not None else None
        if bb is not None and 30.0 <= bb <= 260.0:
            midi_tempo = bb
    except (TypeError, ValueError):
        pass

    logging.info(
        "youtube_melody: starting Basic Pitch on %s (midi_tempo=%.2f; long CPU step for full songs)",
        melody_wav,
        midi_tempo,
    )
    _, midi, _ = predict(melody_wav, midi_tempo=midi_tempo)
    logging.info("youtube_melody: Basic Pitch finished")

    if midi.instruments and midi.instruments[0].notes:
        notes = midi.instruments[0].notes
        notes.sort(key=lambda x: x.start)
        
        first_start = notes[0].start
        for n in notes:
            if n.end - n.start > 0.1:
                first_start = n.start
                break
        
        shift_amount = 0.0
        if hints.get("beat_times_sec"):
            beats = sorted(hints["beat_times_sec"])
            valid_beats = [b for b in beats if b <= first_start + 0.1]
            if valid_beats:
                shift_amount = valid_beats[-1]
            else:
                shift_amount = first_start
        else:
            shift_amount = first_start
            
        if shift_amount > 0.5:
            logging.info("youtube_melody: shifting melody by -%.2fs to remove initial silence and align beats", shift_amount)
            new_notes = []
            for n in notes:
                n.start -= shift_amount
                n.end -= shift_amount
                if n.end > 0:
                    n.start = max(0.0, n.start)
                    new_notes.append(n)
            midi.instruments[0].notes = new_notes
            
            if hints.get("beat_times_sec"):
                hints["beat_times_sec"] = [
                    b - shift_amount for b in hints["beat_times_sec"] if b - shift_amount >= -0.1
                ]

    buf = io.BytesIO()
    midi.write(buf)
    return buf.getvalue(), hints
