import io
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import logging
import time

import pretty_midi
from flask import Flask, request, send_from_directory, send_file, make_response, jsonify

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import chorderator as cdt
from Sessions import Sessions
from construct_midi_seg import construct_midi_seg, MIDI_FOLDER
import melody_analyze
import youtube_melody

app = Flask(__name__, static_url_path='')
app.secret_key = 'AccoMontage2-GUI'
saved_data = cdt.load_data()
APP_ROUTE = '/api/chorderator_back_end'
sessions = Sessions()
logging.basicConfig(level=logging.DEBUG)


def session_from_request(req):
    """(session, session_id) or (None, None) if cookie missing or unknown (never unpack bare get_session)."""
    p = sessions.get_session(req)
    return (None, None) if p is None else p


def resp(msg=None, session_id=None, more=()):
    body = {'status': 'ok' if not msg else msg}
    for item in more:
        body[item[0]] = item[1]
    r = make_response(jsonify(body))
    if session_id:
        r.set_cookie('session', session_id, max_age=3600)
    return r


def send_file_from_session(file, name=None):
    return send_file(
        io.BytesIO(file),
        as_attachment=True,
        download_name=name,
    )


def begin_generate_thread(core, session_id):
    try:
        core.generate_save(session_id, log=True)
    except Exception as e:
        logging.exception("generate_save failed session=%s", session_id)
        if session_id in sessions.sessions:
            sessions.sessions[session_id].generate_error = (
                f"{type(e).__name__}: {e}. "
                "Often caused by a phrase slice length the chord library does not cover "
                "(e.g. 12 bars in the last segment). Use even 8-bar phrases such as A8B8A8B8, "
                "or trim the MIDI grid."
            )[:900]
        try:
            core.state = 7
        except Exception:
            pass


@app.route(APP_ROUTE + '/upload_melody', methods=['POST'])
def upload_melody():
    if sessions.get_session(request) is None:
        session, session_id = sessions.create_session()
        logging.debug('create new session {}'.format(session_id))
    else:
        session, session_id = sessions.get_session(request)
        logging.debug('request is in session {}'.format(session_id))
    session.melody = request.files['midi'].read()
    analysis = melody_analyze.analyze_melody_bytes(session.melody, key_hints=None)
    return resp(session_id=session_id, more=melody_analyze.build_response_more({}, analysis))


@app.route(APP_ROUTE + '/upload_youtube', methods=['POST'])
def upload_youtube():
    data = request.get_json(silent=True) or {}
    url = (data.get('url') or '').strip()
    use_vocal_only = bool(data.get('use_vocal_only', True))
    if not url:
        return resp(msg='missing url')
    if sessions.get_session(request) is None:
        session, session_id = sessions.create_session()
        logging.debug('create new session {}'.format(session_id))
    else:
        session, session_id = sessions.get_session(request)
        logging.debug('request is in session {}'.format(session_id))
    work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), session_id)
    try:
        midi_bytes, hints = youtube_melody.youtube_url_to_midi_bytes(url, work_dir, use_vocal_only=use_vocal_only)
        session.melody = midi_bytes
    except Exception as e:
        logging.exception('upload_youtube failed')
        return resp(msg=str(e)[:900])
    analysis = melody_analyze.analyze_melody_bytes(session.melody, key_hints=hints)
    if analysis.get("detected_tempo") is not None:
        session.tempo = analysis.get("detected_tempo")
    return resp(
        session_id=session_id,
        more=melody_analyze.build_response_more(hints, analysis),
    )


@app.route(APP_ROUTE + '/generate', methods=['POST'])
def generate():
    session, session_id = session_from_request(request)
    if session is None:
        return resp(msg='session expired')
    if not session.melody:
        return resp(msg='load melody first (use YouTube Load melody on the previous step)')
    logging.debug('request is in session {}'.format(session_id))
    params = json.loads(request.data)
    session.load_params(params)
    if not session.tempo:
        session.tempo = 120
    session.generate_error = None

    logging.info(
        'generate meta tonic=%s mode=%s meter=%s tempo=%s segmentation=%s',
        session.tonic,
        session.mode,
        session.meter,
        session.tempo,
        session.segmentation,
    )

    session.core = cdt.get_chorderator()
    session.core.set_cache(**saved_data)
    os.makedirs(session_id, exist_ok=True)
    with open(f'{session_id}/melody.mid', 'wb') as f:
        f.write(session.melody)
    session.core.set_melody(f'{session_id}/melody.mid')
    session.core.set_output_style(session.chord_style)
    session.core.set_texture_prefilter(session.texture_style)
    session.core.set_meta(tonic=session.tonic, meter=session.meter, mode=session.mode, tempo=session.tempo)
    session.core.set_segmentation(session.segmentation)
    threading.Thread(target=begin_generate_thread, args=(session.core, session_id)).start()
    return resp(session_id=session_id)


@app.route(APP_ROUTE + '/stage_query', methods=['GET'])
def answer_stage():
    session, session_id = session_from_request(request)
    if session is None:
        return resp(msg='session expired')
    logging.debug('request is in session {}'.format(session_id))
    if getattr(session, "generate_error", None):
        return resp(
            session_id=session_id,
            more=[
                ["stage", "0"],
                ["generate_error", session.generate_error],
            ],
        )
    return resp(session_id=session_id, more=[['stage', str(session.core.get_state())]])


@app.route(APP_ROUTE + '/generated_query', methods=['GET'])
def answer_gen():
    session, session_id = session_from_request(request)
    if session is None:
        return resp(msg='session expired')
    logging.debug('request is in session {}'.format(session_id))
    for file in os.listdir(session_id):
        if file == 'chord_gen_log.json':
            session.generate_log = json.load(open(session_id + '/chord_gen_log.json', 'r'))
        elif file == 'textured_chord_gen.mid':
            with open(session_id + '/textured_chord_gen.mid', 'rb') as f:
                session.generate_midi = f.read()
        elif file == 'textured_chord_gen.wav':
            with open(session_id + '/textured_chord_gen.wav', 'rb') as f:
                session.generate_wav = f.read()
    session.generate_midi_seg = construct_midi_seg(session, session_id)
    chord_midi_name = session_id + '_' + str(time.time()) + '_chord_gen.mid'
    acc_midi_name = session_id + '_' + str(time.time()) + '_textured_chord_gen.mid'
    pretty_midi.PrettyMIDI(session_id + '/chord_gen.mid').write(MIDI_FOLDER + '/' + chord_midi_name)
    # shutil.copy(session_id + '/chord_gen.mid', MIDI_FOLDER + chord_midi_name)
    pretty_midi.PrettyMIDI(session_id + '/textured_chord_gen.mid').write(MIDI_FOLDER + '/' + acc_midi_name)
    # shutil.copy(session_id + '/textured_chord_gen.mid', MIDI_FOLDER + acc_midi_name)
    shutil.rmtree(session_id)
    new_log = []
    for i in range(len(session.generate_log)):
        new_log.append(session.generate_log[i])
        new_log[i]['midi_name'] = session.generate_midi_seg[i]
    return resp(session_id=session_id,
                more=[['log', new_log], ['chord_midi_name', chord_midi_name], ['acc_midi_name', acc_midi_name]])


@app.route(APP_ROUTE + '/wav/<ran>', methods=['GET'])
def wav(ran):
    session, session_id = session_from_request(request)
    if session is None:
        return resp(msg='session expired')
    logging.debug('request is in session {}'.format(session_id))
    return send_file_from_session(session.generate_wav, 'accomontage2.wav')


@app.route(APP_ROUTE + '/midi/<ran>', methods=['GET'])
def midi(ran):
    # Serve the exact rendered MIDI requested by filename from static/midi.
    # This avoids returning a single cached in-memory MIDI for every download link.
    safe_name = os.path.basename(ran)
    if not safe_name.endswith('.mid'):
        return resp(msg='invalid midi name')
    midi_path = os.path.join(MIDI_FOLDER, safe_name)
    if not os.path.isfile(midi_path):
        return resp(msg='midi not found')
    return send_from_directory(MIDI_FOLDER, safe_name, as_attachment=True, download_name=safe_name)


@app.route(APP_ROUTE + '/midi-seg/<idx>', methods=['GET'])
def midi_seg(idx):
    session, session_id = session_from_request(request)
    if session is None:
        return resp(msg='session expired')
    logging.debug('request is in session {}'.format(session_id))
    return send_file_from_session(session.generate_midi_seg[idx], f'accomontage2-{idx}.mid')


@app.errorhandler(404)
def index(error):
    return make_response(send_from_directory('static', 'index.html'))


def _kill_python_listeners_on_port(port):
    """SIGTERM stale Flask/Python listeners so restart works. Skips non-Python (e.g. AirPlay)."""
    try:
        r = subprocess.run(
            ['lsof', '-i', f':{port}', '-sTCP:LISTEN', '-n', '-P', '-t'],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return
    for pid_s in r.stdout.strip().split():
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        try:
            comm = subprocess.check_output(
                ['ps', '-p', str(pid), '-o', 'comm='],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            continue
        base = os.path.basename((comm or '').split()[0]).lower()
        if 'python' not in base:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            logging.info('stopped prior Python listener pid=%s on port %s', pid, port)
            time.sleep(0.25)
        except ProcessLookupError:
            pass
        except PermissionError:
            logging.warning('could not stop pid=%s on port %s (permission)', pid, port)


if __name__ == '__main__':
    # Default avoids macOS AirPlay Receiver (listens on :5000; returns HTTP 403 as AirTunes).
    port = int(os.environ.get('PORT', '8765'))
    host = os.environ.get('HOST', '127.0.0.1')
    if sys.platform == 'darwin' and port == 5000:
        logging.warning(
            'macOS: port 5000 is often taken by AirPlay (AirTunes) and returns HTTP 403 in the browser. '
            'Prefer the default port 8765, or use http://127.0.0.1:5000/ only if AirPlay is off, '
            'or set PORT to another free port.',
        )
    if os.environ.get('SKIP_FREE_PORT', '').lower() not in ('1', 'true', 'yes'):
        _kill_python_listeners_on_port(port)
    app.run(host=host, port=port)
