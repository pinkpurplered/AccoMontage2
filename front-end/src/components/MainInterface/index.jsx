import React, { useState, useEffect, useRef } from 'react';
import { Typography, Input, Button, message, Progress, Card } from 'antd';
import { YoutubeOutlined, DownloadOutlined, CheckCircleFilled, SyncOutlined } from '@ant-design/icons';
import axios from 'axios';
import { myServer, myRoot } from '../../utils';
import Icon from '../Icon';
import './index.css';

const { Title, Text } = Typography;
const YT_UPLOAD_TIMEOUT_MS = 50 * 60 * 1000;

const statusText = [
    'Preparing...',
    'Extracting melody & analyzing meta...',
    'Constructing chord progressions...',
    'Refining progressions...',
    'Generating textures...',
    'Synthesizing...',
    'Complete!'
];

export default function MainInterface() {
    const [url, setUrl] = useState('');
    const [status, setStatus] = useState('idle'); // idle, processing, done, error
    const [stage, setStage] = useState(0);
    const [errorMsg, setErrorMsg] = useState(null);
    const [chordName, setChordName] = useState(null);
    const [accName, setAccName] = useState(null);
    const intervalRef = useRef(null);

    useEffect(() => {
        return () => {
            if (intervalRef.current) clearInterval(intervalRef.current);
        };
    }, []);

    const onGenerate = () => {
        const u = url.trim();
        if (!u) {
            message.warn('Please enter a YouTube URL.');
            return;
        }

        setStatus('processing');
        setStage(0);
        setErrorMsg(null);

        axios.post(`${myServer}/upload_youtube`, { url: u, use_vocal_only: true }, {
            withCredentials: true,
            timeout: YT_UPLOAD_TIMEOUT_MS,
        })
        .then((res) => {
            if (res.data && res.data.status === 'ok') {
                startGeneration(res.data);
            } else {
                setStatus('error');
                setErrorMsg(res.data?.status || 'YouTube load failed');
            }
        })
        .catch((e) => {
            setStatus('error');
            setErrorMsg(e.message || 'Request failed');
        });
    };

    const startGeneration = (d) => {
        const phrases = (d.auto_phrases && d.auto_phrases.length) ? d.auto_phrases : [{ phrase_name: 'A', phrase_length: 8 }];
        const tonic = d.suggested_tonic || 'C';
        const mode = d.suggested_mode || 'maj';
        const tempo = d.detected_tempo ? Math.max(30, Math.min(260, Math.round(Number(d.detected_tempo)))) : 120;

        const payload = {
            tonic, mode, tempo, meter: '4/4', phrases,
            chord_style: 'pop_standard', rhythm_density: 2, voice_number: 2,
            enable_texture_style: true, enable_chord_style: true, use_vocal_only: true
        };

        axios.post(`${myServer}/generate`, payload, { withCredentials: true })
            .then(() => {
                intervalRef.current = setInterval(askStage, 2000);
            })
            .catch(e => {
                setStatus('error');
                setErrorMsg('Failed to start generation');
            });
    };

    const askStage = () => {
        axios.get(`${myServer}/stage_query`, { withCredentials: true })
            .then(res => {
                const data = res.data;
                if (data.status !== 'ok') return;
                
                if (data.generate_error) {
                    clearInterval(intervalRef.current);
                    setStatus('error');
                    setErrorMsg(data.generate_error);
                    return;
                }
                
                const currentStage = parseInt(data.stage || 0);
                setStage(currentStage);
                
                if (currentStage === 7) {
                    clearInterval(intervalRef.current);
                    fetchResults();
                }
            })
            .catch(() => {
                // Ignore transient errors
            });
    };

    const fetchResults = () => {
        axios.get(`${myServer}/generated_query`, { withCredentials: true })
            .then(res => {
                if (res.data.status === 'ok') {
                    setChordName(res.data.chord_midi_name);
                    setAccName(res.data.acc_midi_name);
                    setStatus('done');
                } else {
                    setStatus('error');
                    setErrorMsg('Failed to fetch generated files');
                }
            })
            .catch(() => {
                setStatus('error');
                setErrorMsg('Failed to fetch generated files');
            });
    };

    return (
        <div className="app-container">
            <div className="main-card">
                <Title level={1} className="title-gradient">Piano Accompaniment Generator</Title>
                <div className="subtitle">Turn any YouTube song into a piano accompaniment MIDI automatically.</div>
                
                <Input.Search
                    className="yt-input"
                    size="large"
                    placeholder="Paste YouTube URL here..."
                    enterButton="Generate"
                    value={url}
                    onChange={e => setUrl(e.target.value)}
                    onSearch={onGenerate}
                    loading={status === 'processing'}
                    disabled={status === 'processing'}
                    prefix={<YoutubeOutlined style={{ color: '#ff0000', fontSize: '20px', marginRight: '8px' }} />}
                />

                {status === 'processing' && (
                    <div className="status-container">
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                            <Text strong>{statusText[Math.min(stage, 6)]}</Text>
                            <Text type="secondary">{Math.round((stage / 6) * 100)}%</Text>
                        </div>
                        <Progress percent={Math.round((stage / 6) * 100)} status="active" strokeColor={{ '0%': '#108ee9', '100%': '#87d068' }} />
                        <Text type="secondary" style={{ display: 'block', marginTop: '15px', fontSize: '13px' }}>
                            {stage === 0 ? "Downloading and isolating vocals (this may take a few minutes)..." : "AI is composing your accompaniment..."}
                        </Text>
                    </div>
                )}

                {status === 'error' && (
                    <div className="status-container" style={{ background: '#fff1f0', border: '1px solid #ffa39e' }}>
                        <Title level={4} style={{ color: '#cf1322', marginTop: 0 }}>Generation Failed</Title>
                        <Text style={{ color: '#cf1322' }}>{errorMsg}</Text>
                        <div style={{ marginTop: '15px' }}>
                            <Button onClick={() => setStatus('idle')} type="primary" danger>Try Again</Button>
                        </div>
                    </div>
                )}

                {status === 'done' && (
                    <div className="result-container">
                        <a href={`${myRoot}/midi/${chordName}`} download="chord.mid" style={{ flex: 1, textDecoration: 'none' }}>
                            <Card className="download-card" bodyStyle={{ padding: '30px 20px' }}>
                                <DownloadOutlined style={{ fontSize: '32px', color: '#1890ff', marginBottom: '15px' }} />
                                <Title level={4} style={{ margin: '0 0 10px 0' }}>Download Chords</Title>
                                <Text type="secondary">Melody + Block Chords</Text>
                            </Card>
                        </a>
                        <a href={`${myRoot}/midi/${accName}`} download="accompaniment.mid" style={{ flex: 1, textDecoration: 'none' }}>
                            <Card className="download-card" bodyStyle={{ padding: '30px 20px' }}>
                                <DownloadOutlined style={{ fontSize: '32px', color: '#722ed1', marginBottom: '15px' }} />
                                <Title level={4} style={{ margin: '0 0 10px 0' }}>Download Accompaniment</Title>
                                <Text type="secondary">Melody + Textured Piano</Text>
                            </Card>
                        </a>
                    </div>
                )}
                
                {status === 'done' && (
                    <Button type="link" onClick={() => { setStatus('idle'); setUrl(''); }} style={{ marginTop: '30px' }}>
                        Generate Another Song
                    </Button>
                )}
            </div>
        </div>
    );
}