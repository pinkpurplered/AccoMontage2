# Piano Accompaniment Generator

Turn any YouTube song into a piano accompaniment MIDI automatically.

## Quick Start

1. Install backend requirements:
```bash
cd back-end
pip install -r requirements.txt
```

2. Start the backend server:
```bash
python app.py
```

3. Open your browser and go to `http://localhost:8765`.

## Development

To edit the React UI:
```bash
cd front-end
npm install
npm start
```

To build the UI for production (copies it to the backend):
```bash
./build.sh
```