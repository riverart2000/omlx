# oMLX Video Polish

A private, local video-finishing project for oMLX. It imports one or more
videos, trims and stitches them, balances light and colour, isolates speech,
removes background noise, corrects programme loudness, and uses timed speech
transcription plus a local oMLX language model to suggest filler-word and
false-start cuts.

## What is modular

- `backend/speech.py` — MLX Whisper word timing and deterministic edit detection
- `backend/llm_editor.py` — conservative false-start review through local oMLX
- `backend/audio.py` — MLX DeepFilterNet isolation and FFmpeg dialogue mastering
- `backend/media.py` — probing, video filters, range logic and encoding profiles
- `backend/render_pipeline.py` — trim, stitch, transition and loudness stages
- `backend/storage.py` — durable local upload records
- `backend/jobs.py` — persistent background job state and diagnostics
- `server.py` — a thin local HTTP/API layer
- `static/` — the self-contained editor interface

Each ML task runs separately from the web service. DeepFilterNet failure falls
back to spectral noise reduction, and VideoToolbox failure retries with a
software encoder. Job errors are written to `data/jobs/<job-id>.error.log`.

## Local services and models

- UI/API: `http://127.0.0.1:8950`
- FFmpeg and VideoToolbox for image processing, transitions and export
- `mlx-community/DeepFilterNet-mlx` for voice isolation
- `mlx-community/whisper-large-v3-turbo` for word timestamps
- the configured oMLX chat model on port 8000 for false-start decisions

All footage and output remain on this Mac. Finished files are written to
`~/Movies/oMLX Video Polish`.

## Service controls

```bash
bash ~/omlx-video-polish/video-polish.sh start
bash ~/omlx-video-polish/video-polish.sh status
bash ~/omlx-video-polish/video-polish.sh logs
bash ~/omlx-video-polish/video-polish.sh restart
```

The included LaunchAgent starts the service at login and restarts it if it
exits unexpectedly.

## Tests

```bash
cd ~/omlx-video-polish
python3 -m unittest discover -s tests -v
```

