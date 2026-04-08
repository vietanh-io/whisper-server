# whisper-server (Local FastAPI + Silero VAD)

Local-first transcription server using FastAPI and `faster-whisper`.

## Features

- `POST /transcribe`: accepts file upload or `media_url`
- `GET /health`: service health check
- Built-in Silero VAD via `faster-whisper` (`vad_filter=True`)
- Output persistence per job (`txt`, `srt`, optional `json`)
- Download output files via `GET /outputs/{job_id}/{filename}`

## Requirements

- Python 3.10+
- FFmpeg available in PATH

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

## Run server

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## API usage

### 1) Transcribe local file

```powershell
python scripts\client.py --file "C:\path\to\video.mp4"
```

### 2) Transcribe from URL

```powershell
python scripts\client.py --media-url "https://example.com/media.mp4"
```

### 3) Tune VAD

```powershell
python scripts\client.py --file "C:\path\to\audio.wav" --vad-threshold 0.45 --min-silence-duration-ms 700 --speech-pad-ms 450
```

## Direct curl-like request (multipart)

```bash
curl -X POST "http://127.0.0.1:8000/transcribe" \
  -F "file=@sample.wav" \
  -F "task=transcribe" \
  -F "output_formats=txt,srt,json" \
  -F "vad_filter=true" \
  -F "vad_threshold=0.5" \
  -F "min_silence_duration_ms=500" \
  -F "speech_pad_ms=400"
```

## Notes

- On CPU, use `WHISPER_MODEL=small` or `medium` for reasonable speed.
- For GPU (later), set `WHISPER_DEVICE=cuda` and a suitable compute type.
- If FFmpeg is missing, service startup will fail with a clear error.
