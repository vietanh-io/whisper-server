# whisper-server (Local FastAPI + Silero VAD)

Local-first transcription server using FastAPI with two selectable backends:
- `faster-whisper` (supports Silero VAD)
- `openai-whisper` (`whisper` package)

## Features

- `POST /transcribe`: accepts file upload or `media_url`
- `GET /health`: service health check
- Select backend per request (`backend=faster-whisper|whisper`)
- Built-in Silero VAD for `faster-whisper` backend (`vad_filter=True`)
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

Most defaults are configured via `.env` (backend, model, task, output formats, VAD defaults).

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

### 4) Choose backend per request

```powershell
python scripts\client.py --file "C:\path\to\audio.wav" --backend whisper
python scripts\client.py --file "C:\path\to\audio.wav" --backend faster-whisper
```

## Direct curl-like request (multipart)

```bash
curl -X POST "http://127.0.0.1:8000/transcribe" \
  -F "file=@sample.wav" \
  -F "backend=faster-whisper" \
  -F "task=transcribe" \
  -F "output_formats=txt,srt,json" \
  -F "vad_filter=true" \
  -F "vad_threshold=0.5" \
  -F "min_silence_duration_ms=500" \
  -F "speech_pad_ms=400"
```

## Notes

- For `faster-whisper`, tune `FASTER_WHISPER_*` vars in `.env`.
- For `whisper`, tune `WHISPER_*` vars in `.env`.
- On CPU, use `small` or `tiny` models for faster iteration.
- If FFmpeg is missing, `/transcribe` returns a clear `503`.
