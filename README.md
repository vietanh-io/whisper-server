# whisper-server (Developer-first Faster-Whisper API)

FastAPI service focused on `faster-whisper` for production/dev workflows.

## What it supports

- Check available models and download model weights on demand
- Direct uploads (`multipart/form-data`) for single or multiple files
- Remote links (`application/json`) with arrays of HTTP/S3/YouTube links
- Transcribe by streaming input through `ffmpeg` into normalized audio
- Batch long media by chunking and merging transcripts
- Supports both tasks: `transcribe` and `translate`
- Per-request model/runtime controls: `model`, `device`, `compute_type`, `beam_size`, VAD, word timestamps
- Per-request batching mode: `batch_mode=auto|on|off`
- Per-request Silero VAD mode: `vad_mode=auto|on|off`
- Advanced decoding: `temperature` fallback, `best_of`, `patience`, `length_penalty`, `repetition_penalty`, `no_repeat_ngram_size`
- Quality thresholds: `compression_ratio_threshold`, `log_prob_threshold`, `no_speech_threshold`
- Context/prompting: `condition_on_previous_text`, `initial_prompt`, `hotwords`, `prefix`
- Hallucination control: `hallucination_silence_threshold`, `suppress_blank`

## Requirements

- Python 3.10+
- FFmpeg + ffprobe in PATH

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## Main endpoints

- `GET /health`
- `GET /models`
- `POST /models/download`
- `POST /transcribe` (single upload or single remote link)
- `POST /transcribe/uploads` (multiple uploads)
- `POST /transcribe/links` (JSON list of links)
- `GET /outputs/{job_id}/{filename}`

## Quick examples

### 1) Single file upload

```bash
curl -X POST "http://127.0.0.1:8000/transcribe" \
  -F "file=@sample.mp4" \
  -F "source_language=vi" \
  -F "model=small" \
  -F "device=cpu" \
  -F "compute_type=int8" \
  -F "batch_mode=off" \
  -F "vad_mode=on" \
  -F "task=transcribe" \
  -F "output_formats=txt,srt,json" \
  -F "temperature=0.0" \
  -F "condition_on_previous_text=false" \
  -F "initial_prompt=Technical meeting about AI"
```

### 2) Multiple file uploads

```bash
curl -X POST "http://127.0.0.1:8000/transcribe/uploads" \
  -F "files=@a.mp4" \
  -F "files=@b.mp4" \
  -F "source_language=ja" \
  -F "task=translate" \
  -F "batch_size=8"
```

### 3) Remote links JSON (HTTP/S3/YouTube)

```bash
curl -X POST "http://127.0.0.1:8000/transcribe/links" \
  -H "Content-Type: application/json" \
  -d '{
    "links": [
      "https://example.com/media.mp4",
      "s3://bucket/path/video.mp4",
      "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    ],
    "source_language": "ko",
    "task": "transcribe",
    "model": "small",
    "device": "cpu",
    "compute_type": "int8",
    "batch_size": 8
  }'
```

## Docker

```bash
docker build -t whisper-server:latest .
docker run --rm -p 8000:8000 --env-file .env whisper-server:latest
```

## Notes

- `task=transcribe`: keep original source language.
- `task=translate`: translate to English.
- If `task=translate` and source is English, the API returns `400` (configurable).
- `batch_mode=auto`: use server default + long-media threshold.
- `batch_mode=on`: force batching.
- `batch_mode=off`: disable batching for that request.
- `vad_mode=auto`: follow request/default VAD toggle.
- `vad_mode=on`: force Silero VAD on.
- `vad_mode=off`: force Silero VAD off.
- For long media, chunking behavior is controlled by:
  - `BATCH_ENABLED`
  - `BATCH_THRESHOLD_SECONDS`
  - `BATCH_CHUNK_SECONDS`

## Advanced transcription parameters

All parameters below can be sent per-request (form field or JSON) and have server-level defaults via env vars.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | comma-separated floats | `0.0,0.2,0.4,0.6,0.8,1.0` | Temperature fallback sequence. Single `0.0` = greedy decoding. |
| `best_of` | int | `5` | Candidates when sampling with temperature > 0 |
| `patience` | float | `1.0` | Beam search patience factor |
| `length_penalty` | float | `1.0` | Exponential length penalty |
| `repetition_penalty` | float | `1.0` | Penalize repeated tokens |
| `no_repeat_ngram_size` | int | `0` | Prevent n-gram repetitions (0 = disabled) |
| `compression_ratio_threshold` | float or null | `2.4` | Max gzip compression ratio before fallback |
| `log_prob_threshold` | float or null | `-1.0` | Min avg log probability before fallback |
| `no_speech_threshold` | float or null | `0.6` | Skip segment when no-speech prob exceeds this |
| `condition_on_previous_text` | bool | `true` | Use previous segment as context. Set `false` to reduce hallucination propagation. |
| `initial_prompt` | string or null | `null` | Guide transcription style/vocabulary |
| `prompt_reset_on_temperature` | float | `0.5` | Reset prompt when fallback temperature exceeds this |
| `hotwords` | string or null | `null` | Boost specific words/phrases |
| `prefix` | string or null | `null` | Text prefix for the first window (sequential mode only) |
| `hallucination_silence_threshold` | float or null | `null` | Skip silent periods to avoid hallucination (sequential mode only) |
| `suppress_blank` | bool | `true` | Suppress blank outputs at sampling start |
| `without_timestamps` | bool | `false` | Only sample text tokens |
| `max_initial_timestamp` | float | `1.0` | Max allowed initial timestamp (seconds, sequential mode only) |
| `max_new_tokens` | int or null | `null` | Max tokens per chunk (sequential mode only) |

Parameters marked "sequential mode only" are ignored when batched inference is active.
