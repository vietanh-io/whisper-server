# whisper-server (Developer-first Faster-Whisper API)

FastAPI service wrapping [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for production and development transcription workflows.

## Features

- Check available models and download model weights on demand
- Direct uploads (`multipart/form-data`) for single or multiple files
- Remote links (`application/json`) with arrays of HTTP/S3/YouTube URLs
- Transcribe by streaming input through `ffmpeg` into normalized 16 kHz mono WAV
- Batch long media by chunking and merging transcripts automatically
- Both tasks: `transcribe` (keep source language) and `translate` (to English)
- Per-request model/runtime controls: `model`, `device`, `compute_type`, `beam_size`, VAD, word timestamps
- Per-request batching mode: `batch_mode=auto|on|off`
- Per-request Silero VAD mode: `vad_mode=auto|on|off`
- Advanced decoding: `temperature` fallback, `best_of`, `patience`, `length_penalty`, `repetition_penalty`, `no_repeat_ngram_size`
- Quality thresholds: `compression_ratio_threshold`, `log_prob_threshold`, `no_speech_threshold`
- Context/prompting: `condition_on_previous_text`, `initial_prompt`, `hotwords`, `prefix`
- Hallucination control: `hallucination_silence_threshold`, `suppress_blank`
- CLI client (`scripts/client.py`) with full parameter support

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

## Project structure

```
whisper-server/
  app/
    main.py                  # FastAPI app factory, lifespan, global singletons
    api/router.py            # Top-level router that mounts all sub-routers
    core/config.py           # Pydantic settings loaded from .env
    transcription/
      schemas.py             # Request/response Pydantic models
      service.py             # Whisper inference logic and output file writing
      media.py               # ffmpeg transcoding, chunking, workspace management
      router.py              # HTTP endpoints (transcribe, models, outputs)
      dependencies.py        # FastAPI dependency injection helpers
  scripts/
    client.py                # CLI client for the server
  Dockerfile                 # Container build
  .env.example               # All available env vars with defaults
  requirements.txt           # Python dependencies
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server health check (includes ffmpeg status) |
| `GET` | `/models` | List available and downloaded model names |
| `POST` | `/models/download` | Download a model by name |
| `POST` | `/transcribe` | Transcribe a single file upload or remote URL |
| `POST` | `/transcribe/uploads` | Transcribe multiple file uploads |
| `POST` | `/transcribe/links` | Transcribe a JSON array of remote URLs |
| `GET` | `/outputs/{job_id}/{filename}` | Download a transcript output file |

## API examples

### Single file upload

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

### Multiple file uploads

```bash
curl -X POST "http://127.0.0.1:8000/transcribe/uploads" \
  -F "files=@a.mp4" \
  -F "files=@b.mp4" \
  -F "source_language=ja" \
  -F "task=translate" \
  -F "batch_size=8"
```

### Remote links (HTTP/S3/YouTube)

```bash
curl -X POST "http://127.0.0.1:8000/transcribe/links" \
  -H "Content-Type: application/json" \
  -d '{
    "links": [
      "https://example.com/media.mp4",
      "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    ],
    "source_language": "ko",
    "task": "transcribe",
    "model": "small",
    "device": "cpu",
    "compute_type": "int8",
    "batch_size": 8,
    "condition_on_previous_text": false,
    "hotwords": "AI transformer attention"
  }'
```

## CLI client (`scripts/client.py`)

A command-line client that talks to the running server. Supports all transcription parameters.

### Basic usage

```bash
# Transcribe a local file
python scripts/client.py --file recording.mp4

# Transcribe from a URL
python scripts/client.py --media-url "https://example.com/audio.mp3"

# Translate Japanese audio to English
python scripts/client.py --file interview.wav --task translate --source-language ja
```

### Model and device options

```bash
python scripts/client.py --file audio.mp3 \
  --model large-v3 \
  --device cuda \
  --compute-type float16
```

### Advanced decoding options

```bash
# Greedy decoding (temperature=0, no fallback)
python scripts/client.py --file audio.mp3 --temperature 0.0

# Custom temperature fallback sequence
python scripts/client.py --file audio.mp3 --temperature "0.0,0.4,0.8"

# Reduce hallucination by disabling context carry-over
python scripts/client.py --file audio.mp3 --no-condition-on-previous-text

# Guide transcription with domain-specific vocabulary
python scripts/client.py --file meeting.mp3 \
  --initial-prompt "Meeting about Kubernetes, Docker, CI/CD pipelines" \
  --hotwords "Kubernetes Docker Jenkins"
```

### VAD and batching options

```bash
# Force VAD on, disable batching
python scripts/client.py --file audio.mp3 --vad-mode on --batch-mode off

# Force batching on with custom batch size
python scripts/client.py --file long_podcast.mp3 --batch-mode on --batch-size 16
```

### Output format options

```bash
# Get all three output formats
python scripts/client.py --file audio.mp3 --output-formats txt,srt,json

# Enable word-level timestamps
python scripts/client.py --file audio.mp3 --word-timestamps --output-formats json
```

### Full CLI reference

```
python scripts/client.py [OPTIONS]

Required (one of):
  --file PATH              Local audio/video file to transcribe
  --media-url URL          Remote URL to transcribe (HTTP, S3, YouTube)

Connection:
  --base-url URL           Server URL (default: http://127.0.0.1:8000)
  --timeout SECONDS        Request timeout (default: 3600)

Task:
  --task {transcribe,translate}   Transcribe or translate to English
  --source-language LANG          ISO language code (e.g. en, vi, ja)
  --output-formats FORMATS        Comma-separated: txt,srt,json

Model:
  --model NAME             Model name (e.g. tiny, small, medium, large-v3)
  --device {cpu,cuda,auto}
  --compute-type {int8,float16,int8_float16,...}

Decoding:
  --beam-size N                         Beam search width (default: 5)
  --temperature FLOATS                  Comma-separated fallback temps (default: 0.0,0.2,...,1.0)
  --best-of N                           Candidates when temperature > 0 (default: 5)
  --patience FLOAT                      Beam patience factor (default: 1.0)
  --length-penalty FLOAT                Length penalty (default: 1.0)
  --repetition-penalty FLOAT            Repetition penalty (default: 1.0)
  --no-repeat-ngram-size N              Prevent n-gram repeats (default: 0 = off)

Quality thresholds:
  --compression-ratio-threshold FLOAT   Max compression ratio (default: 2.4)
  --log-prob-threshold FLOAT            Min avg log-prob (default: -1.0)
  --no-speech-threshold FLOAT           No-speech skip threshold (default: 0.6)

Context and prompting:
  --condition-on-previous-text / --no-condition-on-previous-text  (default: on)
  --initial-prompt TEXT                 Prompt to guide style/vocabulary
  --prompt-reset-on-temperature FLOAT   Reset prompt above this temp (default: 0.5)
  --hotwords TEXT                       Boost specific words/phrases
  --prefix TEXT                         Prefix for first window

Hallucination and token control:
  --hallucination-silence-threshold FLOAT   Skip silent hallucinations (seconds)
  --suppress-blank / --no-suppress-blank    (default: on)
  --without-timestamps / --no-without-timestamps  (default: off)
  --max-initial-timestamp FLOAT             Max first timestamp (default: 1.0)
  --max-new-tokens N                        Max tokens per chunk

VAD:
  --vad-filter / --no-vad-filter            (default: on)
  --vad-threshold FLOAT                     (default: 0.5)
  --min-silence-duration-ms MS              (default: 500)
  --speech-pad-ms MS                        (default: 400)
  --vad-mode {auto,on,off}                  (default: auto)

Batching:
  --batch-mode {auto,on,off}                (default: auto)
  --batch-size N                            (default: 8)
  --use-batch / --no-use-batch              Override batch decision
  --word-timestamps / --no-word-timestamps  (default: off)
```

## Docker

```bash
docker build -t whisper-server:latest .
docker run --rm -p 8000:8000 --env-file .env whisper-server:latest
```

For GPU support, use the NVIDIA runtime:

```bash
docker run --rm --gpus all -p 8000:8000 --env-file .env whisper-server:latest
```

## Configuration

All settings can be overridden via environment variables or a `.env` file. See [.env.example](.env.example) for the full list.

### Task behavior

- `task=transcribe`: keep original source language.
- `task=translate`: translate to English (Whisper only supports translation to English).
- If `task=translate` and source is English, the API returns `400` (configurable via `REJECT_ENGLISH_SOURCE_ON_TRANSLATE`).

### Batching behavior

- `batch_mode=auto`: use server default + long-media threshold (`BATCH_THRESHOLD_SECONDS`).
- `batch_mode=on`: force batching (splits audio into `BATCH_CHUNK_SECONDS` chunks).
- `batch_mode=off`: disable batching for that request.

### VAD behavior

- `vad_mode=auto`: follow the request-level `vad_filter` toggle.
- `vad_mode=on`: force Silero VAD on (removes silence before transcription).
- `vad_mode=off`: force Silero VAD off.

## Advanced transcription parameters

All parameters below can be sent per-request (form field or JSON body) and have server-level defaults via env vars.

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
| `prefix` | string or null | `null` | Text prefix for the first window (sequential only) |
| `hallucination_silence_threshold` | float or null | `null` | Skip silent periods to avoid hallucination (sequential only) |
| `suppress_blank` | bool | `true` | Suppress blank outputs at sampling start |
| `without_timestamps` | bool | `false` | Only sample text tokens |
| `max_initial_timestamp` | float | `1.0` | Max allowed initial timestamp in seconds (sequential only) |
| `max_new_tokens` | int or null | `null` | Max tokens per chunk (sequential only) |

Parameters marked "sequential only" are ignored when batched inference is active.
