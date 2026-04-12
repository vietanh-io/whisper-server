# whisper-server (Developer-first Faster-Whisper API)

FastAPI service wrapping [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for production and development transcription workflows, with built-in translation via [argostranslate](https://github.com/argosopentech/argos-translate).

## Features

- Check available models and download model weights on demand
- Direct uploads (`multipart/form-data`) for single or multiple files
- Remote links (`application/json`) with arrays of HTTP/S3/YouTube URLs
- Transcribe by streaming input through `ffmpeg` into normalized 16 kHz mono WAV
- Batch long media by chunking and merging transcripts automatically
- Both tasks: `transcribe` (keep source language) and `translate` (to any target language via argostranslate)
- Per-request model/runtime controls: `model`, `device`, `compute_type`, `beam_size`, VAD, word timestamps
- Per-request batching mode: `batch_mode=auto|on|off`
- Per-request Silero VAD mode: `vad_mode=auto|on|off`
- Advanced decoding: `temperature` fallback, `best_of`, `patience`, `length_penalty`, `repetition_penalty`, `no_repeat_ngram_size`
- Quality thresholds: `compression_ratio_threshold`, `log_prob_threshold`, `no_speech_threshold`
- Context/prompting: `condition_on_previous_text`, `initial_prompt`, `hotwords`, `prefix`
- Hallucination control: `hallucination_silence_threshold`, `suppress_blank`
- Translation to any language (not just English) with `target_language` parameter
- Language pair management: list, download, and cache argostranslate packages
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
    main.py                  # FastAPI app factory, lifespan, router aggregation
    config.py                # Pydantic settings loaded from .env
    transcribe/
      router.py              # HTTP endpoints (health, models, transcribe, outputs)
      schemas.py             # Request/response Pydantic models
      service.py             # Whisper inference logic and output file writing
      dependencies.py        # FastAPI dependency injection helpers
    media/
      service.py             # ffmpeg transcoding, chunking, workspace management
    translation/
      router.py              # HTTP endpoints (languages list & download)
      schemas.py             # Language pair Pydantic models
      service.py             # argostranslate wrapper (TranslationService)
      dependencies.py        # FastAPI dependency injection helpers
  scripts/
    client.py                # CLI client for the server
  Dockerfile                 # Container build
  .env.example               # All available env vars with defaults
  requirements.txt           # Python dependencies
```

## Translation architecture

When `task=translate`, the server uses a two-step pipeline:

1. **Whisper transcribes** the audio in its source language (always `task=transcribe` internally)
2. **argostranslate translates** the resulting text to `target_language`

This replaces Whisper's built-in translate mode and supports translation to **any language** that argostranslate supports (not just English). Language pair packages are downloaded automatically on first use and cached locally.

```
Audio --> [Whisper: transcribe] --> Source text --> [argostranslate] --> Translated text
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server health check (includes ffmpeg status) |
| `GET` | `/models` | List available and downloaded Whisper model names |
| `POST` | `/models/download` | Download a Whisper model by name |
| `GET` | `/languages` | List available and installed translation language pairs |
| `POST` | `/languages/download` | Pre-download a translation language pair |
| `POST` | `/transcribe` | Transcribe a single file upload or remote URL |
| `POST` | `/transcribe/uploads` | Transcribe multiple file uploads |
| `POST` | `/transcribe/links` | Transcribe a JSON array of remote URLs |
| `GET` | `/outputs/{job_id}/{filename}` | Download a transcript output file |

## API examples

### Health check

```bash
curl http://127.0.0.1:8000/health
```

Response:

```json
{
  "status": "ok",
  "service": "whisper-server",
  "env": "dev",
  "ffmpeg": "available"
}
```

### List available models

```bash
curl http://127.0.0.1:8000/models
```

Response:

```json
{
  "available_model_names": ["base", "base.en", "distil-large-v2", "distil-large-v3", "large-v1", "large-v2", "large-v3", "medium", "medium.en", "small", "small.en", "tiny", "tiny.en"],
  "downloaded_models": ["models--Systran--faster-whisper-small"]
}
```

### Download a model

Pre-download a model so the first transcription request is fast.

```bash
curl -X POST "http://127.0.0.1:8000/models/download" \
  -H "Content-Type: application/json" \
  -d '{"model": "small"}'
```

You can also specify device and compute type:

```bash
curl -X POST "http://127.0.0.1:8000/models/download" \
  -H "Content-Type: application/json" \
  -d '{"model": "large-v3", "device": "cuda", "compute_type": "float16"}'
```

Response:

```json
{
  "model": "small",
  "device": "cpu",
  "compute_type": "int8",
  "download_root": "models"
}
```

### Transcribe a single file (minimal)

The simplest call -- just upload a file, server uses all defaults.

```bash
curl -X POST "http://127.0.0.1:8000/transcribe" \
  -F "file=@recording.mp3"
```

Response:

```json
{
  "job_id": "a1b2c3d4-...",
  "model": "small",
  "device": "cpu",
  "compute_type": "int8",
  "language": "en",
  "duration": 125.4,
  "text": "Hello, welcome to the meeting...",
  "translated_text": null,
  "segments": [
    {"start": 0.0, "end": 3.52, "text": "Hello, welcome to the meeting."},
    {"start": 3.52, "end": 7.84, "text": "Today we will discuss the roadmap."}
  ],
  "output_files": {
    "txt": "/outputs/a1b2c3d4-.../transcript.txt",
    "srt": "/outputs/a1b2c3d4-.../transcript.srt"
  },
  "used_batching": false,
  "chunk_count": 1
}
```

### Transcribe with full options

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

### Transcribe with auto-detect language

Omit `source_language` and Whisper will detect the spoken language automatically.

```bash
curl -X POST "http://127.0.0.1:8000/transcribe" \
  -F "file=@unknown_language.wav" \
  -F "output_formats=txt,srt,json"
```

### Transcribe a remote URL

Pass a `media_url` instead of a file. Supports HTTP/HTTPS links, S3 presigned URLs, and YouTube.

```bash
curl -X POST "http://127.0.0.1:8000/transcribe" \
  -F "media_url=https://example.com/podcast.mp3" \
  -F "model=small" \
  -F "output_formats=txt,srt"
```

YouTube URLs are resolved automatically via yt-dlp:

```bash
curl -X POST "http://127.0.0.1:8000/transcribe" \
  -F "media_url=https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
  -F "source_language=en"
```

### Translate Japanese audio to English

```bash
curl -X POST "http://127.0.0.1:8000/transcribe" \
  -F "file=@interview.wav" \
  -F "task=translate" \
  -F "source_language=ja" \
  -F "target_language=en"
```

Response includes both original and translated text:

```json
{
  "job_id": "...",
  "language": "ja",
  "text": "こんにちは、本日の会議へようこそ...",
  "translated_text": "Hello, welcome to today's meeting...",
  "output_files": {
    "txt": "/outputs/.../transcript.txt",
    "txt_original": "/outputs/.../transcript_original.txt",
    "srt": "/outputs/.../transcript.srt"
  }
}
```

### Translate to any target language

Not limited to English -- translate to any language argostranslate supports.

```bash
curl -X POST "http://127.0.0.1:8000/transcribe" \
  -F "file=@lecture.mp3" \
  -F "task=translate" \
  -F "source_language=en" \
  -F "target_language=vi"
```

### Transcribe multiple files at once

Each file is processed sequentially with the same settings.

```bash
curl -X POST "http://127.0.0.1:8000/transcribe/uploads" \
  -F "files=@meeting_part1.mp4" \
  -F "files=@meeting_part2.mp4" \
  -F "files=@meeting_part3.mp4" \
  -F "source_language=ja" \
  -F "task=translate" \
  -F "target_language=en" \
  -F "model=large-v3" \
  -F "batch_size=8"
```

Response:

```json
{
  "items": [
    {"job_id": "...", "text": "...", "duration": 600.0, "...": "..."},
    {"job_id": "...", "text": "...", "duration": 450.0, "...": "..."},
    {"job_id": "...", "text": "...", "duration": 300.0, "...": "..."}
  ]
}
```

### Transcribe multiple remote links

```bash
curl -X POST "http://127.0.0.1:8000/transcribe/links" \
  -H "Content-Type: application/json" \
  -d '{
    "links": [
      "https://example.com/episode1.mp3",
      "https://example.com/episode2.mp3",
      "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    ],
    "source_language": "en",
    "task": "transcribe",
    "model": "small",
    "output_formats": ["txt", "srt", "json"]
  }'
```

### Download output files

After transcription, download the generated files using the paths from the response.

```bash
# Download the transcript text file
curl -O http://127.0.0.1:8000/outputs/{job_id}/transcript.txt

# Download the SRT subtitle file
curl -O http://127.0.0.1:8000/outputs/{job_id}/transcript.srt

# Download the JSON file (includes segments, metadata)
curl -O http://127.0.0.1:8000/outputs/{job_id}/transcript.json
```

### List available translation pairs

```bash
curl http://127.0.0.1:8000/languages
```

Response:

```json
{
  "available": [
    {"from_code": "en", "from_name": "English", "to_code": "ja", "to_name": "Japanese"},
    {"from_code": "ja", "from_name": "Japanese", "to_code": "en", "to_name": "English"},
    {"from_code": "en", "from_name": "English", "to_code": "vi", "to_name": "Vietnamese"}
  ],
  "installed": [
    {"from_code": "ja", "from_name": "Japanese", "to_code": "en", "to_name": "English"}
  ]
}
```

### Pre-download a language pair

Download a translation pair ahead of time so the first translate request is fast.

```bash
curl -X POST "http://127.0.0.1:8000/languages/download" \
  -H "Content-Type: application/json" \
  -d '{"from_code": "ja", "to_code": "en"}'
```

Response:

```json
{"from_code": "ja", "to_code": "en", "status": "installed"}
```

## CLI client (`scripts/client.py`)

A command-line client that talks to the running server. Supports all transcription and translation parameters.

### Basic usage

```bash
# Transcribe a local file
python scripts/client.py --file recording.mp4

# Transcribe from a URL
python scripts/client.py --media-url "https://example.com/audio.mp3"

# Translate Japanese audio to English
python scripts/client.py --file interview.wav --task translate --source-language ja

# Translate to Vietnamese
python scripts/client.py --file lecture.mp3 --task translate --source-language en --target-language vi
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
  --task {transcribe,translate}   Transcribe or translate to target language
  --source-language LANG          ISO language code (e.g. en, vi, ja)
  --target-language LANG          Translation target language (default: en)
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

## Model selection guide

Models are downloaded automatically from [Hugging Face](https://huggingface.co/Systran) on first use and cached locally in the `models/` directory. After the initial download, no internet connection is required.

### Available models

| Model | Parameters | Size on disk | Best for |
|-------|-----------|-------------|----------|
| `tiny` / `tiny.en` | 39M | ~75 MB | Quick testing, low-resource devices |
| `base` / `base.en` | 74M | ~140 MB | Light workloads, acceptable quality |
| `small` / `small.en` | 244M | ~460 MB | **Recommended default** -- good balance of speed and accuracy |
| `medium` / `medium.en` | 769M | ~1.5 GB | Higher accuracy, slower |
| `large-v1` | 1550M | ~3 GB | Best accuracy (first generation) |
| `large-v2` | 1550M | ~3 GB | Best accuracy (improved) |
| `large-v3` | 1550M | ~3 GB | Best accuracy (latest) |
| `distil-large-v2` | ~756M | ~1.5 GB | Near large-v2 quality at 2x speed |
| `distil-large-v3` | ~756M | ~1.5 GB | Near large-v3 quality at 2x speed |

`.en` variants are English-only and slightly more accurate for English than their multilingual counterparts.

### Hardware requirements

| Model | Min RAM | GPU VRAM | CPU speed (10 min audio) | GPU speed (10 min audio) |
|-------|---------|----------|--------------------------|--------------------------|
| `tiny` | ~1 GB | ~1 GB | ~1 min | ~5 sec |
| `small` | ~2 GB | ~2 GB | ~4 min | ~15 sec |
| `medium` | ~5 GB | ~4 GB | ~10 min | ~30 sec |
| `large-v3` | ~10 GB | ~6 GB | ~20 min | ~45 sec |
| `distil-large-v3` | ~6 GB | ~4 GB | ~10 min | ~20 sec |

Speed estimates are approximate and vary by hardware. GPU times assume an NVIDIA T4 or better.

### Choosing a model

- **Development / testing**: use `tiny` or `base` for fast iteration
- **Production (CPU, cost-sensitive)**: use `small` with `compute_type=int8` -- the server default
- **Production (GPU available)**: use `large-v3` or `distil-large-v3` with `compute_type=float16`
- **English only**: use `.en` variants (e.g. `small.en`) for slightly better accuracy
- **Best speed/quality tradeoff**: use `distil-large-v3` -- close to `large-v3` quality at roughly half the compute cost

### Pre-downloading models

To avoid download delays on the first request, pre-download models at deploy time:

```bash
curl -X POST "http://127.0.0.1:8000/models/download" \
  -H "Content-Type: application/json" \
  -d '{"model": "small"}'
```

Or in Docker, mount a volume so models persist across container restarts:

```bash
docker run --rm -p 8000:8000 --env-file .env -v ./models:/app/models whisper-server:latest
```

Set `FASTER_WHISPER_LOCAL_FILES_ONLY=true` in `.env` to prevent any automatic downloads (useful for air-gapped deployments).

## Configuration

All settings can be overridden via environment variables or a `.env` file. See [.env.example](.env.example) for the full list.

### Task behavior

- `task=transcribe`: keep original source language (no translation).
- `task=translate`: Whisper transcribes in source language, then argostranslate translates to `target_language`.
- `target_language` defaults to `en` for backward compatibility.
- Translation supports any language pair that argostranslate has packages for. Language packages are auto-downloaded on first use.

### Batching behavior

- `batch_mode=auto`: use server default + long-media threshold (`BATCH_THRESHOLD_SECONDS`).
- `batch_mode=on`: force batching (splits audio into `BATCH_CHUNK_SECONDS` chunks).
- `batch_mode=off`: disable batching for that request.

### VAD behavior

- `vad_mode=auto`: follow the request-level `vad_filter` toggle.
- `vad_mode=on`: force Silero VAD on (removes silence before transcription).
- `vad_mode=off`: force Silero VAD off.

### Translation output files

When `task=translate`, the output files contain:
- `transcript.txt` -- the translated text (primary output)
- `transcript_original.txt` -- the source-language transcription
- `transcript.srt` -- SRT subtitles (source-language segments with timestamps)
- `transcript.json` -- includes both `text` (translated) and `original_text` (source)

## Advanced transcription parameters

All parameters below can be sent per-request (form field or JSON body) and have server-level defaults via env vars.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_language` | string | `en` | ISO 639 target language for translation (only used when task=translate) |
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
