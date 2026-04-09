"""
Application settings loaded from environment variables / .env file.

Every field maps 1:1 to an env var (case-insensitive). For example
``faster_whisper_model`` is read from ``FASTER_WHISPER_MODEL``.
Pydantic-settings handles the parsing; see .env.example for the full list.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── App identity ────────────────────────────────────────────────────
    app_name: str = "whisper-server"
    app_env: str = "dev"
    host: str = "127.0.0.1"
    port: int = 8000

    # ── File-system paths ───────────────────────────────────────────────
    output_dir: Path = Path("outputs")    # final transcript files per job
    temp_dir: Path = Path(".tmp")         # temporary input files, cleaned after each job
    models_dir: Path = Path("models")     # HuggingFace model cache

    # ── Request defaults (can be overridden per-request) ────────────────
    default_output_formats: str = "txt,srt"
    default_source_language: str | None = None  # None = auto-detect
    default_task: str = "transcribe"

    # ── Faster-whisper model loading ────────────────────────────────────
    faster_whisper_model: str = "small"
    faster_whisper_device: str = "cpu"           # "cpu", "cuda", or "auto"
    faster_whisper_compute_type: str = "int8"    # CTranslate2 quantization type
    faster_whisper_cpu_threads: int = 0          # 0 = use OMP_NUM_THREADS
    faster_whisper_num_workers: int = 1          # parallel model workers
    faster_whisper_download_root: Path = Path("models")
    faster_whisper_local_files_only: bool = False

    # ── Silero VAD defaults ─────────────────────────────────────────────
    default_vad_filter: bool = True
    default_vad_threshold: float = 0.5           # speech probability threshold
    default_min_silence_duration_ms: int = 500   # min silence to split on
    default_speech_pad_ms: int = 400             # padding around speech segments

    # ── Basic transcription defaults ────────────────────────────────────
    default_beam_size: int = 5
    default_word_timestamps: bool = False
    default_batch_mode: str = "auto"             # "auto" | "on" | "off"
    default_vad_mode: str = "auto"               # "auto" | "on" | "off"

    # ── Decoding strategy defaults ──────────────────────────────────────
    # Temperature fallback: comma-separated floats.  On quality failure the
    # decoder retries with the next temperature in the list.  A single "0.0"
    # means pure greedy decoding with no fallback.
    default_temperature: str = "0.0,0.2,0.4,0.6,0.8,1.0"
    default_best_of: int = 5                     # candidates when temperature > 0
    default_patience: float = 1.0                # beam search patience factor
    default_length_penalty: float = 1.0          # exponential length penalty
    default_repetition_penalty: float = 1.0      # penalise repeated tokens
    default_no_repeat_ngram_size: int = 0        # 0 = disabled

    # ── Quality threshold defaults ──────────────────────────────────────
    # Set to None (empty env var) to disable the corresponding check.
    default_compression_ratio_threshold: float | None = 2.4   # gzip ratio above this = bad
    default_log_prob_threshold: float | None = -1.0           # avg log-prob below this = bad
    default_no_speech_threshold: float | None = 0.6           # skip if no-speech prob > this

    # ── Context / prompting defaults ────────────────────────────────────
    default_condition_on_previous_text: bool = True   # carry context across segments
    default_initial_prompt: str | None = None         # domain vocabulary hint
    default_prompt_reset_on_temperature: float = 0.5  # reset prompt above this temp
    default_hotwords: str | None = None               # boost specific phrases
    default_prefix: str | None = None                 # prefix for the first window

    # ── Hallucination / token control defaults ──────────────────────────
    default_hallucination_silence_threshold: float | None = None  # seconds; None = off
    default_suppress_blank: bool = True
    default_without_timestamps: bool = False
    default_max_initial_timestamp: float = 1.0  # seconds
    default_max_new_tokens: int | None = None   # None = no limit

    # ── Batching (long-media chunking) ──────────────────────────────────
    batch_enabled: bool = True
    batch_threshold_seconds: int = 900   # auto-batch if duration >= this
    batch_chunk_seconds: int = 300       # each chunk length when batching
    default_batch_size: int = 8          # parallel batch size for BatchedInferencePipeline

    # ── Translation guard ───────────────────────────────────────────────
    force_translate_to_english: bool = True
    reject_english_source_on_translate: bool = True  # 400 if source=en + task=translate

    # ── Safety limits ───────────────────────────────────────────────────
    request_timeout_seconds: int = 1800
    max_upload_size_mb: int = 1024


settings = Settings()
