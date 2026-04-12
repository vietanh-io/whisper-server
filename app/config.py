"""
Application settings loaded from environment variables / .env file.

Every field maps 1:1 to an env var (case-insensitive). For example
``faster_whisper_model`` is read from ``FASTER_WHISPER_MODEL``.
Pydantic-settings handles the parsing; see .env.example for the full list.
"""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator(
        "default_compression_ratio_threshold",
        "default_log_prob_threshold",
        "default_no_speech_threshold",
        "default_hallucination_silence_threshold",
        "default_max_new_tokens",
        mode="before",
    )
    @classmethod
    def _empty_str_to_none(cls, v: object) -> object:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    # ── App identity ────────────────────────────────────────────────────
    app_name: str = "whisper-server"
    app_env: str = "dev"
    host: str = "127.0.0.1"
    port: int = 8000

    # ── File-system paths ───────────────────────────────────────────────
    output_dir: Path = Path("outputs")
    temp_dir: Path = Path(".tmp")
    models_dir: Path = Path("models")

    # ── Request defaults (can be overridden per-request) ────────────────
    default_output_formats: str = "txt,srt"
    default_source_language: str | None = None
    default_task: str = "transcribe"

    # ── Faster-whisper model loading ────────────────────────────────────
    faster_whisper_model: str = "small"
    faster_whisper_device: str = "cpu"
    faster_whisper_compute_type: str = "int8"
    faster_whisper_cpu_threads: int = 0
    faster_whisper_num_workers: int = 1
    faster_whisper_download_root: Path = Path("models")
    faster_whisper_local_files_only: bool = False

    # ── Silero VAD defaults ─────────────────────────────────────────────
    default_vad_filter: bool = True
    default_vad_threshold: float = 0.5
    default_min_silence_duration_ms: int = 500
    default_speech_pad_ms: int = 400

    # ── Basic transcription defaults ────────────────────────────────────
    default_beam_size: int = 5
    default_word_timestamps: bool = False
    default_batch_mode: str = "auto"
    default_vad_mode: str = "auto"

    # ── Decoding strategy defaults ──────────────────────────────────────
    default_temperature: str = "0.0,0.2,0.4,0.6,0.8,1.0"
    default_best_of: int = 5
    default_patience: float = 1.0
    default_length_penalty: float = 1.0
    default_repetition_penalty: float = 1.0
    default_no_repeat_ngram_size: int = 0

    # ── Quality threshold defaults ──────────────────────────────────────
    default_compression_ratio_threshold: float | None = 2.4
    default_log_prob_threshold: float | None = -1.0
    default_no_speech_threshold: float | None = 0.6

    # ── Context / prompting defaults ────────────────────────────────────
    default_condition_on_previous_text: bool = True
    default_initial_prompt: str | None = None
    default_prompt_reset_on_temperature: float = 0.5
    default_hotwords: str | None = None
    default_prefix: str | None = None

    # ── Hallucination / token control defaults ──────────────────────────
    default_hallucination_silence_threshold: float | None = None
    default_suppress_blank: bool = True
    default_without_timestamps: bool = False
    default_max_initial_timestamp: float = 1.0
    default_max_new_tokens: int | None = None

    # ── Batching (long-media chunking) ──────────────────────────────────
    batch_enabled: bool = True
    batch_threshold_seconds: int = 900
    batch_chunk_seconds: int = 300
    default_batch_size: int = 8

    # ── Translation (argostranslate) ────────────────────────────────────
    default_target_language: str = "en"
    argos_models_dir: Path = Path("models/argos")

    # ── Safety limits ───────────────────────────────────────────────────
    request_timeout_seconds: int = 1800
    max_upload_size_mb: int = 1024


settings = Settings()
