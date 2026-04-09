from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "whisper-server"
    app_env: str = "dev"
    host: str = "127.0.0.1"
    port: int = 8000

    output_dir: Path = Path("outputs")
    temp_dir: Path = Path(".tmp")
    models_dir: Path = Path("models")

    default_output_formats: str = "txt,srt"
    default_source_language: str | None = None
    default_task: str = "transcribe"

    faster_whisper_model: str = "small"
    faster_whisper_device: str = "cpu"
    faster_whisper_compute_type: str = "int8"
    faster_whisper_cpu_threads: int = 0
    faster_whisper_num_workers: int = 1
    faster_whisper_download_root: Path = Path("models")
    faster_whisper_local_files_only: bool = False

    default_vad_filter: bool = True
    default_vad_threshold: float = 0.5
    default_min_silence_duration_ms: int = 500
    default_speech_pad_ms: int = 400

    default_beam_size: int = 5
    default_word_timestamps: bool = False
    default_batch_mode: str = "auto"
    default_vad_mode: str = "auto"

    default_temperature: str = "0.0,0.2,0.4,0.6,0.8,1.0"
    default_best_of: int = 5
    default_patience: float = 1.0
    default_length_penalty: float = 1.0
    default_repetition_penalty: float = 1.0
    default_no_repeat_ngram_size: int = 0

    default_compression_ratio_threshold: float | None = 2.4
    default_log_prob_threshold: float | None = -1.0
    default_no_speech_threshold: float | None = 0.6

    default_condition_on_previous_text: bool = True
    default_initial_prompt: str | None = None
    default_prompt_reset_on_temperature: float = 0.5
    default_hotwords: str | None = None
    default_prefix: str | None = None

    default_hallucination_silence_threshold: float | None = None
    default_suppress_blank: bool = True
    default_without_timestamps: bool = False
    default_max_initial_timestamp: float = 1.0
    default_max_new_tokens: int | None = None

    batch_enabled: bool = True
    batch_threshold_seconds: int = 900
    batch_chunk_seconds: int = 300
    default_batch_size: int = 8

    force_translate_to_english: bool = True
    reject_english_source_on_translate: bool = True

    request_timeout_seconds: int = 1800
    max_upload_size_mb: int = 1024


settings = Settings()

