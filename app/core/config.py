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

    default_backend: str = "faster-whisper"
    default_output_formats: str = "txt,srt"

    default_language: str | None = None
    default_task: str = "transcribe"

    faster_whisper_model: str = "small"
    faster_whisper_device: str = "cpu"
    faster_whisper_compute_type: str = "int8"

    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_fp16: bool = False

    default_vad_filter: bool = True
    default_vad_threshold: float = 0.5
    default_min_silence_duration_ms: int = 500
    default_speech_pad_ms: int = 400

    request_timeout_seconds: int = 1800
    max_upload_size_mb: int = 1024


settings = Settings()

