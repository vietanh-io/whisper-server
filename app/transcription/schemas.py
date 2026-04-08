from typing import Literal

from pydantic import BaseModel, Field, model_validator

OutputFormat = Literal["txt", "srt", "json"]
WhisperTask = Literal["transcribe", "translate"]
TranscriptionBackend = Literal["faster-whisper", "whisper"]


class VADConfig(BaseModel):
    vad_filter: bool = True
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_silence_duration_ms: int = Field(default=500, ge=0)
    speech_pad_ms: int = Field(default=400, ge=0)


class TranscribeRequest(BaseModel):
    media_url: str | None = None
    language: str | None = None
    task: WhisperTask = "transcribe"
    backend: TranscriptionBackend | None = None
    output_formats: list[OutputFormat] = Field(default_factory=lambda: ["txt", "srt"])
    vad: VADConfig = Field(default_factory=VADConfig)


class TranscribeFormInput(BaseModel):
    media_url: str | None = None
    language: str | None = None
    task: WhisperTask = "transcribe"
    backend: TranscriptionBackend | None = None
    output_formats: str = "txt,srt"
    vad_filter: bool = True
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_silence_duration_ms: int = Field(default=500, ge=0)
    speech_pad_ms: int = Field(default=400, ge=0)

    @model_validator(mode="after")
    def ensure_media_url_is_clean(self) -> "TranscribeFormInput":
        if self.media_url is not None and not self.media_url.strip():
            self.media_url = None
        return self

    def parsed_formats(self) -> list[OutputFormat]:
        values = [x.strip().lower() for x in self.output_formats.split(",") if x.strip()]
        if not values:
            return ["txt", "srt"]
        valid: list[OutputFormat] = []
        for value in values:
            if value in {"txt", "srt", "json"}:
                valid.append(value)  # type: ignore[arg-type]
        return valid or ["txt", "srt"]

    def to_request(self) -> TranscribeRequest:
        return TranscribeRequest(
            media_url=self.media_url,
            language=self.language,
            task=self.task,
            backend=self.backend,
            output_formats=self.parsed_formats(),
            vad=VADConfig(
                vad_filter=self.vad_filter,
                threshold=self.vad_threshold,
                min_silence_duration_ms=self.min_silence_duration_ms,
                speech_pad_ms=self.speech_pad_ms,
            ),
        )


class SegmentOut(BaseModel):
    start: float
    end: float
    text: str


class TranscribeResponse(BaseModel):
    job_id: str
    backend: TranscriptionBackend
    model: str
    language: str | None
    duration: float
    text: str
    segments: list[SegmentOut]
    output_files: dict[str, str]


class HealthResponse(BaseModel):
    status: str
    service: str
    env: str
    default_backend: str
    ffmpeg: str

