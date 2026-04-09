from typing import Literal

from pydantic import BaseModel, Field, model_validator

OutputFormat = Literal["txt", "srt", "json"]
WhisperTask = Literal["transcribe", "translate"]
SwitchMode = Literal["auto", "on", "off"]
ComputeType = Literal[
    "default",
    "auto",
    "int8",
    "int8_float16",
    "int8_bfloat16",
    "int16",
    "float16",
    "bfloat16",
    "float32",
]
DeviceType = Literal["cpu", "cuda", "auto"]
SupportedSourceLanguage = str


class VADConfig(BaseModel):
    vad_filter: bool = True
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_silence_duration_ms: int = Field(default=500, ge=0)
    speech_pad_ms: int = Field(default=400, ge=0)


class TranscribeRequest(BaseModel):
    media_url: str | None = None
    source_language: SupportedSourceLanguage | None = None
    task: WhisperTask = "transcribe"
    output_formats: list[OutputFormat] = Field(default_factory=lambda: ["txt", "srt"])
    vad: VADConfig = Field(default_factory=VADConfig)
    model: str | None = None
    device: DeviceType | None = None
    compute_type: ComputeType | None = None
    beam_size: int = Field(default=5, ge=1, le=20)
    word_timestamps: bool = False
    batch_size: int = Field(default=8, ge=1, le=64)
    batch_mode: SwitchMode = "auto"
    vad_mode: SwitchMode = "auto"
    use_batch: bool | None = None

    temperature: list[float] = Field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    best_of: int = Field(default=5, ge=1, le=20)
    patience: float = Field(default=1.0, ge=0.0)
    length_penalty: float = Field(default=1.0, ge=0.0)
    repetition_penalty: float = Field(default=1.0, ge=0.0)
    no_repeat_ngram_size: int = Field(default=0, ge=0)

    compression_ratio_threshold: float | None = 2.4
    log_prob_threshold: float | None = -1.0
    no_speech_threshold: float | None = 0.6

    condition_on_previous_text: bool = True
    initial_prompt: str | None = None
    prompt_reset_on_temperature: float = 0.5
    hotwords: str | None = None
    prefix: str | None = None

    hallucination_silence_threshold: float | None = None
    suppress_blank: bool = True
    without_timestamps: bool = False
    max_initial_timestamp: float = Field(default=1.0, ge=0.0)
    max_new_tokens: int | None = None


class TranscribeFormInput(BaseModel):
    media_url: str | None = None
    source_language: SupportedSourceLanguage | None = None
    task: WhisperTask = "transcribe"
    output_formats: str = "txt,srt"
    vad_filter: bool = True
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_silence_duration_ms: int = Field(default=500, ge=0)
    speech_pad_ms: int = Field(default=400, ge=0)
    model: str | None = None
    device: DeviceType | None = None
    compute_type: ComputeType | None = None
    beam_size: int = Field(default=5, ge=1, le=20)
    word_timestamps: bool = False
    batch_size: int = Field(default=8, ge=1, le=64)
    batch_mode: SwitchMode = "auto"
    vad_mode: SwitchMode = "auto"
    use_batch: bool | None = None

    temperature: str = "0.0,0.2,0.4,0.6,0.8,1.0"
    best_of: int = Field(default=5, ge=1, le=20)
    patience: float = Field(default=1.0, ge=0.0)
    length_penalty: float = Field(default=1.0, ge=0.0)
    repetition_penalty: float = Field(default=1.0, ge=0.0)
    no_repeat_ngram_size: int = Field(default=0, ge=0)

    compression_ratio_threshold: float | None = 2.4
    log_prob_threshold: float | None = -1.0
    no_speech_threshold: float | None = 0.6

    condition_on_previous_text: bool = True
    initial_prompt: str | None = None
    prompt_reset_on_temperature: float = 0.5
    hotwords: str | None = None
    prefix: str | None = None

    hallucination_silence_threshold: float | None = None
    suppress_blank: bool = True
    without_timestamps: bool = False
    max_initial_timestamp: float = Field(default=1.0, ge=0.0)
    max_new_tokens: int | None = None

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

    def parsed_temperature(self) -> list[float]:
        raw = self.temperature.strip()
        if not raw:
            return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        return [float(t.strip()) for t in raw.split(",") if t.strip()]

    def to_request(self) -> TranscribeRequest:
        return TranscribeRequest(
            media_url=self.media_url,
            source_language=self.source_language,
            task=self.task,
            output_formats=self.parsed_formats(),
            vad=VADConfig(
                vad_filter=self.vad_filter,
                threshold=self.vad_threshold,
                min_silence_duration_ms=self.min_silence_duration_ms,
                speech_pad_ms=self.speech_pad_ms,
            ),
            model=self.model,
            device=self.device,
            compute_type=self.compute_type,
            beam_size=self.beam_size,
            word_timestamps=self.word_timestamps,
            batch_size=self.batch_size,
            batch_mode=self.batch_mode,
            vad_mode=self.vad_mode,
            use_batch=self.use_batch,
            temperature=self.parsed_temperature(),
            best_of=self.best_of,
            patience=self.patience,
            length_penalty=self.length_penalty,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            compression_ratio_threshold=self.compression_ratio_threshold,
            log_prob_threshold=self.log_prob_threshold,
            no_speech_threshold=self.no_speech_threshold,
            condition_on_previous_text=self.condition_on_previous_text,
            initial_prompt=self.initial_prompt,
            prompt_reset_on_temperature=self.prompt_reset_on_temperature,
            hotwords=self.hotwords,
            prefix=self.prefix,
            hallucination_silence_threshold=self.hallucination_silence_threshold,
            suppress_blank=self.suppress_blank,
            without_timestamps=self.without_timestamps,
            max_initial_timestamp=self.max_initial_timestamp,
            max_new_tokens=self.max_new_tokens,
        )


class SegmentOut(BaseModel):
    start: float
    end: float
    text: str


class TranscribeResponse(BaseModel):
    job_id: str
    model: str
    device: str
    compute_type: str
    language: str | None
    duration: float
    text: str
    segments: list[SegmentOut]
    output_files: dict[str, str]
    used_batching: bool = False
    chunk_count: int = 1


class HealthResponse(BaseModel):
    status: str
    service: str
    env: str
    ffmpeg: str


class RemoteLinksRequest(BaseModel):
    links: list[str] = Field(default_factory=list, min_length=1)
    source_language: SupportedSourceLanguage | None = None
    task: WhisperTask = "transcribe"
    output_formats: list[OutputFormat] = Field(default_factory=lambda: ["txt", "srt"])
    model: str | None = None
    device: DeviceType | None = None
    compute_type: ComputeType | None = None
    beam_size: int = Field(default=5, ge=1, le=20)
    word_timestamps: bool = False
    batch_size: int = Field(default=8, ge=1, le=64)
    batch_mode: SwitchMode = "auto"
    vad_mode: SwitchMode = "auto"
    use_batch: bool | None = None
    vad: VADConfig = Field(default_factory=VADConfig)

    temperature: list[float] = Field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    best_of: int = Field(default=5, ge=1, le=20)
    patience: float = Field(default=1.0, ge=0.0)
    length_penalty: float = Field(default=1.0, ge=0.0)
    repetition_penalty: float = Field(default=1.0, ge=0.0)
    no_repeat_ngram_size: int = Field(default=0, ge=0)

    compression_ratio_threshold: float | None = 2.4
    log_prob_threshold: float | None = -1.0
    no_speech_threshold: float | None = 0.6

    condition_on_previous_text: bool = True
    initial_prompt: str | None = None
    prompt_reset_on_temperature: float = 0.5
    hotwords: str | None = None
    prefix: str | None = None

    hallucination_silence_threshold: float | None = None
    suppress_blank: bool = True
    without_timestamps: bool = False
    max_initial_timestamp: float = Field(default=1.0, ge=0.0)
    max_new_tokens: int | None = None


class BatchTranscribeResponse(BaseModel):
    items: list[TranscribeResponse]


class DownloadModelRequest(BaseModel):
    model: str
    device: DeviceType | None = None
    compute_type: ComputeType | None = None


class AvailableModelsResponse(BaseModel):
    available_model_names: list[str]
    downloaded_models: list[str]

