import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.transcription.schemas import SegmentOut, TranscribeRequest, TranscriptionBackend


@dataclass
class TranscriptArtifacts:
    backend: TranscriptionBackend
    model_name: str
    text: str
    language: str | None
    duration: float
    segments: list[SegmentOut]
    output_files: dict[str, Path]


class WhisperService:
    def __init__(self) -> None:
        self._faster_model: Any | None = None
        self._openai_whisper_model: Any | None = None

    def transcribe(
        self,
        input_path: Path,
        request_data: TranscribeRequest,
        job_dir: Path,
    ) -> TranscriptArtifacts:
        backend = self._resolve_backend(request_data.backend)
        language = request_data.language or settings.default_language
        if isinstance(language, str):
            language = language.strip() or None

        if backend == "faster-whisper":
            text, detected_language, duration, segments_list, model_name = self._transcribe_faster_whisper(
                input_path=input_path,
                request_data=request_data,
                language=language,
            )
        else:
            text, detected_language, duration, segments_list, model_name = self._transcribe_whisper(
                input_path=input_path,
                request_data=request_data,
                language=language,
            )

        return write_outputs(
            backend=backend,
            model_name=model_name,
            job_dir=job_dir,
            text=text,
            language=detected_language,
            duration=duration,
            segments=segments_list,
            formats=request_data.output_formats,
            task=request_data.task,
            vad=request_data.vad.model_dump(),
        )

    def _resolve_backend(self, requested_backend: str | None) -> TranscriptionBackend:
        backend = (requested_backend or settings.default_backend).strip().lower()
        if backend not in {"faster-whisper", "whisper"}:
            raise ValueError("backend must be one of: faster-whisper, whisper")
        return backend  # type: ignore[return-value]

    def _get_faster_whisper_model(self) -> Any:
        if self._faster_model is None:
            from faster_whisper import WhisperModel

            self._faster_model = WhisperModel(
                settings.faster_whisper_model,
                device=settings.faster_whisper_device,
                compute_type=settings.faster_whisper_compute_type,
            )
        return self._faster_model

    def _get_openai_whisper_model(self) -> Any:
        if self._openai_whisper_model is None:
            import whisper

            self._openai_whisper_model = whisper.load_model(
                settings.whisper_model,
                device=settings.whisper_device,
            )
        return self._openai_whisper_model

    def _transcribe_faster_whisper(
        self,
        *,
        input_path: Path,
        request_data: TranscribeRequest,
        language: str | None,
    ) -> tuple[str, str | None, float, list[SegmentOut], str]:
        model = self._get_faster_whisper_model()
        segments_iter, info = model.transcribe(
            str(input_path),
            language=language,
            task=request_data.task,
            vad_filter=request_data.vad.vad_filter,
            vad_parameters={
                "threshold": request_data.vad.threshold,
                "min_silence_duration_ms": request_data.vad.min_silence_duration_ms,
                "speech_pad_ms": request_data.vad.speech_pad_ms,
            },
        )
        segments_list: list[SegmentOut] = []
        text_parts: list[str] = []
        for segment in segments_iter:
            cleaned = segment.text.strip()
            segments_list.append(SegmentOut(start=float(segment.start), end=float(segment.end), text=cleaned))
            text_parts.append(cleaned)

        text = "\n".join(part for part in text_parts if part)
        detected_language = info.language if info else None
        duration = float(info.duration or 0.0) if info else 0.0
        return text, detected_language, duration, segments_list, settings.faster_whisper_model

    def _transcribe_whisper(
        self,
        *,
        input_path: Path,
        request_data: TranscribeRequest,
        language: str | None,
    ) -> tuple[str, str | None, float, list[SegmentOut], str]:
        model = self._get_openai_whisper_model()
        result = model.transcribe(
            str(input_path),
            language=language,
            task=request_data.task,
            fp16=settings.whisper_fp16,
        )

        segments_list: list[SegmentOut] = []
        for segment in result.get("segments", []):
            segments_list.append(
                SegmentOut(
                    start=float(segment.get("start", 0.0)),
                    end=float(segment.get("end", 0.0)),
                    text=str(segment.get("text", "")).strip(),
                )
            )

        duration = 0.0
        if segments_list:
            duration = max(segment.end for segment in segments_list)

        text = str(result.get("text", "")).strip()
        detected_language = result.get("language")
        return text, detected_language, duration, segments_list, settings.whisper_model


def format_timestamp(seconds: float) -> str:
    ms_total = int(round(seconds * 1000))
    hours, remainder = divmod(ms_total, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_srt(segments: list[SegmentOut]) -> str:
    chunks: list[str] = []
    for idx, segment in enumerate(segments, start=1):
        chunks.append(str(idx))
        chunks.append(f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}")
        chunks.append(segment.text)
        chunks.append("")
    return "\n".join(chunks).strip() + "\n"


def write_outputs(
    *,
    backend: TranscriptionBackend,
    job_dir: Path,
    text: str,
    language: str | None,
    duration: float,
    segments: list[SegmentOut],
    formats: list[str],
    model_name: str,
    task: str,
    vad: dict[str, Any],
) -> TranscriptArtifacts:
    output_files: dict[str, Path] = {}

    if "txt" in formats:
        txt_path = job_dir / "transcript.txt"
        txt_path.write_text(text, encoding="utf-8")
        output_files["txt"] = txt_path

    if "srt" in formats:
        srt_path = job_dir / "transcript.srt"
        srt_path.write_text(build_srt(segments), encoding="utf-8")
        output_files["srt"] = srt_path

    if "json" in formats:
        json_path = job_dir / "transcript.json"
        payload = {
            "backend": backend,
            "text": text,
            "language": language,
            "duration": duration,
            "model": model_name,
            "task": task,
            "vad": vad,
            "segments": [segment.model_dump() for segment in segments],
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        output_files["json"] = json_path

    return TranscriptArtifacts(
        backend=backend,
        model_name=model_name,
        text=text,
        language=language,
        duration=duration,
        segments=segments,
        output_files=output_files,
    )

