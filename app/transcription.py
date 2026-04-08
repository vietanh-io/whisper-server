import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from faster_whisper import WhisperModel

from app.config import settings
from app.schemas import SegmentOut, TranscribeRequest


@dataclass
class TranscriptArtifacts:
    text: str
    language: str | None
    duration: float
    segments: list[SegmentOut]
    output_files: dict[str, Path]


class WhisperService:
    def __init__(self) -> None:
        self._model: WhisperModel | None = None

    def get_model(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(
                settings.whisper_model,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
            )
        return self._model

    def transcribe(
        self,
        input_path: Path,
        request_data: TranscribeRequest,
        job_dir: Path,
    ) -> TranscriptArtifacts:
        model = self.get_model()
        language = request_data.language or settings.default_language
        if isinstance(language, str):
            language = language.strip() or None

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
            segments_list.append(
                SegmentOut(start=float(segment.start), end=float(segment.end), text=cleaned)
            )
            text_parts.append(cleaned)

        text = "\n".join(part for part in text_parts if part)
        return write_outputs(
            job_dir=job_dir,
            text=text,
            language=info.language if info else None,
            duration=float(info.duration or 0.0) if info else 0.0,
            segments=segments_list,
            formats=request_data.output_formats,
            model_name=settings.whisper_model,
            task=request_data.task,
            vad=request_data.vad.model_dump(),
        )


def ffmpeg_available() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_media(url: str, target_path: Path, timeout_seconds: int) -> None:
    with requests.get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        with target_path.open("wb") as file_handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_handle.write(chunk)


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
        text=text,
        language=language,
        duration=duration,
        segments=segments,
        output_files=output_files,
    )

