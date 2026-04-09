import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.utils import available_models

from app.core.config import settings
from app.transcription.schemas import SegmentOut, TranscribeRequest


@dataclass
class TranscriptArtifacts:
    model_name: str
    device: str
    compute_type: str
    text: str
    language: str | None
    duration: float
    segments: list[SegmentOut]
    output_files: dict[str, Path]
    used_batching: bool
    chunk_count: int


class WhisperService:
    def __init__(self) -> None:
        self._models: dict[tuple[str, str, str], WhisperModel] = {}

    def transcribe(
        self,
        input_path: Path,
        request_data: TranscribeRequest,
        job_dir: Path,
        *,
        chunk_index: int = 0,
        chunk_offset_seconds: float = 0.0,
    ) -> TranscriptArtifacts:
        source_language = request_data.source_language or settings.default_source_language
        if isinstance(source_language, str):
            source_language = source_language.strip().lower() or None
        if (
            request_data.task == "translate"
            and source_language == "en"
            and settings.reject_english_source_on_translate
        ):
            raise ValueError("English source is not supported. Only non-English to English is supported.")

        model_name = request_data.model or settings.faster_whisper_model
        device = request_data.device or settings.faster_whisper_device
        compute_type = request_data.compute_type or settings.faster_whisper_compute_type
        model = self._get_model(model_name=model_name, device=device, compute_type=compute_type)

        text, detected_language, duration, segments_list, used_batching = self._transcribe_faster_whisper(
            model=model,
            input_path=input_path,
            request_data=request_data,
            source_language=source_language,
        )
        if (
            request_data.task == "translate"
            and detected_language == "en"
            and settings.reject_english_source_on_translate
        ):
            raise ValueError("Detected source language is English. Only non-English to English is supported.")

        if chunk_offset_seconds > 0:
            shifted: list[SegmentOut] = []
            for seg in segments_list:
                shifted.append(
                    SegmentOut(
                        start=seg.start + chunk_offset_seconds,
                        end=seg.end + chunk_offset_seconds,
                        text=seg.text,
                    )
                )
            segments_list = shifted

        return write_outputs(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            job_dir=job_dir,
            text=text,
            language=detected_language,
            duration=duration,
            segments=segments_list,
            formats=request_data.output_formats,
            task=request_data.task,
            vad=request_data.vad.model_dump(),
            chunk_index=chunk_index,
            used_batching=used_batching,
            chunk_count=1,
        )

    def get_available_models(self) -> list[str]:
        return sorted(available_models())

    def get_downloaded_models(self) -> list[str]:
        root = settings.faster_whisper_download_root
        if not root.exists():
            return []
        return sorted([path.name for path in root.iterdir() if path.is_dir()])

    def download_model(
        self,
        model_name: str,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> dict[str, str]:
        resolved_device = device or settings.faster_whisper_device
        resolved_compute_type = compute_type or settings.faster_whisper_compute_type
        self._get_model(
            model_name=model_name,
            device=resolved_device,
            compute_type=resolved_compute_type,
        )
        return {
            "model": model_name,
            "device": resolved_device,
            "compute_type": resolved_compute_type,
            "download_root": str(settings.faster_whisper_download_root),
        }

    def _get_model(self, *, model_name: str, device: str, compute_type: str) -> WhisperModel:
        key = (model_name, device, compute_type)
        if key not in self._models:
            self._models[key] = WhisperModel(
                model_size_or_path=model_name,
                device=device,
                compute_type=compute_type,
                cpu_threads=settings.faster_whisper_cpu_threads or 0,
                num_workers=settings.faster_whisper_num_workers,
                download_root=str(settings.faster_whisper_download_root),
                local_files_only=settings.faster_whisper_local_files_only,
            )
        return self._models[key]

    def _transcribe_faster_whisper(
        self,
        *,
        model: WhisperModel,
        input_path: Path,
        request_data: TranscribeRequest,
        source_language: str | None,
    ) -> tuple[str, str | None, float, list[SegmentOut], bool]:
        use_batch = self._resolve_batch_enabled(request_data)
        use_vad = self._resolve_vad_enabled(request_data)
        vad_parameters = {
            "threshold": request_data.vad.threshold,
            "min_silence_duration_ms": request_data.vad.min_silence_duration_ms,
            "speech_pad_ms": request_data.vad.speech_pad_ms,
        }
        # faster-whisper accepts float (greedy) or list[float] (fallback sequence)
        temperature: float | list[float] = request_data.temperature
        if len(request_data.temperature) == 1:
            temperature = request_data.temperature[0]

        # Params shared by both WhisperModel.transcribe() and BatchedInferencePipeline.transcribe()
        common_params: dict[str, Any] = {
            "language": source_language,
            "task": request_data.task,
            "beam_size": request_data.beam_size,
            "best_of": request_data.best_of,
            "patience": request_data.patience,
            "length_penalty": request_data.length_penalty,
            "repetition_penalty": request_data.repetition_penalty,
            "no_repeat_ngram_size": request_data.no_repeat_ngram_size,
            "temperature": temperature,
            "compression_ratio_threshold": request_data.compression_ratio_threshold,
            "log_prob_threshold": request_data.log_prob_threshold,
            "no_speech_threshold": request_data.no_speech_threshold,
            "condition_on_previous_text": request_data.condition_on_previous_text,
            "initial_prompt": request_data.initial_prompt,
            "prompt_reset_on_temperature": request_data.prompt_reset_on_temperature,
            "hotwords": request_data.hotwords,
            "word_timestamps": request_data.word_timestamps,
            "suppress_blank": request_data.suppress_blank,
            "without_timestamps": request_data.without_timestamps,
            "vad_filter": use_vad,
            "vad_parameters": vad_parameters,
        }

        if use_batch:
            pipeline = BatchedInferencePipeline(model=model)
            segments_iter, info = pipeline.transcribe(
                str(input_path),
                batch_size=request_data.batch_size,
                **common_params,
            )
            used_batching = True
        else:
            # Sequential-only params: prefix, max_initial_timestamp,
            # hallucination_silence_threshold, max_new_tokens
            segments_iter, info = model.transcribe(
                str(input_path),
                prefix=request_data.prefix,
                max_initial_timestamp=request_data.max_initial_timestamp,
                hallucination_silence_threshold=request_data.hallucination_silence_threshold,
                max_new_tokens=request_data.max_new_tokens,
                **common_params,
            )
            used_batching = False

        segments_list: list[SegmentOut] = []
        text_parts: list[str] = []
        duration = 0.0
        for segment in segments_iter:
            cleaned = segment.text.strip()
            start = float(segment.start)
            end = float(segment.end)
            segments_list.append(SegmentOut(start=start, end=end, text=cleaned))
            text_parts.append(cleaned)
            duration = max(duration, end)

        text = "\n".join(part for part in text_parts if part)
        detected_language = info.language if info else None
        if info and info.duration:
            duration = max(duration, float(info.duration))
        return text, detected_language, duration, segments_list, used_batching

    def _resolve_batch_enabled(self, request_data: TranscribeRequest) -> bool:
        if request_data.batch_mode == "on":
            return True
        if request_data.batch_mode == "off":
            return False
        if request_data.use_batch is not None:
            return request_data.use_batch
        return settings.batch_enabled

    def _resolve_vad_enabled(self, request_data: TranscribeRequest) -> bool:
        if request_data.vad_mode == "on":
            return True
        if request_data.vad_mode == "off":
            return False
        return request_data.vad.vad_filter

    def merge_chunk_outputs(
        self,
        *,
        job_dir: Path,
        chunk_results: list[TranscriptArtifacts],
        formats: list[str],
        task: str,
    ) -> TranscriptArtifacts:
        if not chunk_results:
            raise ValueError("No chunk results to merge.")

        all_segments: list[SegmentOut] = []
        full_text_parts: list[str] = []
        total_duration = 0.0
        for chunk in chunk_results:
            all_segments.extend(chunk.segments)
            if chunk.text:
                full_text_parts.append(chunk.text)
            total_duration = max(total_duration, chunk.duration)

        merged_text = "\n".join(full_text_parts).strip()
        merged = write_outputs(
            model_name=chunk_results[0].model_name,
            device=chunk_results[0].device,
            compute_type=chunk_results[0].compute_type,
            job_dir=job_dir,
            text=merged_text,
            language=chunk_results[0].language,
            duration=total_duration,
            segments=all_segments,
            formats=formats,
            task=task,
            vad={},
            chunk_index=0,
            used_batching=True,
            chunk_count=len(chunk_results),
        )
        return merged


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
    device: str,
    compute_type: str,
    task: str,
    vad: dict[str, Any],
    chunk_index: int,
    used_batching: bool,
    chunk_count: int,
) -> TranscriptArtifacts:
    output_files: dict[str, Path] = {}
    suffix = f"_chunk_{chunk_index:04d}" if chunk_index > 0 else ""

    if "txt" in formats:
        txt_path = job_dir / f"transcript{suffix}.txt"
        txt_path.write_text(text, encoding="utf-8")
        output_files["txt"] = txt_path

    if "srt" in formats:
        srt_path = job_dir / f"transcript{suffix}.srt"
        srt_path.write_text(build_srt(segments), encoding="utf-8")
        output_files["srt"] = srt_path

    if "json" in formats:
        json_path = job_dir / f"transcript{suffix}.json"
        payload = {
            "text": text,
            "language": language,
            "duration": duration,
            "model": model_name,
            "device": device,
            "compute_type": compute_type,
            "task": task,
            "vad": vad,
            "segments": [segment.model_dump() for segment in segments],
            "used_batching": used_batching,
            "chunk_count": chunk_count,
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        output_files["json"] = json_path

    return TranscriptArtifacts(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        text=text,
        language=language,
        duration=duration,
        segments=segments,
        output_files=output_files,
        used_batching=used_batching,
        chunk_count=chunk_count,
    )

