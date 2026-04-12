"""
Whisper transcription service -- the core inference layer.

Responsibilities
----------------
* Load / cache faster-whisper ``WhisperModel`` instances (keyed by
  model name + device + compute_type so the same combo is never loaded
  twice).
* Run transcription via either **sequential** (``model.transcribe``) or
  **batched** (``BatchedInferencePipeline.transcribe``) mode, forwarding
  all decoding, quality-threshold, prompting, and VAD parameters from
  the request.
* When ``task=translate``, Whisper **always transcribes** in the source
  language, then argostranslate translates the text to the requested
  ``target_language``.  This replaces Whisper's built-in translate mode
  and supports translation to any language, not just English.
* Write transcript output files (txt, srt, json) into the job directory.
* Merge chunked results when long-media batching is used.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.utils import available_models

from app.config import settings
from app.transcribe.schemas import SegmentOut, TranscribeRequest
from app.translation.service import TranslationService


@dataclass
class TranscriptArtifacts:
    """Container for everything produced by a single transcription run."""

    model_name: str
    device: str
    compute_type: str
    text: str
    translated_text: str | None
    language: str | None
    duration: float
    segments: list[SegmentOut]
    output_files: dict[str, Path]
    used_batching: bool
    chunk_count: int


class WhisperService:
    """Manages faster-whisper model instances, runs inference, and
    delegates post-transcription translation to TranslationService."""

    def __init__(self, translation_service: TranslationService) -> None:
        self._models: dict[tuple[str, str, str], WhisperModel] = {}
        self._translation = translation_service

    # ── Public API ──────────────────────────────────────────────────────

    def transcribe(
        self,
        input_path: Path,
        request_data: TranscribeRequest,
        job_dir: Path,
        *,
        chunk_index: int = 0,
        chunk_offset_seconds: float = 0.0,
    ) -> TranscriptArtifacts:
        """Transcribe a single WAV file and write outputs to *job_dir*.

        When ``task=translate``, Whisper still runs in transcribe mode
        (producing source-language text), and argostranslate translates
        the output to ``target_language`` afterward.

        When processing chunked audio, *chunk_index* > 0 indicates which
        chunk this is, and *chunk_offset_seconds* shifts all segment
        timestamps so they reflect their position in the original media.
        """
        source_language = request_data.source_language or settings.default_source_language
        if isinstance(source_language, str):
            source_language = source_language.strip().lower() or None

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

        translated_text: str | None = None
        if request_data.task == "translate" and text:
            resolved_source = detected_language or source_language
            target = request_data.target_language or settings.default_target_language
            if resolved_source and resolved_source != target:
                translated_text = self._translation.translate(text, resolved_source, target)

        return write_outputs(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            job_dir=job_dir,
            text=text,
            translated_text=translated_text,
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
        """Return all model names known to faster-whisper."""
        return sorted(available_models())

    def get_downloaded_models(self) -> list[str]:
        """Return model directory names already present in the download root."""
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
        """Download (or ensure cached) a model and return metadata."""
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

    def merge_chunk_outputs(
        self,
        *,
        job_dir: Path,
        chunk_results: list[TranscriptArtifacts],
        formats: list[str],
        task: str,
        target_language: str = "en",
    ) -> TranscriptArtifacts:
        """Merge multiple chunk transcription results into a single output.

        Called after long-media chunking to produce a unified transcript
        file set that covers the entire original media.  When
        ``task=translate``, the merged text is translated as a whole for
        better coherence than per-chunk translation.
        """
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

        translated_text: str | None = None
        if task == "translate" and merged_text:
            source_lang = chunk_results[0].language
            if source_lang and source_lang != target_language:
                translated_text = self._translation.translate(merged_text, source_lang, target_language)

        merged = write_outputs(
            model_name=chunk_results[0].model_name,
            device=chunk_results[0].device,
            compute_type=chunk_results[0].compute_type,
            job_dir=job_dir,
            text=merged_text,
            translated_text=translated_text,
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

    # ── Private helpers ─────────────────────────────────────────────────

    def _get_model(self, *, model_name: str, device: str, compute_type: str) -> WhisperModel:
        """Return a cached model or load a new one.

        Models are keyed by (name, device, compute_type) so requesting
        the same combination twice returns the existing instance.
        """
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
        """Run faster-whisper transcription (batched or sequential).

        Returns (text, detected_language, duration, segments, used_batching).
        """
        use_batch = self._resolve_batch_enabled(request_data)
        use_vad = self._resolve_vad_enabled(request_data)
        vad_parameters = {
            "threshold": request_data.vad.threshold,
            "min_silence_duration_ms": request_data.vad.min_silence_duration_ms,
            "speech_pad_ms": request_data.vad.speech_pad_ms,
        }

        temperature: float | list[float] = request_data.temperature
        if len(request_data.temperature) == 1:
            temperature = request_data.temperature[0]

        common_params: dict[str, Any] = {
            "language": source_language,
            "task": "transcribe",
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
        """Determine whether batched inference should be used.

        Priority: batch_mode explicit > use_batch flag > server default.
        """
        if request_data.batch_mode == "on":
            return True
        if request_data.batch_mode == "off":
            return False
        if request_data.use_batch is not None:
            return request_data.use_batch
        return settings.batch_enabled

    def _resolve_vad_enabled(self, request_data: TranscribeRequest) -> bool:
        """Determine whether Silero VAD should be applied.

        Priority: vad_mode explicit > request-level vad_filter toggle.
        """
        if request_data.vad_mode == "on":
            return True
        if request_data.vad_mode == "off":
            return False
        return request_data.vad.vad_filter


# ── Output file writers (module-level helpers) ──────────────────────────

def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format ``HH:MM:SS,mmm``."""
    ms_total = int(round(seconds * 1000))
    hours, remainder = divmod(ms_total, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_srt(segments: list[SegmentOut]) -> str:
    """Build an SRT subtitle string from a list of segments."""
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
    translated_text: str | None,
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
    """Write requested output files (txt/srt/json) into *job_dir* and
    return a ``TranscriptArtifacts`` summary.

    When *translated_text* is provided (task=translate), output files
    contain the translated text as the primary content and the original
    source-language text is preserved in a ``_original`` companion file.
    """
    output_files: dict[str, Path] = {}
    suffix = f"_chunk_{chunk_index:04d}" if chunk_index > 0 else ""
    primary_text = translated_text if translated_text else text

    if "txt" in formats:
        txt_path = job_dir / f"transcript{suffix}.txt"
        txt_path.write_text(primary_text, encoding="utf-8")
        output_files["txt"] = txt_path
        if translated_text:
            original_path = job_dir / f"transcript{suffix}_original.txt"
            original_path.write_text(text, encoding="utf-8")
            output_files["txt_original"] = original_path

    if "srt" in formats:
        srt_path = job_dir / f"transcript{suffix}.srt"
        srt_path.write_text(build_srt(segments), encoding="utf-8")
        output_files["srt"] = srt_path

    if "json" in formats:
        json_path = job_dir / f"transcript{suffix}.json"
        payload: dict[str, Any] = {
            "text": primary_text,
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
        if translated_text:
            payload["original_text"] = text
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        output_files["json"] = json_path

    return TranscriptArtifacts(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        text=text,
        translated_text=translated_text,
        language=language,
        duration=duration,
        segments=segments,
        output_files=output_files,
        used_batching=used_batching,
        chunk_count=chunk_count,
    )
