"""
HTTP endpoints for the transcription API.

Endpoint summary
----------------
GET  /health                  Server health + ffmpeg status
GET  /models                  List available / downloaded whisper models
POST /models/download         Download a model by name
POST /transcribe              Transcribe a single file or URL (form-data)
POST /transcribe/uploads      Transcribe multiple file uploads (form-data)
POST /transcribe/links        Transcribe a list of remote URLs (JSON body)
GET  /outputs/{job_id}/{fn}   Download a transcript output file

Data flow
---------
1. Form fields are collected into ``TranscribeFormInput`` via
   ``_build_request_from_form`` and converted to a canonical
   ``TranscribeRequest``.
2. The request is routed to either ``_transcribe_single_upload`` (file)
   or ``_transcribe_single_remote_link`` (URL).
3. Both paths normalise input to a 16 kHz WAV via ``MediaService``, then
   delegate to ``_transcribe_prepared_wav`` which decides whether to
   chunk (long-media batching) or transcribe directly.
4. When ``task=translate``, Whisper always transcribes in the source
   language, then argostranslate translates the text to ``target_language``.
"""

import logging
from typing import Annotated
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import settings
from app.media.service import MediaService
from app.transcribe.dependencies import (
    FfmpegReady,
    get_media_service,
    get_whisper_service,
)
from app.transcribe.schemas import (
    AvailableModelsResponse,
    BatchTranscribeResponse,
    DownloadModelRequest,
    HealthResponse,
    RemoteLinksRequest,
    TranscribeFormInput,
    TranscribeRequest,
    TranscribeResponse,
)
from app.transcribe.service import WhisperService

router = APIRouter(tags=["transcription"])
logger = logging.getLogger(__name__)


# ── Health & model management ───────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health(
    media_service: Annotated[MediaService, Depends(get_media_service)],
) -> HealthResponse:
    ffmpeg_ok = media_service.ffmpeg_available()
    return HealthResponse(
        status="ok" if ffmpeg_ok else "degraded",
        service=settings.app_name,
        env=settings.app_env,
        ffmpeg="available" if ffmpeg_ok else "missing",
    )


@router.get("/models", response_model=AvailableModelsResponse)
def list_models(
    whisper_service: Annotated[WhisperService, Depends(get_whisper_service)],
) -> AvailableModelsResponse:
    return AvailableModelsResponse(
        available_model_names=whisper_service.get_available_models(),
        downloaded_models=whisper_service.get_downloaded_models(),
    )


@router.post("/models/download")
def download_model(
    payload: DownloadModelRequest,
    whisper_service: Annotated[WhisperService, Depends(get_whisper_service)],
) -> dict[str, str]:
    try:
        return whisper_service.download_model(
            model_name=payload.model,
            device=payload.device,
            compute_type=payload.compute_type,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Model download failed: {exc}") from exc


# ── Form-data → TranscribeRequest adapter ───────────────────────────────

def _build_request_from_form(
    *,
    source_language: str | None,
    target_language: str,
    task: str,
    output_formats: str,
    vad_filter: bool,
    vad_threshold: float,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
    model: str | None,
    device: str | None,
    compute_type: str | None,
    beam_size: int,
    word_timestamps: bool,
    batch_size: int,
    batch_mode: str,
    vad_mode: str,
    use_batch: bool | None,
    temperature: str,
    best_of: int,
    patience: float,
    length_penalty: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    compression_ratio_threshold: float | None,
    log_prob_threshold: float | None,
    no_speech_threshold: float | None,
    condition_on_previous_text: bool,
    initial_prompt: str | None,
    prompt_reset_on_temperature: float,
    hotwords: str | None,
    prefix: str | None,
    hallucination_silence_threshold: float | None,
    suppress_blank: bool,
    without_timestamps: bool,
    max_initial_timestamp: float,
    max_new_tokens: int | None,
) -> TranscribeRequest:
    """Construct a validated TranscribeRequest from raw form field values."""
    return TranscribeFormInput(
        media_url=None,
        source_language=source_language,
        target_language=target_language,
        task=task,
        output_formats=output_formats,
        vad_filter=vad_filter,
        vad_threshold=vad_threshold,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        model=model,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        batch_size=batch_size,
        batch_mode=batch_mode,
        vad_mode=vad_mode,
        use_batch=use_batch,
        temperature=temperature,
        best_of=best_of,
        patience=patience,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        compression_ratio_threshold=compression_ratio_threshold,
        log_prob_threshold=log_prob_threshold,
        no_speech_threshold=no_speech_threshold,
        condition_on_previous_text=condition_on_previous_text,
        initial_prompt=initial_prompt,
        prompt_reset_on_temperature=prompt_reset_on_temperature,
        hotwords=hotwords,
        prefix=prefix,
        hallucination_silence_threshold=hallucination_silence_threshold,
        suppress_blank=suppress_blank,
        without_timestamps=without_timestamps,
        max_initial_timestamp=max_initial_timestamp,
        max_new_tokens=max_new_tokens,
    ).to_request()


# ── Batching decision helpers ───────────────────────────────────────────

def _request_wants_batch(request_data: TranscribeRequest) -> bool:
    """Check if the request explicitly or implicitly wants batching."""
    if request_data.batch_mode == "on":
        return True
    if request_data.batch_mode == "off":
        return False
    if request_data.use_batch is not None:
        return request_data.use_batch
    return settings.batch_enabled


def _should_batch(duration_seconds: float, request_data: TranscribeRequest) -> bool:
    """Decide whether to chunk-and-batch based on duration and request flags."""
    if request_data.batch_mode == "on":
        return True
    return _request_wants_batch(request_data) and duration_seconds >= settings.batch_threshold_seconds


# ── Core transcription pipeline ─────────────────────────────────────────

def _transcribe_prepared_wav(
    *,
    media_service: MediaService,
    whisper_service: WhisperService,
    request_data: TranscribeRequest,
    workspace_input_wav: Path,
    workspace_job_dir: Path,
) -> TranscribeResponse:
    """Transcribe a normalised WAV file, optionally chunking for long media."""
    duration_seconds = media_service.probe_duration_seconds(workspace_input_wav)
    if _should_batch(duration_seconds, request_data):
        chunk_dir = workspace_job_dir / "chunks"
        chunks = media_service.split_audio_chunks(
            source_path=workspace_input_wav,
            chunk_seconds=settings.batch_chunk_seconds,
            chunk_dir=chunk_dir,
        )
        if not chunks:
            raise HTTPException(status_code=500, detail="Chunking produced no output files.")

        chunk_results = []
        for idx, chunk in enumerate(chunks):
            chunk_results.append(
                whisper_service.transcribe(
                    input_path=chunk,
                    request_data=request_data,
                    job_dir=workspace_job_dir,
                    chunk_index=idx + 1,
                    chunk_offset_seconds=float(idx * settings.batch_chunk_seconds),
                )
            )
        merged = whisper_service.merge_chunk_outputs(
            job_dir=workspace_job_dir,
            chunk_results=chunk_results,
            formats=request_data.output_formats,
            task=request_data.task,
            target_language=request_data.target_language,
        )
        response_files = media_service.build_output_links_by_job_dir(
            job_dir=workspace_job_dir,
            output_files=merged.output_files,
        )
        return TranscribeResponse(
            job_id=workspace_job_dir.name,
            model=merged.model_name,
            device=merged.device,
            compute_type=merged.compute_type,
            language=merged.language,
            duration=merged.duration,
            text=merged.text,
            translated_text=merged.translated_text,
            segments=merged.segments,
            output_files=response_files,
            used_batching=True,
            chunk_count=merged.chunk_count,
        )

    single = whisper_service.transcribe(
        input_path=workspace_input_wav,
        request_data=request_data,
        job_dir=workspace_job_dir,
    )
    response_files = media_service.build_output_links_by_job_dir(
        job_dir=workspace_job_dir,
        output_files=single.output_files,
    )
    return TranscribeResponse(
        job_id=workspace_job_dir.name,
        model=single.model_name,
        device=single.device,
        compute_type=single.compute_type,
        language=single.language,
        duration=single.duration,
        text=single.text,
        translated_text=single.translated_text,
        segments=single.segments,
        output_files=response_files,
        used_batching=single.used_batching,
        chunk_count=single.chunk_count,
    )


# ── Single-item transcription helpers ───────────────────────────────────

def _transcribe_single_upload(
    *,
    file: UploadFile,
    request_data: TranscribeRequest,
    media_service: MediaService,
    whisper_service: WhisperService,
) -> TranscribeResponse:
    """Save an uploaded file, transcode to WAV, and transcribe it."""
    workspace = media_service.create_workspace(file.filename)
    uploaded_path = workspace.input_dir / (file.filename or "upload.bin")
    wav_path = workspace.input_dir / "normalized.wav"
    try:
        media_service.save_uploaded_file(file, uploaded_path)
        media_service.transcode_to_wav(str(uploaded_path), wav_path)
        response = _transcribe_prepared_wav(
            media_service=media_service,
            whisper_service=whisper_service,
            request_data=request_data,
            workspace_input_wav=wav_path,
            workspace_job_dir=workspace.job_dir,
        )
        return response
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        media_service.cleanup_workspace_input(workspace)


def _transcribe_single_remote_link(
    *,
    media_url: str,
    request_data: TranscribeRequest,
    media_service: MediaService,
    whisper_service: WhisperService,
) -> TranscribeResponse:
    """Resolve a remote URL, transcode to WAV via ffmpeg, and transcribe."""
    workspace = media_service.create_workspace("remote_link.wav")
    wav_path = workspace.input_dir / "normalized.wav"
    try:
        normalized_url = media_service.normalize_remote_url(media_url)
        media_service.transcode_to_wav(normalized_url, wav_path)
        response = _transcribe_prepared_wav(
            media_service=media_service,
            whisper_service=whisper_service,
            request_data=request_data,
            workspace_input_wav=wav_path,
            workspace_job_dir=workspace.job_dir,
        )
        return response
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        media_service.cleanup_workspace_input(workspace)


# ── HTTP endpoints ──────────────────────────────────────────────────────

@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_single(
    _ffmpeg_ready: Annotated[bool, FfmpegReady],
    media_service: Annotated[MediaService, Depends(get_media_service)],
    whisper_service: Annotated[WhisperService, Depends(get_whisper_service)],
    file: UploadFile | None = File(default=None),
    media_url: str | None = Form(default=None),
    source_language: str | None = Form(default=settings.default_source_language),
    target_language: str = Form(default=settings.default_target_language),
    task: str = Form(default=settings.default_task),
    output_formats: str = Form(default=settings.default_output_formats),
    model: str | None = Form(default=None),
    device: str | None = Form(default=None),
    compute_type: str | None = Form(default=None),
    beam_size: int = Form(default=settings.default_beam_size),
    word_timestamps: bool = Form(default=settings.default_word_timestamps),
    batch_size: int = Form(default=settings.default_batch_size),
    batch_mode: str = Form(default=settings.default_batch_mode),
    vad_mode: str = Form(default=settings.default_vad_mode),
    use_batch: bool | None = Form(default=None),
    vad_filter: bool = Form(default=settings.default_vad_filter),
    vad_threshold: float = Form(default=settings.default_vad_threshold),
    min_silence_duration_ms: int = Form(default=settings.default_min_silence_duration_ms),
    speech_pad_ms: int = Form(default=settings.default_speech_pad_ms),
    temperature: str = Form(default=settings.default_temperature),
    best_of: int = Form(default=settings.default_best_of),
    patience: float = Form(default=settings.default_patience),
    length_penalty: float = Form(default=settings.default_length_penalty),
    repetition_penalty: float = Form(default=settings.default_repetition_penalty),
    no_repeat_ngram_size: int = Form(default=settings.default_no_repeat_ngram_size),
    compression_ratio_threshold: float | None = Form(default=settings.default_compression_ratio_threshold),
    log_prob_threshold: float | None = Form(default=settings.default_log_prob_threshold),
    no_speech_threshold: float | None = Form(default=settings.default_no_speech_threshold),
    condition_on_previous_text: bool = Form(default=settings.default_condition_on_previous_text),
    initial_prompt: str | None = Form(default=settings.default_initial_prompt),
    prompt_reset_on_temperature: float = Form(default=settings.default_prompt_reset_on_temperature),
    hotwords: str | None = Form(default=settings.default_hotwords),
    prefix: str | None = Form(default=settings.default_prefix),
    hallucination_silence_threshold: float | None = Form(default=settings.default_hallucination_silence_threshold),
    suppress_blank: bool = Form(default=settings.default_suppress_blank),
    without_timestamps: bool = Form(default=settings.default_without_timestamps),
    max_initial_timestamp: float = Form(default=settings.default_max_initial_timestamp),
    max_new_tokens: int | None = Form(default=settings.default_max_new_tokens),
) -> TranscribeResponse:
    """Transcribe a single file upload or a single remote URL."""
    if file is None and not media_url:
        raise HTTPException(status_code=400, detail="Provide either file or media_url.")

    request_data = _build_request_from_form(
        source_language=source_language,
        target_language=target_language,
        task=task,
        output_formats=output_formats,
        vad_filter=vad_filter,
        vad_threshold=vad_threshold,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        model=model,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        batch_size=batch_size,
        batch_mode=batch_mode,
        vad_mode=vad_mode,
        use_batch=use_batch,
        temperature=temperature,
        best_of=best_of,
        patience=patience,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        compression_ratio_threshold=compression_ratio_threshold,
        log_prob_threshold=log_prob_threshold,
        no_speech_threshold=no_speech_threshold,
        condition_on_previous_text=condition_on_previous_text,
        initial_prompt=initial_prompt,
        prompt_reset_on_temperature=prompt_reset_on_temperature,
        hotwords=hotwords,
        prefix=prefix,
        hallucination_silence_threshold=hallucination_silence_threshold,
        suppress_blank=suppress_blank,
        without_timestamps=without_timestamps,
        max_initial_timestamp=max_initial_timestamp,
        max_new_tokens=max_new_tokens,
    )
    try:
        if file is not None:
            return _transcribe_single_upload(
                file=file,
                request_data=request_data,
                media_service=media_service,
                whisper_service=whisper_service,
            )
        else:
            return _transcribe_single_remote_link(
                media_url=media_url or "",
                request_data=request_data,
                media_service=media_service,
                whisper_service=whisper_service,
            )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Single transcription failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc


@router.post("/transcribe/uploads", response_model=BatchTranscribeResponse)
async def transcribe_uploads(
    _ffmpeg_ready: Annotated[bool, FfmpegReady],
    media_service: Annotated[MediaService, Depends(get_media_service)],
    whisper_service: Annotated[WhisperService, Depends(get_whisper_service)],
    files: list[UploadFile] = File(default_factory=list),
    source_language: str | None = Form(default=settings.default_source_language),
    target_language: str = Form(default=settings.default_target_language),
    task: str = Form(default=settings.default_task),
    output_formats: str = Form(default=settings.default_output_formats),
    model: str | None = Form(default=None),
    device: str | None = Form(default=None),
    compute_type: str | None = Form(default=None),
    beam_size: int = Form(default=settings.default_beam_size),
    word_timestamps: bool = Form(default=settings.default_word_timestamps),
    batch_size: int = Form(default=settings.default_batch_size),
    batch_mode: str = Form(default=settings.default_batch_mode),
    vad_mode: str = Form(default=settings.default_vad_mode),
    use_batch: bool | None = Form(default=None),
    vad_filter: bool = Form(default=settings.default_vad_filter),
    vad_threshold: float = Form(default=settings.default_vad_threshold),
    min_silence_duration_ms: int = Form(default=settings.default_min_silence_duration_ms),
    speech_pad_ms: int = Form(default=settings.default_speech_pad_ms),
    temperature: str = Form(default=settings.default_temperature),
    best_of: int = Form(default=settings.default_best_of),
    patience: float = Form(default=settings.default_patience),
    length_penalty: float = Form(default=settings.default_length_penalty),
    repetition_penalty: float = Form(default=settings.default_repetition_penalty),
    no_repeat_ngram_size: int = Form(default=settings.default_no_repeat_ngram_size),
    compression_ratio_threshold: float | None = Form(default=settings.default_compression_ratio_threshold),
    log_prob_threshold: float | None = Form(default=settings.default_log_prob_threshold),
    no_speech_threshold: float | None = Form(default=settings.default_no_speech_threshold),
    condition_on_previous_text: bool = Form(default=settings.default_condition_on_previous_text),
    initial_prompt: str | None = Form(default=settings.default_initial_prompt),
    prompt_reset_on_temperature: float = Form(default=settings.default_prompt_reset_on_temperature),
    hotwords: str | None = Form(default=settings.default_hotwords),
    prefix: str | None = Form(default=settings.default_prefix),
    hallucination_silence_threshold: float | None = Form(default=settings.default_hallucination_silence_threshold),
    suppress_blank: bool = Form(default=settings.default_suppress_blank),
    without_timestamps: bool = Form(default=settings.default_without_timestamps),
    max_initial_timestamp: float = Form(default=settings.default_max_initial_timestamp),
    max_new_tokens: int | None = Form(default=settings.default_max_new_tokens),
) -> BatchTranscribeResponse:
    """Transcribe multiple uploaded files. Each file is processed sequentially."""
    if not files:
        raise HTTPException(status_code=400, detail="Provide at least one file in multipart/form-data.")
    request_data = _build_request_from_form(
        source_language=source_language,
        target_language=target_language,
        task=task,
        output_formats=output_formats,
        vad_filter=vad_filter,
        vad_threshold=vad_threshold,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        model=model,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        batch_size=batch_size,
        batch_mode=batch_mode,
        vad_mode=vad_mode,
        use_batch=use_batch,
        temperature=temperature,
        best_of=best_of,
        patience=patience,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        compression_ratio_threshold=compression_ratio_threshold,
        log_prob_threshold=log_prob_threshold,
        no_speech_threshold=no_speech_threshold,
        condition_on_previous_text=condition_on_previous_text,
        initial_prompt=initial_prompt,
        prompt_reset_on_temperature=prompt_reset_on_temperature,
        hotwords=hotwords,
        prefix=prefix,
        hallucination_silence_threshold=hallucination_silence_threshold,
        suppress_blank=suppress_blank,
        without_timestamps=without_timestamps,
        max_initial_timestamp=max_initial_timestamp,
        max_new_tokens=max_new_tokens,
    )
    items: list[TranscribeResponse] = []
    for file in files:
        items.append(
            _transcribe_single_upload(
                file=file,
                request_data=request_data,
                media_service=media_service,
                whisper_service=whisper_service,
            )
        )
    return BatchTranscribeResponse(items=items)


@router.post("/transcribe/links", response_model=BatchTranscribeResponse)
async def transcribe_links(
    _ffmpeg_ready: Annotated[bool, FfmpegReady],
    payload: RemoteLinksRequest,
    media_service: Annotated[MediaService, Depends(get_media_service)],
    whisper_service: Annotated[WhisperService, Depends(get_whisper_service)],
) -> BatchTranscribeResponse:
    """Transcribe a list of remote media URLs (JSON body)."""
    request_data = TranscribeRequest(
        source_language=payload.source_language,
        target_language=payload.target_language,
        task=payload.task,
        output_formats=payload.output_formats,
        model=payload.model,
        device=payload.device,
        compute_type=payload.compute_type,
        beam_size=payload.beam_size,
        word_timestamps=payload.word_timestamps,
        batch_size=payload.batch_size,
        batch_mode=payload.batch_mode,
        vad_mode=payload.vad_mode,
        use_batch=payload.use_batch,
        vad=payload.vad,
        temperature=payload.temperature,
        best_of=payload.best_of,
        patience=payload.patience,
        length_penalty=payload.length_penalty,
        repetition_penalty=payload.repetition_penalty,
        no_repeat_ngram_size=payload.no_repeat_ngram_size,
        compression_ratio_threshold=payload.compression_ratio_threshold,
        log_prob_threshold=payload.log_prob_threshold,
        no_speech_threshold=payload.no_speech_threshold,
        condition_on_previous_text=payload.condition_on_previous_text,
        initial_prompt=payload.initial_prompt,
        prompt_reset_on_temperature=payload.prompt_reset_on_temperature,
        hotwords=payload.hotwords,
        prefix=payload.prefix,
        hallucination_silence_threshold=payload.hallucination_silence_threshold,
        suppress_blank=payload.suppress_blank,
        without_timestamps=payload.without_timestamps,
        max_initial_timestamp=payload.max_initial_timestamp,
        max_new_tokens=payload.max_new_tokens,
    )
    items: list[TranscribeResponse] = []
    for link in payload.links:
        try:
            items.append(
                _transcribe_single_remote_link(
                    media_url=link,
                    request_data=request_data,
                    media_service=media_service,
                    whisper_service=whisper_service,
                )
            )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Link transcription failed for %s: %s", link, exc)
            raise HTTPException(status_code=500, detail=f"Transcription failed for {link}: {exc}") from exc
    return BatchTranscribeResponse(items=items)


# ── Output file download ───────────────────────────────────────────────

@router.get("/outputs/{job_id}/{filename}")
def get_output(
    job_id: str,
    filename: str,
    media_service: Annotated[MediaService, Depends(get_media_service)],
) -> FileResponse:
    """Serve a transcript output file (txt, srt, or json) for download."""
    try:
        candidate = media_service.resolve_output_file(job_id, filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path=str(candidate), filename=filename)
