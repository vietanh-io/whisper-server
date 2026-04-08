import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import settings
from app.transcription.dependencies import (
    FfmpegReady,
    get_media_service,
    get_whisper_service,
)
from app.transcription.media import MediaService
from app.transcription.schemas import HealthResponse, TranscribeFormInput, TranscribeRequest, TranscribeResponse
from app.transcription.service import WhisperService

router = APIRouter(tags=["transcription"])
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
def health(
    media_service: Annotated[MediaService, Depends(get_media_service)],
) -> HealthResponse:
    ffmpeg_ok = media_service.ffmpeg_available()
    return HealthResponse(
        status="ok" if ffmpeg_ok else "degraded",
        service=settings.app_name,
        env=settings.app_env,
        default_backend=settings.default_backend,
        ffmpeg="available" if ffmpeg_ok else "missing",
    )


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    _ffmpeg_ready: Annotated[bool, FfmpegReady],
    media_service: Annotated[MediaService, Depends(get_media_service)],
    whisper_service: Annotated[WhisperService, Depends(get_whisper_service)],
    file: UploadFile | None = File(default=None),
    media_url: str | None = Form(default=None),
    language: str | None = Form(default=None),
    task: str = Form(default=settings.default_task),
    backend: str | None = Form(default=None),
    output_formats: str = Form(default=settings.default_output_formats),
    vad_filter: bool = Form(default=settings.default_vad_filter),
    vad_threshold: float = Form(default=settings.default_vad_threshold),
    min_silence_duration_ms: int = Form(default=settings.default_min_silence_duration_ms),
    speech_pad_ms: int = Form(default=settings.default_speech_pad_ms),
) -> TranscribeResponse:
    if file is None and not media_url:
        raise HTTPException(status_code=400, detail="Provide either file or media_url.")

    form_input = TranscribeFormInput(
        media_url=media_url,
        language=language,
        task=task,
        backend=backend,
        output_formats=output_formats,
        vad_filter=vad_filter,
        vad_threshold=vad_threshold,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )
    request_data: TranscribeRequest = form_input.to_request()

    workspace = media_service.create_workspace(file.filename if file else "remote_media.bin")
    try:
        if file is not None:
            media_service.save_uploaded_file(file, workspace.input_path)
        else:
            media_service.download_media(
                str(request_data.media_url),
                workspace.input_path,
                settings.request_timeout_seconds,
            )
    except Exception as exc:  # noqa: BLE001
        media_service.cleanup_workspace_input(workspace)
        raise HTTPException(status_code=400, detail=f"Cannot prepare media input: {exc}") from exc

    try:
        result = whisper_service.transcribe(
            input_path=workspace.input_path,
            request_data=request_data,
            job_dir=workspace.job_dir,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Transcription failed for job_id=%s: %s", workspace.job_id, exc)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    finally:
        media_service.cleanup_workspace_input(workspace)

    return TranscribeResponse(
        job_id=workspace.job_id,
        backend=result.backend,
        model=result.model_name,
        language=result.language,
        duration=result.duration,
        text=result.text,
        segments=result.segments,
        output_files=media_service.build_output_links(workspace, result.output_files),
    )


@router.get("/outputs/{job_id}/{filename}")
def get_output(
    job_id: str,
    filename: str,
    media_service: Annotated[MediaService, Depends(get_media_service)],
) -> FileResponse:
    try:
        candidate = media_service.resolve_output_file(job_id, filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path=str(candidate), filename=filename)

