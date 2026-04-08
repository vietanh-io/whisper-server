from typing import cast

from fastapi import Depends, HTTPException, Request

from app.transcription.media import MediaService
from app.transcription.service import WhisperService


def get_media_service(request: Request) -> MediaService:
    return cast(MediaService, request.app.state.media_service)


def get_whisper_service(request: Request) -> WhisperService:
    return cast(WhisperService, request.app.state.whisper_service)


def require_ffmpeg(request: Request) -> bool:
    ffmpeg_ok = bool(getattr(request.app.state, "ffmpeg_ok", False))
    if not ffmpeg_ok:
        raise HTTPException(
            status_code=503,
            detail="FFmpeg is required for transcription but was not found in PATH.",
        )
    return ffmpeg_ok


FfmpegReady = Depends(require_ffmpeg)

