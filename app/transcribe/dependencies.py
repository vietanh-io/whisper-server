"""
FastAPI dependency-injection helpers for the transcribe module.

Services are created once at startup and stored on ``app.state``.
These functions retrieve them so endpoint handlers can declare them
via ``Depends()``.

``FfmpegReady`` is a reusable dependency that gates any endpoint
requiring ffmpeg -- it returns 503 if ffmpeg was not found at startup.
"""

from typing import cast

from fastapi import Depends, HTTPException, Request

from app.media.service import MediaService
from app.transcribe.service import WhisperService


def get_media_service(request: Request) -> MediaService:
    """Retrieve the global MediaService instance from app state."""
    return cast(MediaService, request.app.state.media_service)


def get_whisper_service(request: Request) -> WhisperService:
    """Retrieve the global WhisperService instance from app state."""
    return cast(WhisperService, request.app.state.whisper_service)


def require_ffmpeg(request: Request) -> bool:
    """Dependency that rejects the request if ffmpeg is unavailable."""
    ffmpeg_ok = bool(getattr(request.app.state, "ffmpeg_ok", False))
    if not ffmpeg_ok:
        raise HTTPException(
            status_code=503,
            detail="FFmpeg is required for transcription but was not found in PATH.",
        )
    return ffmpeg_ok


FfmpegReady = Depends(require_ffmpeg)
