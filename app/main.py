"""
Application entry-point -- FastAPI app factory and startup lifecycle.

Global singletons (WhisperService, MediaService) are created at module
level and attached to ``app.state`` during the lifespan context so they
are accessible from dependency injection in endpoint handlers.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import settings
from app.transcription.media import MediaService
from app.transcription.service import WhisperService

# Global service singletons -- created once, shared across all requests
whisper_service = WhisperService()
media_service = MediaService(output_dir=settings.output_dir, temp_dir=settings.temp_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle hook.

    On startup: ensure runtime directories exist, check ffmpeg
    availability, and attach services to app.state for DI access.
    """
    media_service.ensure_runtime_dirs()
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.faster_whisper_download_root.mkdir(parents=True, exist_ok=True)
    app.state.ffmpeg_ok = media_service.ffmpeg_available()
    app.state.media_service = media_service
    app.state.whisper_service = whisper_service
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(api_router)
