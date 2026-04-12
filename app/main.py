"""
Application entry-point -- FastAPI app factory and startup lifecycle.

Global singletons (TranslationService, WhisperService, MediaService) are
created at module level and attached to ``app.state`` during the lifespan
context so they are accessible from dependency injection in endpoint
handlers.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.media.service import MediaService
from app.transcribe.router import router as transcribe_router
from app.transcribe.service import WhisperService
from app.translation.router import router as translation_router
from app.translation.service import TranslationService

# Global service singletons -- created once, shared across all requests
translation_service = TranslationService(models_dir=settings.argos_models_dir)
whisper_service = WhisperService(translation_service=translation_service)
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
    settings.argos_models_dir.mkdir(parents=True, exist_ok=True)
    app.state.ffmpeg_ok = media_service.ffmpeg_available()
    app.state.media_service = media_service
    app.state.whisper_service = whisper_service
    app.state.translation_service = translation_service
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(transcribe_router)
app.include_router(translation_router)
