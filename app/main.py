from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import settings
from app.transcription.media import MediaService
from app.transcription.service import WhisperService

whisper_service = WhisperService()
media_service = MediaService(output_dir=settings.output_dir, temp_dir=settings.temp_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    media_service.ensure_runtime_dirs()
    app.state.ffmpeg_ok = media_service.ffmpeg_available()
    app.state.media_service = media_service
    app.state.whisper_service = whisper_service
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(api_router)

