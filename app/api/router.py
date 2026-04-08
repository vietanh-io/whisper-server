from fastapi import APIRouter

from app.transcription.router import router as transcription_router

api_router = APIRouter()
api_router.include_router(transcription_router)

