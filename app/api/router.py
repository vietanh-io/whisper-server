"""
Top-level API router -- aggregates all sub-module routers.

Routes are mounted without a URL prefix so endpoints live at the
application root (e.g. ``/health``, ``/transcribe``).
"""

from fastapi import APIRouter

from app.transcription.router import router as transcription_router

api_router = APIRouter()
api_router.include_router(transcription_router)
