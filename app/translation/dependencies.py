"""FastAPI dependency-injection helpers for the translation module."""

from typing import cast

from fastapi import Request

from app.translation.service import TranslationService


def get_translation_service(request: Request) -> TranslationService:
    """Retrieve the global TranslationService instance from app state."""
    return cast(TranslationService, request.app.state.translation_service)
