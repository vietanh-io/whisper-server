"""
HTTP endpoints for the translation (argostranslate) API.

Endpoint summary
----------------
GET  /languages               List available / installed translation pairs
POST /languages/download      Pre-download a translation language pair
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.translation.dependencies import get_translation_service
from app.translation.schemas import (
    DownloadLanguageRequest,
    LanguagePairOut,
    LanguagesResponse,
)
from app.translation.service import TranslationService

router = APIRouter(tags=["translation"])


@router.get("/languages", response_model=LanguagesResponse)
def list_languages(
    translation_service: Annotated[TranslationService, Depends(get_translation_service)],
) -> LanguagesResponse:
    """List available and installed argostranslate language pairs."""
    try:
        available = [
            LanguagePairOut(
                from_code=p.from_code, from_name=p.from_name,
                to_code=p.to_code, to_name=p.to_name,
            )
            for p in translation_service.get_available_pairs()
        ]
    except Exception:  # noqa: BLE001
        available = []
    installed = [
        LanguagePairOut(
            from_code=p.from_code, from_name=p.from_name,
            to_code=p.to_code, to_name=p.to_name,
        )
        for p in translation_service.get_installed_pairs()
    ]
    return LanguagesResponse(available=available, installed=installed)


@router.post("/languages/download")
def download_language(
    payload: DownloadLanguageRequest,
    translation_service: Annotated[TranslationService, Depends(get_translation_service)],
) -> dict[str, str]:
    """Pre-download a language pair so the first translate request is fast."""
    try:
        translation_service.ensure_package(payload.from_code, payload.to_code)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Language download failed: {exc}") from exc
    return {"from_code": payload.from_code, "to_code": payload.to_code, "status": "installed"}
