"""Pydantic models for the translation domain."""

from pydantic import BaseModel


class LanguagePairOut(BaseModel):
    """A single source->target translation pair."""

    from_code: str
    from_name: str
    to_code: str
    to_name: str


class LanguagesResponse(BaseModel):
    available: list[LanguagePairOut]
    installed: list[LanguagePairOut]


class DownloadLanguageRequest(BaseModel):
    from_code: str
    to_code: str
