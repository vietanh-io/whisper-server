"""
Translation service powered by argostranslate (the engine behind LibreTranslate).

Runs entirely in-process -- no external server needed.  Language pair
packages are downloaded on demand and cached under ``argos_models_dir``.

Typical flow:
    1. ``ensure_package("ja", "en")`` -- download Japanese->English if missing
    2. ``translate("...", "ja", "en")`` -- translate the text

The package index is fetched once at init time and cached.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import argostranslate.package
import argostranslate.translate

logger = logging.getLogger(__name__)


@dataclass
class LanguagePairInfo:
    """Metadata for a single argostranslate language pair."""

    from_code: str
    from_name: str
    to_code: str
    to_name: str


class TranslationService:
    """Manages argostranslate language packages and performs text translation."""

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._models_dir.mkdir(parents=True, exist_ok=True)
        os.environ["ARGOS_PACKAGES_DIR"] = str(self._models_dir)
        self._index_fetched = False

    def _ensure_index(self) -> None:
        """Fetch the remote package index once per process lifetime."""
        if self._index_fetched:
            return
        try:
            argostranslate.package.update_package_index()
            self._index_fetched = True
        except Exception:  # noqa: BLE001
            logger.warning("Failed to fetch argostranslate package index; using cached index if available")

    def ensure_package(self, from_code: str, to_code: str) -> None:
        """Download and install a language pair if not already installed.

        Raises ValueError if the requested pair is not available.
        """
        installed = argostranslate.translate.get_installed_languages()
        from_lang = next((lang for lang in installed if lang.code == from_code), None)
        if from_lang:
            translation = from_lang.get_translation(
                next((lang for lang in installed if lang.code == to_code), None)
            )
            if translation is not None:
                return

        self._ensure_index()
        available = argostranslate.package.get_available_packages()
        package = next(
            (p for p in available if p.from_code == from_code and p.to_code == to_code),
            None,
        )
        if package is None:
            raise ValueError(
                f"No argostranslate package available for {from_code} -> {to_code}. "
                f"Use GET /languages to see available pairs."
            )
        logger.info("Downloading argostranslate package: %s -> %s", from_code, to_code)
        download_path = package.download()
        argostranslate.package.install_from_path(download_path)
        logger.info("Installed argostranslate package: %s -> %s", from_code, to_code)

    def translate(self, text: str, from_code: str, to_code: str) -> str:
        """Translate *text* from one language to another.

        Auto-downloads the language pair package if it hasn't been installed yet.
        """
        if from_code == to_code:
            return text
        self.ensure_package(from_code, to_code)
        return argostranslate.translate.translate(text, from_code, to_code)

    def get_installed_pairs(self) -> list[LanguagePairInfo]:
        """Return all installed language pairs."""
        pairs: list[LanguagePairInfo] = []
        installed_packages = argostranslate.package.get_installed_packages()
        for pkg in installed_packages:
            pairs.append(LanguagePairInfo(
                from_code=pkg.from_code,
                from_name=pkg.from_name,
                to_code=pkg.to_code,
                to_name=pkg.to_name,
            ))
        return pairs

    def get_available_pairs(self) -> list[LanguagePairInfo]:
        """Return all downloadable language pairs from the remote index."""
        self._ensure_index()
        pairs: list[LanguagePairInfo] = []
        for pkg in argostranslate.package.get_available_packages():
            pairs.append(LanguagePairInfo(
                from_code=pkg.from_code,
                from_name=pkg.from_name,
                to_code=pkg.to_code,
                to_name=pkg.to_name,
            ))
        return pairs
