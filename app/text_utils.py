from __future__ import annotations

from typing import Any


_GERMAN_ASCII_TRANSLATION = str.maketrans(
    {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "Ä": "Ae",
        "Ö": "Oe",
        "Ü": "Ue",
        "ß": "ss",
    }
)


def normalize_german_text(value: Any) -> str:
    """Normalize German text so ASCII transliterations and umlauts compare equally."""
    if value is None:
        return ""
    return str(value).translate(_GERMAN_ASCII_TRANSLATION).casefold()


def contains_german_text(value: Any, tokens: list[str] | tuple[str, ...] | set[str]) -> bool:
    normalized = normalize_german_text(value)
    return any(normalize_german_text(token) in normalized for token in tokens)
