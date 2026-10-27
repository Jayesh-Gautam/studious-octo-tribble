"""Light, lossless-in-meaning cleaning of the SDR ``Discrepancy`` text.

Heavier normalisation (abbreviations, spelling, stemming) is Phase 2. Here we only
make the text consistent so deduplication and EDA are meaningful.
"""
from __future__ import annotations

import re
import unicodedata

from aerofix.config import load_config

_WS = re.compile(r"\s+")
_DEDUP_STRIP = re.compile(r"[^A-Z0-9 ]")


def basic_clean(text: str | None) -> str:
    """Uppercase, fold accents/odd characters to ASCII, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return _WS.sub(" ", text.upper()).strip()


def strip_boilerplate(text: str, prefixes: dict[str, str] | None = None) -> tuple[str, str]:
    """Remove a leading situation phrase and return (text, context_label).

    e.g. "AIRCRAFT WAS NOT GROUNDED: L1 DOOR ..." -> ("L1 DOOR ...", "NOT_GROUNDED").
    The label is kept as a feature because it carries real information (severity),
    but it is not part of the fault description.
    """
    prefixes = prefixes or load_config()["preprocess"]["boilerplate_prefixes"]
    for phrase, label in prefixes.items():
        if text.startswith(phrase):
            return text[len(phrase):].lstrip(" :-"), label
    return text, "NONE"


def dedup_key(operator: str | None, text: str) -> str:
    """Key for near-duplicate detection: same operator + same text ignoring
    punctuation and spacing (e.g. 'R/H' vs 'RH', double spaces)."""
    norm = _WS.sub(" ", _DEDUP_STRIP.sub(" ", text)).strip()
    return f"{(operator or '').strip().upper()}|{norm}"
