"""Tokeniser for uppercase maintenance text.

Generic tokenisers split "R&R", "P/N" or "737-700" into pieces, which destroys
their meaning. Here a token is:
    [REF]                       the reference mask token
    W/  B/                      single letter + slash ("with", "by" shorthand)
    R&R  P/N  A/C  737-700  O-RING  3.5   alphanumerics joined by & / ' . -
Punctuation between tokens is dropped.
"""
from __future__ import annotations

import re

TOKEN = re.compile(r"\[REF\]|\b[A-Z]/(?=\s|$)|[A-Z0-9]+(?:[&/'.\-][A-Z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.upper()) if text else []


def is_word(tok: str) -> bool:
    """Plain alphabetic token: the only kind spelling correction may touch."""
    return tok.isalpha()
