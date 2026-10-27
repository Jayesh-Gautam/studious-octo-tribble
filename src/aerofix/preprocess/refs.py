"""Find, extract and mask maintenance-manual references in SDR text.

Why this matters: technicians write e.g. "REPLACED PER AMM 52-10-00". The first two
digits of the reference are the ATA chapter, i.e. the label we want to predict. A
classifier trained on raw text can "cheat" by reading the reference. So:

* ``extract_refs`` keeps the references as structured data (for the reference
  recommender in Phase 6), and
* ``mask_refs`` replaces them with ``[REF]`` in classifier / retriever inputs.

Formats handled (all seen in the 2025 data):
    AMM 52-10-00             SRM 53-30-15-2R-1         M/M 33-51-06 (-> AMM)
    FIM 27-11 TASK 804       AMM TASK 32-11-00-200-801
    SRM 51-10-02-0G; 51-40-02-0G    (chained: second ref inherits "SRM")
    PER 32-51-11 / IAW 32-51-11     (bare chapter numbers after PER/IAW/REF -> type REF)
    SRM B737-700 51-20-01 / SRM-51-10-02 / NDT PT 6-51-00-00   (aircraft / part in between)

Safety net for masking only: any leftover NN-NN-NN number is also masked, because
it is almost always an ATA reference in an unusual format. It is NOT added to the
extracted refs (a few could be dates), and masking a date loses nothing.
"""
from __future__ import annotations

import re
from functools import lru_cache

from aerofix.config import load_config

_NUM = r"(?P<num>\d{2}-\d{2}(?:-\d{2,3})?)(?P<suffix>(?:-[0-9A-Z]{1,4})*)"
_ALIASES = {"MM": "AMM", "M/M": "AMM"}


_SEP = r"[\s\-:#.]*"
_AIRCRAFT = r"(?:(?:B?737|A3[12]\d)(?:-\d{1,3}[A-Z]{0,2})?[\s\-]+)?"  # "B737-700 " between type and number
_KEYWORD = r"(?:(?:TASK|REF|CHAPTER|CHAP|CH|FIG|SECTION|SECT|PART|PT)\.?\s*\d?" + _SEP + ")?"
_LEFTOVER = re.compile(r"\b\d{2}-\d{2}-\d{2}(?:-[0-9A-Z]{1,4})*\b")


@lru_cache(maxsize=1)
def _patterns() -> tuple[re.Pattern, re.Pattern, re.Pattern]:
    types = sorted(load_config()["preprocess"]["ref_types"], key=len, reverse=True)
    alt = "|".join(re.escape(t) for t in types)  # longest first so SWPM beats SWP...
    main = re.compile(
        rf"\b(?P<type>{alt})\b{_SEP}{_AIRCRAFT}{_KEYWORD}{_NUM}(?:\s*TASK\s*\d{{3,4}})?"
    )
    # A further number joined by ; , & / AND right after a match: same manual type.
    chain = re.compile(rf"\s*(?:[;,&/]|AND)\s*(?P<num>\d{{2}}-\d{{2}}-\d{{2,3}})(?P<suffix>(?:-[0-9A-Z]{{1,4}})*)")
    bare = re.compile(rf"\b(?:PER|IAW|I\.A\.W\.?|REF(?:ERENCE)?\.?|IN ACCORDANCE WITH)\s*:?\s*(?:TASK\s*)?{_NUM}")
    return main, chain, bare


def find_refs(text: str) -> list[tuple[int, int, str, str]]:
    """Return (start, end, manual_type, chapter_number) for each reference, in text order."""
    if not text:
        return []
    main, chain, bare = _patterns()
    found: list[tuple[int, int, str, str]] = []
    for m in main.finditer(text):
        rtype = _ALIASES.get(m.group("type"), m.group("type"))
        found.append((m.start(), m.end(), rtype, m.group("num")))
        pos = m.end()
        while c := chain.match(text, pos):
            found.append((c.start("num"), c.end(), rtype, c.group("num")))
            pos = c.end()
    taken = [(s, e) for s, e, *_ in found]
    for m in bare.finditer(text):
        s = m.start("num")
        if not any(a <= s < b for a, b in taken):  # already covered by a typed ref
            found.append((s, m.end(), "REF", m.group("num")))
    return sorted(found)


def extract_refs(text: str) -> list[str]:
    """Unique references like 'AMM 52-10-00', in order of first appearance."""
    return list(dict.fromkeys(f"{t} {n}" for _, _, t, n in find_refs(text)))


def mask_refs(text: str, token: str | None = None) -> str:
    """Replace every reference span with the mask token (default from config)."""
    token = token or load_config()["preprocess"]["mask_token"]
    out, last = [], 0
    for s, e, *_ in find_refs(text):
        if s < last:  # overlapping span (chain inside a typed ref): already masked
            continue
        out.append(text[last:s])
        out.append(token)
        last = e
    out.append(text[last:])
    return _LEFTOVER.sub(token, "".join(out))


def ref_chapters(refs: list[str]) -> list[str]:
    """Two-digit chapter of each reference: ['AMM 52-10-00'] -> ['52']."""
    return [r.split()[-1][:2] for r in refs]
