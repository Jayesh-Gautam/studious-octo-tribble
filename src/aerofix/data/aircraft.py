"""Map FAA SDR aircraft model codes to family / variant / generation.

FAA model codes are compact: ``7378H4`` is a 737-800 built for customer code H4,
``A321271NX`` is an A321neo. The exact rules live in ``configs/config.yaml``
(``variants``); this module only applies them.
"""
from __future__ import annotations

import re

from aerofix.config import load_config

_BOEING = re.compile(r"^737(\d)([0-9A-Z]{2})(ER)?$")   # 737 + series + customer [+ER]
_AIRBUS = re.compile(r"^A3(19|20|21)(\d{3})?(NX?)?$")   # A3xx + engine code [+N/NX]

_GENERATION = {
    "737-100": "Original", "737-200": "Original",
    "737-300": "Classic", "737-400": "Classic", "737-500": "Classic",
    "737-600": "NG", "737-700": "NG", "737-800": "NG", "737-900": "NG", "737-900ER": "NG",
}


def family_of(make: str, model: str, cfg: dict | None = None) -> str | None:
    """Return 'A320' or 'B737' if the row belongs to one of our fleets, else None."""
    cfg = cfg or load_config()
    make, model = (make or "").strip().upper(), (model or "").strip().upper()
    for fam, rule in cfg["filters"].items():
        if make == rule["make"] and re.match(rule["model_regex"], model):
            return fam
    return None


def parse_variant(make: str, model: str, cfg: dict | None = None) -> tuple[str, str]:
    """Return (variant, generation) for an A320-family or 737 model code.

    Unrecognised codes return an '(unspecified)' variant rather than guessing.
    """
    cfg = cfg or load_config()
    v = cfg["variants"]
    make, model = (make or "").strip().upper(), (model or "").strip().upper()

    if make == "BOEING":
        if model in v["boeing_exact"]:
            variant = v["boeing_exact"][model]
        elif m := _BOEING.match(model):
            variant = v["boeing_series"][m.group(1)] + ("ER" if m.group(3) else "")
        else:
            variant = "737 (unspecified)"
        if "MAX" in variant:
            return variant, "MAX"
        return variant, _GENERATION.get(variant, "Unspecified")

    if model in v["airbus_exact"]:
        return v["airbus_exact"][model], "Unspecified"
    if m := _AIRBUS.match(model):
        neo = bool(m.group(3))
        return f"A3{m.group(1)}{'neo' if neo else ''}", "neo" if neo else "ceo"
    return "A320 family (unspecified)", "Unspecified"
