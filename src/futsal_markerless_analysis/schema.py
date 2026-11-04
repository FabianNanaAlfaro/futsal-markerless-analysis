"""Coordinate and anatomical naming conventions shared by pipeline adapters."""

from __future__ import annotations

from typing import Optional

FUNCTIONAL_POINTS = ("hip", "knee", "ankle", "foot")
SIDES = ("left", "right")

_ALIASES = {
    "hip": "hip", "pelvis": "hip", "hipcenter": "hip",
    "knee": "knee", "kneecenter": "knee",
    "ankle": "ankle", "anklecenter": "ankle",
    "foot": "foot", "footindex": "foot", "bigtoe": "foot", "toe": "foot",
    "lhip": "hip", "rhip": "hip", "lknee": "knee", "rknee": "knee",
    "lankle": "ankle", "rankle": "ankle", "lfootindex": "foot",
    "rfootindex": "foot", "lbigtoe": "foot", "rbigtoe": "foot",
    "cadera": "hip", "caderaizq": "hip", "caderader": "hip",
    "rodilla": "knee", "rodillaizq": "knee", "rodillader": "knee",
    "tobillo": "ankle", "tobilloizq": "ankle", "tobilloder": "ankle",
    "pie": "foot", "pieizq": "foot", "pieder": "foot",
}


def homologate_landmark(name: str, selected_side: Optional[str] = None) -> Optional[str]:
    """Map common Sports2D, MediaPipe and DLC labels to functional points.

    When a label encodes laterality, rows for the unselected side are ignored.
    The side must be supplied explicitly for labels without laterality.
    """
    if selected_side is not None and selected_side.lower() not in SIDES:
        raise ValueError("selected_side must be 'left' or 'right'")
    raw = "".join(character for character in str(name).lower() if character.isalnum())
    side = None
    for suffix, candidate in (("izq", "left"), ("der", "right"), ("left", "left"), ("right", "right")):
        if raw.endswith(suffix):
            side = candidate
            raw = raw[:-len(suffix)]
            break
    for prefix, candidate in (("left", "left"), ("right", "right"), ("l", "left"), ("r", "right")):
        if side is None and raw.startswith(prefix) and raw[len(prefix):] in {"hip", "knee", "ankle", "foot", "footindex", "bigtoe", "toe"}:
            side = candidate
            raw = raw[len(prefix):]
            break
    if side is not None and selected_side is not None and side != selected_side.lower():
        return None
    return _ALIASES.get(raw)
