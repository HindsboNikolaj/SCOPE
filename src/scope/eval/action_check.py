"""Check camera actions in code, before any judge sees the answer.

An LLM judge reads words. Told "Zoom out 2x" and shown "The camera has been zoomed out 2x", it
passes the row even when the call was ``ptz_adjust(zoom_factor=2)``, which zooms in. So the
camera commands a question states explicitly are checked against the recorded tool calls here,
and a record whose calls do not carry out the command fails whatever the judges say.

Only explicit commands count: "pan right 20 degrees", "after zooming out 50%", "go to road1",
"return Home". Conditional or optional moves ("zoom if needed", "if yes, zoom") and moves the
question only mentions ("before zooming in, how many...") are not required.

Directions follow ``src/scope/tools/blender_tools.py``: ``pan_deg > 0`` turns right,
``tilt_deg > 0`` tilts up, ``zoom_factor > 1``, ``zoom_percent > 0``, ``zoom_full="in"`` and
``zoom_bounding`` zoom in. A half turn (180 degrees) may go either way.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

PRESETS = {"diff-view-1", "eor-viewpoint", "hotel-m", "mailbox", "road1", "sor-viewpoint",
           "store-front", "store-front2", "street-view-1", "street-view-2"}

_OPTIONAL = re.compile(r"\b(?:if needed|if necessary|if yes|if so|if one is|if any)\b", re.I)
_BEFORE = re.compile(r"\bbefore\s+(?:zoom|pan|tilt|turn)\w*", re.I)
_DEG = r"(?:(\d+(?:\.\d+)?)\s*(?:°|degrees?|deg)\s*)?"


def _clauses(question: str) -> List[str]:
    """Split a command into its clauses so 'if yes, zoom' stays conditional."""
    q = question.replace("’", "'").replace("‘", "'")
    return [c for c in re.split(r"[;,.]|\bthen\b|\band\b", q, flags=re.I) if c.strip()]


def required_actions(question: str) -> List[Tuple[str, Any]]:
    """(kind, detail) pairs the question explicitly commands, in order of appearance."""
    req: List[Tuple[str, Any]] = []
    q = question or ""
    conditional = False
    for clause in _clauses(q):
        c = clause.strip()
        if _OPTIONAL.search(c):
            conditional = True
        if conditional:
            # Everything after a condition ("if yes, zoom and read") is optional.
            continue
        if _BEFORE.search(c):
            continue
        low = c.lower()
        if re.search(r"\b(?:half-turn|half turn|turn around|180\s*(?:°|degrees?|degree)\s*turn)\b", low):
            req.append(("pan", "any180"))
        else:
            m = re.search(r"\b(?:pan(?:ning|ned)?|turn(?:ing|ed)?)\s+(?:by\s+)?" + _DEG
                          + r"(?:to the\s+)?(left|right)\b", low)
            m2 = re.search(r"\b(?:pan(?:ning|ned)?|turn(?:ing|ed)?)\s+(left|right)\b", low)
            m3 = re.search(r"\b\d+(?:\.\d+)?\s*(?:°|degrees?)\s*turn to the\s+(left|right)\b", low)
            if m or m2 or m3:
                side = (m.group(2) if m else (m2.group(1) if m2 else m3.group(1)))
                deg = re.search(r"(\d+(?:\.\d+)?)\s*(?:°|degrees?)", low)
                if deg and abs(float(deg.group(1))) >= 179:
                    req.append(("pan", "any180"))
                else:
                    req.append(("pan", side))
            elif re.search(r"\bpan\s+180\b", low):
                req.append(("pan", "any180"))
        m = re.search(r"\btilt(?:ing|ed)?\s+(up|down)\b", low)
        if m:
            req.append(("tilt", m.group(1)))
        if re.search(r"\bzoom(?:ing|ed)?\s+(?:out|back)\b", low) or re.search(r"\bzoom out\b", low):
            req.append(("zoom", "out"))
        elif re.search(r"\bzoom(?:ing|ed|s)?\b", low) and re.search(
                r"\bzoom(?:ing|ed)?\s+(?:in|into|to|on)\b|\bzoom\s+\d|\b\d+(?:\.\d+)?\s*[x×]\s*zoom in\b|\bzoom in\b", low):
            req.append(("zoom", "in"))
        elif re.search(r"\bzoom(?:ing|ed)?\b", low):
            req.append(("zoom", "any"))
        if re.search(r"\breturn\b.*\bhome\b", low):
            req.append(("home", None))
        m = re.search(r"\b(?:go|move|switch|return)\s+to\s+(?:the\s+)?'?([a-z0-9][a-z0-9-]*)'?", low)
        if m and m.group(1) in PRESETS:
            req.append(("preset", m.group(1)))
    return req


def _num(args: Dict[str, Any], key: str) -> float:
    try:
        return float(args.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def satisfied(kind: str, detail: Any, calls: Iterable[Dict[str, Any]]) -> bool:
    for call in calls:
        name = call.get("name") or ""
        args = call.get("arguments")
        args = args if isinstance(args, dict) else {}
        if kind == "pan" and name == "ptz_adjust":
            deg = _num(args, "pan_deg")
            if detail == "any180" and abs(deg) >= 90:
                return True
            if (detail == "right" and deg > 0) or (detail == "left" and deg < 0):
                return True
        elif kind == "tilt" and name == "ptz_adjust":
            deg = _num(args, "tilt_deg")
            if (detail == "up" and deg > 0) or (detail == "down" and deg < 0):
                return True
        elif kind == "zoom":
            if name == "zoom_bounding" and detail in ("in", "any"):
                return True
            if name == "ptz_adjust":
                factor, percent = _num(args, "zoom_factor"), _num(args, "zoom_percent")
                full = str(args.get("zoom_full") or "").lower()
                zin = factor > 1 or percent > 0 or full == "in"
                zout = 0 < factor < 1 or percent < 0 or full == "out"
                if (detail == "in" and zin) or (detail == "out" and zout) or (detail == "any" and (zin or zout)):
                    return True
        elif kind == "home":
            if name == "home_action":
                return True
            if name == "go_to_preset" and str(args.get("preset_name") or args.get("name") or "").lower() == "home":
                return True
        elif kind == "preset" and name == "go_to_preset":
            target = str(args.get("preset_name") or args.get("name") or "").strip().strip("'\"").lower()
            if target == detail:
                return True
    return False


def check(question: str, calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return {'required': [...], 'missing': [...], 'ok': bool} for one record."""
    req = required_actions(question)
    missing = [f"{k}:{d}" if d is not None else k for k, d in req if not satisfied(k, d, calls)]
    return {"required": [f"{k}:{d}" if d is not None else k for k, d in req],
            "missing": missing, "ok": not missing}
