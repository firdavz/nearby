"""THE CONTRACT — shared function shapes. Changes need everyone's OK.

Every module exposes exactly these functions with exactly these shapes.
Each module has a FAKE and a REAL implementation selected by a flag in config.py,
so the whole app runs on fakes from minute 20 and real pieces drop in one by one.
"""
from __future__ import annotations

from typing import Literal, Optional, TypedDict

Side = Literal["left", "centre", "right"]
Distance = Literal["near", "mid", "far"]
Urgency = Literal["info", "warn", "urgent"]


class Detection(TypedDict):
    label: str
    side: Side
    distance: Distance
    closing: bool
    box: list[float]          # [x1, y1, x2, y2] in pixels of the frame we were given
    track_id: int             # -1 if no tracking


class MemoryHit(TypedDict):
    t: float                  # unix seconds
    caption: str
    clip_url: str             # playable URL served by main.py


class AgentLine(TypedDict):
    text: str
    side: Optional[Side]      # where to pan the voice; None = centre
    urgency: Urgency


class AgentAnswer(TypedDict):
    text: str
    clip_url: Optional[str]


# ---- the seven functions -------------------------------------------------
# fast_loop.detect(frame: np.ndarray) -> list[Detection]
# slow_loop.caption(clip_path: str) -> str
# memory.store(clip_path: str, caption: str, objects: list[str], t: float) -> None
# memory.search(query: str) -> list[MemoryHit]
# memory.last_seen(label: str) -> MemoryHit | None
# agent.agent_on_events(events: list[Detection], caption: str) -> AgentLine | None
# agent.agent_ask(question: str) -> AgentAnswer


# ---- the hazard rule (fast path, no LLM) ----------------------------------
def is_hazard(d: Detection) -> bool:
    return d["closing"] and d["distance"] == "near"


def fixed_alert(d: Detection) -> AgentLine:
    """Urgent alerts use fixed text so a slow model never delays a warning."""
    return {"text": f"{d['label']}, {d['side']}, {d['distance']}", "side": d["side"], "urgency": "urgent"}


# ---- scene-level rule: crowding (fast path, no LLM) ---------------------------
# Several near/mid objects ahead → "busy ahead" with the clearer side. Added after the
# contract was agreed; it adds a function, it changes no shapes.
CROWD_AHEAD = 2      # near/mid objects in the centre third
CROWD_TOTAL = 4      # or this many near/mid objects anywhere
_crowd_state = {"t": 0.0}
CROWD_REPEAT_S = 5.0


def clearer_side(dets: list[Detection]) -> Side:
    load = {"left": 0, "right": 0}
    for d in dets:
        if d["distance"] != "far" and d["side"] in load:
            load[d["side"]] += 1
    return "left" if load["left"] <= load["right"] else "right"


def crowd_alert(dets: list[Detection]) -> AgentLine | None:
    import time
    close = [d for d in dets if d["distance"] in ("near", "mid")]
    ahead = [d for d in close if d["side"] == "centre"]
    if len(ahead) < CROWD_AHEAD and len(close) < CROWD_TOTAL:
        return None
    now = time.time()
    if now - _crowd_state["t"] < CROWD_REPEAT_S:
        return None
    _crowd_state["t"] = now
    side = clearer_side(dets)
    return {"text": f"busy ahead, slow down. clearer on the {side}", "side": side, "urgency": "warn"}
