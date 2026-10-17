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
