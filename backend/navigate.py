"""NAVIGATE mode — steer the walker around what's in the corridor ahead.  Owner: D.

The camera points where the person walks (chest/head mounted). We look at a vertical
corridor down the middle of the frame and at walking height (lower part of the image):

  columns:   |  L2  |  L1  |  C  |  R1  |  R2  |      five bands across the frame
  blockage:  how "close" the nearest obstacle in each band is (box height ~ proximity)

Rules (fast path, fixed text, no LLM):
  * corridor C clear               → nothing (silence is the default), or "straight on" after a correction
  * C blocked, L1 or R1 free       → "slightly left" / "slightly right"
  * C and both L1/R1 blocked       → "left" / "right" towards the freer half, else "stop"
  * every cue is counted; when the corridor is clear again we nudge back ("ease back right")
  * if the phone gives a compass heading, drifting > HEADING_TOL degrees from the heading
    captured at "start walk" also triggers a nudge back, even with nothing in view.

Spoken no more than once per CUE_REPEAT_S, except STOP which may repeat every STOP_REPEAT_S.
Object chatter is suppressed in this mode — only steering and urgent hazards speak.
"""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field

from backend.contracts import AgentLine, Detection


def _f(name: str, default: float) -> float:
    """Tunable: module default, overridable by an env var of the same name (so thresholds can be tuned live)."""
    return float(os.getenv(name, default))


BANDS = 5                      # L2 L1 C R1 R2
# Proximity. The camera is chest-high looking slightly down, so an obstacle's bottom edge drops toward the
# bottom of the frame as we get close: that edge is the primary nearness signal (tables, chairs, bins are
# low and never get tall). Box height is kept as a second term for tall things (people, doors).
BOTTOM_START = _f("BOTTOM_START", 0.55)   # bottom edge at this fraction of H → proximity 0
BOTTOM_FULL = _f("BOTTOM_FULL", 0.95)     # … at this fraction → proximity 1
FLOOR_FROM = _f("FLOOR_FROM", 0.45)       # ignore a box whose bottom is above this: it's not on our floor path
BLOCK_MID = _f("BLOCK_MID", 0.35)         # corridor band C counts as blocked at this proximity
BLOCK_NEAR = _f("BLOCK_NEAR", 0.65)       # … and as near (firm cue / stop logic) at this
# Time-to-contact from the rate the bottom edge is dropping (per band, last TTC_WINDOW frames).
TTC_WARN = _f("TTC_WARN", 1.8)            # C blocked if contact is sooner than this (s)
TTC_STOP = _f("TTC_STOP", 1.0)            # C near if contact is sooner than this (s)
TTC_CONTACT = 0.97                        # bottom_frac that means we're touching it
TTC_WINDOW = 8                            # frames of history per band
CUE_REPEAT_S = 2.5
STOP_REPEAT_S = 1.5
CLEAR_FRAMES = 12              # ~1.2 s of clear corridor before we suggest coming back
COMMIT_S = 6.0                 # keep steering the same way for this long while the obstacle persists
HEADING_TOL = 20.0             # degrees


@dataclass
class NavState:
    active: bool = False
    start_heading: float | None = None
    heading: float | None = None
    offset: int = 0                 # +1 per right cue, -1 per left cue (how far we've pushed them)
    clear_streak: int = 0
    last_cue_t: float = 0.0
    last_cue: str = ""
    committed_side: str | None = None   # side we told them to go while the current obstacle persists
    committed_t: float = 0.0
    history: list[str] = field(default_factory=list)
    bottoms: list[deque] = field(default_factory=lambda: [deque(maxlen=TTC_WINDOW) for _ in range(BANDS)])  # (t, bottom_frac)
    ttc_ahead: float | None = None      # seconds to contact in band C from the last frame, None if not closing

    def start(self, heading: float | None) -> None:
        self.active, self.start_heading, self.heading = True, heading, heading
        self.offset, self.clear_streak, self.last_cue_t, self.last_cue = 0, 0, 0.0, ""
        self.committed_side, self.committed_t = None, 0.0
        for h in self.bottoms:
            h.clear()
        self.ttc_ahead = None

    def stop(self) -> None:
        self.active = False


def _proximity(y1: float, y2: float, H: float) -> float:
    """max(bottom-edge term, height term): bottom edge at BOTTOM_START·H → 0, at BOTTOM_FULL·H → 1."""
    bottom_term = min(1.0, max(0.0, (y2 / H - BOTTOM_START) / (BOTTOM_FULL - BOTTOM_START)))
    return max(bottom_term, (y2 - y1) / H)


def _scan(dets: list[Detection], W: float, H: float) -> tuple[list[float], list[float]]:
    """Per band: max proximity and max bottom_frac, for boxes on our floor path (bottom below FLOOR_FROM·H)."""
    prox = [0.0] * BANDS
    bottom = [0.0] * BANDS
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        if y2 < H * FLOOR_FROM:
            continue
        p = _proximity(y1, y2, H)
        b0 = max(0, min(BANDS - 1, int(x1 / W * BANDS)))
        b1 = max(0, min(BANDS - 1, int(x2 / W * BANDS)))
        for b in range(b0, b1 + 1):
            prox[b] = max(prox[b], p)
            bottom[b] = max(bottom[b], y2 / H)
    return prox, bottom


def _band_blockage(dets: list[Detection], W: float, H: float) -> list[float]:
    return _scan(dets, W, H)[0]


def _ttc(hist: deque, bottom_now: float) -> float | None:
    """Seconds until the band's bottom edge reaches TTC_CONTACT, from a least-squares slope over the window.
    None if nothing is there, the window is too short, or the edge is not dropping."""
    if bottom_now <= 0 or len(hist) < 3:
        return None
    ts = [t for t, _ in hist]
    bs = [b for _, b in hist]
    dt = ts[-1] - ts[0]
    if dt < 0.2:
        return None
    mt, mb = sum(ts) / len(ts), sum(bs) / len(bs)
    var = sum((t - mt) ** 2 for t in ts)
    rate = sum((t - mt) * (b - mb) for t, b in zip(ts, bs)) / var if var else 0.0
    if rate <= 1e-3:
        return None
    return max(0.0, (TTC_CONTACT - bottom_now) / rate)


def assess(dets: list[Detection], W: float, H: float, state: NavState) -> dict:
    """One frame's corridor picture: band proximities, C's time-to-contact, and whether C is blocked / near.
    Updates state's bottom-edge history and ttc_ahead."""
    prox, bottom = _scan(dets, W, H)
    now = time.time()
    for b in range(BANDS):
        state.bottoms[b].append((now, bottom[b]))
    ttc = _ttc(state.bottoms[2], bottom[2])
    state.ttc_ahead = ttc
    C = prox[2]
    return {
        "prox": prox, "bottom": bottom, "ttc": ttc,
        "blocked": C >= BLOCK_MID or (ttc is not None and ttc < TTC_WARN),
        "near": C >= BLOCK_NEAR or (ttc is not None and ttc < TTC_STOP),
    }


def _say(state: NavState, text: str, side: str | None, urgency: str, repeat_s: float) -> AgentLine | None:
    now = time.time()
    if text == state.last_cue and now - state.last_cue_t < repeat_s:
        return None
    if text != state.last_cue and now - state.last_cue_t < 0.8 and text != "stop":
        return None  # don't machine-gun different cues either; "stop" may always cut in
    state.last_cue, state.last_cue_t = text, now
    state.history.append(text)
    return {"text": text, "side": side, "urgency": urgency}


def steer(dets: list[Detection], W: float, H: float, state: NavState) -> AgentLine | None:
    if not state.active:
        return None
    a = assess(dets, W, H, state)
    L2, L1, C, R1, R2 = a["prox"]
    c_blocked, c_near = a["blocked"], a["near"]
    l1_free, r1_free = L1 < BLOCK_NEAR, R1 < BLOCK_NEAR
    left_load, right_load = L1 + L2, R1 + R2

    if c_blocked:
        state.clear_streak = 0
        if c_near and not l1_free and not r1_free and L2 >= BLOCK_NEAR and R2 >= BLOCK_NEAR:
            return _say(state, "stop", None, "urgent", STOP_REPEAT_S)
        now = time.time()
        # stick with the side we already committed to while this obstacle persists (no flip-flopping)
        if state.committed_side and now - state.committed_t < COMMIT_S:
            side = state.committed_side
            if (side == "left" and l1_free) or (side == "right" and r1_free):
                state.committed_t = now
                word = ("slightly " if not c_near else "") + side
                return _say(state, word, side, "warn", CUE_REPEAT_S)
        # otherwise pick the freer side; ties go toward the original line
        prefer_left = (left_load < right_load) or (left_load == right_load and state.offset > 0)
        for side, free in (("left", l1_free), ("right", r1_free)) if prefer_left else (("right", r1_free), ("left", l1_free)):
            if free:
                state.offset += -1 if side == "left" else 1
                state.committed_side, state.committed_t = side, now
                word = ("slightly " if not c_near else "") + side
                return _say(state, word, side, "warn", CUE_REPEAT_S)
        # inner bands blocked too: step out wider if an outer band is free
        for side, free in (("left", L2 < BLOCK_NEAR), ("right", R2 < BLOCK_NEAR)) if prefer_left else (("right", R2 < BLOCK_NEAR), ("left", L2 < BLOCK_NEAR)):
            if free:
                state.offset += -2 if side == "left" else 2
                state.committed_side, state.committed_t = side, now
                return _say(state, side, side, "warn", CUE_REPEAT_S)
        return _say(state, "stop", None, "urgent", STOP_REPEAT_S)

    # corridor clear
    state.clear_streak += 1
    if state.clear_streak >= CLEAR_FRAMES:
        state.committed_side = None
    if state.clear_streak < CLEAR_FRAMES:
        return None

    # come back to the original line: by cue history, or by compass if we have one
    drift = None
    if state.heading is not None and state.start_heading is not None:
        drift = ((state.heading - state.start_heading + 540) % 360) - 180   # -180..180, +ve = turned right
    if drift is not None and abs(drift) > HEADING_TOL:
        back = "left" if drift > 0 else "right"
        return _say(state, f"ease back {back}", back, "info", CUE_REPEAT_S * 2)
    if state.offset != 0:
        back = "right" if state.offset < 0 else "left"
        state.offset = 0 if abs(state.offset) <= 2 else state.offset + (2 if state.offset < 0 else -2)
        return _say(state, f"clear, ease back {back}", back, "info", CUE_REPEAT_S * 2)
    if state.last_cue and state.last_cue != "straight on":
        return _say(state, "straight on", None, "info", CUE_REPEAT_S * 3)
    return None


# --------------------------------------------------------------------------- reasoned guidance
CORRIDOR = (1, 2, 3)           # L1 C R1: the bands that matter for walking
_WHERE = {0: "on your left", 1: "on your left", 2: "ahead", 3: "on your right", 4: "on your right"}


def guide_side(dets: list[Detection], W: float, H: float) -> str:
    """Clearer side from the same band blockage steer() uses; ties go right."""
    L2, L1, C, R1, R2 = _band_blockage(dets, W, H)
    return "left" if (L1 + L2) < (R1 + R2) else "right"


def guidance_sentence(dets: list[Detection], W: float, H: float, caption: str = "") -> str | None:
    """Short template phrase for a corridor that is mid-blocked but not urgent, e.g.
    "person on your left and a table ahead, bear right" (≤ 12 words). None if the corridor is clear
    or if the centre is near-blocked (steer() owns that: left/right/stop). `caption` is unused for
    now; it is here so a model could enrich the phrase later without changing the call."""
    block = _band_blockage(dets, W, H)
    if block[2] >= BLOCK_NEAR or max(block[b] for b in CORRIDOR) < BLOCK_MID:
        return None
    # nearest object per corridor band (walking height only), closest first, at most two
    seen: dict[str, tuple[float, int]] = {}
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        if y2 < H * FLOOR_FROM:
            continue
        band = max(0, min(BANDS - 1, int((x1 + x2) / 2 / W * BANDS)))
        if band not in CORRIDOR:
            continue
        prox = _proximity(y1, y2, H)
        if d["label"] not in seen or prox > seen[d["label"]][0]:
            seen[d["label"]] = (prox, band)
    if not seen:
        return None
    top = sorted(seen.items(), key=lambda kv: -kv[1][0])[:2]
    parts = []
    for label, (_, band) in top:
        noun = label if label == "person" else f"a {label}"
        parts.append(f"{noun} {_WHERE[band]}")
    return f"{' and '.join(parts)}, bear {guide_side(dets, W, H)}"
