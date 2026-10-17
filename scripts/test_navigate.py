"""Synthetic checks for backend/navigate.py (no camera, no server):  python scripts/test_navigate.py
Time is simulated so the time-to-contact logic is deterministic."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import navigate as nv  # noqa: E402

W, H = 640, 480
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def det(label, x1, y1, x2, y2):
    return {"label": label, "side": "centre", "distance": "mid", "closing": False, "box": [x1, y1, x2, y2], "track_id": 1}


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


clock = Clock()
nv.time.time = clock          # navigate reads time.time() for history, repeats and commitment


def fresh() -> nv.NavState:
    s = nv.NavState(); s.start(None); return s


def walk(state, box_fn, seconds, fps=10):
    """Feed frames at `fps` for `seconds`; box_fn(frac) → list of boxes. Returns [(frac, bottom_frac, cue)]."""
    out = []
    n = int(seconds * fps)
    for i in range(n + 1):
        frac = i / n
        dets = box_fn(frac)
        line = nv.steer(dets, W, H, state)
        out.append((frac, max((d["box"][3] for d in dets), default=0) / H, line["text"] if line else None, state.ttc_ahead))
        clock.t += 1.0 / fps
    return out


# 1. guidance / empty
check(nv.guidance_sentence([], W, H) is None, "empty view → no guidance")
check(nv.steer([], W, H, fresh()) is None, "empty view → silence")

# 2. low table edge across the path: bottom edge 0.60H → 0.90H over 1.5 s, centre band, height 0.12H
def table(frac):
    y2 = H * (0.60 + 0.30 * frac)
    return [det("dining table", W * 0.42, y2 - H * 0.12, W * 0.58, y2)]
trace = walk(fresh(), table, 1.5)
first_cue = next(((b, c) for _, b, c, _ in trace if c), None)
check(first_cue is not None and first_cue[0] <= 0.75, f"centre table: first cue {first_cue} by 0.75·H")
near_cue = next(((b, c) for _, b, c, _ in trace if c in ("left", "right", "stop")), None)
check(near_cue is not None and near_cue[0] < 0.92, f"centre table: firm cue {near_cue} before 0.92·H")
ttcs = [t for *_, t in trace if t is not None]
check(bool(ttcs) and min(ttcs) < nv.TTC_STOP, f"ttc_ahead computed and fell below TTC_STOP (min {min(ttcs) if ttcs else None})")

# 3. same edge but a wide table (spans every band) → must end in "stop" before 0.92·H
def wide(frac):
    y2 = H * (0.60 + 0.30 * frac)
    return [det("dining table", 0, y2 - H * 0.12, W, y2)]
trace = walk(fresh(), wide, 1.5)
stop_at = next((b for _, b, c, _ in trace if c == "stop"), None)
check(stop_at is not None and stop_at < 0.92, f"wide table: 'stop' at bottom {stop_at} (< 0.92·H)")

# 4. tall far person: bottom 0.5·H, height 0.6·H, standing still → blocked at most softly, never near
s = fresh()
person = [det("person", W * 0.45, 0, W * 0.55, H * 0.5)]
for _ in range(10):
    a = nv.assess(person, W, H, s); clock.t += 0.1
check(not a["near"], f"tall far person: not near (prox {a['prox'][2]:.2f}, ttc {a['ttc']})")
check(a["ttc"] is None, "tall far person standing still: no time-to-contact")

# 5. walking parallel past a table on the right: it stays in R1/R2, bottom edge steady → silence
def beside(frac):
    return [det("dining table", W * 0.65, H * 0.55, W * 0.95, H * 0.80)]
trace = walk(fresh(), beside, 2.0)
check(all(c is None for _, _, c, _ in trace), "table beside the path (R1/R2, steady) → silence")

# 6. floor filter: a box whose bottom is above 0.45·H is ignored even if tall
s = fresh()
a = nv.assess([det("tv", W * 0.4, 0, W * 0.6, H * 0.40)], W, H, s)
check(a["prox"][2] == 0.0, "box ending above 0.45·H ignored (not on the floor path)")

# 7. env override of a threshold
import os, importlib
os.environ["BLOCK_NEAR"] = "0.9"
m = importlib.reload(nv); check(m.BLOCK_NEAR == 0.9, "BLOCK_NEAR overridable from env")
del os.environ["BLOCK_NEAR"]; importlib.reload(nv)

print("\nALL OK" if not fails else f"\n{len(fails)} FAILED")
sys.exit(1 if fails else 0)
