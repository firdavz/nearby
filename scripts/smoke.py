"""Run before every push:  python scripts/smoke.py
Boots the app on whatever flags .env has, streams 3 s of frames, asks a question.
Exits non-zero if anything in the contract is broken."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("CLIP_SECONDS", "1.5")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from backend import config  # noqa: E402

REQUIRED_DET = {"label", "side", "distance", "closing", "box", "track_id"}
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


flags = {k: getattr(config, k) for k in ("USE_REAL_YOLO", "USE_REAL_COSMOS", "USE_REAL_MEMORY", "USE_REAL_LLM")}
print("flags:", json.dumps(flags))

with TestClient(main.app) as c:
    frame = np.full((480, 640, 3), 40, np.uint8)
    cv2.circle(frame, (320, 240), 80, (0, 200, 255), -1)
    _, jpg = cv2.imencode(".jpg", frame)
    n, dets_ok, said = 0, True, 0
    with c.websocket_connect("/ws/frames") as ws:
        t0 = time.time()
        while time.time() - t0 < 3.2:
            ws.send_bytes(jpg.tobytes())
            r = json.loads(ws.receive_text())
            n += 1
            for d in r["detections"]:
                dets_ok &= REQUIRED_DET <= set(d) and d["side"] in ("left", "centre", "right") and d["distance"] in ("near", "mid", "far")
            said += bool(r["say"])
            time.sleep(0.08)
    check(n > 10, f"frame loop alive ({n} frames)")
    check(dets_ok, "detect() returns contract shape")
    time.sleep(1.2)
    s = c.get("/state").json()
    check(s["clips"] >= 1, f"slow path wrote {s['clips']} clip(s) and captioned")
    caps = [l for l in s["log"] if l["kind"] == "caption"]
    if caps:
        r = c.get(caps[-1]["clip_url"])
        check(r.status_code == 200 and "video" in r.headers.get("content-type", ""), "clip is served as video")
    a = c.post("/ask", json={"question": "what do you see?"}).json()
    check(isinstance(a.get("text"), str) and "clip_url" in a, "agent_ask() returns contract shape")
    hits = c.get("/search", params={"q": "person"}).json()
    check(isinstance(hits, list), "search() returns a list")
    if not config.USE_REAL_YOLO:
        check(said >= 1, "fake walker produced a spoken line (hazard path works)")

print("\nALL OK" if not fails else f"\n{len(fails)} FAILED — do not push")
sys.exit(1 if fails else 0)
