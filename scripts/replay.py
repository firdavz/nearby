"""Replay an mp4 into the running server as if it were the webcam.  Owner: D (for B's YOLO tuning).

  python scripts/replay.py data/clips/some.mp4 [--fps 10] [--url ws://localhost:8000/ws/frames] [--loop]

Frames are resized to 640 wide, JPEG q60, sent one at a time over the websocket; the next frame
goes only after the server's JSON reply (same lock-step the browser uses). Prints each non-null
`say` line with its time offset, and a summary at the end. Ctrl-C stops a --loop run.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import websockets


def frames_of(path: str, width: int = 640):
    """Yield (offset_seconds, jpeg_bytes) for every frame of the clip, resized to `width`."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"cannot open {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w != width:
            frame = cv2.resize(frame, (width, int(round(h * width / w))))
        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        yield i / src_fps, jpg.tobytes()
        i += 1
    cap.release()


async def replay(path: str, fps: float, url: str, loop: bool) -> None:
    period = 1.0 / fps if fps > 0 else 0.0
    sent = said = hazards = 0
    labels: Counter = Counter()
    lat: list[float] = []
    t_start = time.time()
    try:
        async with websockets.connect(url, max_size=None) as ws:
            while True:
                for offset, jpg in frames_of(path):
                    t0 = time.time()
                    await ws.send(jpg)
                    reply = json.loads(await ws.recv())
                    lat.append(time.time() - t0)
                    sent += 1
                    labels.update(d["label"] for d in reply.get("detections", []))
                    hazards += bool(reply.get("hazard"))
                    if reply.get("say"):
                        s = reply["say"]
                        said += 1
                        print(f"[{time.time() - t_start:7.2f}s  clip {offset:6.2f}s] {s['urgency']:6s} {s.get('side') or '-':6s} {s['text']}")
                    wait = period - (time.time() - t0)
                    if wait > 0:
                        await asyncio.sleep(wait)
                if not loop:
                    break
    except KeyboardInterrupt:
        pass
    wall = time.time() - t_start
    print("\n--- summary ---")
    print(f"frames sent      {sent}  in {wall:.1f}s  ({sent / wall if wall else 0:.1f} fps effective)")
    if lat:
        lat.sort()
        print(f"round trip ms    median {1000 * lat[len(lat) // 2]:.0f}   p95 {1000 * lat[int(len(lat) * 0.95)]:.0f}")
    print(f"say lines        {said}")
    print(f"hazard frames    {hazards}")
    print(f"labels seen      {dict(labels.most_common(8)) or '{}'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mp4", help="path to the clip to replay")
    ap.add_argument("--fps", type=float, default=10.0, help="send rate cap (default 10)")
    ap.add_argument("--url", default="ws://localhost:8000/ws/frames")
    ap.add_argument("--loop", action="store_true", help="repeat the clip until Ctrl-C")
    a = ap.parse_args()
    if not Path(a.mp4).exists():
        sys.exit(f"no such file: {a.mp4}")
    asyncio.run(replay(a.mp4, a.fps, a.url, a.loop))


if __name__ == "__main__":
    main()
