"""UNDERSTANDING — caption(clip_path) -> str, plus the clip writer.  Owner: D.

FAKE: rotates through plausible captions.
REAL: NVIDIA Cosmos Reason via OpenAI-compatible NIM; clip sent as base64 mp4.
Keep clips short and small (5 s, 640 px wide) so the base64 payload stays well under limits.
"""
from __future__ import annotations

import base64
import itertools
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from backend import config

# --------------------------------------------------------------------------- clip writer
class ClipBuffer:
    """Collects frames; every CLIP_SECONDS writes an mp4 and returns its path."""

    def __init__(self):
        self.frames: list[np.ndarray] = []
        self.started = time.time()

    def add(self, frame: np.ndarray) -> Path | None:
        self.frames.append(frame)
        if time.time() - self.started < config.CLIP_SECONDS:
            return None
        path = self._flush()
        return path

    def _flush(self) -> Path | None:
        frames, self.frames = self.frames, []
        elapsed = max(0.5, time.time() - self.started)
        self.started = time.time()
        if len(frames) < 3:
            return None
        fps = max(1.0, len(frames) / elapsed)
        H, W = frames[0].shape[:2]
        config.CLIP_DIR.mkdir(parents=True, exist_ok=True)
        path = config.CLIP_DIR / f"clip_{int(time.time() * 1000)}.mp4"
        return write_mp4(frames, path, fps, W, H)


def write_mp4(frames: list[np.ndarray], path: Path, fps: float, W: int, H: int) -> Path:
    """Browser-playable H.264 via ffmpeg (stdin rawvideo). Falls back to OpenCV mp4v if ffmpeg is missing."""
    try:
        p = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{W}x{H}", "-r", f"{fps:.2f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for f in frames:
            p.stdin.write(np.ascontiguousarray(f).tobytes())
        p.stdin.close(); p.wait(timeout=30)
        if p.returncode == 0 and path.exists():
            return path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()
    return path


# --------------------------------------------------------------------------- FAKE
_fake_captions = itertools.cycle([
    "A person stands at a desk; a cup and a laptop are on the table.",
    "A set of keys is placed on the table next to the cup.",
    "The person walks away from the table toward the door on the left.",
    "An empty room with a chair and a table; a bag hangs on the chair.",
])


def caption_fake(clip_path: str) -> str:
    time.sleep(0.3)  # pretend it's slow so the UI proves it never blocks
    return next(_fake_captions)


# --------------------------------------------------------------------------- REAL
PROMPT = ("You are the eyes of a person with sight loss. Describe this short clip in one or two plain sentences: "
          "the main objects and where they are (left/right/ahead, near/far), any people and what they are doing, "
          "and anything that could be a hazard. Name small personal items explicitly if visible "
          "(keys, phone, wallet, glasses, cup, bag). No preamble.")


def caption_real(clip_path: str) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=config.COSMOS_BASE_URL, api_key=config.COSMOS_API_KEY or "none")
    b64 = base64.b64encode(Path(clip_path).read_bytes()).decode()
    resp = client.chat.completions.create(
        model=config.COSMOS_MODEL, max_tokens=120, temperature=0.2,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}}]}],
        extra_body={"fps": 2},
    )
    return (resp.choices[0].message.content or "").strip()


caption = caption_real if config.USE_REAL_COSMOS else caption_fake
