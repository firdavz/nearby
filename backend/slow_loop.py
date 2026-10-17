"""UNDERSTANDING — caption(clip_path) -> str, plus the clip writer.  Owner: D.

FAKE: rotates through plausible captions.
REAL: NVIDIA Cosmos Reason, two ways of reaching it (COSMOS_MODE in .env):

  nvcf    — NVIDIA's hosted API on build.nvidia.com. Three steps: upload the mp4 to NVCF
            assets, invoke with the asset id embedded in the message, delete the asset.
            Needs only an nvapi- key. Default.
  openai  — a self-hosted NIM container or vLLM serving Cosmos (OpenAI-compatible):
            video_url with a base64 data URI + media_io_kwargs for frame sampling.

Keep clips short and small (5 s, 640 px wide) so uploads stay fast.
"""
from __future__ import annotations

import base64
import itertools
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import requests

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
        return self._flush()

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
          "(keys, phone, wallet, glasses, cup, bag). Answer directly, no preamble, no reasoning steps.")


def _clean(text: str) -> str:
    """Cosmos Reason may wrap output in <think>…</think><answer>…</answer>. Keep only the answer."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.S)
    if m:
        text = m.group(1)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return " ".join(text.split()).strip()


# ---- mode: nvcf (hosted build.nvidia.com) ---------------------------------------
NVCF_ASSETS = "https://api.nvcf.nvidia.com/v2/nvcf/assets"


def _nvcf_upload(path: Path) -> str:
    auth = {"Authorization": f"Bearer {config.COSMOS_API_KEY}"}
    r = requests.post(NVCF_ASSETS, headers={**auth, "Content-Type": "application/json", "Accept": "application/json"},
                      json={"contentType": "video/mp4", "description": "second-sight clip"}, timeout=30)
    r.raise_for_status()
    info = r.json()
    put = requests.put(info["uploadUrl"], data=path.read_bytes(),
                       headers={"content-type": "video/mp4", "x-amz-meta-nvcf-asset-description": "second-sight clip"},
                       timeout=60)
    put.raise_for_status()
    return info["assetId"]


def _nvcf_delete(asset_id: str) -> None:
    try:
        requests.delete(f"{NVCF_ASSETS}/{asset_id}", headers={"Authorization": f"Bearer {config.COSMOS_API_KEY}"}, timeout=15)
    except Exception:
        pass


def caption_nvcf(clip_path: str) -> str:
    asset_id = _nvcf_upload(Path(clip_path))
    try:
        payload = {
            "model": config.COSMOS_MODEL,
            "messages": [{"role": "user", "content": f'{PROMPT} <video src="data:video/mp4;asset_id,{asset_id}" />'}],
            "max_tokens": config.COSMOS_MAX_TOKENS, "temperature": 0.2, "top_p": 0.7,
            "num_frames_per_inference": config.COSMOS_NUM_FRAMES, "stream": False,
        }
        headers = {"Authorization": f"Bearer {config.COSMOS_API_KEY}", "Content-Type": "application/json",
                   "Accept": "application/json", "NVCF-INPUT-ASSET-REFERENCES": asset_id, "NVCF-FUNCTION-ASSET-IDS": asset_id}
        r = requests.post(config.COSMOS_INVOKE_URL, headers=headers, json=payload, timeout=120)
        if r.status_code >= 400:
            raise RuntimeError(f"Cosmos {r.status_code}: {r.text[:300]}")
        return _clean(r.json()["choices"][0]["message"]["content"])
    finally:
        _nvcf_delete(asset_id)


# ---- mode: openai (self-hosted NIM / vLLM) --------------------------------------
def _resolve_model(client) -> str:
    """Hackathon servers may expose a different id than the docs say; trust /v1/models."""
    if config.COSMOS_MODEL and config.COSMOS_MODEL != "auto":
        return config.COSMOS_MODEL
    return client.models.list().data[0].id


def clip_fps(clip_path: str) -> float:
    """Frame rate the clip was written at (live clips through a slow tunnel can be exactly 1 fps)."""
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    cap.release()
    return fps if fps and fps > 0 else 1.0


def _fps_from_error(msg: str) -> float | None:
    """vLLM says 'Target fps (4.0) must be <= original fps (1.0)'; pull out the original."""
    m = re.search(r"original fps \(([\d.]+)\)", msg)
    return float(m.group(1)) if m else None


def caption_openai(clip_path: str) -> str:
    """Self-hosted Cosmos (NIM / vLLM / the hackathon's Cosmos3-Reason). Bearer = COSMOS_API_KEY.
    The sampling hint is capped at the clip's own fps: Cosmos rejects a target above the source rate,
    and without any hint it defaults to 4 fps, which fails the same way."""
    from openai import OpenAI, BadRequestError
    client = OpenAI(base_url=config.COSMOS_BASE_URL, api_key=config.COSMOS_API_KEY or "none", timeout=90)
    model = _resolve_model(client)
    b64 = base64.b64encode(Path(clip_path).read_bytes()).decode()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}}]}]
    kwargs = dict(model=model, messages=messages, max_tokens=config.COSMOS_MAX_TOKENS, temperature=0.2)
    fps = min(float(config.COSMOS_FPS), clip_fps(clip_path))
    hint = lambda f: {"media_io_kwargs": {"video": {"fps": round(f, 3)}}}
    try:
        resp = client.chat.completions.create(**kwargs, extra_body=hint(fps))
    except BadRequestError as e:
        src = _fps_from_error(str(e))
        if src is not None and src < fps:           # our probe of the clip fps was wrong: trust the server
            resp = client.chat.completions.create(**kwargs, extra_body=hint(src))
        elif "media_io_kwargs" in str(e):           # server has no such parameter at all: go plain
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    return _clean(resp.choices[0].message.content or "")


# ---- mode: yolo (no Cosmos at all) ------------------------------------------------
def caption_from_detections(dets: list[dict]) -> str:
    """Template caption from the YOLO detections seen during the clip — no model call.
    dets: list of {label, side, distance} (one per detection per frame)."""
    if not dets:
        return "Nothing recognised in view."
    from collections import Counter
    seen: Counter = Counter((d["label"], d["side"], d["distance"]) for d in dets)
    # keep the most frequent placement per label, most common labels first
    best: dict[str, tuple[str, str, int]] = {}
    for (label, side, dist), n in seen.items():
        if label not in best or n > best[label][2]:
            best[label] = (side, dist, n)
    order = sorted(best.items(), key=lambda kv: -kv[1][2])[:5]
    where = {"left": "on your left", "right": "on your right", "centre": "ahead of you"}
    parts = []
    for label, (side, dist, _) in order:
        article = "a person" if label == "person" else f"a {label}"
        parts.append(f"{article} {where[side]}, {dist}")
    return (parts[0].capitalize() + ("; " + "; ".join(parts[1:]) if len(parts) > 1 else "") + ".")


def caption_real(clip_path: str) -> str:
    if config.COSMOS_MODE == "nvcf":
        return caption_nvcf(clip_path)
    if config.COSMOS_MODE == "openai":
        return caption_openai(clip_path)
    raise RuntimeError("COSMOS_MODE=yolo captions are built in main.py from detections, not from the clip")


caption = caption_real if (config.USE_REAL_COSMOS and config.COSMOS_MODE != "yolo") else caption_fake


if __name__ == "__main__":  # python -m backend.slow_loop data/clips/some.mp4
    import sys
    t = time.time()
    print(caption_real(sys.argv[1]))
    print(f"latency {time.time() - t:.1f}s  mode={config.COSMOS_MODE} model={config.COSMOS_MODEL}")
