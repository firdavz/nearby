"""GLUE — one FastAPI process.  Owner: D.

  uvicorn main:app --reload --port 8000
  open http://localhost:8000

Wiring:  browser webcam → JPEG frames over WebSocket → detect() every frame →
         hazards/agent line back to the browser immediately (fast path)
         frames also go into a ClipBuffer → every CLIP_SECONDS a clip → caption() → store()
         in a background thread (slow path), so the video never freezes.
         POST /ask → agent_ask().  GET /clips/<name> serves saved clips.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import agent, config, fast_loop, memory, slow_loop
from backend.contracts import is_hazard

app = FastAPI(title="Second Sight")
config.CLIP_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/clips", StaticFiles(directory=config.CLIP_DIR), name="clips")
FRONTEND = Path(__file__).parent / "frontend"

_slow = ThreadPoolExecutor(max_workers=2)
SLOW_MAX_PENDING = 2           # clips waiting for caption+store; beyond this, skip new ones so memory never falls behind
FLASH_HOLD_S = 1.5             # after a spoken urgent warning, keep the red flash while the danger is still there
_log: list[dict] = []          # everything said, for the screen
_state = {"persona": config.PERSONA, "latest_caption": "", "fps": 0.0, "clips": 0, "clips_skipped": 0}
_lock = threading.Lock()
_slow_pending = 0              # guarded by _lock


def _submit_slow(clip_path: Path, objects: list[str], t: float, dets: list[dict]) -> None:
    """Queue a clip for caption+store, unless SLOW_MAX_PENDING are already waiting (Cosmos is slow):
    then drop it, so the queue can't grow without limit and memory stays close to live."""
    global _slow_pending
    with _lock:
        if _slow_pending >= SLOW_MAX_PENDING:
            _state["clips_skipped"] += 1
            skip = True
        else:
            _slow_pending += 1
            skip = False
    if skip:
        clip_path.unlink(missing_ok=True)
        return

    def run() -> None:
        global _slow_pending
        try:
            _slow_path(clip_path, objects, t, dets)
        except Exception as e:
            print(f"[slow_path] clip failed: {e}")
        finally:
            with _lock:
                _slow_pending -= 1

    _slow.submit(run)


def _slow_path(clip_path: Path, objects: list[str], t: float, dets: list[dict] | None = None) -> None:
    """Runs in a worker thread: caption → store. Never blocks the frame loop.
    COSMOS_MODE=yolo → caption is a template built from the clip's detections (no Cosmos).
    Any Cosmos failure also falls back to that template, so the log never shows an error."""
    try:
        if config.COSMOS_MODE == "yolo" or not config.USE_REAL_COSMOS:
            cap = slow_loop.caption_from_detections(dets or []) if config.COSMOS_MODE == "yolo" else slow_loop.caption(str(clip_path))
        else:
            cap = slow_loop.caption(str(clip_path))
    except Exception as e:
        print(f"[slow_path] Cosmos failed ({e}); using YOLO caption")
        cap = slow_loop.caption_from_detections(dets or [])
    # where each object was last seen in this clip, so last_seen() can say "on your right"
    sides = {d["label"]: d["side"] for d in (dets or [])}
    memory.store(str(clip_path), cap, objects, t, sides=sides)
    agent.set_latest_caption(cap)
    with _lock:
        _state["latest_caption"] = cap
        _state["clips"] += 1
        _log.append({"t": t, "kind": "caption", "text": cap, "clip_url": memory.clip_url_for(str(clip_path))})


@app.get("/", response_class=HTMLResponse)
def index():
    return (FRONTEND / "index.html").read_text(encoding="utf-8")   # Windows defaults to cp1252 and crashes


@app.get("/state")
def state():
    with _lock:
        return {**_state, "log": _log[-30:]}


class Ask(BaseModel):
    question: str


@app.post("/ask")
def ask(body: Ask):
    ans = agent.agent_ask(body.question)
    with _lock:
        _log.append({"t": time.time(), "kind": "answer", "text": ans["text"], "clip_url": ans.get("clip_url")})
    return ans


class Persona(BaseModel):
    persona: str


@app.post("/persona")
def persona(body: Persona):
    config.PERSONA = body.persona if body.persona in agent.PERSONAS else "calm"
    with _lock:
        _state["persona"] = config.PERSONA
    return {"persona": config.PERSONA}


@app.get("/search")
def search(q: str):
    return memory.search(q)


@app.websocket("/ws/frames")
async def ws_frames(ws: WebSocket):
    """One frame in → one JSON reply out, always. The browser keeps a few frames in flight to hide the
    tunnel round trip; if they pile up faster than detect() runs, we skip to the newest and tell the
    client how many were skipped so it can keep its in-flight count honest."""
    await ws.accept()
    buf = slow_loop.ClipBuffer()
    seen_labels: set[str] = set()
    clip_dets: list[dict] = []
    n, t_fps = 0, time.time()
    last_urgent = 0.0
    q: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def reader() -> None:
        try:
            while True:
                q.put_nowait(await ws.receive_bytes())
        except (WebSocketDisconnect, RuntimeError, KeyError):
            q.put_nowait(None)              # KeyError: a non-binary message; treat like a close

    reader_task = asyncio.create_task(reader())
    try:
        while True:
            data = await q.get()
            dropped = 0
            while data is not None and not q.empty():   # stale frames waiting → keep only the newest
                nxt = q.get_nowait()
                if nxt is None:
                    data = None
                    break
                data, dropped = nxt, dropped + 1
            if data is None:
                break
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                await ws.send_text(json.dumps({"detections": [], "hazard": False, "say": None,
                                               "caption": _state["latest_caption"], "fps": round(_state["fps"], 1),
                                               "dropped": dropped}))
                continue
            t = time.time()

            # ---- fast path --------------------------------------------------
            dets = await asyncio.to_thread(fast_loop.detect, frame)
            seen_labels.update(d["label"] for d in dets)
            clip_dets.extend({"label": d["label"], "side": d["side"], "distance": d["distance"]} for d in dets)
            line = agent.agent_on_events(dets, _state["latest_caption"])
            if line:
                with _lock:
                    _log.append({"t": t, "kind": line["urgency"], "text": line["text"]})

            # ---- slow path --------------------------------------------------
            clip = buf.add(frame)
            if clip is not None:
                _submit_slow(clip, sorted(seen_labels), t, clip_dets)
                seen_labels, clip_dets = set(), []

            # Red flash follows the spoken warning (not raw is_hazard), held while that danger lasts,
            # so leaning in without a warning doesn't flash.
            if line and line.get("urgency") == "urgent":
                last_urgent = t
            flash = t - last_urgent < FLASH_HOLD_S and any(is_hazard(d) for d in dets)

            n += 1
            if t - t_fps >= 1:
                _state["fps"], n, t_fps = n / (t - t_fps), 0, t

            await ws.send_text(json.dumps({
                "detections": dets, "hazard": flash,
                "say": line, "caption": _state["latest_caption"], "fps": round(_state["fps"], 1),
                "dropped": dropped,
            }))
    except WebSocketDisconnect:
        pass
    finally:
        reader_task.cancel()


@app.on_event("startup")
def _seed():
    if not config.USE_REAL_MEMORY:
        memory.seed_fake_memory()
