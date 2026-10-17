"""GLUE — one FastAPI process.  Owner: D.

  uvicorn main:app --reload --port 8000
  open http://localhost:8000

Wiring:  browser webcam → JPEG frames over WebSocket → detect() every frame →
         hazards/agent line back to the browser immediately (fast path)
         frames also go into a ClipBuffer → every CLIP_SECONDS a clip → caption() → store()
         in a background thread (slow path), so the video never freezes.
         POST /ask → agent_ask().  GET /clips/<name> serves saved clips.
         POST /upload → the same per-frame path (process_frame) over a recorded video in a background
         thread, paced in real time, writing an annotated mp4 + cue timeline under /uploads/.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import agent, config, fast_loop, memory, slow_loop
from backend.contracts import crowd_alert, fixed_alert, is_hazard
from backend.navigate import NavState, guidance_sentence, guide_side, steer

app = FastAPI(title="Second Sight")
config.CLIP_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/clips", StaticFiles(directory=config.CLIP_DIR), name="clips")
UPLOAD_DIR = config.ROOT / "data/uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
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


def _nav_json(nav: NavState) -> dict:
    return {"active": nav.active, "offset": nav.offset, "last": nav.last_cue,
            "ttc_ahead": round(nav.ttc_ahead, 2) if nav.ttc_ahead is not None else None}


@dataclass
class SessionState:
    """Everything one frame source carries between frames: a websocket connection or an uploaded video.
    Guide mode is on from the first frame unless the client says otherwise."""
    live: bool = True                                   # live sessions publish their fps to /state
    buf: slow_loop.ClipBuffer = field(default_factory=slow_loop.ClipBuffer)
    seen_labels: set[str] = field(default_factory=set)
    clip_dets: list[dict] = field(default_factory=list)
    nav: NavState = field(default_factory=NavState)
    last_guidance: float = 0.0
    last_hazard_text: str = ""                          # guide mode repeats an urgent alert at most every 2 s
    last_hazard_t: float = 0.0
    last_urgent: float = 0.0
    n: int = 0
    t_fps: float = field(default_factory=time.time)
    fps: float = 0.0

    def __post_init__(self) -> None:
        self.nav.start(None)


def empty_reply(s: SessionState, dropped: int = 0) -> dict:
    return {"detections": [], "hazard": False, "say": None, "caption": _state["latest_caption"],
            "fps": round(s.fps, 1), "dropped": dropped, "crowded": False, "nav": _nav_json(s.nav)}


def process_frame(frame: np.ndarray, s: SessionState, dropped: int = 0) -> dict:
    """One frame through the whole pipeline: detect → (guide steering | describe chatter) → clip buffer →
    slow path. Returns the JSON the websocket sends. Same code for the camera and for uploaded video."""
    t = time.time()
    nav = s.nav

    # ---- fast path --------------------------------------------------------------
    dets = fast_loop.detect(frame)
    s.seen_labels.update(d["label"] for d in dets)
    s.clip_dets.extend({"label": d["label"], "side": d["side"], "distance": d["distance"]} for d in dets)
    H, W = frame.shape[:2]
    if nav.active:
        # guide mode: steering cues, else urgent hazards, else a reasoned phrase at most every 6 s.
        # No crowding rule, no agent chatter: objects are only described when asked.
        line = steer(dets, W, H, nav)
        if line is None:
            hz = [d for d in dets if is_hazard(d)]
            if hz:
                alert = fixed_alert(hz[0])
                if alert["text"] != s.last_hazard_text or t - s.last_hazard_t >= 2.0:
                    s.last_hazard_text, s.last_hazard_t = alert["text"], t
                    line = alert
        if line is None and t - s.last_guidance >= 6.0:
            g = guidance_sentence(dets, W, H, _state["latest_caption"])
            if g:
                s.last_guidance = t
                line = {"text": g, "side": guide_side(dets, W, H), "urgency": "info"}
    else:
        # scene-level crowding rule first (fast path, fixed text); otherwise C's per-object hazards + chatter
        line = crowd_alert(dets) or agent.agent_on_events(dets, _state["latest_caption"])
    if line:
        with _lock:
            _log.append({"t": t, "kind": line["urgency"], "text": line["text"]})

    # ---- slow path --------------------------------------------------------------
    clip = s.buf.add(frame)
    if clip is not None:
        _submit_slow(clip, sorted(s.seen_labels), t, s.clip_dets)
        s.seen_labels, s.clip_dets = set(), []

    # Red flash follows the spoken warning (not raw is_hazard), held while that danger lasts,
    # so leaning in without a warning doesn't flash.
    if line and line.get("urgency") == "urgent":
        s.last_urgent = t
    flash = t - s.last_urgent < FLASH_HOLD_S and any(is_hazard(d) for d in dets)

    s.n += 1
    if t - s.t_fps >= 1:
        s.fps, s.n, s.t_fps = s.n / (t - s.t_fps), 0, t
        if s.live:
            _state["fps"] = s.fps

    return {
        "detections": dets, "hazard": flash,
        "say": line, "caption": _state["latest_caption"], "fps": round(s.fps, 1),
        "dropped": dropped, "crowded": sum(d["distance"] != "far" for d in dets) >= 3,
        "nav": _nav_json(nav),
    }


@app.websocket("/ws/frames")
async def ws_frames(ws: WebSocket):
    """One frame in → one JSON reply out, always. The browser keeps a few frames in flight to hide the
    tunnel round trip; if they pile up faster than detect() runs, we skip to the newest and tell the
    client how many were skipped so it can keep its in-flight count honest."""
    await ws.accept()
    s = SessionState(live=True)
    q: asyncio.Queue[bytes | None] = asyncio.Queue()

    def control(text: str) -> None:
        """Text messages are JSON controls: nav start/stop and compass headings."""
        try:
            m = json.loads(text)
        except ValueError:
            return
        if m.get("type") == "nav":
            if m.get("action") == "start":
                h = m.get("heading")
                s.nav.start(float(h) if h is not None else None)
            elif m.get("action") == "stop":
                s.nav.stop()
        elif m.get("type") == "heading" and m.get("heading") is not None:
            s.nav.heading = float(m["heading"])

    async def reader() -> None:
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    q.put_nowait(msg["bytes"])
                elif msg.get("text"):
                    control(msg["text"])
        except (WebSocketDisconnect, RuntimeError):
            pass
        q.put_nowait(None)

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
                await ws.send_text(json.dumps(empty_reply(s, dropped)))
                continue
            reply = await asyncio.to_thread(process_frame, frame, s, dropped)
            await ws.send_text(json.dumps(reply))
    except WebSocketDisconnect:
        pass
    finally:
        reader_task.cancel()


# ---- uploaded video through the same pipeline ------------------------------------------
_uploads: dict[str, dict] = {}         # id → status dict (guarded by _lock)
BAND_COLOUR = (120, 120, 120)
COLOURS = {"haz": (92, 92, 255), "near": (52, 177, 242), "far": (163, 209, 90)}   # BGR of the UI colours


def _annotate(frame: np.ndarray, r: dict, cue: str, caption: str) -> np.ndarray:
    """Boxes + labels, the five band lines, the current cue big at the bottom, the caption as a subtitle."""
    f = frame.copy()
    H, W = f.shape[:2]
    for i in range(1, 5):
        x = int(W * i / 5)
        cv2.line(f, (x, 0), (x, H), BAND_COLOUR, 1, cv2.LINE_AA)
    cv2.line(f, (W // 2, 0), (W // 2, H), (163, 209, 90), 1, cv2.LINE_AA)
    for d in r["detections"]:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        haz = d["closing"] and d["distance"] == "near"
        col = COLOURS["haz"] if haz else COLOURS["near"] if d["distance"] == "near" else COLOURS["far"]
        cv2.rectangle(f, (x1, y1), (x2, y2), col, 3 if haz else 2)
        label = f"{d['label']} {d['side']} {d['distance']}" + (" closing" if d["closing"] else "")
        cv2.putText(f, label, (x1 + 4, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    if r["hazard"]:
        cv2.rectangle(f, (0, 0), (W - 1, H - 1), COLOURS["haz"], 12)
    y = H - 12
    if caption:
        cv2.rectangle(f, (0, y - 22), (W, H), (0, 0, 0), -1)
        cv2.putText(f, caption[:95], (10, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        y -= 26
    if cue:
        (tw, th), _ = cv2.getTextSize(cue, cv2.FONT_HERSHEY_DUPLEX, 1.3, 2)
        x = max(8, (W - tw) // 2)
        cv2.putText(f, cue, (x, y - 8), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(f, cue, (x, y - 8), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)
    return f


def _set_status(uid: str, **kw) -> None:
    with _lock:
        _uploads[uid].update(kw)


def _process_upload(uid: str, path: Path, fps: float, mode: str) -> None:
    """Background thread. Reads the video at ~fps, pacing frames in real time so the clip buffer, the
    time-to-contact estimate and every cooldown behave exactly as they do with a live camera."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        _set_status(uid, state="error", error="cannot open video")
        return
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, int(round(src_fps / max(0.5, fps))))
    eff_fps = src_fps / step
    s = SessionState(live=False)
    if mode != "guide":
        s.nav.stop()
    out = UPLOAD_DIR / f"{uid}.annotated.mp4"
    writer: slow_loop.Mp4Writer | None = None
    timeline: list[dict] = []
    cues = 0
    last_say, last_say_t = "", 0.0
    t_start = time.time()
    i = k = 0
    _set_status(uid, state="running")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if (i - 1) % step:
                continue
            h, w = frame.shape[:2]
            if w != config.FRAME_W:
                frame = cv2.resize(frame, (config.FRAME_W, int(round(h * config.FRAME_W / w))))
            target = t_start + k / eff_fps              # real-time pacing
            now = time.time()
            if target > now:
                time.sleep(target - now)
            r = process_frame(frame, s)
            t_video = round(k / eff_fps, 2)
            if r["say"]:
                cues += 1
                last_say, last_say_t = r["say"]["text"], time.time()
            timeline.append({"t": t_video, "say": r["say"], "hazard": r["hazard"], "nav": r["nav"],
                             "n_dets": len(r["detections"])})
            cue = last_say if time.time() - last_say_t < 2.5 else (s.nav.last_cue if s.nav.active else "")
            if writer is None:
                writer = slow_loop.Mp4Writer(out, eff_fps, frame.shape[1], frame.shape[0])
            writer.add(_annotate(frame, r, cue, r["caption"]))
            k += 1
            if k % 5 == 0:
                _set_status(uid, percent=min(99, int(100 * i / total)) if total else 0, frames=k, cues=cues)
        # flush the clip buffer so the tail of the video gets captioned and stored before we say done
        clip = s.buf.flush()
        if clip is not None:
            _slow_path(clip, sorted(s.seen_labels), time.time(), s.clip_dets)
    except Exception as e:
        _set_status(uid, state="error", error=str(e))
        print(f"[upload {uid}] failed: {e}")
    finally:
        cap.release()
        if writer is not None:
            writer.close()
    if _uploads[uid].get("state") == "error":
        return
    (UPLOAD_DIR / f"{uid}.timeline.json").write_text(json.dumps(timeline))
    _set_status(uid, state="done", percent=100, frames=k, cues=cues, duration=round(k / eff_fps, 2),
                video_url=f"/uploads/{uid}.annotated.mp4", timeline_url=f"/uploads/{uid}.timeline.json")


@app.post("/upload")
async def upload(file: UploadFile = File(...), fps: float = Form(10.0), mode: str = Form("guide")):
    uid = uuid.uuid4().hex[:10]
    path = UPLOAD_DIR / f"{uid}.mp4"
    path.write_bytes(await file.read())
    with _lock:
        _uploads[uid] = {"id": uid, "state": "queued", "percent": 0, "frames": 0, "cues": 0,
                         "mode": mode, "fps": fps, "name": file.filename}
    threading.Thread(target=_process_upload, args=(uid, path, fps, mode), daemon=True).start()
    return {"id": uid, "status_url": f"/upload/{uid}/status"}


@app.get("/upload/{uid}/status")
def upload_status(uid: str):
    with _lock:
        st = _uploads.get(uid)
        if st is None:
            raise HTTPException(404, "no such upload")
        return dict(st)


@app.on_event("startup")
def _seed():
    if not config.USE_REAL_MEMORY:
        memory.seed_fake_memory()
