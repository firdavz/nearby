# Second Sight — live AI companion for people with sight loss

A phone or laptop camera acts as eyes. It warns about hazards with directional sound, stays quiet
otherwise, and remembers everything it saw — so you can ask *"where are my keys?"* and get the answer
plus the clip.

**Sponsor tools:** YOLO (fast eyes) · NVIDIA Cosmos (scene understanding) · VAST (memory) ·
W&B Inference + Weave (agent brain) · CoreWeave GPUs · Cursor.

## Run it (works on fakes with no keys at all)

```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8000
# open http://localhost:8000 → Start camera
```

Phone camera needs HTTPS. Quickest: `pip install pyopenssl` then
`uvicorn main:app --port 8443 --ssl-keyfile key.pem --ssl-certfile cert.pem`
(make the pair with `openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 2 -subj /CN=localhost`),
accept the warning on the phone. If that fights you, use the laptop webcam — the demo is the same.

## Architecture — two speeds, one memory

```
browser camera ──JPEG/WebSocket──▶ main.py
                                     ├─ FAST  detect(frame) → hazard rule → fixed alert ─▶ browser: earcon (panned) + speech   <1 s
                                     └─ SLOW  ClipBuffer → every 5 s an H.264 clip → caption() (Cosmos) → store() (VAST)      background
POST /ask ────────────────────────▶ agent_ask() → tools: search_memory · last_seen · describe_now ─▶ text + clip_url
```

Everything is one Python process. Modules are plain functions. Slow calls run in a thread pool so video never freezes.

## The contract (`backend/contracts.py`) — don't change without telling the group

| Function | Owner | File |
|---|---|---|
| `detect(frame) -> list[Detection]` | B | `backend/fast_loop.py` |
| `caption(clip_path) -> str` | D | `backend/slow_loop.py` |
| `store(clip_path, caption, objects, t)` · `search(query)` · `last_seen(label)` | A | `backend/memory.py` |
| `agent_on_events(events, caption) -> AgentLine \| None` · `agent_ask(question) -> AgentAnswer` | C | `backend/agent.py` |

Every module has a **fake** and a **real** implementation. Flags in `.env`:
`USE_REAL_YOLO`, `USE_REAL_COSMOS`, `USE_REAL_MEMORY`, `USE_REAL_LLM`. Flip each to `1` once it's proven; flip back for the demo if it misbehaves.

Captions without Cosmos: `USE_REAL_COSMOS=1` + `COSMOS_MODE=yolo` builds captions from the detections seen in each clip
("A cup on your right, mid; a person on your left, near."). `COSMOS_MODE=nvcf` (hosted) or `openai` (self-hosted NIM) call
Cosmos Reason and fall back to the YOLO template on any error.

## Behaviour rules already built in

- **Hazard = closing + near.** Fixed text, never waits for the LLM. Repeats at most every 2 s while the danger persists.
- **Default is silence.** Non-urgent lines are rate-limited (`SPEAK_COOLDOWN_S`, default 6 s) and never repeat the last sentence.
- **Direction:** browser speech can't be panned, so a short panned earcon (left/right) plays first, then the sentence. Urgent lines cancel any queued speech.
- **Memory stores YOLO labels per clip**, so `last_seen("cup")` is exact; captions give semantic search.
- **Persona:** `calm` or `attenborough` — dropdown in the UI, `POST /persona`.

## What to test in the first hour

1. Cosmos and VAST reachable from the venue, and latency per call → `USE_REAL_COSMOS=1`, watch the log.
2. YOLO on your demo objects. COCO has no `keys` class — use **cup, bottle, cell phone, backpack, book or remote** as the demo item (the agent's fake path already knows these words).
3. Phone camera over HTTPS, else laptop webcam.

## Demo (3 minutes)

1. Walk toward the camera → red border, panned earcon, *"person, left, near"*.
2. Put the cup down, walk away, ask *"where is my cup?"* → answer + clip.
3. Switch to Attenborough.
4. Close on impact and which sponsor did what.

## Files
```
main.py                 FastAPI glue: /ws/frames, /ask, /persona, /search, /state, /clips
frontend/index.html     camera, boxes, hazard flash, panned earcon + speech, ask box, log
backend/contracts.py    shapes + hazard rule + fixed alert
backend/fast_loop.py    detect(): fake walker | YOLO+ByteTrack, side/distance/closing
backend/slow_loop.py    ClipBuffer → H.264 via ffmpeg; caption(): fake | Cosmos
backend/memory.py       store/search/last_seen: in-memory | VAST S3 + SQLite + embeddings
backend/agent.py        agent_on_events/agent_ask: rules | W&B LLM with tools, Weave-traced
```
