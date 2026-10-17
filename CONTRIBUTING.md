# Second Sight — instructions for anyone working in this repo

Read this before touching code. Every teammate reads this file, so it is the shared truth.

## What we're building
Live AI companion for people with sight loss. Camera → YOLO (fast hazards) + Cosmos (scene captions) →
VAST memory → W&B LLM agent → spoken, direction-panned output. One FastAPI process. See README.md.

## The contract is law
`backend/contracts.py` defines the seven function shapes. **Never change a shape** — if you think you must,
stop and tell the human to raise it with the group. Add fields only if every consumer tolerates them missing.

## Ownership — only edit your owner's files
| Role | Files you may edit |
|---|---|
| A memory | `backend/memory.py` |
| B eyes | `backend/fast_loop.py` |
| C brain + voice | `backend/agent.py`, the audio/speech section of `frontend/index.html` |
| D glue | `main.py`, `frontend/index.html`, `backend/slow_loop.py`, `README.md`, `CONTRIBUTING.md` |
Shared, change-with-care: `backend/config.py` (add new env vars at the bottom, with defaults; never rename), `.env.example` (mirror config.py).
Never commit `.env`, `data/`, or `*.pt`.

## Fake/real discipline
Every module keeps BOTH a `_fake` and a `_real` implementation selected by a `USE_REAL_*` flag.
Never delete a fake. Never make the app import a heavy dependency (torch, ultralytics, boto3, weave)
at module top level unless the real flag is on — import lazily inside the real function.

## Before you push (every 30 minutes)
1. `python -m py_compile main.py backend/*.py`
2. `python scripts/smoke.py` — must print ALL OK on fakes
3. `git pull --rebase origin main` → resolve → `git push origin main`
Main must always run on fakes. If you broke it, fix or revert within 5 minutes.

## Style
Plain functions, type hints, no classes unless holding state. Short docstrings. No new frameworks.
Slow calls (Cosmos, LLM, S3) never run on the frame loop — use the thread pool in main.py.
