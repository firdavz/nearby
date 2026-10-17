
# ════════════════════════════════════════════════════════════
# 🧠  Person A — Memory (VAST)
# ════════════════════════════════════════════════════════════

> You own **`backend/memory.py`**.
> **Your job:** everything the camera sees is stored in VAST and can be found again, by meaning and by object. You are the sponsor-tool the host cares most about, so make VAST visibly do the work.

<br>

## ⚙️  Setup  `[ 0:00 – 0:15 ]`

- Get the VAST S3 endpoint, access key, secret and bucket name from the organisers or Discord. Put them in your `.env` as `VAST_S3_ENDPOINT`, `VAST_ACCESS_KEY`, `VAST_SECRET_KEY`, `VAST_BUCKET`.

- `pip install boto3` (already in requirements). Test with five lines of boto3: list buckets, put a small file, get it back.

- Ask **D** which embedding endpoint is available (NVIDIA NIM `nv-embedqa-e5-v5` via the Cosmos key, or a W&B-served model). Put it in `EMBED_API_KEY`, `EMBED_BASE_URL`, `EMBED_MODEL`. If nothing is available, the local hashed embedding already works — search is weaker but functional.

<br>

## 🔧  Phase 1  `[ 0:15 – 2:00 ]`  — make RealMemory work

1. `USE_REAL_MEMORY=1`. Run the app; the slow path now calls `RealMemory.store()` every 5 s. Watch for the upload succeeding in the VAST bucket.

2. `search("person at a desk")` returns the right clip with a working presigned URL that plays in the browser log.

3. `last_seen("cup")` returns the most recent clip whose YOLO labels contain cup. This is exact, not fuzzy — it reads the objects list **B**'s detections give you.

4. Replace the local hashed embedding with the real embedding model and confirm `search("someone walking out of the room")` finds the caption about walking to the door.

5. *Optional, only if the cluster has VAST DB enabled:* move the metadata table from SQLite to VAST DB with the `vastdb` SDK. Keep SQLite as the mirror. **Do not start this before step 4 works.**

<br>

## 🎯  Phase 2  `[ 2:00 – 3:30 ]`  — the "where is my cup?" feature, end to end

- The answer sentence should read like a person: *"I last saw your cup about two minutes ago, on the table to the right."* Work with **C** so `agent_ask` uses your `last_seen` result and turns the timestamp into "two minutes ago".

- The clip plays inline in the log on the page. Confirm presigned URLs survive long enough for the demo (one hour is set).

- Store a per-clip side for the demo object (left/centre/right, from **B**'s detection) so the answer can say where, not just when. Add it as an optional field in your own table; **do not touch the contract.**

- Semantic search quality: try five natural questions a blind user would ask and make sure the top hit is right for each.

<br>

## ✅  Tests that prove you are done

- [ ] Store 3 clips, `search("keys")` returns the right one with a playable link — **by 1:00.**
- [ ] The live app has stored 20+ clips in VAST and "where is my cup?" plays the right one — **by 2:00.**

<br>

## 🛟  Fallback

`USE_REAL_MEMORY=0` returns to the in-memory keyword store, which already answers the demo question from seeded clips. The demo survives without VAST; the pitch does not, so **get Tier 1 (S3 upload) working first** and worry about embeddings second.

## 🤝  Hand-offs

None incoming. You depend on nobody — the fake slow path feeds you clips and labels from minute one.

<br>

## 🤖  Task brief

```text
Read CONTRIBUTING.md and backend/contracts.py. I am person A and I own backend/memory.py only. Set USE_REAL_MEMORY=1 and make RealMemory work against the VAST S3 endpoint in my .env: upload a clip, store its caption and object labels, then make search("keys") and last_seen("cup") return the right clip with a working presigned URL. Keep the fake implementation untouched. Run python scripts/smoke.py after every change and fix anything you broke. Do not edit other files except appending env vars to backend/config.py.
```

<br><br>

# ════════════════════════════════════════════════════════════
# 👁️  Person B — Eyes (YOLO)
# ════════════════════════════════════════════════════════════

> You own **`backend/fast_loop.py`**.
> **Your job:** within one second of someone walking at the camera, the app says *"person, left, near"* — and says nothing when they stand still. The whole demo opens on this, so it is the one part that cannot be flaky.

<br>

## ⚙️  Setup  `[ 0:00 – 0:15 ]`

- `pip install ultralytics`. Download weights once: `python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"`. If you have a CoreWeave GPU box, set `YOLO_DEVICE=0`; on a laptop cpu runs yolov8n at 15–30 fps at 640 px, which is enough.

- Check the webcam works in the browser page on fakes before touching anything.

<br>

## 🔧  Phase 1  `[ 0:15 – 2:00 ]`  — make detect_real solid

1. `USE_REAL_YOLO=1`. Boxes with labels appear over the live video. Track IDs persist while a person walks across the frame (ByteTrack is already configured).

2. **Side.** Box centre < 38% of width = left, > 62% = right. Check it matches what a person wearing the camera would call left and right — if the camera is mirrored, flip it here, nowhere else.

3. **Distance.** Box height as a fraction of frame height: `NEAR_FRAC` (default 0.45) and `MID_FRAC` (0.22). Stand at 1 m, 2 m, 4 m from the camera and set these so 1 m is near and 4 m is far.

4. **Closing.** Area now vs area `SMOOTH_FRAMES` ago (default 5) must exceed `CLOSING_GROWTH` (default 1.15). Walk toward the camera at normal pace: closing should flip true for the last 1–2 m of the approach. Stand still, sway, turn around: it must stay false. Tune growth and window until both hold.

5. The hazard rule is **closing AND near** (in `contracts.py`, not yours). Confirm a normal approach produces exactly one urgent line, repeated at most every 2 s if the person keeps standing close and moving.

6. Hold the demo object (cup or bottle) at arm's length in each third of the frame: label correct, side correct. If the label flickers between cup and bowl, pick the one YOLO prefers and **tell A and C the exact string.**

<br>

## 🎯  Phase 2  `[ 2:00 – 3:30 ]`  — the hazard feature, end to end

- Run the demo-walk clip **D** recorded through `scripts/replay.py` ten times. Count false alarms and misses. **Target: 10 of 10 warnings, 0 extra.** Adjust only your thresholds.

- **Non-person hazards:** a chair or table edge that grows in the frame as the wearer walks toward it is also closing + near. Test walking the camera toward a chair; make sure it fires and the label is sensible.

- **Camera FOV:** if **D** switches to the phone or the glasses feed (narrower or wider lens), re-check `NEAR_FRAC` once — one number, two minutes.

- **Latency:** time from the frame arriving to the warning line in the browser. Under 300 ms on GPU, under 800 ms on CPU. If worse, lower `imgsz` to 480 in `model.track(...)` or raise `conf`.

<br>

## ✅  Tests that prove you are done

- [ ] Walk at the camera → "person, centre, near" within 1 s; stand still → silence — **by 1:00.**
- [ ] 10/10 on the replay clip, no extras — **by 2:30.**

<br>

## 🛟  Fallback

`USE_REAL_YOLO=0` brings back the fake walker, which triggers the warning on a timer. It demos the pipeline but not the detection, so if real YOLO is unstable at 3:00, **fix detection before anything else — it is the headline.**

## 🤝  Hand-offs

**D** gives you `scripts/replay.py` by 1:30 and the demo-walk clip by 2:00. Until then, test on your webcam. Tell **A** and **C** the exact label string for the demo object as soon as you know it.

<br>

## 🤖  Task brief

```text
Read CONTRIBUTING.md and backend/contracts.py. I am person B and I own backend/fast_loop.py only. Set USE_REAL_YOLO=1 and make detect_real work on the laptop webcam: boxes appear, track IDs persist, side/distance/closing are filled per the contract. Then help me tune NEAR_FRAC, MID_FRAC, CLOSING_GROWTH and SMOOTH_FRAMES so walking toward the camera triggers exactly one "person, centre, near" within a second and standing still triggers nothing. Keep detect_fake untouched. Run python scripts/smoke.py after every change. Do not edit other files.
```
