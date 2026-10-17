"""Preflight for the demo machine: is each real part on, and does it actually work here?

  python scripts/doctor.py          check everything (read-only; spends ~3 ElevenLabs characters)
  python scripts/doctor.py --fix    also install lap + download the YOLO weights if missing

Prints OK / WARN / FAIL per item with the exact fix. Secrets are never printed.
Run it on the machine that will run uvicorn for the demo (the lab VM), well before the slot.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIX = "--fix" in sys.argv
fails = warns = 0


def report(level: str, what: str, fix: str = "") -> None:
    global fails, warns
    fails += level == "FAIL"
    warns += level == "WARN"
    print(f"  {level:4s}  {what}" + (f"\n        -> {fix}" if fix and level != "OK" else ""))


def has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def pip(*pkgs: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *pkgs], check=False)


if not (ROOT / ".env").exists():
    print("No .env in the repo root. Start with:  cp .env.example .env")
    sys.exit(1)

from backend import config  # noqa: E402

print("flags:", {k: int(getattr(config, k)) for k in ("USE_REAL_YOLO", "USE_REAL_COSMOS", "USE_REAL_MEMORY", "USE_REAL_LLM")})

# ---------------------------------------------------------------- B: eyes (YOLO)
print("\nB  eyes (YOLO)")
if not config.USE_REAL_YOLO:
    report("WARN", "USE_REAL_YOLO=0: the fake walker is running, not the camera", "set USE_REAL_YOLO=1 in .env")
for mod, pkg in (("ultralytics", "ultralytics"), ("lap", "lap")):
    if not has(mod) and FIX:
        print(f"        installing {pkg} ...")
        pip(pkg)
    report("OK" if has(mod) else "FAIL", f"python package {pkg}", f"pip install {pkg}   (or rerun with --fix)")
weights = ROOT / config.YOLO_WEIGHTS
if not weights.exists() and FIX and has("ultralytics"):
    print(f"        downloading {config.YOLO_WEIGHTS} ...")
    subprocess.run([sys.executable, "-c", f"from ultralytics import YOLO; YOLO({config.YOLO_WEIGHTS!r})"], cwd=ROOT, check=False)
report("OK" if weights.exists() else "FAIL", f"weights {config.YOLO_WEIGHTS} on disk",
       "rerun with --fix (needs internet once), or copy the .pt file into the repo root")
if config.USE_REAL_YOLO and has("ultralytics") and has("lap") and weights.exists():
    import numpy as np
    from backend import fast_loop
    frame = np.zeros((480, 640, 3), np.uint8)
    fast_loop.detect(frame)
    t0 = time.time()
    for _ in range(5):
        fast_loop.detect(frame)
    ms = (time.time() - t0) / 5 * 1000
    report("OK" if ms < 150 else "WARN", f"detect() runs: {ms:.0f} ms/frame on {config.YOLO_DEVICE}",
           "slow: set YOLO_DEVICE=0 if this machine has a GPU")

# ---------------------------------------------------------------- D: clips + Cosmos
print("\nD  clips + Cosmos")
report("OK" if shutil.which("ffmpeg") else "WARN", "ffmpeg on PATH",
       "install ffmpeg (sudo apt install -y ffmpeg), else saved clips may not play in the browser")
if config.USE_REAL_COSMOS and config.COSMOS_MODE == "openai":
    import httpx
    try:
        t0 = time.time()
        r = httpx.get(config.COSMOS_BASE_URL.rstrip("/") + "/models",
                      headers={"Authorization": f"Bearer {config.COSMOS_API_KEY}"}, timeout=8)
        report("OK" if r.status_code == 200 else "FAIL", f"Cosmos at {config.COSMOS_BASE_URL}: HTTP {r.status_code} in {time.time() - t0:.1f}s",
               "check COSMOS_API_KEY (GPU_BEARER_TOKEN from /config/team-48.config)")
    except Exception as e:
        report("FAIL", f"Cosmos unreachable: {type(e).__name__}", "is COSMOS_BASE_URL right, and is this machine on the venue network?")
else:
    report("WARN" if not config.USE_REAL_COSMOS else "OK",
           f"USE_REAL_COSMOS={int(config.USE_REAL_COSMOS)} COSMOS_MODE={config.COSMOS_MODE}",
           "see README 'Running on the lab VM' to switch Cosmos on")

# ---------------------------------------------------------------- A: memory
print("\nA  memory (VAST)")
report("OK" if config.USE_REAL_MEMORY else "WARN", f"USE_REAL_MEMORY={int(config.USE_REAL_MEMORY)}",
       "set USE_REAL_MEMORY=1 on the VM (laptops can't reach VAST S3); smoke.py exercises it")

# ---------------------------------------------------------------- C: brain + voice
print("\nC  brain (W&B) + voice (ElevenLabs)")
from backend import agent  # noqa: E402

key = (config.WANDB_API_KEY or "").strip()
if not config.USE_REAL_LLM:
    report("WARN", "USE_REAL_LLM=0: answers come from rules, not the W&B model", "bash scripts/enable_brain.sh")
if not key:
    report("FAIL" if config.USE_REAL_LLM else "WARN", "WANDB_API_KEY missing", "bash scripts/enable_brain.sh  (type the key on this machine)")
else:
    if config.WANDB_PROJECT and not agent.wandb_project():
        report("WARN", f"WANDB_PROJECT={config.WANDB_PROJECT!r} is a placeholder; ignored (it made W&B reject the key)",
               "leave it blank, or set your real entity/project")
    try:
        t0 = time.time()
        text = agent._complete("Reply with exactly: ok", "ok?")
        report("OK" if text else "FAIL", f"W&B {config.AGENT_MODEL}: replied {text[:20]!r} in {time.time() - t0:.1f}s",
               "empty reply: use AGENT_MODEL=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B")
    except Exception as e:
        report("FAIL", f"W&B call failed: {type(e).__name__}: {str(e)[:120]}", "check the key / credits at wandb.ai")
from backend import voice  # noqa: E402

if not voice.enabled():
    report("WARN", "ElevenLabs off: browser voice will be used", "bash scripts/enable_brain.sh  (optional)")
else:
    t0 = time.time()
    url = voice.tts("OK.", config.PERSONA)
    report("OK" if url else "FAIL", f"ElevenLabs voice rendered in {time.time() - t0:.1f}s",
           voice._off["why"] or "check ELEVENLABS_API_KEY / credits")

# ---------------------------------------------------------------- demo clips
print("\nDemo clips")
clips = sorted((ROOT / "data/demo").glob("*.mp4"))
report("OK" if clips else "FAIL", f"data/demo/*.mp4: {', '.join(c.name for c in clips) or 'none'}",
       "film walk.mp4 (someone walks toward you) and cup.mp4 (put a cup down, walk away) at eye level; "
       "data/ is not in git, copy them onto this machine")

print(f"\n{'READY' if not fails else f'{fails} FAIL'}, {warns} WARN")
sys.exit(1 if fails else 0)
