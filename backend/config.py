"""Flags and endpoints. All from .env; every USE_REAL_* defaults to 0 (fake)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

def flag(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in ("1", "true", "yes")

USE_REAL_YOLO = flag("USE_REAL_YOLO")
USE_REAL_COSMOS = flag("USE_REAL_COSMOS")
USE_REAL_MEMORY = flag("USE_REAL_MEMORY")   # VAST S3 + semantic search
USE_REAL_LLM = flag("USE_REAL_LLM")

CLIP_DIR = ROOT / "data/clips"
CLIP_SECONDS = float(os.getenv("CLIP_SECONDS", "5"))
FRAME_W = int(os.getenv("FRAME_W", "640"))

# YOLO
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8n.pt")
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cpu")
NEAR_FRAC = float(os.getenv("NEAR_FRAC", "0.45"))   # box height / frame height → near
MID_FRAC = float(os.getenv("MID_FRAC", "0.22"))
CLOSING_GROWTH = float(os.getenv("CLOSING_GROWTH", "1.15"))  # box area ratio over window → closing
SMOOTH_FRAMES = int(os.getenv("SMOOTH_FRAMES", "5"))

# Cosmos (OpenAI-compatible NIM)
COSMOS_BASE_URL = os.getenv("COSMOS_BASE_URL", "https://integrate.api.nvidia.com/v1")
COSMOS_API_KEY = os.getenv("COSMOS_API_KEY") or os.getenv("NVIDIA_API_KEY", "")
COSMOS_MODEL = os.getenv("COSMOS_MODEL", "nvidia/cosmos-reason1-7b")

# W&B Inference + Weave
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "")
WANDB_BASE_URL = os.getenv("WANDB_BASE_URL", "https://api.inference.wandb.ai/v1")
AGENT_MODEL = os.getenv("AGENT_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

# Embeddings (OpenAI-compatible; blank key → local hashed fallback)
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", COSMOS_BASE_URL)
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")

# VAST S3
VAST_S3_ENDPOINT = os.getenv("VAST_S3_ENDPOINT", "")
VAST_ACCESS_KEY = os.getenv("VAST_ACCESS_KEY", "")
VAST_SECRET_KEY = os.getenv("VAST_SECRET_KEY", "")
VAST_BUCKET = os.getenv("VAST_BUCKET", "second-sight")

# Agent behaviour
PERSONA = os.getenv("PERSONA", "calm")          # calm | attenborough
SPEAK_COOLDOWN_S = float(os.getenv("SPEAK_COOLDOWN_S", "6"))

# ElevenLabs voices (owner C). Blank key → browser speech, as before.
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVEN_MODEL = os.getenv("ELEVEN_MODEL", "eleven_flash_v2_5")             # lowest latency
ELEVEN_VOICE_CALM = os.getenv("ELEVEN_VOICE_CALM", "pFZP5JQG7iQjIQuC4Bku")  # Lily: warm British female
ELEVEN_VOICE_DOC = os.getenv("ELEVEN_VOICE_DOC", "JBFqnCBsd6RMkjVDRZzb")    # George: warm British storyteller
