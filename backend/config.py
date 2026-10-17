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

# Cosmos Reason. COSMOS_MODE=nvcf → NVIDIA hosted API (build.nvidia.com, needs only an nvapi- key)
#                COSMOS_MODE=openai → self-hosted NIM/vLLM, OpenAI-compatible at COSMOS_BASE_URL
#                COSMOS_MODE=yolo   → no Cosmos: captions are templates built from YOLO detections
COSMOS_MODE = os.getenv("COSMOS_MODE", "nvcf")
COSMOS_API_KEY = os.getenv("COSMOS_API_KEY") or os.getenv("NVIDIA_API_KEY", "")
COSMOS_MODEL = os.getenv("COSMOS_MODEL", "nvidia/cosmos-reason1-7b")
COSMOS_INVOKE_URL = os.getenv("COSMOS_INVOKE_URL") or f"https://ai.api.nvidia.com/v1/vlm/{COSMOS_MODEL}"
COSMOS_BASE_URL = os.getenv("COSMOS_BASE_URL", "http://127.0.0.1:8001/v1")
COSMOS_NUM_FRAMES = int(os.getenv("COSMOS_NUM_FRAMES", "8"))   # nvcf: frames sampled per clip
COSMOS_FPS = float(os.getenv("COSMOS_FPS", "2"))               # openai: sampling fps
COSMOS_MAX_TOKENS = int(os.getenv("COSMOS_MAX_TOKENS", "512"))

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
SPEAK_COOLDOWN_S = float(os.getenv("SPEAK_COOLDOWN_S", "10"))   # min gap between chatter lines (hazards ignore it)

# ElevenLabs voices (owner C). Blank key → browser speech, as before.
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVEN_MODEL = os.getenv("ELEVEN_MODEL", "eleven_v3")                     # most human; ~3 s (hazards are pre-rendered)
ELEVEN_VOICE_CALM = os.getenv("ELEVEN_VOICE_CALM", "cjVigY5qzO86Huf0OWal")  # Eric: smooth, trustworthy, casual
ELEVEN_VOICE_DOC = os.getenv("ELEVEN_VOICE_DOC", "JBFqnCBsd6RMkjVDRZzb")    # George: warm British storyteller

# W&B Inference wants the OpenAI-Project header as entity/project ("vastdata/team-48"). The lab VM exports
# WANDB_TEAM=vastdata and the short WANDB_PROJECT=team-48, which gives "400 Invalid Authentication"; join them.
WANDB_TEAM = os.getenv("WANDB_TEAM", "")
if WANDB_PROJECT and "/" not in WANDB_PROJECT and WANDB_TEAM:
    WANDB_PROJECT = f"{WANDB_TEAM}/{WANDB_PROJECT}"

# YOLO closing (owner B): box area now vs this many seconds ago. Replaces SMOOTH_FRAMES, which meant a
# different time at every frame rate (5 fps through the tunnel vs 30 locally).
CLOSING_WINDOW_S = float(os.getenv("CLOSING_WINDOW_S", "0.6"))
