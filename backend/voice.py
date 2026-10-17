"""VOICE — natural speech via ElevenLabs.  Owner: C.

tts(text, persona) -> "/clips/tts_<hash>.mp3" or None.
The mp3 lands in CLIP_DIR, which main.py already serves at /clips, so no new route is needed.
Cached by (voice, text): a repeated line costs nothing and plays instantly.
None means "use browser speech" — no key, network down, or out of credits. Never raises.

Hazard lines are pre-rendered at startup (warm_hazards) so warnings stay instant.
"""
from __future__ import annotations

import hashlib
import threading

from backend import config

HAZARD_LABELS = ("person", "bicycle", "car", "motorcycle", "bus", "truck", "dog", "chair", "bench", "door")
_lock = threading.Lock()
_off = {"why": ""}   # set after a bad key / no credits, so we stop calling for the rest of the run


def enabled() -> bool:
    key = config.ELEVENLABS_API_KEY.strip()
    return bool(key) and not key.startswith("PASTE") and not _off["why"]


def _voice_id(persona: str) -> str:
    return config.ELEVEN_VOICE_DOC if persona == "attenborough" else config.ELEVEN_VOICE_CALM


def _path(text: str, voice_id: str):
    h = hashlib.sha1(f"{voice_id}|{config.ELEVEN_MODEL}|{text}".encode()).hexdigest()[:16]
    return config.CLIP_DIR / f"tts_{h}.mp3"


def cached_url(text: str, persona: str) -> str | None:
    """URL if this line is already rendered, else None. Never calls the network."""
    p = _path(text, _voice_id(persona))
    return f"/clips/{p.name}" if p.exists() else None


def tts(text: str, persona: str = "calm") -> str | None:
    if not enabled() or not text:
        return None
    voice_id = _voice_id(persona)
    p = _path(text, voice_id)
    if p.exists():
        return f"/clips/{p.name}"
    try:
        import httpx
        r = httpx.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            params={"output_format": "mp3_44100_64"},
            headers={"xi-api-key": config.ELEVENLABS_API_KEY},
            json={"text": text, "model_id": config.ELEVEN_MODEL,
                  "voice_settings": {"stability": 0.5, "similarity_boost": 0.75,
                                     "style": 0.35 if persona == "attenborough" else 0.0}},
            timeout=6,
        )
        if r.status_code in (401, 402, 403):   # bad key or out of credits: retrying won't help
            _off["why"] = f"HTTP {r.status_code}: {r.text[:160]}"
            print("voice: ElevenLabs disabled for this run, using browser speech —", _off["why"])
            return None
        r.raise_for_status()
    except Exception as e:
        print("voice.tts failed, browser speech will be used:", str(e)[:200])
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        p.write_bytes(r.content)
    return f"/clips/{p.name}"


def warm_hazards() -> None:
    """Pre-render every fixed hazard line in both voices, in the background, so warnings sound
    natural with zero delay. ~60 short lines, rendered once and cached on disk."""
    if not enabled():
        return

    def run():
        for persona in ("calm", "attenborough"):
            for label in HAZARD_LABELS:
                for side in ("left", "centre", "right"):
                    tts(f"{label}, {side}, near", persona)

    threading.Thread(target=run, daemon=True).start()
