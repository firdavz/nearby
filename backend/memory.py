"""MEMORY — store / search / last_seen.  Owner: A.

FAKE: in-process list + keyword search (works from minute one).
REAL: clips uploaded to VAST S3 (team DB bucket, second-sight/clips/); captions + YOLO object
labels + embeddings kept in SQLite (VAST DB mirror next); search is cosine over NVIDIA Cosmos
Embed1 caption vectors (local hashed embedding if the GPU host is unreachable). Clips play via
main.py's /clips/, pulled back from VAST if the local copy is missing.

Both implementations share the same tiny interface, and both record the *YOLO labels* seen in
each clip — that's what makes last_seen("cup") exact rather than a fuzzy caption match.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

from backend import config
from backend.contracts import MemoryHit

# --------------------------------------------------------------------------- embeddings
DIM = 1024


def _local_embed(text: str) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1
    n = np.linalg.norm(v)
    return v / n if n else v


def embed(text: str, kind: str = "passage") -> np.ndarray:
    if not config.EMBED_API_KEY:
        return _local_embed(text)
    from openai import OpenAI
    client = OpenAI(base_url=config.EMBED_BASE_URL, api_key=config.EMBED_API_KEY)
    r = client.embeddings.create(model=config.EMBED_MODEL, input=[text],
                                 extra_body={"input_type": kind, "truncate": "END"})
    v = np.array(r.data[0].embedding, dtype=np.float32)
    return v / np.linalg.norm(v)


def clip_url_for(path_or_key: str) -> str:
    """main.py serves data/clips at /clips/<name>. S3 keys are presigned by the real store."""
    return f"/clips/{Path(path_or_key).name}"


# --------------------------------------------------------------------------- FAKE
_fake: list[dict] = []


def store_fake(clip_path: str, caption: str, objects: list[str], t: float) -> None:
    _fake.append({"t": t, "caption": caption, "objects": [o.lower() for o in objects], "clip": clip_path})


def search_fake(query: str) -> list[MemoryHit]:
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = []
    for m in _fake:
        words = set(re.findall(r"[a-z0-9]+", m["caption"].lower())) | set(m["objects"])
        s = len(q & words)
        if s:
            scored.append((s, m["t"], m))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    return [{"t": m["t"], "caption": m["caption"], "clip_url": clip_url_for(m["clip"])} for _, _, m in scored[:5]]


def last_seen_fake(label: str) -> MemoryHit | None:
    label = label.lower().rstrip("s")
    for m in reversed(_fake):
        if any(label in o for o in m["objects"]) or label in m["caption"].lower():
            return {"t": m["t"], "caption": m["caption"], "clip_url": clip_url_for(m["clip"])}
    return None


# --------------------------------------------------------------------------- REAL
# On the lab VM the team credentials live in /config/<team>.config under VAST's own names
# (S3_ENDPOINT, ACCESS_KEY, ...). Read those directly so no secrets get copied into .env.
def _team_config() -> dict[str, str]:
    out: dict[str, str] = {}
    for f in sorted(Path("/config").glob("*.config")) if Path("/config").is_dir() else []:
        for line in f.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip().removeprefix("export ").strip()] = v.strip().strip('"')
    return out


_CFG = _team_config()


def _env(*names: str, default: str = "") -> str:
    """First non-empty value among our .env names, then VAST's names in /config."""
    for n in names:
        v = getattr(config, n, "") or os.getenv(n, "") or _CFG.get(n, "")
        if v:
            return v
    return default


S3_ENDPOINT = _env("VAST_S3_ENDPOINT", "S3_ENDPOINT")
S3_ACCESS = _env("VAST_ACCESS_KEY", "ACCESS_KEY")
S3_SECRET = _env("VAST_SECRET_KEY", "SECRET_KEY")
# Clips go in the team's own DB bucket under a prefix: the chunks/segments buckets trigger
# VAST's ingest pipeline, and creating new buckets isn't allowed for team users.
S3_BUCKET = _env("VASTDB_BUCKET") or config.VAST_BUCKET
S3_PREFIX = "second-sight/clips/"

# Cosmos Embed1 on the shared GPU host (256-dim). Override with EMBED1_URL if it moves.
EMBED1_URL = _env("EMBED1_URL", "COSMOS_EMBED1_URL", default="http://166.19.38.112:8003")
GPU_TOKEN = _env("GPU_BEARER_TOKEN")


def _cosmos_embed(text: str, kind: str) -> np.ndarray | None:
    """Cosmos Embed1 vector, or None if the GPU host can't be reached."""
    if not GPU_TOKEN:
        return None
    import httpx
    try:
        r = httpx.post(f"{EMBED1_URL}/v1/embeddings", timeout=15,
                       headers={"Authorization": f"Bearer {GPU_TOKEN}"},
                       json={"input": text, "model": "nvidia/cosmos-embed1",
                             # Embed1 only accepts query / bulk_text / bulk_video
                             "request_type": "query" if kind == "query" else "bulk_text",
                             "encoding_format": "float"})
        r.raise_for_status()
        v = np.array(r.json()["data"][0]["embedding"], dtype=np.float32)
        return v / np.linalg.norm(v)
    except Exception as e:
        print(f"[memory] Cosmos Embed failed, using local embedding: {e}")
        return None


def real_embed(text: str, kind: str = "passage") -> np.ndarray:
    """Cosmos Embed1 when the GPU token is there; otherwise the .env embedding / local fallback."""
    if GPU_TOKEN and not config.EMBED_API_KEY:
        v = _cosmos_embed(text, kind)
        if v is not None:
            return v
    return embed(text, kind)


class RealMemory:
    def __init__(self):
        db = config.ROOT / "data/memory.sqlite"
        db.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()   # store() runs on the thread pool, search() on requests
        self.con = sqlite3.connect(db, check_same_thread=False)
        self.con.execute("CREATE TABLE IF NOT EXISTS clips (t REAL, caption TEXT, objects TEXT, key TEXT, emb BLOB)")
        self.con.commit()
        self.s3 = None
        if S3_ENDPOINT and S3_ACCESS:
            import boto3
            import urllib3
            urllib3.disable_warnings()   # the lab cluster uses a self-signed cert (VAST's skills do the same)
            self.s3 = boto3.client("s3", endpoint_url=S3_ENDPOINT, aws_access_key_id=S3_ACCESS,
                                   aws_secret_access_key=S3_SECRET, verify=False)
            print(f"[memory] VAST S3 on, bucket {S3_BUCKET}/{S3_PREFIX}")
        else:
            print("[memory] no VAST credentials found, clips stay local only")

    def store(self, clip_path: str, caption: str, objects: list[str], t: float) -> None:
        key = clip_path
        if self.s3:
            try:
                key = S3_PREFIX + Path(clip_path).name
                self.s3.upload_file(clip_path, S3_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
            except Exception as e:
                print(f"[memory] S3 upload failed, keeping clip local: {e}")
                key = clip_path
        text = caption + " Objects: " + ", ".join(objects)
        emb = real_embed(text, "passage")
        with self.lock:
            self.con.execute("INSERT INTO clips VALUES (?,?,?,?,?)",
                             (t, caption, json.dumps([o.lower() for o in objects]), key, emb.tobytes()))
            self.con.commit()

    def _url(self, key: str) -> str:
        """Browsers can't reach the VAST endpoint, so clips are always played through main.py's
        /clips/. If the local copy is gone, pull it back down from VAST first."""
        local = config.CLIP_DIR / Path(key).name
        if self.s3 and key.startswith(S3_PREFIX) and not local.exists():
            try:
                local.parent.mkdir(parents=True, exist_ok=True)
                self.s3.download_file(S3_BUCKET, key, str(local))
            except Exception as e:
                print(f"[memory] couldn't fetch {key} from VAST: {e}")
        return clip_url_for(key)

    def _rows(self, sql: str) -> list[tuple]:
        with self.lock:
            return self.con.execute(sql).fetchall()

    def search(self, query: str) -> list[MemoryHit]:
        q = real_embed(query, "query")
        # Only compare vectors from the same model (the local fallback has a different size).
        rows = [r for r in self._rows("SELECT t, caption, objects, key, emb FROM clips")
                if len(r[4]) == q.nbytes]
        if not rows:
            return []
        M = np.stack([np.frombuffer(r[4], dtype=np.float32) for r in rows])
        scores = M @ q
        top = np.argsort(-scores)[:5]
        return [{"t": rows[i][0], "caption": rows[i][1], "clip_url": self._url(rows[i][3])} for i in top if scores[i] > 0]

    def last_seen(self, label: str) -> MemoryHit | None:
        label = label.lower().rstrip("s")
        for t, cap, objs, key, _ in self._rows("SELECT * FROM clips ORDER BY t DESC"):
            if any(label in o for o in json.loads(objs)) or label in cap.lower():
                return {"t": t, "caption": cap, "clip_url": self._url(key)}
        return None


if config.USE_REAL_MEMORY:
    _real = RealMemory()
    store, search, last_seen = _real.store, _real.search, _real.last_seen
else:
    store, search, last_seen = store_fake, search_fake, last_seen_fake


def seed_fake_memory() -> None:
    """Three test clips so 'where are my keys?' works before any camera runs."""
    now = time.time()
    store("data/clips/seed_1.mp4", "A person sits at a desk with a laptop and a blue cup.", ["person", "laptop", "cup"], now - 120)
    store("data/clips/seed_2.mp4", "A set of keys is placed on the kitchen table next to a bowl.", ["person", "keys", "bowl"], now - 60)
    store("data/clips/seed_3.mp4", "The person walks toward the door; a backpack hangs on the chair.", ["person", "backpack", "chair"], now - 20)
