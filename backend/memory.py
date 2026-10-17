"""MEMORY — store / search / last_seen.  Owner: A.

FAKE: in-process list + keyword search (works from minute one).
REAL: clips uploaded to a VAST S3 bucket; captions + YOLO object labels + embeddings kept in
SQLite (swap for VAST DB when the cluster has it); search is cosine over caption embeddings
(NIM/W&B embedding model, or a local hashed embedding if no key is set).

Both implementations share the same tiny interface, and both record the *YOLO labels* seen in
each clip — that's what makes last_seen("cup") exact rather than a fuzzy caption match.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
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
class RealMemory:
    def __init__(self):
        db = config.ROOT / "data/memory.sqlite"
        db.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(db, check_same_thread=False)
        self.con.execute("CREATE TABLE IF NOT EXISTS clips (t REAL, caption TEXT, objects TEXT, key TEXT, emb BLOB)")
        self.con.commit()
        self.s3 = None
        if config.VAST_S3_ENDPOINT and config.VAST_ACCESS_KEY:
            import boto3
            self.s3 = boto3.client("s3", endpoint_url=config.VAST_S3_ENDPOINT,
                                   aws_access_key_id=config.VAST_ACCESS_KEY,
                                   aws_secret_access_key=config.VAST_SECRET_KEY)
            try:
                self.s3.head_bucket(Bucket=config.VAST_BUCKET)
            except Exception:
                self.s3.create_bucket(Bucket=config.VAST_BUCKET)

    def store(self, clip_path: str, caption: str, objects: list[str], t: float) -> None:
        key = clip_path
        if self.s3:
            key = f"clips/{Path(clip_path).name}"
            self.s3.upload_file(clip_path, config.VAST_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
        text = caption + " Objects: " + ", ".join(objects)
        self.con.execute("INSERT INTO clips VALUES (?,?,?,?,?)",
                         (t, caption, json.dumps([o.lower() for o in objects]), key, embed(text).tobytes()))
        self.con.commit()

    def _url(self, key: str) -> str:
        if self.s3 and key.startswith("clips/"):
            return self.s3.generate_presigned_url("get_object", Params={"Bucket": config.VAST_BUCKET, "Key": key}, ExpiresIn=3600)
        return clip_url_for(key)

    def search(self, query: str) -> list[MemoryHit]:
        rows = self.con.execute("SELECT t, caption, objects, key, emb FROM clips").fetchall()
        if not rows:
            return []
        q = embed(query, "query")
        M = np.stack([np.frombuffer(r[4], dtype=np.float32) for r in rows])
        scores = M @ q
        top = np.argsort(-scores)[:5]
        return [{"t": rows[i][0], "caption": rows[i][1], "clip_url": self._url(rows[i][3])} for i in top if scores[i] > 0]

    def last_seen(self, label: str) -> MemoryHit | None:
        label = label.lower().rstrip("s")
        for t, cap, objs, key, _ in self.con.execute("SELECT * FROM clips ORDER BY t DESC").fetchall():
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
