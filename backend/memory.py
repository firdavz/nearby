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


NOTE_PREFIX = "note:"   # key of a remembered user note (no video clip)


def clip_url_for(path_or_key: str) -> str:
    """main.py serves data/clips at /clips/<name>. User notes have no clip, so their url is ""."""
    if not path_or_key or path_or_key.startswith(NOTE_PREFIX):
        return ""
    return f"/clips/{Path(path_or_key).name}"


# --------------------------------------------------------------------------- FAKE
_fake: list[dict] = []


def store_fake(clip_path: str, caption: str, objects: list[str], t: float,
               sides: dict[str, str] | None = None) -> None:
    _fake.append({"t": t, "caption": caption, "objects": [o.lower() for o in objects], "clip": clip_path,
                  "sides": {k.lower(): v for k, v in (sides or {}).items()}})


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


def note_caption(note: str) -> str:
    """'my pills are in the drawer' -> 'You told me your pills are in the drawer.' (read back to the user)"""
    swaps = {"my": "your", "mine": "yours", "i": "you", "i'm": "you're", "im": "you're", "me": "you", "myself": "yourself",
             "i am": "you are", "i was": "you were"}
    text = re.sub(r"\b(i am|i was|i'm|im|myself|mine|my|me|i)\b", lambda m: swaps[m.group(1).lower()],
                  note.strip().rstrip("."), flags=re.I)
    return f"You told me {text}."


def remember_fake(note: str, t: float | None = None) -> None:
    t = time.time() if t is None else t
    _fake.append({"t": t, "caption": note_caption(note), "objects": [],
                  "clip": f"{NOTE_PREFIX}{t:.3f}", "sides": {}})


def timeline_fake(since: float, until: float | None = None) -> list[MemoryHit]:
    until = time.time() if until is None else until
    return [{"t": m["t"], "caption": m["caption"], "clip_url": clip_url_for(m["clip"])}
            for m in sorted(_fake, key=lambda m: m["t"]) if since <= m["t"] <= until]


def recall_fake(question: str) -> list[MemoryHit]:
    window = parse_time_window(question)
    if not window:
        return search_fake(question)
    q = set(re.findall(r"[a-z0-9]+", question.lower()))
    inside = [m for m in _fake if window[0] <= m["t"] <= window[1]]
    inside.sort(key=lambda m: (-len(q & (set(re.findall(r"[a-z0-9]+", m["caption"].lower())) | set(m["objects"]))), -m["t"]))
    return [{"t": m["t"], "caption": m["caption"], "clip_url": clip_url_for(m["clip"])} for m in inside[:5]] \
        or search_fake(question)


def last_seen_fake(label: str) -> MemoryHit | None:
    label = label.lower().rstrip("s")
    for m in reversed(_fake):
        if any(label in o for o in m["objects"]) or label in m["caption"].lower():
            hit: MemoryHit = {"t": m["t"], "caption": m["caption"], "clip_url": clip_url_for(m["clip"])}
            side = next((s for o, s in m["sides"].items() if label in o), None)
            if side:
                hit["side"] = side
            return hit
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


KEYWORD_BOOST = 0.06
STOPWORDS = {"a", "an", "the", "i", "my", "me", "is", "are", "was", "were", "did", "do", "does", "where", "what",
             "who", "when", "how", "there", "anything", "something", "someone", "in", "on", "at", "to", "of", "it",
             "you", "your", "see", "saw", "last", "put", "leave", "left", "doing", "any", "have", "has"}
# Words a blind user would say -> words Cosmos captions and YOLO labels tend to use
SYNONYMS = {
    "way": {"walkway", "path", "blocking", "middle"}, "path": {"walkway", "way", "blocking"},
    "drink": {"cup", "mug", "bottle", "glass"}, "cup": {"mug"}, "mug": {"cup"},
    "bag": {"backpack", "handbag", "suitcase"}, "phone": {"smartphone", "cell"},
    "sofa": {"couch"}, "couch": {"sofa"}, "tv": {"television", "remote"},
    "came": {"enters", "walks", "opens", "door"}, "come": {"enters", "walks", "opens", "door"},
}


def _query_concepts(text: str) -> list[set[str]]:
    """One set per meaningful query word: the word plus its synonyms. Each set counts once."""
    words = {w.rstrip("s") if len(w) > 3 else w for w in re.findall(r"[a-z0-9]+", text.lower())} - STOPWORDS
    return [{w} | SYNONYMS.get(w, set()) for w in words]


def _clip_words(caption: str, objects_json: str) -> set[str]:
    text = caption + " " + " ".join(json.loads(objects_json))
    return {w.rstrip("s") if len(w) > 3 else w for w in re.findall(r"[a-z0-9]+", text.lower())}


VDB_SCHEMA = "second_sight"   # our own schema in the team's VastDB bucket, next to vss-schema
VDB_TABLE = "clips"
EMB_DIM = 256                 # Cosmos Embed1, same as VAST's own vss-collection vectors


def _vdb_columns():
    import pyarrow as pa
    return pa.schema([
        ("t", pa.float64()),
        ("caption", pa.utf8()),
        ("objects", pa.utf8()),     # JSON list of YOLO labels
        ("sides", pa.utf8()),       # JSON {label: left|centre|right}
        ("clip_key", pa.utf8()),    # S3 key of the clip
        ("vectors", pa.list_(pa.field("item", pa.float32(), nullable=False), EMB_DIM)),
    ])


class RealMemory:
    def __init__(self):
        db = config.ROOT / "data/memory.sqlite"
        db.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()   # store() runs on the thread pool, search() on requests
        self.con = sqlite3.connect(db, check_same_thread=False)
        self.con.execute("CREATE TABLE IF NOT EXISTS clips (t REAL, caption TEXT, objects TEXT, key TEXT, emb BLOB)")
        if "sides" not in [c[1] for c in self.con.execute("PRAGMA table_info(clips)")]:
            self.con.execute("ALTER TABLE clips ADD COLUMN sides TEXT")
        self.con.commit()
        self.s3 = None
        self.vdb = None
        if S3_ENDPOINT and S3_ACCESS:
            import boto3
            import urllib3
            urllib3.disable_warnings()   # the lab cluster uses a self-signed cert (VAST's skills do the same)
            self.s3 = boto3.client("s3", endpoint_url=S3_ENDPOINT, aws_access_key_id=S3_ACCESS,
                                   aws_secret_access_key=S3_SECRET, verify=False)
            print(f"[memory] VAST S3 on, bucket {S3_BUCKET}/{S3_PREFIX}")
            self._vdb_connect()
        else:
            print("[memory] no VAST credentials found, clips stay local only")

    # ---- VAST DB: the durable copy of every memory row (SQLite is the fast local mirror)
    def _vdb_connect(self) -> None:
        try:
            import logging

            import vastdb
            # The SDK logs a harmless "sorted columns ... without elysium" RPC error on every
            # table open on this cluster (VAST's own vss-collection does the same). Hide it.
            logging.getLogger("vastdb").setLevel(logging.CRITICAL)
            self.vdb = vastdb.connect(endpoint=S3_ENDPOINT, access=S3_ACCESS, secret=S3_SECRET, ssl_verify=False)
            with self.vdb.transaction() as tx:
                bucket = tx.bucket(S3_BUCKET)
                schema = bucket.schema(VDB_SCHEMA, fail_if_missing=False) or bucket.create_schema(VDB_SCHEMA, fail_if_exists=False)
                table = schema.table(VDB_TABLE, fail_if_missing=False)
                if table is None:
                    schema.create_table(VDB_TABLE, columns=_vdb_columns(), fail_if_exists=False)
            print(f"[memory] VAST DB on, table {S3_BUCKET}/{VDB_SCHEMA}/{VDB_TABLE}")
            if not self._rows("SELECT t FROM clips LIMIT 1"):
                self._restore_from_vdb()
        except Exception as e:
            print(f"[memory] VAST DB unavailable, SQLite only: {e}")
            self.vdb = None

    def _restore_from_vdb(self) -> None:
        """After a restart the local mirror is empty, so reload every memory from VAST DB."""
        with self.vdb.transaction() as tx:
            rows = tx.bucket(S3_BUCKET).schema(VDB_SCHEMA).table(VDB_TABLE).select().read_all().to_pylist()
        with self.lock:
            for r in rows:
                emb = np.array(r["vectors"], dtype=np.float32) if r["vectors"] else real_embed(r["caption"])
                self.con.execute("INSERT INTO clips (t, caption, objects, key, emb, sides) VALUES (?,?,?,?,?,?)",
                                 (r["t"], r["caption"], r["objects"], r["clip_key"], emb.tobytes(), r["sides"]))
            self.con.commit()
        print(f"[memory] restored {len(rows)} memories from VAST DB")

    def _vdb_insert(self, row: dict, emb: np.ndarray) -> None:
        import pyarrow as pa
        vec = emb.tolist() if emb.shape == (EMB_DIM,) else None   # local fallback vectors aren't stored
        try:
            with self.vdb.transaction() as tx:
                tx.bucket(S3_BUCKET).schema(VDB_SCHEMA).table(VDB_TABLE).insert(
                    pa.table({**{k: [v] for k, v in row.items()}, "vectors": [vec]}, schema=_vdb_columns()))
        except Exception as e:
            print(f"[memory] VAST DB insert failed (kept in SQLite): {e}")

    def store(self, clip_path: str, caption: str, objects: list[str], t: float,
              sides: dict[str, str] | None = None) -> None:
        """sides is optional ({label: left|centre|right} from B's detections) so last_seen can say where."""
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
        objs = json.dumps([o.lower() for o in objects])
        sides_json = json.dumps({k.lower(): v for k, v in (sides or {}).items()})
        with self.lock:
            self.con.execute("INSERT INTO clips (t, caption, objects, key, emb, sides) VALUES (?,?,?,?,?,?)",
                             (t, caption, objs, key, emb.tobytes(), sides_json))
            self.con.commit()
        if self.vdb:
            self._vdb_insert({"t": t, "caption": caption, "objects": objs, "sides": sides_json, "clip_key": key}, emb)

    def _url(self, key: str) -> str:
        """Browsers can't reach the VAST endpoint, so clips are always played through main.py's
        /clips/. If the local copy is gone, pull it back down from VAST first."""
        if key.startswith(NOTE_PREFIX):
            return ""
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
        """Hybrid search: Cosmos Embed1 similarity plus a small boost per exact word match
        (with everyday synonyms), so "at my desk" or "in my way" don't lose to vaguer clips."""
        q = real_embed(query, "query")
        # Only compare vectors from the same model (the local fallback has a different size).
        rows = [r for r in self._rows("SELECT t, caption, objects, key, emb FROM clips")
                if len(r[4]) == q.nbytes]
        if not rows:
            return []
        M = np.stack([np.frombuffer(r[4], dtype=np.float32) for r in rows])
        concepts = _query_concepts(query)
        boost = np.array([KEYWORD_BOOST * min(3, sum(bool(c & cw) for c in concepts))
                          for cw in (_clip_words(r[1], r[2]) for r in rows)])
        scores = M @ q + boost
        top = np.argsort(-scores)[:5]
        return [{"t": rows[i][0], "caption": rows[i][1], "clip_url": self._url(rows[i][3])} for i in top if scores[i] > 0]

    def last_seen(self, label: str) -> MemoryHit | None:
        """Newest clip with the label in YOLO's objects or the caption. Adds an optional "side"
        (left|centre|right) when B's detections recorded where the object was."""
        label = label.lower().rstrip("s")
        for t, cap, objs, key, sides in self._rows("SELECT t, caption, objects, key, sides FROM clips ORDER BY t DESC"):
            if any(label in o for o in json.loads(objs)) or label in cap.lower():
                hit: MemoryHit = {"t": t, "caption": cap, "clip_url": self._url(key)}
                side = next((s for o, s in json.loads(sides or "{}").items() if label in o), None)
                if side:
                    hit["side"] = side   # extra field; consumers that don't know it ignore it
                return hit
        return None

    # ---- memory layer 2: notes, time-aware recall, timeline (used by recap)
    def remember(self, note: str, t: float | None = None) -> None:
        """Save something the user told us ("my medication is in the top drawer") as a memory with
        no clip. It's found by search/last_seen/recall like any caption, and lands in VAST DB too."""
        t = time.time() if t is None else t
        caption = note_caption(note)
        key = f"{NOTE_PREFIX}{t:.3f}"
        emb = real_embed(caption, "passage")
        with self.lock:
            self.con.execute("INSERT INTO clips (t, caption, objects, key, emb, sides) VALUES (?,?,?,?,?,?)",
                             (t, caption, "[]", key, emb.tobytes(), "{}"))
            self.con.commit()
        if self.vdb:
            self._vdb_insert({"t": t, "caption": caption, "objects": "[]", "sides": "{}", "clip_key": key}, emb)

    def timeline(self, since: float, until: float | None = None) -> list[MemoryHit]:
        """Every memory between since and until (unix seconds), oldest first."""
        until = time.time() if until is None else until
        with self.lock:
            rows = self.con.execute("SELECT t, caption, key FROM clips WHERE t >= ? AND t <= ? ORDER BY t",
                                    (since, until)).fetchall()
        return [{"t": t, "caption": cap, "clip_url": self._url(key)} for t, cap, key in rows]

    def recall(self, question: str) -> list[MemoryHit]:
        """search(), but a time phrase in the question ("10 minutes ago", "in the last hour",
        "this morning") limits it to memories from then. No time phrase = plain search()."""
        window = parse_time_window(question)
        if not window:
            return self.search(question)
        q = real_embed(question, "query")
        with self.lock:
            rows = self.con.execute("SELECT t, caption, objects, key, emb FROM clips WHERE t >= ? AND t <= ?",
                                    window).fetchall()
        rows = [r for r in rows if len(r[4]) == q.nbytes]
        if not rows:
            return self.search(question)
        M = np.stack([np.frombuffer(r[4], dtype=np.float32) for r in rows])
        concepts = _query_concepts(question)
        boost = np.array([KEYWORD_BOOST * min(3, sum(bool(c & cw) for c in concepts))
                          for cw in (_clip_words(r[1], r[2]) for r in rows)])
        top = np.argsort(-(M @ q + boost))[:5]
        return [{"t": rows[i][0], "caption": rows[i][1], "clip_url": self._url(rows[i][3])} for i in top]


if config.USE_REAL_MEMORY:
    _real = RealMemory()
    store, search, last_seen = _real.store, _real.search, _real.last_seen
    remember, timeline, recall = _real.remember, _real.timeline, _real.recall
else:
    store, search, last_seen = store_fake, search_fake, last_seen_fake
    remember, timeline, recall = remember_fake, timeline_fake, recall_fake


def seed_fake_memory() -> None:
    """Three test clips so 'where are my keys?' works before any camera runs."""
    now = time.time()
    store("data/clips/seed_1.mp4", "A person sits at a desk with a laptop and a blue cup.", ["person", "laptop", "cup"], now - 120)
    store("data/clips/seed_2.mp4", "A set of keys is placed on the kitchen table next to a bowl.", ["person", "keys", "bowl"], now - 60)
    store("data/clips/seed_3.mp4", "The person walks toward the door; a backpack hangs on the chair.", ["person", "backpack", "chair"], now - 20)


# --------------------------------------------------------------------------- memory layer 2
# Extra functions on top of the contract (store/search/last_seen are unchanged). C's agent can use:
#   parse_remember(text) -> note | None   "remember that my pills are in the drawer" -> "my pills are in the drawer"
#   remember(note)                         save it; later search/last_seen/recall find it (clip_url "")
#   recall(question) -> list[MemoryHit]    search() that honours "10 minutes ago", "in the last hour", "this morning"
#   recap(minutes=10) -> str               "what happened while I was away?" in two or three spoken sentences
#   timeline(since, until=None)            raw memories in a time range, oldest first
_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10, "fifteen": 15,
        "twenty": 20, "thirty": 30, "few": 3, "couple": 2, "couple of": 2}
_UNIT = {"sec": 1, "second": 1, "min": 60, "minute": 60, "hour": 3600, "hr": 3600}
_N = r"(\d+|an?|one|two|three|four|five|ten|fifteen|twenty|thirty|few|couple(?: of)?)"
_U = r"(sec(?:ond)?|min(?:ute)?|hour|hr)s?"


def parse_time_window(text: str, now: float | None = None) -> tuple[float, float] | None:
    """Turn a time phrase into (since, until) unix seconds, or None if the text has no time phrase."""
    now = time.time() if now is None else now
    s = text.lower()
    lt = time.localtime(now)
    midnight = now - (lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec)
    m = re.search(rf"(?:in|over|during)?\s*the\s+(?:last|past)\s+(?:{_N}\s+)?{_U}", s) or \
        re.search(rf"\b(?:last|past)\s+(?:{_N}\s+)?{_U}", s)
    if m:
        n = _NUM.get(m.group(1), None) if m.group(1) and not m.group(1).isdigit() else int(m.group(1) or 1)
        return now - (n or 1) * _UNIT[m.group(2)], now
    m = re.search(rf"\b{_N}\s+{_U}\s+ago\b", s)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else _NUM.get(m.group(1), 1)
        centre = now - n * _UNIT[m.group(2)]
        half = max(60.0, 0.5 * n * _UNIT[m.group(2)])
        return centre - half, min(now, centre + half)
    if re.search(r"\b(just now|a moment ago|a second ago)\b", s):
        return now - 60, now
    if "this morning" in s:
        return midnight, min(now, midnight + 12 * 3600)
    if "this afternoon" in s:
        return midnight + 12 * 3600, min(now, midnight + 18 * 3600)
    if "this evening" in s or "tonight" in s:
        return midnight + 18 * 3600, now
    if re.search(r"\b(earlier today|today)\b", s):
        return midnight, now
    return None


def parse_remember(text: str) -> str | None:
    """'Remember that my keys are on the hook' -> 'my keys are on the hook'. None if it's not a note."""
    m = re.match(r"^\s*(?:hey,?\s+)?(?:please\s+|can you\s+|could you\s+)?(?:remember|note|don'?t forget)"
                 r"(?:\s+that)?[\s,:]+(.+?)[\s.!?]*$", text, re.I)
    return m.group(1).strip() if m and len(m.group(1).split()) >= 2 else None


def _ago(t: float, now: float) -> str:
    d = max(0, now - t)
    if d < 45:
        return "just now"
    if d < 90:
        return "about a minute ago"
    if d < 3600:
        return f"about {round(d / 60)} minutes ago"
    return f"about {round(d / 3600)} hour{'s' if d >= 5400 else ''} ago"


def _distinct_events(hits: list[MemoryHit], limit: int = 12) -> list[MemoryHit]:
    """Clips repeat every 5 s, so keep only captions that differ from the one before."""
    out: list[MemoryHit] = []
    for h in hits:
        key = re.sub(r"[^a-z ]", "", h["caption"].lower())
        if not out or key != re.sub(r"[^a-z ]", "", out[-1]["caption"].lower()):
            out.append(h)
    return out[-limit:]


def recap(minutes: float = 10, since: float | None = None) -> str:
    """'What happened while I was away?' Spoken summary of recent memories. Uses the W&B LLM when
    USE_REAL_LLM is on; otherwise (or if it fails) a plain list of the distinct moments."""
    now = time.time()
    since = now - minutes * 60 if since is None else since
    events = _distinct_events(timeline(since, now))
    if not events:
        mins = round((now - since) / 60)
        return "Nothing new in the last minute." if mins <= 1 else f"Nothing new in the last {mins} minutes."
    lines = [f"{_ago(e['t'], now)}: {e['caption']}" for e in events]
    if config.USE_REAL_LLM and config.WANDB_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(base_url=config.WANDB_BASE_URL, api_key=config.WANDB_API_KEY.strip(), timeout=8,
                            default_headers={"OpenAI-Project": config.WANDB_PROJECT} if config.WANDB_PROJECT else None)
            thinking = any(k in config.AGENT_MODEL.lower() for k in ("nemotron", "qwen"))
            r = client.chat.completions.create(
                model=config.AGENT_MODEL, temperature=0, max_tokens=150,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}} if thinking else None,
                messages=[{"role": "system", "content":
                           "You tell a blind person what their camera saw while they were away. Two or three short "
                           "spoken sentences, oldest to newest, keep the 'about N minutes ago' times, mention people "
                           "arriving or leaving and objects put down or left behind. Only use the notes given. No lists."},
                          {"role": "user", "content": "\n".join(lines)}])
            text = (r.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as e:
            print(f"[memory] recap LLM failed, using plain recap: {e}")
    return "Here's what I saw. " + " ".join(f"{e['caption'].rstrip('.')}, {_ago(e['t'], now)}." for e in events[-4:])
