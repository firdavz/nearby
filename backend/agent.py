"""BRAIN — agent_on_events / agent_ask.  Owner: C.

Default is silence. Urgent hazards never touch the LLM (fixed text, see contracts.fixed_alert).
The LLM only speaks for non-urgent, noteworthy changes (rate-limited) and to answer questions.

FAKE: rules + canned phrasing, instant.
REAL: memory_lookup (last_seen + search + live caption) runs in code, then a W&B Inference LLM
      words the answer from those facts only. Chatter decisions run off the frame loop. Traced in Weave.
"""
from __future__ import annotations

import json
import re
import threading
import time

from backend import config, memory, voice
from backend.contracts import AgentAnswer, AgentLine, Detection, fixed_alert, is_hazard

PERSONAS = {
    "calm": "You are a calm, concise guide for a person with sight loss. One short sentence. Plain words. Never repeat yourself.",
    "attenborough": "You narrate in the warm, wry style of a veteran natural-history broadcaster — but still one short sentence, and still useful.",
}
# Appended LAST to every system prompt so the voice survives the task rules. Each must still say where.
# No locations in these: the model copied example locations into real answers.
VOICE = {
    "calm": ("Voice: relaxed and natural, like a friend talking — contractions (they're, you've), "
             "everyday words, no formal phrasing."),
    "attenborough": ("Voice: a hushed, awe-struck nature-documentary narrator, treating everyday objects as rare "
                     "wildlife (words like 'behold', 'remarkable', 'in its natural habitat'). Vary your phrasing."),
}
VOICE_RULES = ("Whatever the voice: every place and time must come from the input, never from imagination; "
               "if something was never seen, say so; the rules about when to stay SILENT still apply.")


def _system(task: str) -> str:
    p = config.PERSONA if config.PERSONA in PERSONAS else "calm"
    return f"{PERSONAS[p]} {task}\n{VOICE[p]} {VOICE_RULES}"

_last_spoke = 0.0
_last_text = ""
_latest_caption = ""
_quiet_until = 0.0      # no chatter before this (set around a user question)
HAZARD_REPEAT_S = 8.0   # the same danger, still there, is repeated at most this often
HAZARD_GONE_S = 2.0     # if a danger vanishes for this long and comes back, warn again at once
HAZARD_MIN_GAP_S = 4.0  # never two warnings closer than this, whatever they are (2.5 s still felt like nagging)
ANSWER_QUIET_S = 8.0    # chatter holds off this long around a question, so the answer stands alone

# Priority among dangers seen at once: fast movers, then bikes, then people/animals, then obstacles;
# in your path (centre) before off to the side; the bigger (closer) box first.
_DANGER_CLASS = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "bicycle": 1, "person": 2, "dog": 2}
_hazards: dict[str, dict] = {}            # per object: when it was last seen as a hazard / last warned
_last_warning = 0.0
# Someone who has been right in front of the camera for a while (the presenter at a laptop) is not a
# new danger: leaning toward the camera made their box grow and fired "coming straight at you".
# A real approach comes from further away, so it has only just become near when it closes in.
NEAR_FRESH_S = 3.0      # only warn about something that became near within the last few seconds
_near: dict[str, dict] = {}               # per object: since when it has been near, and last seen near


# What deserves an urgent warning. Things that move can hit you from any side. Things that don't move
# only matter in your path, and only if you could walk into them: on the VM, "a laptop just to your
# left", "a chair just to your right" and a misdetected indoor "train" were all urgent warnings.
_MOVERS = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "dog"}
_OBSTACLES = {"chair", "couch", "bench", "dining table", "potted plant", "bed", "suitcase", "fire hydrant",
              "stop sign", "parking meter", "refrigerator", "toilet"}


def _worth_warning(d: Detection) -> bool:
    if d["label"] in _MOVERS:
        return True
    return d["label"] in _OBSTACLES and d["side"] == "centre"


def _key(d: Detection) -> str:
    return f"{d['label']}:{d['track_id']}" if d["track_id"] >= 0 else d["label"]


def _danger_rank(d: Detection) -> tuple:
    x1, y1, x2, y2 = d["box"]
    return (_DANGER_CLASS.get(d["label"], 3), 0 if d["side"] == "centre" else 1, -(x2 - x1) * (y2 - y1))


def _hazard_line(events: list[Detection]) -> AgentLine | None:
    """Fast path, no model. Picks the single most important danger and warns once per object,
    not every frame: continuous warnings drowned out everything else, including answers."""
    global _last_warning
    now = time.time()
    for d in events:                      # track how long each object has been near
        if d["distance"] == "near":
            st = _near.setdefault(_key(d), {"since": now, "last": now})
            if now - st["last"] > 1.0:    # it had moved away: a fresh arrival
                st["since"] = now
            st["last"] = now
    due = []
    for d in sorted((e for e in events if is_hazard(e) and _worth_warning(e)), key=_danger_rank):
        key = _key(d)
        h = _hazards.setdefault(key, {"warned": 0.0, "seen": 0.0, "eligible": False})
        returning = now - h["seen"] > HAZARD_GONE_S
        if returning:
            # Decide once, when it becomes a danger: a fresh arrival, or someone already sat right there?
            # Deciding on every frame dropped a second danger that had to wait out HAZARD_MIN_GAP_S.
            h["eligible"] = now - _near[key]["since"] <= NEAR_FRESH_S
        h["seen"] = now
        if h["eligible"] and (returning or now - h["warned"] > HAZARD_REPEAT_S):
            due.append((d, h))
    if not due or now - _last_warning < HAZARD_MIN_GAP_S:
        return None
    d, h = due[0]
    h["warned"] = _last_warning = now
    line = fixed_alert(d)
    line["text"] = voice.hazard_phrase(d["label"], d["side"])  # human wording, still no model
    _said(line["text"])
    return line


def set_latest_caption(c: str) -> None:
    global _latest_caption
    _latest_caption = c


def _hold_chatter(seconds: float) -> None:
    global _quiet_until
    _quiet_until = max(_quiet_until, time.time() + seconds)
    with _ready_lock:
        _ready.clear()        # anything queued before the question is stale now


# ---------------------------------------------------------------------------- live narration
# It used to say only "someone's coming at you". Two layers now, both below hazards in priority:
#   1. arrivals — instant, from YOLO, no model: "There's a chair in front of you." "Someone's on your left."
#      Said once when something comes into view, not again while it stays.
#   2. actions — from Cosmos's clip description, worded by the W&B model: "The man ahead is waving at you."
# Lines are voiced off the frame loop (ElevenLabs takes ~1-3 s) and handed over on a later frame.
ARRIVAL_STABLE_S = 0.6    # in view at least this long (ignores one-frame flickers)
ARRIVAL_GAP_S = 3.0       # at most one arrival line every few seconds
ARRIVAL_AGAIN_S = 20.0    # something that left and came back is announced again, but not sooner than this
READY_MAX_AGE_S = 7.0     # a voiced line older than this is stale, drop it
_SKIP_LABELS = {"tie", "train", "airplane", "boat", "kite", "frisbee", "surfboard", "skis", "snowboard"}  # indoor misdetections
_FRIENDLY = {"dining table": "table", "potted plant": "plant", "cell phone": "phone", "tv": "TV",
             "handbag": "bag", "wine glass": "glass", "sports ball": "ball"}
_COUNT = {2: "Two", 3: "Three"}
_present: dict[str, dict] = {}            # label -> since / last seen / last announced
_last_arrival = 0.0
_ready: list[dict] = []                   # voiced lines waiting for the next frame
_ready_lock = threading.Lock()


def _deliver(line: AgentLine) -> None:
    """Voice a line in the background, then hand it to the next frame."""
    def run():
        line["audio_url"] = voice.tts(line["text"], config.PERSONA)
        with _ready_lock:
            _ready.append({"line": line, "t": time.time()})
    threading.Thread(target=run, daemon=True).start()


def _take_ready() -> AgentLine | None:
    now = time.time()
    with _ready_lock:
        _ready[:] = [r for r in _ready if now - r["t"] < READY_MAX_AGE_S]
        if not _ready or now < _quiet_until:
            return None
        _ready.sort(key=lambda r: r["line"]["urgency"] != "warn")   # close things first
        line = _ready.pop(0)["line"]
    _said(line["text"])
    return line


def _where(d: Detection) -> str:
    side = d["side"]
    if d["distance"] == "near":
        return "right in front of you" if side == "centre" else f"close on your {side}"
    if d["distance"] == "far":
        return "further ahead" if side == "centre" else f"over on your {side}"
    return "ahead of you" if side == "centre" else f"on your {side}"


def _plural(noun: str) -> str:
    return noun + ("es" if noun.endswith(("s", "sh", "ch", "x")) else "s")


def _arrival_text(label: str, dets: list[Detection]) -> str:
    where = _where(max(dets, key=lambda e: (e["box"][2] - e["box"][0]) * (e["box"][3] - e["box"][1])))
    n = len(dets)
    if label == "person":
        return f"Someone's {where}." if n == 1 else f"{_COUNT.get(n, 'A few')} people {where}."
    noun = _FRIENDLY.get(label, label)
    if n == 1:
        return f"There's {'an' if noun[0] in 'aeiou' else 'a'} {noun} {where}."
    return f"{_COUNT.get(n, 'A few')} {_plural(noun)} {where}."


def _arrival(events: list[Detection]) -> None:
    """Announce what has just come into view: people first, then things in your path, nearest first."""
    global _last_arrival
    now = time.time()
    groups: dict[str, list[Detection]] = {}
    for d in events:
        if d["label"] not in _SKIP_LABELS:
            groups.setdefault(d["label"], []).append(d)
    for label, dets in groups.items():
        p = _present.setdefault(label, {"since": now, "last": now, "announced": 0.0, "n": 0, "n_since": now, "n_told": 0})
        if now - p["last"] > 1.5:             # it had gone: a new arrival
            p["since"], p["n_told"] = now, 0
        p["last"] = now
        if len(dets) != p["n"]:
            p["n"], p["n_since"] = len(dets), now
        if p["n"] < p["n_told"] and now - p["n_since"] >= ARRIVAL_STABLE_S:
            p["n_told"] = p["n"]              # one walked off: quietly lower the bar, so a newcomer counts
    if now - _last_arrival < ARRIVAL_GAP_S or now < _quiet_until:
        return
    found = []                                # (label, dets, kind)
    for label, dets in groups.items():
        p = _present[label]
        if p["announced"] < p["since"] and now - p["since"] >= ARRIVAL_STABLE_S and now - p["announced"] > ARRIVAL_AGAIN_S:
            found.append((label, dets, "arrived"))
        elif 0 < p["n_told"] < p["n"] and now - p["n_since"] >= ARRIVAL_STABLE_S:
            found.append((label, dets, "another"))   # e.g. a second person walks in while the first is still there
    if not found:
        return
    dist = {"near": 0, "mid": 1, "far": 2}
    label, dets, kind = min(found, key=lambda g: (0 if g[0] in _MOVERS else 1 if g[0] in _OBSTACLES else 2,
                                                  min(dist[d["distance"]] for d in g[1]),
                                                  0 if any(d["side"] == "centre" for d in g[1]) else 1))
    p = _present[label]
    added = p["n"] - p["n_told"]
    p["announced"], p["n_told"] = now, p["n"]
    _last_arrival = now
    if kind == "another":
        d = max(dets, key=lambda e: e["track_id"])   # the tracker numbers new objects upward: newest = highest id
        noun = _FRIENDLY.get(label, label)
        if added == 1:
            text = f"Someone else, {_where(d)}." if label == "person" else f"Another {noun}, {_where(d)}."
        else:
            more = "people" if label == "person" else _plural(noun)
            text = f"{_COUNT.get(added, 'A few')} more {more}, {_where(d)}."
    else:
        d = max(dets, key=lambda e: (e["box"][2] - e["box"][0]) * (e["box"][3] - e["box"][1]))
        text = _arrival_text(label, dets)
    _deliver({"text": text, "side": d["side"], "urgency": "warn" if d["distance"] == "near" else "info"})


def _live(events: list[Detection], caption: str, narrate: bool) -> AgentLine | None:
    """Every frame: hazards first (instant), then any voiced line that's ready; meanwhile queue new
    arrivals and, with the real model, narration of what's happening."""
    line = _hazard_line(events)
    if line:
        with _ready_lock:
            _ready.clear()                     # chatter queued before a warning is stale
        return line
    line = _take_ready()
    if line:
        return line
    _arrival(events)
    if narrate:
        _maybe_narrate(caption)
    return None


def _said(text: str) -> None:
    global _last_spoke, _last_text
    _last_spoke, _last_text = time.time(), text


_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty"}


def _words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")


def _fmt_t(t: float) -> str:
    """Plain-spoken 'how long ago': 40 s -> 'about a minute ago', 7 min -> 'seven minutes ago'."""
    ago = max(0, int(time.time() - t))
    if ago < 20:
        return "just now"
    if ago < 90:
        return "about a minute ago"
    mins = round(ago / 60)
    if mins < 60:
        return f"{_words(mins)} minutes ago"
    hours = round(mins / 60)
    return "about an hour ago" if hours == 1 else f"about {_words(min(hours, 59))} hours ago"


# --------------------------------------------------------------------------- FAKE
def agent_on_events_fake(events: list[Detection], caption: str) -> AgentLine | None:
    return _live(events, caption, narrate=False)


# A's newer memory features, used when present (getattr) so this file works before and after A's push:
#   recall(q) -> list[MemoryHit]   like search(), but understands times ("ten minutes ago")
#   parse_remember(q) -> note|None + remember(note)   "remember that I parked on level 3"
#   recap(minutes=15) -> str       "what did I miss?"
# "what happened to my cup?" is a question about the cup, not a recap, hence (?! to\b).
_RECAP_RE = re.compile(r"while i was (away|gone|out)|what('s| has)? happened(?! to\b)|what did i miss|recap", re.I)


def _recall(question: str):
    return (getattr(memory, "recall", None) or memory.search)(question)


def _memory_command(question: str) -> AgentAnswer | None:
    """'Remember ...' notes and 'what did I miss?' recaps, answered straight from A's memory."""
    parse, remember = getattr(memory, "parse_remember", None), getattr(memory, "remember", None)
    if parse and remember:
        note = parse(question)
        if note:
            remember(note)
            return {"text": "Got it, I'll remember that.", "clip_url": None}
    recap = getattr(memory, "recap", None)
    if recap and _RECAP_RE.search(question):
        return {"text": recap(minutes=15), "clip_url": None}
    return None


OBJECT_WORDS = ("keys", "phone", "wallet", "glasses", "cup", "bag", "backpack", "bottle", "laptop")
PLURAL_WORDS = ("keys", "glasses")
# A's memory adds an optional hit["side"]: where the object was in view when it was last seen.
_SIDE_WORDS = {"left": "on your left", "right": "on your right", "centre": "straight ahead of you"}


def _side_words(hit: dict) -> str | None:
    return _SIDE_WORDS.get(hit.get("side") or "")


def agent_ask_fake(question: str) -> AgentAnswer:
    q = question.lower()
    for word in OBJECT_WORDS:
        if word in q:
            hit = memory.last_seen(word)
            if hit:
                was = "were" if word in PLURAL_WORDS else "was"
                where = _side_words(hit)
                head = f"Your {word} {was} {where}, {_fmt_t(hit['t'])}." if where else f"I last saw your {word} {_fmt_t(hit['t'])}."
                cap = hit["caption"].rstrip(".")
                return {"text": f"{head} I saw: {cap[:1].lower() + cap[1:]}.", "clip_url": hit["clip_url"]}
            return {"text": f"I haven't seen your {word} yet.", "clip_url": None}
    if "see" in q or "now" in q or "around" in q:
        return {"text": _latest_caption or "I can't see anything yet.", "clip_url": None}
    hits = _recall(question)
    if hits:
        return {"text": f"{_fmt_t(hits[0]['t'])}: {hits[0]['caption']}", "clip_url": hits[0]["clip_url"]}
    return {"text": "I don't have anything about that.", "clip_url": None}


# --------------------------------------------------------------------------- REAL
# Retrieval happens in code, every time, and the model only words the answer from those facts.
# Letting the model choose tools made it skip the lookup and invent answers ("2 days ago",
# "a living room") — unacceptable for a blind user. One call instead of two is also faster.
_client = None
_weave = False


def wandb_project() -> str:
    """entity/project for W&B, or "" for the key's default. The .env.example placeholder
    ("team/second-sight") makes W&B reject the key with 401 and the agent silently falls back to
    rules, so a placeholder or malformed value counts as unset."""
    p = (config.WANDB_PROJECT or "").strip()
    return "" if (not p or p.startswith("team/") or p.count("/") != 1) else p
_llm_lock = threading.Lock()
_lookup = None   # memory_lookup, wrapped as a Weave op once Weave is up so lookups show in the trace


def _llm():
    with _llm_lock:
        return _llm_init()


def _llm_init():
    global _client, _weave, _lookup
    if _client is None:
        from openai import OpenAI
        project = wandb_project()
        _client = OpenAI(base_url=config.WANDB_BASE_URL, api_key=(config.WANDB_API_KEY or "none").strip(),
                         default_headers={"OpenAI-Project": project} if project else None)
        try:
            import weave
            weave.init(project or "second-sight"); _weave = True
            _lookup = weave.op(memory_lookup)
        except Exception:
            pass
    return _client


# Reasoning models (Nemotron, Qwen) otherwise spend the whole token budget thinking and return nothing.
_NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def _is_thinking_model() -> bool:
    return any(k in config.AGENT_MODEL.lower() for k in ("nemotron", "qwen"))


def _complete(system: str, user: str) -> str:
    r = _llm().chat.completions.create(model=config.AGENT_MODEL, temperature=0, max_tokens=120,  # 0: no invented details
                                       messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                                       extra_body=_NO_THINK if _is_thinking_model() else None)
    return (r.choices[0].message.content or "").strip()


def _is_now_question(q: str) -> bool:
    return any(w in q for w in ("see", "now", "around", "in front", "ahead"))


def memory_lookup(question: str) -> dict:
    """Everything the agent knows that bears on the question: last_seen for any object named
    (exact), else semantic search hits, plus the live caption. Returns facts + the clip to show."""
    q = question.lower()
    facts, clip = [], None
    asked = [w for w in OBJECT_WORDS if w in q]
    for word in asked:
        hit = memory.last_seen(word)
        if hit:
            where = _side_words(hit)
            facts.append(f"last_seen({word}): {hit['caption']}" + (f" [side: {where}]" if where else "")
                         + f" [time: {_fmt_t(hit['t'])}]")
            clip = clip or hit["clip_url"]
        else:
            facts.append(f"last_seen({word}): never seen")
    if not asked and not _is_now_question(q):  # a named object never seen must not borrow another clip
        for h in _recall(question)[:3]:
            facts.append(f"memory: {h['caption']} [time: {_fmt_t(h['t'])}]")
            clip = clip or h["clip_url"]
    if not asked or _is_now_question(q):   # for "where is my X" the live scene only confuses the model
        facts.append(f"camera right now: {_latest_caption or 'nothing yet'}")
    return {"facts": facts, "clip_url": clip}


def agent_on_events_real(events: list[Detection], caption: str) -> AgentLine | None:
    """Called on every frame, so it must return instantly; the model runs in a background thread."""
    return _live(events, caption, narrate=True)


_narrating = {"busy": False, "last": 0.0, "caption": ""}


def _maybe_narrate(caption: str) -> None:
    """On each new Cosmos description, ask the model whether something is happening worth saying."""
    now = time.time()
    if (not caption or caption == _narrating["caption"] or _narrating["busy"] or now < _quiet_until
            or now - _narrating["last"] < config.SPEAK_COOLDOWN_S):
        return
    previous, _narrating["caption"] = _narrating["caption"], caption
    _narrating.update(busy=True, last=now)

    def run():
        try:
            line = _narrate(previous, caption)
            if line:
                _deliver(line)
        finally:
            _narrating["busy"] = False
    threading.Thread(target=run, daemon=True).start()


def _narrate(previous: str, caption: str) -> AgentLine | None:
    sysmsg = _system(
        "You tell a blind person what is HAPPENING around them, from a description of the last few seconds.\n"
        "Say ONE short sentence when something happens they'd want to know: someone waves, points, gestures, "
        "talks to them, holds something out, sits down, stands up, walks toward or away; a door opens; or they're "
        "now somewhere new (another room, a corridor, outside, stairs).\n"
        "Reply with only the word SILENT when nothing new is happening, when the description only lists objects "
        "(those are announced separately), or when it repeats the last thing said.\n"
        "The camera is the person's eyes: 'at the camera' means 'at you'. Keep where it happens (on your left / "
        "on your right / ahead) whenever the description says. Facts from the description only, never 'centre'. "
        "Your reply is spoken aloud as-is.")
    user = f"Previous description: {previous or '(none)'}\nNew description: {caption}\nLast thing said: {_last_text!r}"
    try:
        text = _complete(sysmsg, user)
    except Exception as e:
        print("narration: LLM failed, staying silent:", e)
        return None
    if not text or text.upper().startswith("SILENT") or text == _last_text:
        return None
    side = "left" if "left" in text.lower() else "right" if "right" in text.lower() else None
    return {"text": text, "side": side, "urgency": "info"}


def agent_ask_real(question: str) -> AgentAnswer:
    _llm()  # make sure Weave (and so the traced lookup) is ready
    found = (_lookup or memory_lookup)(question)
    sysmsg = _system(
        "Answer the person's question using ONLY the facts below. Say where it was — the [side: ...] phrase "
        "if given, plus the place from the description — and how long ago using the [time: ...] phrase. "
        "Use those phrases exactly as written, without brackets or labels. If the facts don't answer it, "
        "say you haven't seen it. Never invent places, rooms, sides or times: if the description says 'on the "
        "shelf', say 'on the shelf', not 'on the shelf in the kitchen'. One or two short sentences, spoken aloud as-is.")
    user = f"Question: {question}\nFacts:\n" + "\n".join(f"- {f}" for f in found["facts"])
    try:
        text = _complete(sysmsg, user)
    except Exception as e:
        print("agent_ask_real: LLM failed, using rule-based answer:", e)
        return agent_ask_fake(question)
    if not text:
        return agent_ask_fake(question)
    return {"text": text, "clip_url": found["clip_url"]}


if config.USE_REAL_LLM:
    _on_events, _ask = agent_on_events_real, agent_ask_real
    threading.Thread(target=_llm, daemon=True).start()  # warm up client + Weave so the first question isn't slow
else:
    _on_events, _ask = agent_on_events_fake, agent_ask_fake


# ---- natural voice: an optional "audio_url" on every line (consumers that ignore it are unaffected)
def agent_on_events(events: list[Detection], caption: str) -> AgentLine | None:
    line = _on_events(events, caption)
    if line and "audio_url" not in line:
        line["audio_url"] = voice.cached_url(line["text"], config.PERSONA)  # never a network call on the frame loop
    return line


def agent_ask(question: str) -> AgentAnswer:
    """The user's question outranks chatter: it's held off while we answer and for a while after."""
    _hold_chatter(ANSWER_QUIET_S)
    ans = _memory_command(question) or _ask(question)
    ans["audio_url"] = voice.tts(ans["text"], config.PERSONA)
    _hold_chatter(ANSWER_QUIET_S)
    global _last_text
    _last_text = ans["text"]   # so chatter won't echo the answer
    return ans


voice.warm_hazards()
