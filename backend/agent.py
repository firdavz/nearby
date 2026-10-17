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
HAZARD_MIN_GAP_S = 2.5  # never two warnings closer than this, whatever they are
ANSWER_QUIET_S = 8.0    # chatter holds off this long around a question, so the answer stands alone
NOVELTY_S = 45.0        # chatter won't mention the same kind of object again within this window

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
    for d in sorted((e for e in events if is_hazard(e)), key=_danger_rank):
        key = _key(d)
        if now - _near[key]["since"] > NEAR_FRESH_S:
            continue                      # been right there for a while: not a new danger
        h = _hazards.setdefault(key, {"warned": 0.0, "seen": 0.0})
        returning = now - h["seen"] > HAZARD_GONE_S
        h["seen"] = now
        if returning or now - h["warned"] > HAZARD_REPEAT_S:
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


def _rate_ok() -> bool:
    """May the agent chatter now? Not during the cooldown, and not around a user question."""
    now = time.time()
    return now - _last_spoke > config.SPEAK_COOLDOWN_S and now >= _quiet_until


def _hold_chatter(seconds: float) -> None:
    global _quiet_until
    _quiet_until = max(_quiet_until, time.time() + seconds)
    _pending["line"] = None   # anything queued before the question is stale now


_announced: dict[str, float] = {}   # object label -> when chatter last mentioned it


def _is_new(label: str) -> bool:
    return time.time() - _announced.get(label, 0.0) > NOVELTY_S


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
    line = _hazard_line(events)
    if line:
        return line
    if not _rate_ok():
        return None
    near = [e for e in events if e["distance"] == "near" and _is_new(e["label"])]
    if near:
        e = near[0]
        text = f"{e['label']} close on your {e['side']}." if e["side"] != "centre" else f"{e['label']} directly ahead."
        if text != _last_text:
            _said(text); _announced[e["label"]] = time.time()
            return {"text": text, "side": e["side"], "urgency": "warn"}
    return None


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


_pending: dict = {"line": None, "t": 0.0}
_thinking = {"busy": False, "last": 0.0}
PENDING_MAX_AGE_S = 7.0   # a chatter line older than this is stale, drop it (includes ~3 s voice render)


def agent_on_events_real(events: list[Detection], caption: str) -> AgentLine | None:
    """Called on every frame, so it must return instantly. The LLM runs in a background
    thread; its line (if any) is handed back on a later frame."""
    line = _hazard_line(events)  # never wait for a model
    if line:
        _pending["line"] = None
        return line
    if _pending["line"]:
        line, _pending["line"] = _pending["line"], None
        if time.time() - _pending["t"] < PENDING_MAX_AGE_S and _rate_ok():
            _said(line["text"])
            return line
        return None
    now = time.time()
    if _thinking["busy"] or not _rate_ok() or now - _thinking["last"] < config.SPEAK_COOLDOWN_S:
        return None
    _thinking.update(busy=True, last=now)
    threading.Thread(target=_think, args=(list(events), caption), daemon=True).start()
    return None


def _think(events: list[Detection], caption: str) -> None:
    try:
        line = _decide(events, caption)
        if line:
            line["audio_url"] = voice.tts(line["text"], config.PERSONA)  # render here, off the frame loop
            _pending.update(line=line, t=time.time())
    finally:
        _thinking["busy"] = False


_last_decided_caption = ""


def _decide(events: list[Detection], caption: str) -> AgentLine | None:
    global _last_decided_caption
    # Only things not mentioned recently; and if nothing is new and the scene hasn't changed,
    # there is nothing to say — skip the model call entirely.
    near = [e for e in events if e["distance"] != "far" and _is_new(e["label"])]
    scene_changed = bool(caption) and caption != _last_decided_caption
    _last_decided_caption = caption
    if not near and not scene_changed:
        return None
    sysmsg = _system(
        "Decide if the person needs to hear anything right now.\n"
        "Say one short sentence, with left/right/ahead, when: an object is near them that they may want "
        "or bump into (a cup, keys, phone, bag, chair in their path); a person comes close; or the scene "
        "changes (new room, door, stairs, step, crossing).\n"
        "Reply with only the word SILENT when: things are far away, the scene is unchanged, or it repeats "
        "what was last said.\n"
        "Your reply is spoken aloud as-is: no labels, no quotes, no explanation.")
    user = f"Objects: {json.dumps([{k: e[k] for k in ('label', 'side', 'distance')} for e in near])}\nScene: {caption}\nLast thing said: {_last_text!r}"
    try:
        text = _complete(sysmsg, user)
    except Exception as e:
        print("agent_on_events_real: LLM failed, staying silent:", e)
        return None
    if not text or text.upper().startswith("SILENT") or text == _last_text:
        return None
    side = near[0]["side"] if near else None
    for e in near:
        _announced[e["label"]] = time.time()
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
