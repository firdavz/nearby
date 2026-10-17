"""BRAIN — agent_on_events / agent_ask.  Owner: C.

Default is silence. Urgent hazards never touch the LLM (fixed text, see contracts.fixed_alert).
The LLM only speaks for non-urgent, noteworthy changes (rate-limited) and to answer questions.

FAKE: rules + canned phrasing, instant.
REAL: memory_lookup (last_seen + search + live caption) runs in code, then a W&B Inference LLM
      words the answer from those facts only. Chatter decisions run off the frame loop. Traced in Weave.
"""
from __future__ import annotations

import json
import threading
import time

from backend import config, memory
from backend.contracts import AgentAnswer, AgentLine, Detection, fixed_alert, is_hazard

PERSONAS = {
    "calm": "You are a calm, concise guide for a person with sight loss. One short sentence. Plain words. Never repeat yourself.",
    "attenborough": "You narrate in the warm, wry style of a veteran natural-history broadcaster — but still one short sentence, and still useful.",
}
# Appended LAST to every system prompt so the voice survives the task rules. Each must still say where.
# No locations in these: the model copied example locations into real answers.
VOICE = {
    "calm": "Voice: calm and plain, like a trusted friend.",
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
_last_hazard = {"t": 0.0, "text": "", "seen": 0.0}
HAZARD_REPEAT_S = 8.0   # the same danger, still there, is repeated at most this often
HAZARD_GONE_S = 2.0     # if a danger vanishes for this long and comes back, warn again at once


def _hazard_line(events: list[Detection]) -> AgentLine | None:
    """Fast path. Bypasses the chatter cooldown but warns once per danger, not every frame:
    a continuous warning drowned out everything else, including answers to questions."""
    hazards = [e for e in events if is_hazard(e)]
    if not hazards:
        return None
    line = fixed_alert(hazards[0])
    now = time.time()
    same = line["text"] == _last_hazard["text"]
    continuous = now - _last_hazard["seen"] < HAZARD_GONE_S
    _last_hazard["seen"] = now
    if same and continuous and now - _last_hazard["t"] < HAZARD_REPEAT_S:
        return None
    _last_hazard.update(t=now, text=line["text"])
    _said(line["text"])
    return line


def set_latest_caption(c: str) -> None:
    global _latest_caption
    _latest_caption = c


def _rate_ok() -> bool:
    return time.time() - _last_spoke > config.SPEAK_COOLDOWN_S


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
    near = [e for e in events if e["distance"] == "near"]
    if near:
        e = near[0]
        text = f"{e['label']} close on your {e['side']}." if e["side"] != "centre" else f"{e['label']} directly ahead."
        if text != _last_text:
            _said(text); return {"text": text, "side": e["side"], "urgency": "warn"}
    return None


OBJECT_WORDS = ("keys", "phone", "wallet", "glasses", "cup", "bag", "backpack", "bottle", "laptop")


def agent_ask_fake(question: str) -> AgentAnswer:
    q = question.lower()
    for word in OBJECT_WORDS:
        if word in q:
            hit = memory.last_seen(word)
            if hit:
                return {"text": f"I last saw your {word} {_fmt_t(hit['t'])}: {hit['caption']}", "clip_url": hit["clip_url"]}
            return {"text": f"I haven't seen your {word} yet.", "clip_url": None}
    if "see" in q or "now" in q or "around" in q:
        return {"text": _latest_caption or "I can't see anything yet.", "clip_url": None}
    hits = memory.search(question)
    if hits:
        return {"text": f"{_fmt_t(hits[0]['t'])}: {hits[0]['caption']}", "clip_url": hits[0]["clip_url"]}
    return {"text": "I don't have anything about that.", "clip_url": None}


# --------------------------------------------------------------------------- REAL
# Retrieval happens in code, every time, and the model only words the answer from those facts.
# Letting the model choose tools made it skip the lookup and invent answers ("2 days ago",
# "a living room") — unacceptable for a blind user. One call instead of two is also faster.
_client = None
_weave = False
_llm_lock = threading.Lock()
_lookup = None   # memory_lookup, wrapped as a Weave op once Weave is up so lookups show in the trace


def _llm():
    with _llm_lock:
        return _llm_init()


def _llm_init():
    global _client, _weave, _lookup
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url=config.WANDB_BASE_URL, api_key=config.WANDB_API_KEY or "none",
                         default_headers={"OpenAI-Project": config.WANDB_PROJECT} if config.WANDB_PROJECT else None)
        try:
            import weave
            weave.init(config.WANDB_PROJECT or "second-sight"); _weave = True
            _lookup = weave.op(memory_lookup)
        except Exception:
            pass
    return _client


# Reasoning models (Nemotron, Qwen) otherwise spend the whole token budget thinking and return nothing.
_NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def _is_thinking_model() -> bool:
    return any(k in config.AGENT_MODEL.lower() for k in ("nemotron", "qwen"))


def _complete(system: str, user: str) -> str:
    r = _llm().chat.completions.create(model=config.AGENT_MODEL, temperature=0.2, max_tokens=120,
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
            facts.append(f"last_seen({word}): {hit['caption']} ({_fmt_t(hit['t'])})")
            clip = clip or hit["clip_url"]
        else:
            facts.append(f"last_seen({word}): never seen")
    if not asked and not _is_now_question(q):  # a named object never seen must not borrow another clip
        for h in memory.search(question)[:3]:
            facts.append(f"memory: {h['caption']} ({_fmt_t(h['t'])})")
            clip = clip or h["clip_url"]
    facts.append(f"camera right now: {_latest_caption or 'nothing yet'}")
    return {"facts": facts, "clip_url": clip}


_pending: dict = {"line": None, "t": 0.0}
_thinking = {"busy": False, "last": 0.0}
PENDING_MAX_AGE_S = 4.0   # a chatter line older than this is stale, drop it


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
            _pending.update(line=line, t=time.time())
    finally:
        _thinking["busy"] = False


def _decide(events: list[Detection], caption: str) -> AgentLine | None:
    near = [e for e in events if e["distance"] != "far"]
    if not near and not caption:
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
    return {"text": text, "side": side, "urgency": "info"}


def agent_ask_real(question: str) -> AgentAnswer:
    _llm()  # make sure Weave (and so the traced lookup) is ready
    found = (_lookup or memory_lookup)(question)
    sysmsg = _system(
        "Answer the person's question using ONLY the facts below. Say where, and how long ago using the "
        "time phrase in brackets exactly as written (without the brackets). If the facts don't answer it, "
        "say you haven't seen it. Never invent places or times. One or two short sentences, spoken aloud as-is.")
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
    agent_on_events, agent_ask = agent_on_events_real, agent_ask_real
    threading.Thread(target=_llm, daemon=True).start()  # warm up client + Weave so the first question isn't slow
else:
    agent_on_events, agent_ask = agent_on_events_fake, agent_ask_fake
