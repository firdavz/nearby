"""BRAIN — agent_on_events / agent_ask.  Owner: C.

Default is silence. Urgent hazards never touch the LLM (fixed text, see contracts.fixed_alert).
The LLM only speaks for non-urgent, noteworthy changes (rate-limited) and to answer questions.

FAKE: rules + canned phrasing, instant.
REAL: W&B Inference LLM with tools (search_memory, last_seen, describe_now), traced in Weave.
"""
from __future__ import annotations

import json
import time

from backend import config, memory
from backend.contracts import AgentAnswer, AgentLine, Detection, fixed_alert, is_hazard

PERSONAS = {
    "calm": "You are a calm, concise guide for a person with sight loss. One short sentence. Plain words. Never repeat yourself.",
    "attenborough": "You narrate in the warm, wry style of a veteran natural-history broadcaster — but still one short sentence, and still useful.",
}

_last_spoke = 0.0
_last_text = ""
_latest_caption = ""
_last_hazard = {"t": 0.0, "text": ""}
HAZARD_REPEAT_S = 2.0   # an urgent warning repeats at most this often while the danger persists


def _hazard_line(events: list[Detection]) -> AgentLine | None:
    """Fast path. Bypasses the chatter cooldown but won't machine-gun the same warning."""
    hazards = [e for e in events if is_hazard(e)]
    if not hazards:
        return None
    line = fixed_alert(hazards[0])
    now = time.time()
    if line["text"] == _last_hazard["text"] and now - _last_hazard["t"] < HAZARD_REPEAT_S:
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


def _fmt_t(t: float) -> str:
    ago = int(time.time() - t)
    return f"{ago} seconds ago" if ago < 90 else f"{ago // 60} minutes ago"


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


def agent_ask_fake(question: str) -> AgentAnswer:
    q = question.lower()
    for word in ("keys", "phone", "wallet", "glasses", "cup", "bag", "backpack", "bottle", "laptop"):
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
TOOLS = [
    {"type": "function", "function": {"name": "search_memory", "description": "Semantic search over everything the camera has seen.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "last_seen", "description": "Most recent clip in which an object label was seen.",
     "parameters": {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}}},
    {"type": "function", "function": {"name": "describe_now", "description": "What the camera sees right now.",
     "parameters": {"type": "object", "properties": {}}}},
]

_client = None
_weave = False


def _llm():
    global _client, _weave
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url=config.WANDB_BASE_URL, api_key=config.WANDB_API_KEY or "none",
                         default_headers={"OpenAI-Project": config.WANDB_PROJECT} if config.WANDB_PROJECT else None)
        try:
            import weave
            weave.init(config.WANDB_PROJECT or "second-sight"); _weave = True
        except Exception:
            pass
    return _client


def _call_tool(name: str, args: dict):
    if name == "search_memory":
        return memory.search(args["query"])
    if name == "last_seen":
        return memory.last_seen(args["label"])
    if name == "describe_now":
        return {"caption": _latest_caption}
    return {"error": "unknown tool"}


def _run(messages: list[dict], max_steps: int = 4) -> tuple[str, str | None]:
    client = _llm()
    clip = None
    for _ in range(max_steps):
        r = client.chat.completions.create(model=config.AGENT_MODEL, messages=messages, tools=TOOLS,
                                           tool_choice="auto", temperature=0.2, max_tokens=120)
        m = r.choices[0].message
        messages.append(m.model_dump(exclude_none=True))
        if not m.tool_calls:
            return (m.content or "").strip(), clip
        for tc in m.tool_calls:
            res = _call_tool(tc.function.name, json.loads(tc.function.arguments or "{}"))
            if isinstance(res, dict) and res.get("clip_url"):
                clip = res["clip_url"]
            elif isinstance(res, list) and res and res[0].get("clip_url"):
                clip = res[0]["clip_url"]
            messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": json.dumps(res, default=str)})
    return "I'm not sure.", clip


def agent_on_events_real(events: list[Detection], caption: str) -> AgentLine | None:
    line = _hazard_line(events)  # never wait for a model
    if line:
        return line
    if not _rate_ok():
        return None
    near = [e for e in events if e["distance"] != "far"]
    if not near and not caption:
        return None
    sysmsg = PERSONAS.get(config.PERSONA, PERSONAS["calm"]) + (
        " Decide if the person needs to hear anything. If nothing matters, reply exactly SILENT. "
        "Otherwise one short sentence, mentioning left/right/ahead when relevant.")
    user = f"Objects: {json.dumps([{k: e[k] for k in ('label', 'side', 'distance')} for e in near])}\nScene: {caption}\nLast thing said: {_last_text!r}"
    text, _ = _run([{"role": "system", "content": sysmsg}, {"role": "user", "content": user}], max_steps=1)
    if not text or text.upper().startswith("SILENT") or text == _last_text:
        return None
    _said(text)
    side = near[0]["side"] if near else None
    return {"text": text, "side": side, "urgency": "info"}


def agent_ask_real(question: str) -> AgentAnswer:
    sysmsg = PERSONAS.get(config.PERSONA, PERSONAS["calm"]) + (
        " Answer questions about what the camera has seen using the tools. Say how long ago in plain words. "
        "If you don't know, say so in one sentence.")
    text, clip = _run([{"role": "system", "content": sysmsg}, {"role": "user", "content": question}])
    return {"text": text, "clip_url": clip}


if config.USE_REAL_LLM:
    agent_on_events, agent_ask = agent_on_events_real, agent_ask_real
else:
    agent_on_events, agent_ask = agent_on_events_fake, agent_ask_fake
