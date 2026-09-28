"""Shared helpers for the evaluation runners: model client, native catalog, argument normalisation.

Environment:
    REMOTE_OPENAI_BASE_URL / REMOTE_OPENAI_API_KEY   model endpoint (default: local vLLM)
    SYN_MODEL        model id sent to the endpoint
    SYN_QUERIES      query file (default: benchmarks/synthetic/data/queries.jsonl)
    SYN_API          "chat" (default) or "responses" (OpenAI Responses API)
    SYN_REASONING_EFFORT, SYN_MAX_TOKENS, SYN_TIMEOUT, SYN_RETRIES
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

from openai import OpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root


MODEL = os.environ.get("SYN_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
QPATH = os.environ.get("SYN_QUERIES", os.path.join(ROOT, "benchmarks", "synthetic", "data", "queries.jsonl"))
TOOLS = os.path.join(ROOT, "benchmarks", "synthetic", "data", "tools.json")
client = OpenAI(base_url=os.environ.get("REMOTE_OPENAI_BASE_URL", "http://localhost:8000/v1"),
                api_key=os.environ.get("REMOTE_OPENAI_API_KEY", "EMPTY"),
                **({"timeout": float(os.environ["SYN_TIMEOUT"])} if os.environ.get("SYN_TIMEOUT") else {}))
RETRIES = int(os.environ.get("SYN_RETRIES", "0"))   # transient-error retries (online APIs)
MAX_TOKENS = int(os.environ.get("SYN_MAX_TOKENS", "1024"))   # completion cap (thinking counts toward it)


API = os.environ.get("SYN_API", "chat")              # "responses" -> OpenAI Responses API (reasoning + tools)
REASONING_EFFORT = os.environ.get("SYN_REASONING_EFFORT")   # e.g. "medium"; Responses API only


def _responses_call(**kw):
    """Translate a chat.completions request to /v1/responses and the reply back.

    Reasoning models reject function tools with reasoning in chat.completions, so
    the loop keeps its chat-format history and this adapter converts each turn.
    Reasoning items are not carried across turns (stateless, like the chat path).
    """
    from types import SimpleNamespace as NS
    inp = []
    for m in kw["messages"]:
        r = m["role"]
        if r in ("system", "user"):
            inp.append({"role": r, "content": m["content"]})
        elif r == "assistant":
            if m.get("content"):
                inp.append({"role": "assistant", "content": m["content"]})
            for tc in m.get("tool_calls") or []:
                inp.append({"type": "function_call", "call_id": tc["id"],
                            "name": tc["function"]["name"], "arguments": tc["function"]["arguments"] or "{}"})
        elif r == "tool":
            inp.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": m["content"]})
    tools = [{"type": "function", "name": t["function"]["name"],
              "description": t["function"].get("description", ""),
              "parameters": t["function"].get("parameters", {"type": "object", "properties": {}}),
              "strict": False} for t in kw.get("tools") or []]
    req = {"model": kw["model"], "input": inp, "tools": tools,
           "max_output_tokens": kw.get("max_tokens", MAX_TOKENS)}
    if REASONING_EFFORT:
        req["reasoning"] = {"effort": REASONING_EFFORT}
    if kw.get("tool_choice"):
        req["tool_choice"] = kw["tool_choice"]
    r = client.responses.create(**req)
    text, calls = [], []
    for o in r.output or []:
        if o.type == "message":
            text += [c.text for c in (o.content or []) if getattr(c, "type", "") == "output_text"]
        elif o.type == "function_call":
            calls.append(NS(id=o.call_id, type="function", function=NS(name=o.name, arguments=o.arguments)))
    inc = getattr(getattr(r, "incomplete_details", None), "reason", None)
    finish = "tool_calls" if calls else ("length" if inc == "max_output_tokens" else "stop")
    u = r.usage
    usage = NS(prompt_tokens=u.input_tokens, completion_tokens=u.output_tokens,
               completion_tokens_details=NS(reasoning_tokens=getattr(u.output_tokens_details, "reasoning_tokens", 0)),
               prompt_tokens_details=NS(cached_tokens=getattr(u.input_tokens_details, "cached_tokens", 0)))
    msg = NS(content="".join(text) or None, tool_calls=calls or None)
    return NS(choices=[NS(message=msg, finish_reason=finish)], model=r.model, system_fingerprint=None, usage=usage)


def create_with_retry(**kw):
    """chat.completions.create with backoff on transient errors (rate limit, timeout,
    connection, 5xx). Non-transient errors (e.g. 400) are raised immediately.
    SYN_RETRIES=0 (default) keeps the original single-attempt behaviour."""
    import time
    import openai
    transient = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
                 openai.InternalServerError)
    for attempt in range(RETRIES + 1):
        try:
            return _responses_call(**kw) if API == "responses" else client.chat.completions.create(**kw)
        except transient as e:
            import sys as _s
            print(f"RETRY {type(e).__name__} attempt={attempt + 1}", file=_s.stderr, flush=True)
            if attempt == RETRIES:
                raise
            time.sleep(min(60, 2 ** attempt * 2))


def native_openai_by_domain():
    tools = json.load(open(TOOLS, encoding="utf-8"))
    by = defaultdict(list)
    for t in tools:
        by[t["group"]].append({"type": "function", "function": {
            "name": t["name"], "description": t["description"],
            "parameters": t["parameters"]}})
    return by

NATIVE = native_openai_by_domain()


def norm(v):
    """Normalize an argument value for comparison (numeric-aware, case-insensitive)."""
    if isinstance(v, bool):
        return ("b", v)
    if isinstance(v, (int, float)):
        return ("n", float(v))
    s = str(v).strip()
    if re.fullmatch(r"-?\d+", s):
        return ("n", float(s))
    try:
        return ("n", float(s))
    except ValueError:
        return ("s", s.lower())


def clean(args):
    return {k: norm(v) for k, v in (args or {}).items() if v not in (None, "")}


def match(gold_name, gold_args, name, args):
    return name == gold_name and clean(args) == clean(gold_args)


