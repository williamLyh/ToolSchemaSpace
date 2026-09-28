#!/usr/bin/env python3
"""Schema proxy: apply a schema-variant operator to ANY OpenAI-compatible agent harness.

    harness (AutomationBench, MCP-Atlas, ...)  --/v1/chat/completions-->  proxy  -->  upstream model
                                                <-- native tool calls ---          <-- variant tool calls

The harness keeps its native tools, executor, and scorer. Per request the proxy
  1. compiles the request's native `tools` into a variant catalog (operators.compile_spec,
     or the k-ladder for the enum split) and sends that to the model instead;
  2. decodes every tool call the model makes back into the native call
     (operators.SeqDecoder), so the harness only ever executes native actions;
  3. answers protocol-internal calls itself (transaction begin/set) and rejects
     off-schema calls with a recoverable error (hard control), re-querying the
     model inside the same harness turn until it emits native-decodable calls;
  4. keeps a per-conversation transcript of what the model actually saw and said
     in the variant vocabulary, and splices it back into later requests in place of
     the native tool calls the harness echoes, so the model's history stays in the
     variant form.

Operators (--op): native, merge (one dispatcher over all task tools), merge_app (one
dispatcher per app prefix), split (k-ladder finest enum split), nest (arguments into
one object), rename_opaque, rename_ns (app__tool), strip (descriptions), reorder
(parameter order), transaction (begin/set/commit for tools with >=2 parameters).

Run:
    python -m toolschema.proxy --op merge --upstream http://127.0.0.1:8000/v1 --port 8100
then point the harness at http://127.0.0.1:8100/v1.
"""
import argparse
import asyncio
import copy
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SCHEMA_K", "10")

from aiohttp import ClientSession, ClientTimeout, web  # noqa: E402

from .operators import compile_spec  # noqa: E402
from .schema_adapter import TOOL_INSTRUCTION, SchemaAdapter, _canonicalize, _to_openai  # noqa: E402

log = logging.getLogger("schema_proxy")
OPS = ("native", "merge", "merge_app", "split", "nest", "rename_opaque", "rename_ns",
       "strip", "reorder", "transaction")
PROTOCOL_NOTE = {
    "transaction": ("Some actions are transactional: call the begin_* function first, read "
                    "the returned txn_id, set each argument with the set_* functions (passing "
                    "that exact txn_id), then call commit_* to execute. Nothing happens until "
                    "commit."),
}


# ---------------------------------------------------------------------------
# variant compilation
# ---------------------------------------------------------------------------
def app_of(name):
    return name.split("_", 1)[0] if "_" in name else name


def spec_for(op, canon):
    names = [t["name"] for t in canon]
    if op == "native":
        return {"ops": []}
    if op == "merge":
        return {"ops": [{"op": "merge", "tools": names, "into": "execute"}]}
    if op == "merge_app":
        groups = OrderedDict()
        for n in names:
            groups.setdefault(app_of(n), []).append(n)
        return {"ops": [{"op": "merge", "tools": g, "into": f"{a}_execute"} for a, g in groups.items()]}
    if op == "nest":
        return {"ops": [{"op": "arg_lower", "tool": t["name"], "into": "options",
                         "params": list(t["parameters"]["properties"])}
                        for t in canon if len(t["parameters"]["properties"]) >= 2]}
    if op == "rename_opaque":
        return {"ops": [{"op": "rename", "tool": n,
                         "to": "fn_" + hashlib.sha1(n.encode()).hexdigest()[:6]} for n in names]}
    if op == "rename_ns":
        return {"ops": [{"op": "rename", "tool": n, "to": f"{app_of(n)}__{n}"} for n in names]}
    if op == "strip":
        return {"ops": [{"op": "strip_desc", "tool": n} for n in names]}
    if op == "reorder":
        return {"ops": [{"op": "shuffle_params", "tool": n, "seed": 13} for n in names]}
    if op == "transaction":
        return {"ops": [{"op": "curry", "tool": t["name"]} for t in canon
                        if len(t["parameters"]["properties"]) >= 2]}
    raise ValueError(op)


class Variant:
    """One compiled variant for one native catalog: tools to show + call map to decode."""

    def __init__(self, op, native_tools):
        canon = [_canonicalize(t) for t in native_tools]
        self.native_names = {t["name"] for t in canon}
        if op == "split":
            tools, cmap = SchemaAdapter(k=9, control="hard").transform(native_tools)
            self.tools, self.cmap, self.comp = tools, cmap, None
        else:
            comp = compile_spec(canon, spec_for(op, canon))
            self.comp = comp                       # oracle encoder for the equivalence gate
            self.tools = [_to_openai(t) for t in comp.tools]
            self.cmap = comp.call_map
        self.op = op


# ---------------------------------------------------------------------------
# per-conversation state
# ---------------------------------------------------------------------------
class Conversation:
    def __init__(self, variant):
        from .operators import SeqDecoder
        self.v = variant
        self.dec = SeqDecoder(variant.cmap)
        self.seg_by_call = {}      # first native tool_call id of a returned turn -> variant messages
        self.seg_by_text = {}      # hash of a returned text turn -> variant messages
        self.n_rejected = 0
        self.n_meta = 0
        self.issued = set()        # native tool_call ids handed to the harness


CONVS = OrderedDict()


def conv_key(messages, tools):
    first = [m for m in messages if m.get("role") in ("system", "user")][:2]
    blob = json.dumps(first, sort_keys=True, default=str) + json.dumps(sorted(
        (t.get("function", t)["name"] for t in tools)))
    return hashlib.sha1(blob.encode()).hexdigest()


def get_conv(op, messages, tools):
    k = conv_key(messages, tools)
    if k not in CONVS:
        CONVS[k] = Conversation(Variant(op, tools))
        while len(CONVS) > 4096:
            CONVS.popitem(last=False)
    CONVS.move_to_end(k)
    return CONVS[k]


def text_key(content):
    return hashlib.sha1(json.dumps(content, sort_keys=True, default=str).encode()).hexdigest()


def splice_history(conv, messages):
    """Replace the harness's native assistant turns with what the model really said."""
    out = []
    for m in messages:
        if m.get("role") == "assistant":
            calls = m.get("tool_calls") or []
            if calls and calls[0].get("id") in conv.seg_by_call:
                out.extend(copy.deepcopy(conv.seg_by_call[calls[0]["id"]]))
                continue
            if not calls and m.get("content") and text_key(m["content"]) in conv.seg_by_text:
                out.extend(copy.deepcopy(conv.seg_by_text[text_key(m["content"])]))
                continue
        out.append(m)
    return out


# ---------------------------------------------------------------------------
# decoding one model turn
# ---------------------------------------------------------------------------
def tool_msg(call_id, payload):
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(payload)}


def process_calls(conv, calls, control):
    """-> (native_calls [(id, name, args)], internal_results [tool messages])."""
    from .schema_adapter import classify_call
    natives, internal = [], []
    for c in calls:
        cid = c.get("id") or f"call_{uuid.uuid4().hex[:12]}"
        name = c["function"]["name"]
        try:
            args = json.loads(c["function"].get("arguments") or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
        except Exception as e:                     # malformed JSON: recoverable error
            conv.n_rejected += 1
            internal.append(tool_msg(cid, {"error": f"Could not parse arguments: {e}"}))
            continue
        action, nm, na = classify_call(conv.v.cmap, control, name, args)
        if action == "reject":
            conv.n_rejected += 1
            msg = (f"Unknown function `{name}`. Use only the provided tools."
                   if name not in conv.v.cmap else
                   f"Invalid call to `{name}`: arguments do not match its schema.")
            internal.append(tool_msg(cid, {"error": msg}))
            continue
        entry = conv.v.cmap.get(name, {})
        kind = entry.get("kind")
        if kind in ("curry_begin", "curry_set", "resolver"):
            n_flags = len(conv.dec.flags)
            conv.dec.step(name, args)
            conv.n_meta += 1
            new = conv.dec.flags[n_flags:]
            if kind == "curry_begin":
                internal.append(tool_msg(cid, {"txn_id": conv.dec.last_txn_id}))
            elif new:
                conv.n_rejected += 1
                internal.append(tool_msg(cid, {"error": f"{new[0][0]}: {new[0][1]}"}))
            else:
                internal.append(tool_msg(cid, {"status": "ok"}))
            continue
        if name not in conv.v.cmap:                # loose control: run the off-schema call as is
            natives.append((cid, name, args))
            continue
        n_flags = len(conv.dec.flags)
        ev = conv.dec.step(name, args)
        new = conv.dec.flags[n_flags:]
        if ev is None:                             # e.g. commit of an unknown transaction
            conv.n_rejected += 1
            detail = f"{new[0][0]}: {new[0][1]}" if new else "nothing to execute"
            internal.append(tool_msg(cid, {"error": detail}))
            continue
        natives.append((cid, ev[0], ev[1]))
    return natives, internal


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
async def upstream_chat(app, body):
    url = app["upstream"].rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if app["upstream_key"]:
        headers["Authorization"] = f"Bearer {app['upstream_key']}"
    for attempt in range(6):
        try:
            async with app["session"].post(url, json=body, headers=headers) as r:
                data = await r.json(content_type=None)
                if r.status >= 500 or r.status == 429:
                    raise RuntimeError(f"upstream {r.status}: {str(data)[:200]}")
                return r.status, data
        except Exception as e:                     # transient: retry with backoff
            if attempt == 5:
                return 502, {"error": {"message": f"proxy upstream failure: {e}"}}
            await asyncio.sleep(2 ** attempt)


def sse_from_completion(data):
    """Minimal SSE stream for harnesses that request stream=true."""
    ch = data["choices"][0]
    msg = ch["message"]
    delta = {"role": "assistant", "content": msg.get("content")}
    if msg.get("tool_calls"):
        delta["tool_calls"] = [dict(tc, index=i) for i, tc in enumerate(msg["tool_calls"])]
    base = {k: data.get(k) for k in ("id", "created", "model")}
    chunks = [dict(base, object="chat.completion.chunk",
                   choices=[{"index": 0, "delta": delta, "finish_reason": None}]),
              dict(base, object="chat.completion.chunk",
                   choices=[{"index": 0, "delta": {}, "finish_reason": ch.get("finish_reason")}],
                   usage=data.get("usage"))]
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


async def chat(request):
    app = request.app
    body = await request.json()
    if app["model"]:                               # harness model ids (e.g. "openai/<id>") -> upstream id
        body["model"] = app["model"]
    stream = bool(body.pop("stream", False))
    body.pop("stream_options", None)
    tools = body.get("tools") or []
    if not tools:
        status, data = await upstream_chat(app, body)
        return respond(data, status, stream)

    conv = get_conv(app["op"], body["messages"], tools)
    messages = splice_history(conv, body["messages"])
    if app["instruction"]:
        note = TOOL_INSTRUCTION + ("\n\n" + PROTOCOL_NOTE[app["op"]] if app["op"] in PROTOCOL_NOTE else "")
        if messages and messages[0].get("role") == "system":
            messages = [dict(messages[0], content=f"{messages[0].get('content') or ''}\n\n{note}")] + messages[1:]
        else:
            messages = [{"role": "system", "content": note}] + messages
    req = dict(body, tools=conv.v.tools, messages=messages)
    if app["temperature"] is not None:
        req["temperature"] = app["temperature"]
    req.pop("tool_choice", None) if body.get("tool_choice") not in (None, "auto", "none", "required") else None

    pending = []                                   # variant messages not yet surfaced to the harness
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    data = None
    for _ in range(app["max_inner"]):
        status, data = await upstream_chat(app, dict(req, messages=messages + pending))
        if status >= 400 or "choices" not in data:
            return respond(data, status, stream)
        for k in usage:
            usage[k] += (data.get("usage") or {}).get(k, 0) or 0
        msg = data["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        variant_msg = {"role": "assistant", "content": msg.get("content"),
                       **({"tool_calls": calls} if calls else {})}
        if not calls:                              # final text turn
            conv.seg_by_text[text_key(msg.get("content"))] = pending + [variant_msg]
            break
        natives, internal = process_calls(conv, calls, app["control"])
        if natives:
            seg = pending + [variant_msg] + internal
            conv.seg_by_call[natives[0][0]] = seg
            out = [{"id": cid, "type": "function",
                    "function": {"name": n, "arguments": json.dumps(a)}} for cid, n, a in natives]
            conv.issued.update(cid for cid, _, _ in natives)
            data["choices"][0]["message"] = {"role": "assistant", "content": msg.get("content"),
                                             "tool_calls": out}
            data["choices"][0]["finish_reason"] = "tool_calls"
            break
        pending += [variant_msg] + internal        # only meta/rejected calls: answer and re-query
    else:                                          # inner budget exhausted: end the turn as text
        data["choices"][0]["message"] = {"role": "assistant",
                                         "content": "I was unable to complete the tool calls."}
        data["choices"][0]["finish_reason"] = "stop"
    data["usage"] = usage
    data.setdefault("proxy", {})["variant"] = app["op"]
    return respond(data, 200, stream)


def respond(data, status, stream):
    if stream and status < 400 and "choices" in data:
        return web.Response(text=sse_from_completion(data), content_type="text/event-stream")
    return web.json_response(data, status=status)


async def models(request):
    app = request.app
    headers = {"Authorization": f"Bearer {app['upstream_key']}"} if app["upstream_key"] else {}
    async with app["session"].get(app["upstream"].rstrip("/") + "/models", headers=headers) as r:
        return web.json_response(await r.json(content_type=None), status=r.status)


async def stats(request):
    convs = list(CONVS.values())
    return web.json_response({"op": request.app["op"], "conversations": len(convs),
                              "rejected": sum(c.n_rejected for c in convs),
                              "meta_calls": sum(c.n_meta for c in convs)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", choices=OPS, required=True)
    ap.add_argument("--upstream", required=True, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1")
    ap.add_argument("--upstream-key-env", default="UPSTREAM_API_KEY")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--control", choices=("hard", "loose"), default="hard")
    ap.add_argument("--max-inner", type=int, default=16, help="model calls per harness turn (paper protocol: 16)")
    ap.add_argument("--model", default=None, help="force this upstream model id on every request")
    ap.add_argument("--temperature", type=float, default=None, help="override sampling temperature (0 = paper protocol)")
    ap.add_argument("--no-instruction", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    async def on_start(app):
        app["session"] = ClientSession(timeout=ClientTimeout(total=1800))

    async def on_stop(app):
        await app["session"].close()

    app = web.Application(client_max_size=64 * 1024 ** 2)
    app.update(op=a.op, upstream=a.upstream, upstream_key=os.environ.get(a.upstream_key_env, ""),
               control=a.control, max_inner=a.max_inner, instruction=not a.no_instruction,
               temperature=a.temperature, model=a.model)
    app.on_startup.append(on_start)
    app.on_cleanup.append(on_stop)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/v1/v1/chat/completions", chat)   # harnesses that append /v1 to a /v1 base
    app.router.add_get("/v1/models", models)
    app.router.add_get("/stats", stats)
    web.run_app(app, host="0.0.0.0", port=a.port, print=None)


if __name__ == "__main__":
    main()
