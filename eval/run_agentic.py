"""Agentic eval on the synthetic env via SimEnv — the runtime for the
temporal/reference operators (curry, indirect, progressive disclosure) where
single-shot scoring is undefined, and for agentic calibration of the ladder.

Episode loop: model sees the variant tools; each tool call is executed by
SimEnv (resolvers answer, transactions open/commit, hard control rejects
schema-domain violations with an error the model may recover from); episode
ends when the model stops calling tools or MAX_TURNS is hit. Scoring is
state-based: exact multiset of EXECUTED native actions vs gold_calls.

Env:  SCHEMA_SPEC (operator spec json/path)  or  SCHEMA_VARIANT=0..9 ladder
      SCHEMA_DATASET=synthetic  SCHEMA_CONTROL=hard|loose  SCHEMA_K
      SYN_QUERIES, SYN_MODEL, SYN_WORKERS, SYN_TOOLCHOICE (optional),
      SYN_TEMP, SYN_SEED (optional), MAX_TURNS (default 16), OUT_CSV, OUT_JSONL
Run:  SCHEMA_SPEC=benchmarks/synthetic/data/specs/curry.json python -m eval.run_agentic [N]
"""
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from eval import common as ev             # client, MODEL, clean, NATIVE
from toolschema.schema_adapter import SchemaAdapter
from toolschema.simenv import SimEnv

adapter = SchemaAdapter.from_env()
assert adapter is not None, "set SCHEMA_SPEC or SCHEMA_VARIANT"
control = adapter.control
MAX_TURNS = int(os.environ.get("MAX_TURNS", "16"))
TEMP = float(os.environ.get("SYN_TEMP", "0.0"))
SEED_ENV = os.environ.get("SYN_SEED")
SEED = int(SEED_ENV) if SEED_ENV not in (None, "") else None
WORKERS = int(os.environ.get("SYN_WORKERS", "20"))
TOOLCHOICE = os.environ.get("SYN_TOOLCHOICE") or None
SERIALIZE_PARALLEL = os.environ.get("SYN_SERIALIZE_PARALLEL", "0") == "1"
EXTRA_BODY = json.loads(os.environ.get("SYN_EXTRA_BODY") or "null")  # e.g. chat_template_kwargs
NOTE = os.environ.get("SYN_NOTE") or None           # extra system instruction (ablations)
TAG_SUFFIX = os.environ.get("SYN_TAG_SUFFIX", "")   # e.g. "+hint"
OUT_CSV = os.environ.get("OUT_CSV")
OUT_JSONL = os.environ.get("OUT_JSONL")
N = next((int(a) for a in sys.argv[1:] if a.isdigit()), None)

# Compile every domain's variant up-front (thread-safe read-only afterwards).
VARIANT = {}                                        # domain -> (openai_tools, call_map)
for dom, nat in ev.NATIVE.items():
    VARIANT[dom] = adapter.transform(nat, group=dom)


def call_key(name, args):
    return (name, tuple(sorted(ev.clean(args).items())))


def run_episode(q):
    tools, cmap = VARIANT[q["domain"]]
    env = SimEnv(cmap, control=control)
    sys_txt = q["system"] + "\n\n" + adapter.INSTRUCTION + (("\n" + NOTE) if NOTE else "")
    messages = [{"role": "system", "content": sys_txt},
                {"role": "user", "content": q["query"]}]
    turns = 0
    err = None
    provider_models, system_fingerprints, finish_reasons = set(), set(), []
    argument_parse_errors = 0
    provider_tool_calls = 0
    usage = {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
    called_names = []
    while turns < MAX_TURNS:
        turns += 1
        try:
            resp = ev.create_with_retry(
                model=ev.MODEL, messages=messages, tools=tools,
                temperature=TEMP, max_tokens=ev.MAX_TOKENS,
                **({"seed": SEED} if SEED is not None else {}),
                **({"extra_body": EXTRA_BODY} if EXTRA_BODY else {}),
                **({"tool_choice": TOOLCHOICE} if TOOLCHOICE and turns == 1 else {}))
        except Exception as e:                      # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            break
        choice = resp.choices[0]
        msg = choice.message
        tcs = msg.tool_calls or []
        if getattr(resp, "model", None):
            provider_models.add(resp.model)
        if getattr(resp, "system_fingerprint", None):
            system_fingerprints.add(resp.system_fingerprint)
        finish_reasons.append(getattr(choice, "finish_reason", None))
        provider_tool_calls += len(tcs)
        u = getattr(resp, "usage", None)
        if u is not None:                              # token accounting (online APIs)
            usage["prompt"] += getattr(u, "prompt_tokens", 0) or 0
            usage["completion"] += getattr(u, "completion_tokens", 0) or 0
            usage["reasoning"] += getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
            usage["cached"] += getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        if not tcs:
            break
        # Some parsers (notably llama3_json in newer vLLM) omit tool-call ids.
        # OpenAI-compatible servers reject the next turn when tool_call_id is
        # empty, so assign stable per-turn ids before replaying the assistant
        # message and matching tool responses.
        tc_ids = [tc.id or f"call_{turns}_{i}" for i, tc in enumerate(tcs)]
        call_messages = [{"id": tc_id, "type": "function",
                          "function": {"name": tc.function.name,
                                       "arguments": tc.function.arguments}}
                         for tc_id, tc in zip(tc_ids, tcs)]
        if not SERIALIZE_PARALLEL:
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": call_messages})
        for i, (tc_id, tc) in enumerate(zip(tc_ids, tcs)):
            if SERIALIZE_PARALLEL:
                messages.append({"role": "assistant",
                                 "content": (msg.content or "") if i == 0 else "",
                                 "tool_calls": [call_messages[i]]})
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
                argument_parse_errors += 1
            if not isinstance(args, dict):        # model may emit a bare string/list
                args = {}
                argument_parse_errors += 1
            called_names.append(tc.function.name)
            result = env.call(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc_id, "content": result})
    flags = env.finish()

    golds = [(g["name"], g.get("arguments", {})) for g in q["gold_calls"]]
    gset = Counter(call_key(n, a) for n, a in golds)
    aset = Counter(call_key(n, a) for n, a in env.native_actions)
    exact = int(gset == aset)
    hit = sum(min(gset[k], aset[k]) for k in gset)
    return {
        "id": q["id"], "domain": q["domain"], "type": q.get("type"),
        "exact": exact, "recall": hit / max(1, len(golds)),
        "precision": hit / max(1, sum(aset.values())),
        "n_gold": len(golds), "n_exec": sum(aset.values()),
        "n_calls": env.n_calls, "n_rejected": env.n_rejected,
        "turns": turns, "flags": [f[0] for f in flags], "err": err,
        "provider_models": sorted(provider_models),
        "system_fingerprints": sorted(system_fingerprints),
        "finish_reasons": finish_reasons,
        "provider_tool_calls": provider_tool_calls,
        "usage": usage,
        "argument_parse_errors": argument_parse_errors,
        "actions": [[n, a] for n, a in env.native_actions],
        "called_names": called_names,
        "first_name": called_names[0] if called_names else None,
    }


def main():
    rows = [json.loads(l) for l in open(ev.QPATH, encoding="utf-8") if l.strip()]
    if N:
        rows = rows[:N]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(run_episode, rows))

    jl = open(OUT_JSONL, "w", encoding="utf-8") if OUT_JSONL else None
    by_dom = defaultdict(list)
    for r in results:
        by_dom[r["domain"]].append(r)
        if jl:
            jl.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    if jl:
        jl.close()

    spec_env = os.environ.get("SCHEMA_SPEC", "")
    spec_name = (os.path.basename(spec_env).replace(".json", "") or "spec") \
        if adapter.spec is not None else None
    tag = (spec_name if spec_name else f"v{adapter.k}") + f"_{control}" \
          + (f"+{TOOLCHOICE}" if TOOLCHOICE else "") + TAG_SUFFIX
    lines = ["model,tag,domain,n,exact,recall,precision,rejected_rate,"
             "livelock_rate,dangling_rate,mean_turns,temperature,seed"]

    def agg(rs, dom):
        n = len(rs)
        live = sum(1 for r in rs if r["turns"] >= MAX_TURNS) / n
        dang = sum(1 for r in rs if "dangling_txn" in r["flags"]) / n
        rej = sum(r["n_rejected"] for r in rs) / max(1, sum(r["n_calls"] for r in rs))
        lines.append(f"{ev.MODEL},{tag},{dom},{n},"
                     f"{sum(r['exact'] for r in rs)/n:.4f},"
                     f"{sum(r['recall'] for r in rs)/n:.4f},"
                     f"{sum(r['precision'] for r in rs)/n:.4f},"
                     f"{rej:.4f},{live:.4f},{dang:.4f},"
                     f"{sum(r['turns'] for r in rs)/n:.2f},"
                     f"{TEMP},{'' if SEED is None else SEED}")

    for dom in sorted(by_dom):
        agg(by_dom[dom], dom)
    agg(results, "ALL")
    out = "\n".join(lines)
    print(out)
    if OUT_CSV:
        with open(OUT_CSV, "w", encoding="utf-8") as f:
            f.write(out + "\n")
    errs = Counter(r["err"].split(":", 1)[0] for r in results if r["err"])
    if errs:
        print("REQUEST_ERRORS:", dict(errs), file=sys.stderr)
        examples = list(dict.fromkeys(r["err"] for r in results if r["err"]))[:3]
        print("REQUEST_ERROR_EXAMPLES:", examples, file=sys.stderr)


if __name__ == "__main__":
    main()
