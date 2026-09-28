"""Whole-catalog agentic eval for the class-grouping merge arms (class, random and anti-class groups).

Every request sees all 12 dispatchers / 168 native operations. Scoring is in
class-qualified native space (`<class>.<method>`), matching run_hierarchy.py.

Env:  SCHEMA_SPEC=benchmarks/synthetic/data/specs/class_semantic.json
      SCHEMA_CONTROL=hard  SYN_QUERIES SYN_MODEL SYN_WORKERS MAX_TURNS
      OUT_CSV OUT_JSONL
Run:  SCHEMA_SPEC=benchmarks/synthetic/data/specs/class_neutral.json \
        python -m eval.run_class_grouping [N]
"""
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from eval import common as ev             # client, MODEL, QPATH, clean
from toolschema.class_grouping import QualifyingEnv, compile_spec_file, openai_tools

EXTRA_BODY = json.loads(os.environ.get("SYN_EXTRA_BODY") or "null")
SPEC = os.environ.get("SCHEMA_SPEC")
assert SPEC, "set SCHEMA_SPEC to a class-grouping spec json"
control = os.environ.get("SCHEMA_CONTROL", "hard")
MAX_TURNS = int(os.environ.get("MAX_TURNS", "16"))
WORKERS = int(os.environ.get("SYN_WORKERS", "20"))
TEMP = float(os.environ.get("SYN_TEMP", "0.0"))
SEED_ENV = os.environ.get("SYN_SEED")
SEED = int(SEED_ENV) if SEED_ENV not in (None, "") else None
TOOLCHOICE = os.environ.get("SYN_TOOLCHOICE") or None
SERIALIZE_PARALLEL = os.environ.get("SYN_SERIALIZE_PARALLEL", "0") == "1"
OUT_CSV = os.environ.get("OUT_CSV")
OUT_JSONL = os.environ.get("OUT_JSONL")
NOTE = os.environ.get("SYN_NOTE") or None
TAG_SUFFIX = os.environ.get("SYN_TAG_SUFFIX", "")
N = next((int(a) for a in sys.argv[1:] if a.isdigit()), None)

COMP = compile_spec_file(SPEC)
TOOLS = openai_tools(COMP)
INSTRUCTION = (
    "Use the provided tools to fulfil the request. The catalog spans several "
    "device/service classes; make sure you act on the right one. Call tools "
    "until the request is fully handled, then stop. Use ONLY the provided "
    "functions and their declared arguments."
)
if NOTE:
    INSTRUCTION += "\n" + NOTE


def call_key(name, args):
    return (name, tuple(sorted(ev.clean(args).items())))


def run_episode(q):
    env = QualifyingEnv(COMP, control=control)
    messages = [{"role": "system", "content": q["system"] + "\n\n" + INSTRUCTION},
                {"role": "user", "content": q["query"]}]
    turns = 0
    err = None
    err_detail = None
    provider_models, system_fingerprints, finish_reasons = set(), set(), []
    argument_parse_errors = 0
    provider_tool_calls = 0
    called_names = []
    while turns < MAX_TURNS:
        turns += 1
        try:
            resp = ev.client.chat.completions.create(
                model=ev.MODEL, messages=messages, tools=TOOLS,
                temperature=TEMP, max_tokens=1024,
                **({"seed": SEED} if SEED is not None else {}),
                **({"extra_body": EXTRA_BODY} if EXTRA_BODY else {}),
                **({"tool_choice": TOOLCHOICE} if TOOLCHOICE and turns == 1 else {}))
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            err_detail = str(e)
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
        if not tcs:
            break
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
            if not isinstance(args, dict):
                args = {}
                argument_parse_errors += 1
            called_names.append(tc.function.name)
            result = env.call(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc_id, "content": result})
    flags = env.finish()

    golds = [(f"{q['domain']}.{g['name']}", g.get("arguments", {}))
             for g in q["gold_calls"]]
    gset = Counter(call_key(n, a) for n, a in golds)
    aset = Counter(call_key(n, a) for n, a in env.native_actions)
    exact = int(gset == aset)
    hit = sum(min(gset[k], aset[k]) for k in gset)
    wrong_cls = sum(c for (n, _a), c in aset.items()
                    if "." in n and n.split(".", 1)[0] != q["domain"])
    return {
        "id": q["id"], "domain": q["domain"], "type": q.get("type"),
        "exact": exact, "recall": hit / max(1, len(golds)),
        "precision": hit / max(1, sum(aset.values())),
        "n_gold": len(golds), "n_exec": sum(aset.values()),
        "n_calls": env.n_calls, "n_rejected": env.n_rejected,
        "wrong_class": wrong_cls, "turns": turns, "err": err,
        "err_detail": err_detail, "flags": [f[0] for f in flags],
        "provider_models": sorted(provider_models),
        "system_fingerprints": sorted(system_fingerprints),
        "finish_reasons": finish_reasons,
        "provider_tool_calls": provider_tool_calls,
        "argument_parse_errors": argument_parse_errors,
        "actions": [[n, a] for n, a in env.native_actions],
        "called_names": called_names,
        "first_name": called_names[0] if called_names else None,
    }


def main():
    rows = [json.loads(line) for line in open(ev.QPATH, encoding="utf-8") if line.strip()]
    if N:
        rows = rows[:N]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(run_episode, rows))

    jl = open(OUT_JSONL, "w", encoding="utf-8") if OUT_JSONL else None
    by_dom = defaultdict(list)
    for row in results:
        by_dom[row["domain"]].append(row)
        if jl:
            jl.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    if jl:
        jl.close()

    spec_name = os.path.basename(SPEC).replace(".json", "")
    tag = f"{spec_name}_{control}" + TAG_SUFFIX
    lines = ["model,tag,domain,n,exact,recall,precision,rejected_rate,"
             "wrongclass_rate,livelock_rate,mean_turns,temperature,seed"]

    def agg(rs, dom):
        n = len(rs)
        live = sum(1 for row in rs if row["turns"] >= MAX_TURNS) / n
        rej = sum(row["n_rejected"] for row in rs) / max(1, sum(row["n_calls"] for row in rs))
        wc = sum(1 for row in rs if row["wrong_class"]) / n
        lines.append(f"{ev.MODEL},{tag},{dom},{n},"
                     f"{sum(row['exact'] for row in rs) / n:.4f},"
                     f"{sum(row['recall'] for row in rs) / n:.4f},"
                     f"{sum(row['precision'] for row in rs) / n:.4f},"
                     f"{rej:.4f},{wc:.4f},{live:.4f},"
                     f"{sum(row['turns'] for row in rs) / n:.2f},"
                     f"{TEMP},{'' if SEED is None else SEED}")

    for dom in sorted(by_dom):
        agg(by_dom[dom], dom)
    agg(results, "ALL")
    out = "\n".join(lines)
    print(out)
    if OUT_CSV:
        with open(OUT_CSV, "w", encoding="utf-8") as handle:
            handle.write(out + "\n")
    errs = Counter(row["err"].split(":", 1)[0] for row in results if row["err"])
    if errs:
        print("REQUEST_ERRORS:", dict(errs), file=sys.stderr)
        examples = list(dict.fromkeys(row["err"] for row in results if row["err"]))[:3]
        print("REQUEST_ERROR_EXAMPLES:", examples, file=sys.stderr)


if __name__ == "__main__":
    main()
