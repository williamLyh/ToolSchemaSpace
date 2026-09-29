"""Equivalence gate for the schema proxy on a real tool catalog.

For every task tool set and every proxy operator: build sample native calls per tool, encode each into the
variant's call sequence (oracle), feed that sequence through the proxy's decoder exactly as a model's tool
calls would arrive, and require the decoded native call to equal the original with no rejection. Each tool is
checked with its required arguments only and with up to three optional ones, and numeric values are sampled
on both sides of any interval cut.

    python -m tests.test_proxy                                     # the synthetic catalog, one task per domain
    python -m tests.test_proxy catalog.json task_tools.json [--classes classes.json] [--cuts cuts.json]
catalog.json: list of OpenAI tools; task_tools.json: list of per-task tool-name lists.
"""
import argparse
import json
import os
import sys

import toolschema.proxy as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM = (7, 2.5)
REQUIRED_ONLY = False


def sample_value(pd):
    pd = dict(pd)
    for key in ("anyOf", "oneOf"):
        if key in pd:
            alts = [a for a in pd[key] if a.get("type") != "null"]
            if alts:
                pd = alts[0]
    if "enum" in pd:
        return pd["enum"][-1]
    t = pd.get("type")
    return {"integer": NUM[0], "number": NUM[1], "boolean": True, "array": ["x"],
            "object": {"k": "v"}}.get(t, "sample text")


def sample_call(tool):
    fn = P._canonicalize(tool); props = fn["parameters"].get("properties", {})   # $refs resolved
    req = fn["parameters"].get("required", []) or list(props)[:2]
    names = list(req) if REQUIRED_ONLY else list(dict.fromkeys(list(req) + list(props)[:3]))
    return {"name": fn["name"], "arguments": {k: sample_value(props[k]) for k in names if k in props}}


def check(op, tools):
    v = P.Variant(op, tools)
    bad = []
    for t in tools:
        gold = sample_call(t)
        conv = P.Conversation(v)
        if v.hier is not None:
            seq = v.hier.encode_episode(P.class_of(gold["name"]), [gold])
        else:
            seq = v.comp.encode_call(gold["name"], gold["arguments"])
        got = []
        for i, (n, a) in enumerate(seq):
            calls = [{"id": f"c{i}", "function": {"name": n, "arguments": json.dumps(a)}}]
            if v.cmap.get(n, {}).get("kind") in ("curry_set", "curry_commit"):   # the model echoes the txn id
                a = dict(a, txn_id=conv.dec.last_txn_id); calls[0]["function"]["arguments"] = json.dumps(a)
            natives, internal = P.process_calls(conv, calls, "hard")
            got += [(nm, ar) for _, nm, ar in natives]
            if any("error" in json.loads(m["content"]) for m in internal):
                bad.append((t["function"]["name"], "rejected", internal)); break
        norm = lambda x: json.dumps(x, sort_keys=True)
        if [(gold["name"], norm(gold["arguments"]))] != [(n, norm(a)) for n, a in got]:
            bad.append((gold["name"], gold["arguments"], got))
    return bad


def synthetic_tasks():
    raw = json.load(open(os.path.join(ROOT, "benchmarks", "synthetic", "data", "tools.json")))
    tasks = []
    for group in dict.fromkeys(t["group"] for t in raw):   # names repeat across domains: one catalog per domain
        tasks.append([{"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters")}}
                      for t in raw if t["group"] == group])
    return tasks


def main():
    global NUM, REQUIRED_ONLY
    ap = argparse.ArgumentParser()
    ap.add_argument("catalog", nargs="?"); ap.add_argument("tasks", nargs="?")
    ap.add_argument("--classes"); ap.add_argument("--cuts"); ap.add_argument("--ops", default=",".join(P.OPS))
    a = ap.parse_args()
    if a.classes:
        P.CLASSES.update(json.load(open(a.classes)))
    if a.cuts:
        P.CUTS.update(json.load(open(a.cuts)))
    if a.catalog:
        catalog = {t["function"]["name"]: t for t in json.load(open(a.catalog))}
        tasks = [[catalog[n] for n in tt if n in catalog] for tt in json.load(open(a.tasks))]
    else:
        tasks = synthetic_tasks()
    total, failed = 0, []
    for op in a.ops.split(","):
        fails = 0
        for tools in tasks:
            if not tools:
                continue
            b = []
            for REQUIRED_ONLY in (False, True):                       # also exercise omitted optional arguments
                for NUM in ((7, 2.5), (100000, 100000.5), (-3, -3.5)):   # both sides of any interval cut
                    b += check(op, tools)
            total += len(tools); fails += len(b)
            if b and fails <= 3:
                print(f"  FAIL {op}: {str(b[0])[:300]}")
        print(f"{op:14s} {'OK' if not fails else f'{fails} FAILURES'}", flush=True)
        if fails:
            failed.append(op)
    print(f"checked {total} tool-variant round trips (x6 samples each)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
