"""Equivalence gate for the schema proxy on a real tool catalog.

For every task tool set and every proxy operator: build one sample native call per
tool, encode it into the variant's call sequence (oracle), feed that sequence
through the proxy's decoder exactly as a model's tool calls would arrive, and
require the decoded native call to equal the original with no rejection.
    python -m tests.test_proxy <catalog.json> <task_tools.json>
catalog.json: list of OpenAI tools; task_tools.json: list of per-task tool-name lists.
Without arguments it runs on the synthetic benchmark (one task = one domain).
"""
import json
import os
import sys

import toolschema.proxy as P
from benchmarks.synthetic.extra_surfaces import encode_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    return {"integer": 7, "number": 2.5, "boolean": True, "array": ["x"], "object": {"k": "v"}}.get(t, "sample text")


def sample_call(tool):
    fn = P._canonicalize(tool); props = fn["parameters"].get("properties", {})   # $refs resolved
    req = fn["parameters"].get("required", []) or list(props)[:2]
    names = list(dict.fromkeys(list(req) + list(props)[:3]))
    return {"name": fn["name"], "arguments": {k: sample_value(props[k]) for k in names if k in props}}


def check(op, tools):
    v = P.Variant(op, tools)
    bad = []
    for t in tools:
        gold = sample_call(t)
        conv = P.Conversation(v)
        if v.comp is not None:
            seq = v.comp.encode_call(gold["name"], gold["arguments"])
        else:
            seq = encode_split(v.tools, v.cmap, [gold])
        got = []
        for i, (n, a) in enumerate(seq):
            calls = [{"id": f"c{i}", "function": {"name": n, "arguments": json.dumps(a)}}]
            if v.cmap.get(n, {}).get("kind") in ("curry_set", "curry_commit"):   # the model echoes the issued txn id
                a = dict(a, txn_id=conv.dec.last_txn_id); calls[0]["function"]["arguments"] = json.dumps(a)
            natives, internal = P.process_calls(conv, calls, "hard")
            got += [(nm, ar) for _, nm, ar in natives]
            if any("error" in json.loads(m["content"]) for m in internal):
                bad.append((t["function"]["name"], "rejected", internal)); break
        norm = lambda x: json.dumps(x, sort_keys=True)
        if [(gold["name"], norm(gold["arguments"]))] != [(n, norm(a)) for n, a in got]:
            bad.append((gold["name"], gold["arguments"], got))
    return bad


def load(argv):
    if len(argv) >= 2:
        catalog = {t["function"]["name"]: t for t in json.load(open(argv[0]))}
        return catalog, json.load(open(argv[1]))
    raw = json.load(open(os.path.join(ROOT, "benchmarks", "synthetic", "data", "tools.json")))
    tasks = []
    for group in dict.fromkeys(t["group"] for t in raw):   # names repeat across domains: one catalog per domain
        tasks.append([{"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters")}}
                      for t in raw if t["group"] == group])
    return None, tasks


def main():
    catalog, tasks = load(sys.argv[1:])
    total = failed = 0
    for op in P.OPS:
        fails = 0
        for tt in tasks:
            tools = tt if catalog is None else [catalog[n] for n in tt if n in catalog]
            if not tools:
                continue
            b = check(op, tools)
            total += len(tools); fails += len(b)
            if b and fails <= 3:
                print(f"  FAIL {op}: {str(b[0])[:300]}")
        failed += fails
        print(f"{op:14s} {'OK' if not fails else f'{fails} FAILURES'}")
    print("checked", total, "tool-variant round trips")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
