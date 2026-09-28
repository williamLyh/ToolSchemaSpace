"""Oracle + hard-validation gate for the merge argument structures (flat / nested / union).

Checks, for every domain and every merge-structure spec:
  1. gold trajectories encode/decode with zero flags
  2. reconstructed native actions equal gold
  3. hard SimEnv rejects zero-op, multi-op, and cross-operation arguments
  4. complexity covariates are recorded

Run: python -m tests.test_merge_encodings
"""
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from eval import common as ev
from toolschema.merge_encodings import schema_complexity, validate_merged
from toolschema.operators import compile_spec
from toolschema.simenv import SimEnv

TOOLS = json.load(open(os.path.join(ROOT, "benchmarks", "synthetic", "data", "tools.json"), encoding="utf-8"))
QPATH = os.environ.get("SYN_QUERIES",
                       os.path.join(ROOT, "benchmarks", "synthetic", "data", "queries.jsonl"))
SPEC_DIR = os.path.join(ROOT, "benchmarks", "synthetic", "data", "specs")
TAGS = [
    "merge_full_flat", "merge_full_nested", "merge_full_union",
    "merge_sem_flat", "merge_sem_nested", "merge_sem_union",
]

by_group = defaultdict(list)
for tool in TOOLS:
    by_group[tool["group"]].append({
        "name": tool["name"],
        "description": tool["description"],
        "parameters": tool["parameters"],
    })
queries = defaultdict(list)
for line in open(QPATH, encoding="utf-8"):
    row = json.loads(line)
    queries[row["domain"]].append(row)


def _load(tag):
    return json.load(open(os.path.join(SPEC_DIR, f"{tag}.json"), encoding="utf-8"))


def roundtrip(comp, golds):
    seq = comp.encode_episode(golds)
    native, flags = comp.decode_seq(seq)
    if flags:
        return False, f"flags={flags[:2]}"
    if len(native) != len(golds):
        return False, f"{len(native)} native vs {len(golds)} golds"
    for (name, args), gold in zip(native, golds):
        if name != gold["name"] or ev.clean(args) != ev.clean(gold["arguments"]):
            return False, f"{name}({args}) != {gold['name']}({gold['arguments']})"
    return True, ""


def _first_dispatcher(comp):
    for name, entry in comp.call_map.items():
        if entry.get("kind") == "execute":
            return name, entry
    raise AssertionError("no execute entry")


def test_hard_rejects(comp):
    env = SimEnv(comp.call_map, control="hard")
    name, entry = _first_dispatcher(comp)
    encoding = entry.get("encoding", "flat")
    ops = list(entry["ops"])
    op = ops[0]
    other = ops[1] if len(ops) > 1 else None
    params = entry.get("op_params", {}).get(op, [])
    payload = {params[0]: "x"} if params else {}

    if encoding == "union":
        bad = [
            {},
            {op: payload, other: payload} if other else {op: payload, "not_an_op": payload},
        ]
    else:
        bad = [
            {"operation": "not_a_real_op", "arguments": payload} if encoding == "nested"
            else {"operation": "not_a_real_op"},
        ]
        if encoding == "nested" and other:
            foreign = entry.get("op_params", {}).get(other, [])
            if foreign and foreign[0] not in params:
                bad.append({"operation": op, "arguments": {foreign[0]: "x"}})

    for args in bad:
        if validate_merged(entry, args) is None:
            return False, f"{encoding} failed to flag {args}"
        before = env.n_rejected
        env.call(name, args)
        if env.n_rejected <= before:
            return False, f"{encoding} SimEnv accepted {args}"
    return True, ""


def main():
    fails = []
    checked = Counter()
    complexity = []
    for tag in TAGS:
        spec = _load(tag)
        for domain, tools in by_group.items():
            if not tools:
                continue
            comp = compile_spec(tools, spec, group=domain)
            complexity.append({
                "tag": tag,
                "domain": domain,
                **schema_complexity(comp.tools),
            })
            ok, why = test_hard_rejects(comp)
            checked[f"{tag}_reject"] += 1
            if not ok:
                fails.append((domain, tag, "reject", why))
            for query in queries[domain]:
                ok, why = roundtrip(comp, query["gold_calls"])
                checked[tag] += 1
                if not ok:
                    fails.append((domain, tag, query["id"], why))

    print(f"checked: {dict(checked)} total={sum(checked.values())}")
    if fails:
        print(f"FAILURES: {len(fails)}")
        for row in fails[:20]:
            print(" ", row)
        sys.exit(1)
    print("OK: merge argument structures round-trip and reject illegal calls")


if __name__ == "__main__":
    main()
