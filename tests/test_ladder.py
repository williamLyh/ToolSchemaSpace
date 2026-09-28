"""Ladder gate: scoring must be variant-invariant along the merge/split ladder.

For every gold call in queries.jsonl, for each variant k=0..9, build the ORACLE
call in that variant's vocabulary (the correct way to express the gold action),
decode it via the same classify_call path the eval uses, and assert it matches
the native gold. This guarantees a perfect model scores 1.0 at every granularity,
so any accuracy difference across variants is a real schema-usage effect.

Run:  python -m tests.test_ladder
"""
import json
import os
import sys

from eval import common as ev            # NATIVE, match, QPATH
from toolschema.schema_adapter import K as N_VARIANTS, SchemaAdapter, classify_call


def oracle_call(cmap, gold_name, gold_args):
    """Express (gold_name, gold_args) in the variant vocabulary described by cmap."""
    # execute / namespaced entry
    for name, e in cmap.items():
        if e.get("namespaced") and gold_name in e.get("ops", []):
            args = {"operation": gold_name}
            args.update({f"{gold_name}::{k}": v for k, v in gold_args.items()})
            return name, args
    # named/baked entry whose fixed values agree with the gold
    for name, e in cmap.items():
        if e.get("namespaced") or e["op"] != gold_name:
            continue
        fixed = e.get("fixed", {})
        if all(str(gold_args.get(k)) == str(v) for k, v in fixed.items()):
            rest = {k: v for k, v in gold_args.items() if k not in fixed}
            return name, rest
    raise AssertionError(f"no oracle for {gold_name} in variant vocab")


def main():
    rows = [json.loads(l) for l in open(ev.QPATH, encoding="utf-8") if l.strip()]
    fails = 0
    checked = 0
    for k in range(N_VARIANTS):
        os.environ["SCHEMA_VARIANT"] = str(k)
        os.environ["SCHEMA_CONTROL"] = "hard"
        os.environ["SCHEMA_DATASET"] = "synthetic"
        SchemaAdapter._cached = False                 # reset memoized singleton
        ad = SchemaAdapter.from_env()
        cache = {}
        for r in rows:
            dom = r["domain"]
            if dom not in cache:
                ad.variant_openai_tools(dom, ev.NATIVE[dom])
                cache[dom] = ad.call_map_for(dom)
            cmap = cache[dom]
            for gc in r["gold_calls"]:
                name, args = oracle_call(cmap, gc["name"], gc["arguments"])
                act, nn, na = classify_call(cmap, "hard", name, args)
                checked += 1
                if act == "reject" or not ev.match(gc["name"], gc["arguments"], nn, na):
                    fails += 1
                    if fails <= 10:
                        print(f"FAIL v{k} {dom} gold={gc['name']}{gc['arguments']} "
                              f"oracle=({name},{args}) -> ({act},{nn},{na})")
    print(f"checked {checked} (gold x variant) decodes; failures={fails}")
    assert fails == 0, "variant scoring is NOT invariant"
    print(f"OK: scoring is variant-invariant across all {N_VARIANTS} granularities")


if __name__ == "__main__":
    main()
