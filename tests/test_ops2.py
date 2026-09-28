"""Oracle round-trip gate for the second-wave specs (enum splits, semantic /
anti / random merge partitions, surface tier). Same contract as tests/test_ops:
every gold episode must encode->decode to itself with zero flags under every
spec, per-domain AND under the global group-stamped spec files.

Run:  python -m tests.test_ops2
"""
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from eval import common as ev
from toolschema.operators import compile_spec
from benchmarks.synthetic.opspecs2 import DOMAINS, specs2_for

TOOLS = json.load(open(os.path.join(ROOT, "benchmarks", "synthetic", "data", "tools.json"), encoding="utf-8"))
QPATH = os.environ.get("SYN_QUERIES",
                       os.path.join(ROOT, "benchmarks", "synthetic", "data", "queries.jsonl"))

by_group = defaultdict(list)
for t in TOOLS:
    by_group[t["group"]].append({"name": t["name"], "description": t["description"],
                                 "parameters": t["parameters"]})
queries = defaultdict(list)
for l in open(QPATH, encoding="utf-8"):
    r = json.loads(l)
    queries[r["domain"]].append(r)


def roundtrip(comp, golds):
    seq = comp.encode_episode(golds)
    native, flags = comp.decode_seq(seq)
    if flags:
        return False, f"flags={flags[:2]}"
    if len(native) != len(golds):
        return False, f"{len(native)} native vs {len(golds)} golds"
    for (nm, na), g in zip(native, golds):
        if nm != g["name"] or ev.clean(na) != ev.clean(g["arguments"]):
            return False, f"{nm}({na}) != {g['name']}({g['arguments']})"
    return True, ""


def main():
    checked, fails = Counter(), []
    n_tools = Counter()
    for d in DOMAINS:
        g = d["name"]
        for tag, spec in specs2_for(d).items():
            if not spec["ops"]:
                continue
            comp = compile_spec(by_group[g], spec)
            n_tools[tag] += len(comp.tools)
            # structural sanity per tag
            if tag == "s_obf":
                assert all(f["name"].startswith("fn_") for f in comp.tools), g
            if tag == "s_nodesc":
                assert all(not f.get("description") for f in comp.tools), g
            if tag.startswith("mgroup"):
                assert all("operation" in f["parameters"]["properties"] for f in comp.tools), g
            for q in queries[g]:
                ok, why = roundtrip(comp, q["gold_calls"])
                checked[tag] += 1
                if not ok:
                    fails.append((g, tag, q["id"], why))

    # same-granularity assertion: the three mgroup partitions must have the
    # same catalog size per domain (they differ only in fold membership)
    for d in DOMAINS:
        s = specs2_for(d)
        ks = {t: len(s[t]["ops"]) for t in ("mgroup_sem", "mgroup_anti", "mgroup_rand")}
        assert len(set(ks.values())) == 1, (d["name"], ks)

    # global group-stamped spec files must behave identically under filtering
    merged = defaultdict(list)
    for d in DOMAINS:
        for tag, spec in specs2_for(d).items():
            for o in spec["ops"]:
                o["group"] = d["name"]
            merged[tag].extend(spec["ops"])
    for tag, ops in merged.items():
        for d in DOMAINS:
            g = d["name"]
            comp = compile_spec(by_group[g], {"ops": ops}, group=g)
            for q in queries[g][:25]:
                ok, why = roundtrip(comp, q["gold_calls"])
                checked[f"global_{tag}"] += 1
                if not ok:
                    fails.append((g, f"global_{tag}", q["id"], why))

    print(f"checked: {dict(checked)}  total={sum(checked.values())}")
    print(f"catalog sizes (sum over domains): {dict(n_tools)}")
    if fails:
        print(f"FAILURES: {len(fails)}")
        for f in fails[:15]:
            print("  ", f)
        sys.exit(1)
    print("OK: all second-wave specs round-trip (variant-invariant scoring)")


if __name__ == "__main__":
    main()
