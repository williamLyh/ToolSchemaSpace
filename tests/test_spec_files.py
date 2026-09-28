"""Round-trip gate for every spec file shipped in benchmarks/synthetic/data/specs.

Operator specs ({"ops": [...]}, group-stamped): compiled per domain; every gold episode
must encode -> decode to itself with zero flags.
Class-grouping specs ({"groups": [...]}, whole catalog): every gold episode must decode
to the class-qualified golds, and every dispatcher must reject an unknown operation.

Run:  python -m tests.test_spec_files
"""
import glob
import json
import os
import sys
from collections import Counter

from eval import common as ev
from toolschema.class_grouping import compile_spec_file, hard_reject_ok
from toolschema.operators import compile_spec
from tests.test_ops2 import by_group, queries, roundtrip

SPEC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "benchmarks", "synthetic", "data", "specs")


def check_catalog(path, fails):
    comp = compile_spec_file(path)
    ok, why = hard_reject_ok(comp)
    if not ok:
        fails.append((os.path.basename(path), "reject", why))
    n = 0
    for domain, rows in queries.items():
        for q in rows:
            native, flags = comp.decode_qualified(comp.encode_episode(domain, q["gold_calls"]))
            want = [(f"{domain}.{g['name']}", ev.clean(g.get("arguments", {}))) for g in q["gold_calls"]]
            got = [(n_, ev.clean(a)) for n_, a in native]
            n += 1
            if flags or got != want:
                fails.append((os.path.basename(path), q["id"], f"{got} != {want} flags={flags[:2]}"))
    return n


def main():
    checked, fails = Counter(), []
    for path in sorted(glob.glob(os.path.join(SPEC_DIR, "*.json"))):
        tag = os.path.basename(path)[:-5]
        spec = json.load(open(path, encoding="utf-8"))
        if "groups" in spec:
            checked[tag] = check_catalog(path, fails)
            continue
        if "ops" not in spec:                          # assignment metadata, not a spec
            continue
        for domain, tools in by_group.items():
            comp = compile_spec(tools, spec, group=domain)
            for q in queries[domain]:
                ok, why = roundtrip(comp, q["gold_calls"])
                checked[tag] += 1
                if not ok:
                    fails.append((tag, domain, q["id"], why))
    print(f"{len(checked)} spec files, {sum(checked.values())} episodes checked")
    if fails:
        print(f"FAILURES: {len(fails)}")
        for f in fails[:15]:
            print("  ", f)
        sys.exit(1)
    print("OK: every shipped spec round-trips")


if __name__ == "__main__":
    main()
