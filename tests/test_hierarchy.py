"""Perfect-model gate for the hierarchy arms.

Every gold episode, encoded by the arm's oracle and fed through HierEnv under
hard control, must execute exactly the class-qualified golds with ZERO
rejections — for all four arms. Only then is a measured arm difference a real
schema-usage effect.

Also asserts the collision pairs are load-bearing: the same method name must
exist in two classes, and flat renames exactly those.

Run:  python -m tests.test_hierarchy
"""
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from eval import common as ev            # clean(): normalized arg compare
from toolschema.hierarchy import ARMS, HierEnv, build_arm

CATALOG = json.load(open(os.path.join(ROOT, "benchmarks", "synthetic", "data", "tools.json"),
                         encoding="utf-8"))
QPATH = os.environ.get("SYN_QUERIES",
                       os.path.join(ROOT, "benchmarks", "synthetic", "data", "queries.jsonl"))
ROWS = [json.loads(l) for l in open(QPATH, encoding="utf-8") if l.strip()]

dup = {n for n, c in Counter(t["name"] for t in CATALOG).items() if c > 1}
assert len(dup) == 5, f"expected 5 cross-class collision pairs, got {sorted(dup)}"


def key(name, args):
    return (name, tuple(sorted(ev.clean(args).items())))


def main():
    checked, fails = Counter(), []
    for arm in ARMS:
        comp = build_arm(arm, CATALOG)
        names = {t["name"] for t in comp.tools}
        assert len(names) == len(comp.tools), f"{arm}: duplicate variant names"
        for q in ROWS:
            golds = [(f"{q['domain']}.{g['name']}", g.get("arguments", {}))
                     for g in q["gold_calls"]]
            env = HierEnv(comp, control="hard")
            for vname, vargs in comp.encode_episode(q["domain"], q["gold_calls"]):
                env.call(vname, vargs)
            checked[arm] += 1
            if env.n_rejected:
                fails.append((arm, q["id"], f"{env.n_rejected} rejected"))
            elif Counter(key(*a) for a in env.native_actions) != \
                    Counter(key(n, a) for n, a in golds):
                fails.append((arm, q["id"], f"{env.native_actions} != {golds}"))

    # hard control rejects wrong-class picks on a collision pair
    comp = build_arm("dispatch", CATALOG)
    env = HierEnv(comp, control="hard")
    r = json.loads(env.call("tv", {"operation": "create_event"}))
    assert "error" in r and env.n_rejected == 1, r
    comp = build_arm("disclose", CATALOG)
    env = HierEnv(comp, control="hard")
    r = json.loads(env.call("invoke", {"class": "tv", "method": "no_such", "arguments": {}}))
    assert "error" in r and env.n_rejected == 1, r
    r = json.loads(env.call("list_methods", {"class": "tv"}))
    assert r["methods"][0]["parameters"], "list_methods must return full schemas"

    print(f"checked (episode x arm): {dict(checked)}  total={sum(checked.values())}")
    if fails:
        print(f"FAILURES: {len(fails)}")
        for f in fails[:15]:
            print("  ", f)
        sys.exit(1)
    print("OK: all four hierarchy arms round-trip every gold episode (exact=1.0, 0 rejects)")


if __name__ == "__main__":
    main()
