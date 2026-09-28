"""Third-wave specs: predN dose curve, curry×indirect composite, partial curry.

  pred2/pred4        split_predicate with 2/4 equal-quantile interior cuts
  curryind           curry + indirect on the same tools (both runtime-state axes)
  currypart          curry on even-indexed >=2-param tools only (mixture of
                     transactional and plain conventions — C-axis valley probe)

Gated by tests/test_spec_files (same round-trip contract). Group-stamped global files
via: python -m benchmarks.synthetic.opspecs3
"""
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from .domains import DOMAINS
from .opspecs import specs_for
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")   # benchmarks/synthetic/data


def _cuts(pd, n):
    lo, hi = pd["_range"]
    s = sorted(pd.get("_samples") or [])
    if len(s) >= n + 1:
        qs = [s[int(len(s) * (i + 1) / (n + 1))] for i in range(n)]
    else:
        qs = [lo + (hi - lo) * (i + 1) // (n + 1) for i in range(n)]
    out = sorted({max(lo + 1, min(int(c), hi)) for c in qs})
    return out


def specs3_for(domain, gold_rows):
    base = specs_for(domain, gold_rows)
    pred2, pred4 = [], []
    for t in domain["tools"]:
        for p, pd in t["params"].items():
            if "_range" in pd:
                c2, c4 = _cuts(pd, 2), _cuts(pd, 4)
                if len(c2) >= 2:
                    pred2.append({"op": "split_predicate", "tool": t["name"],
                                  "param": p, "cuts": c2})
                if len(c4) >= 3:
                    pred4.append({"op": "split_predicate", "tool": t["name"],
                                  "param": p, "cuts": c4})
    curry_tools = {o["tool"] for o in base["curry"]["ops"]}
    curryind = base["curry"]["ops"] + [o for o in base["indirect"]["ops"]
                                       if o["tool"] in curry_tools]
    currypart = [o for i, o in enumerate(base["curry"]["ops"]) if i % 2 == 0]
    return {"pred2": {"ops": pred2}, "pred4": {"ops": pred4},
            "curryind": {"ops": curryind}, "currypart": {"ops": currypart}}


if __name__ == "__main__":
    qpath = os.path.join(DATA, "queries.jsonl")
    queries = defaultdict(list)
    for l in open(qpath):
        r = json.loads(l)
        queries[r["domain"]].append(r)
    merged = defaultdict(list)
    for d in DOMAINS:
        for tag, spec in specs3_for(d, queries[d["name"]]).items():
            for o in spec["ops"]:
                o["group"] = d["name"]
            merged[tag].extend(spec["ops"])
    outd = os.path.join(DATA, "specs")
    for tag, ops in merged.items():
        json.dump({"ops": ops}, open(os.path.join(outd, f"{tag}.json"), "w"))
        print(tag, len(ops), "ops")
