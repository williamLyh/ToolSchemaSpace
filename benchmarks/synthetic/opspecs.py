"""Auto-generate operator specs (schema/operators.py) for the synthetic env.

The value metadata the new operators need (_range/_samples for predicate cuts,
_pool for indirection tables) lives only in benchmarks/synthetic/domains.py — it is not
serialized into tools.json — so spec generation is synthetic-specific and
lives here. Each domain gets a battery of single-operator specs plus one
mixed spec; all are gated by tests/test_ops.py before any eval.

    from .opspecs import specs_for, DOMAINS
    specs_for(domain_dict, gold_rows) -> {"pred": spec, "indirect": spec,
                                          "curry": spec, "pack": spec, "mixed": spec}

`gold_rows` (the domain's query records) is REQUIRED for indirection: routine/
scenario golds use slot values outside the param's declared _pool, and the
resolver table must cover every value referable in the task distribution —
otherwise the variant makes tasks unsolvable and violates equivalence.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from .domains import DOMAINS  # noqa: E402


def num_cut(pd):
    """One interior cut for a NUM param: median of _samples if given, else the
    range midpoint — clamped strictly inside (lo, hi] so both sides are live."""
    lo, hi = pd["_range"]
    if pd.get("_samples"):
        vals = sorted(pd["_samples"])
        c = vals[len(vals) // 2]
    else:
        c = (lo + hi) // 2
    return max(lo + 1, min(int(c), hi))


def _gold_values(gold_rows):
    """(tool, param) -> set of gold argument values observed in the query set."""
    seen = {}
    for r in gold_rows or []:
        for g in r["gold_calls"]:
            for p, v in g["arguments"].items():
                seen.setdefault((g["name"], p), set()).add(v)
    return seen


def _tool_meta(domain):
    """Per tool: numeric params (with cut), pool params (with values), enum params."""
    out = []
    for t in domain["tools"]:
        nums, pools, enums = [], [], []
        for p, pd in t["params"].items():
            if "_range" in pd:
                nums.append((p, num_cut(pd)))
            elif "_pool" in pd:
                pools.append((p, list(pd["_pool"])))
            elif "enum" in pd or pd.get("type") == "boolean":
                enums.append(p)
        out.append({"name": t["name"], "n_params": len(t["params"]),
                    "params": list(t["params"]), "nums": nums, "pools": pools,
                    "enums": enums})
    return out


def specs_for(domain, gold_rows=None):
    meta = _tool_meta(domain)
    gv = _gold_values(gold_rows)
    pred, ind, curry, pack = [], [], [], []
    for m in meta:
        for p, cut in m["nums"]:
            pred.append({"op": "split_predicate", "tool": m["name"], "param": p,
                         "cuts": [cut]})
        for p, values in m["pools"]:
            vals = sorted(set(values) | gv.get((m["name"], p), set()), key=str)
            ind.append({"op": "indirect", "tool": m["name"], "param": p,
                        "values": vals})
        if m["n_params"] >= 2:
            curry.append({"op": "curry", "tool": m["name"]})
        if m["n_params"] >= 3:
            pack.append({"op": "arg_lower", "tool": m["name"], "into": "options",
                         "params": m["params"][1:]})

    # mixed: odd-indexed tools merged into one dispatch entry; even-indexed get
    # every applicable non-temporal operator at once (worst-case composition).
    names = [m["name"] for m in meta]
    mixed = [{"op": "merge", "tools": names[1::2]}]
    for m in meta[0::2]:
        if m["enums"]:
            mixed.append({"op": "split_enum", "tool": m["name"], "params": m["enums"]})
        for p, cut in m["nums"]:
            mixed.append({"op": "split_predicate", "tool": m["name"], "param": p,
                          "cuts": [cut]})
        for p, values in m["pools"]:
            vals = sorted(set(values) | gv.get((m["name"], p), set()), key=str)
            mixed.append({"op": "indirect", "tool": m["name"], "param": p,
                          "values": vals})

    return {"pred": {"ops": pred}, "indirect": {"ops": ind},
            "curry": {"ops": curry}, "pack": {"ops": pack},
            "mixed": {"ops": mixed}}
