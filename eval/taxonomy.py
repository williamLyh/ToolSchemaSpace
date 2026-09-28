#!/usr/bin/env python3
"""Failure-mode taxonomy (paper Table 2): assign each failed episode one of seven modes.

Rules fire in priority order; the first that applies labels the episode:
    livelock        the episode hit the turn limit
    txn_handle      a transaction flag fired (unknown / dangling / double / incomplete commit)
    rejected_stuck  a call was rejected as off-schema and too few native actions executed
    wrong_class     a call was routed through the wrong class dispatcher or namespace
    under_exec      fewer native actions executed than required, no rejection
    over_exec       more native actions executed than required
    wrong_args      the right actions executed with at least one wrong argument value

    python -m eval.taxonomy results.jsonl [more.jsonl ...]
prints, per file, the success rate and the rate of each failure mode.
"""
import json
import sys
from collections import Counter

MODES = ["livelock", "txn_handle", "rejected_stuck", "wrong_class", "under_exec", "over_exec", "wrong_args"]
MAX_TURNS = 16
TXN = {"set_unknown_txn", "commit_unknown_txn", "dangling_txn", "commit_missing_required",
       "double_commit", "set_after_commit"}


def signature(row: dict) -> str:
    """Failure mode of one failed episode record (as written by eval.run_agentic / run_hierarchy)."""
    if row.get("err") and row.get("n_gold") is None:
        return "other"                             # no recorded outcome (request error)
    if row.get("turns", 0) >= MAX_TURNS:
        return "livelock"
    if TXN & set(row.get("flags", [])):
        return "txn_handle"
    if row.get("n_rejected", 0) > 0 and row.get("n_exec", 0) < row.get("n_gold", 0):
        return "rejected_stuck"
    if row.get("wrong_class", 0):
        return "wrong_class"
    if row.get("n_exec", 0) < row.get("n_gold", 0):
        return "under_exec"
    if row.get("n_exec", 0) > row.get("n_gold", 0):
        return "over_exec"
    return "wrong_args"


def profile(rows):
    n = len(rows)
    fails = Counter(signature(r) for r in rows if not r.get("exact"))
    out = {"n": n, "success": sum(1 for r in rows if r.get("exact")) / max(1, n)}
    out.update({m: fails[m] / max(1, n) for m in MODES + ["other"]})
    return out


if __name__ == "__main__":
    for path in sys.argv[1:]:
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        p = profile(rows)
        print(path, " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in p.items()))
