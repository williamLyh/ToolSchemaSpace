"""Second-wave operator specs (user-directed 2026-07-15).

  enum1       split_enum on each tool's FIRST enumerable param (fine direction, mild)
  enumfull    split_enum on as many enumerable params as fit under a 24-name cap
  mgroup_sem  SEMANTIC merge: agglomerative clusters (name-token+param Jaccard),
              ~4 tools per dispatch tool, cluster-derived names
  mgroup_anti same k and sizes, members drawn round-robin ACROSS semantic
              clusters — maximally incoherent folds at identical granularity
  mgroup_rand same sizes, seeded random partition (control)
  s_obf       rename every function to an opaque fn_<hash> (descriptions intact)
  s_nodesc    strip every function/param description (names intact)
  s_shuffle   deterministic permutation of parameter order

The three mgroup partitions share granularity (same #dispatch tools, same
sizes) and differ ONLY in which tools share a fold — the agentic version of
the single-shot partition-scatter finding. All specs are gated by
tests/test_ops2.py before any eval.

    from .opspecs2 import specs2_for, DOMAINS
"""
import hashlib
import random
from collections import Counter

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from .domains import DOMAINS  # noqa: E402

GENERIC = {"set", "get", "the", "a", "to", "of"}
ENUM_CAP = 24


def _enumerable(pd):
    return "enum" in pd or pd.get("type") == "boolean"


def _enum_card(pd):
    return len(pd["enum"]) if "enum" in pd else 2


def _sim(ta, tb):
    na, pa = set(ta["name"].split("_")) - GENERIC, set(ta["params"])
    nb, pb = set(tb["name"].split("_")) - GENERIC, set(tb["params"])
    jn = len(na & nb) / max(1, len(na | nb))
    jp = len(pa & pb) / max(1, len(pa | pb))
    return 0.6 * jn + 0.4 * jp


def _agglomerate(tools, k):
    """Average-linkage agglomerative clustering down to k clusters."""
    clusters = [[t] for t in tools]
    while len(clusters) > k:
        best, bi, bj = -1.0, 0, 1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                s = sum(_sim(a, b) for a in clusters[i] for b in clusters[j]) \
                    / (len(clusters[i]) * len(clusters[j]))
                if s > best:
                    best, bi, bj = s, i, j
        clusters[bi] += clusters.pop(bj)
    return clusters


def _cluster_name(members, used):
    toks = Counter(t for m in members for t in m["name"].split("_") if t not in GENERIC)
    top = [t for t, _ in toks.most_common(2)]
    nm = ("_".join(top) if top else "misc") + "_ops"
    while nm in used:
        nm += "_x"
    used.add(nm)
    return nm


def _merge_spec(partition):
    used = set()
    return [{"op": "merge", "tools": [m["name"] for m in members],
             "into": _cluster_name(members, used)} for members in partition]


def specs2_for(domain):
    tools = domain["tools"]
    n = len(tools)
    k = max(2, round(n / 4))

    # --- enum splits ------------------------------------------------------ #
    enum1, enumfull = [], []
    for t in tools:
        eps = [(p, pd) for p, pd in t["params"].items() if _enumerable(pd)]
        if not eps:
            continue
        enum1.append({"op": "split_enum", "tool": t["name"], "params": [eps[0][0]]})
        sel, prod = [], 1
        for p, pd in eps:
            c = _enum_card(pd)
            if prod * c > ENUM_CAP:
                break
            sel.append(p)
            prod *= c
        if sel:
            enumfull.append({"op": "split_enum", "tool": t["name"], "params": sel})

    # --- merge partitions (same k, same sizes, different coherence) ------- #
    sem = _agglomerate(list(tools), k)
    sizes = [len(c) for c in sem]
    anti = [[] for _ in range(k)]
    i = 0
    for cluster in sem:                    # round-robin across semantic clusters
        for m in cluster:
            anti[i % k].append(m)
            i += 1
    rng = random.Random(20260715)
    shuffled = list(tools)
    rng.shuffle(shuffled)
    rand, pos = [], 0
    for s in sizes:
        rand.append(shuffled[pos:pos + s])
        pos += s

    # --- surface tier ------------------------------------------------------ #
    s_obf = [{"op": "rename", "tool": t["name"],
              "to": "fn_" + hashlib.sha1(f"{domain['name']}:{t['name']}".encode()).hexdigest()[:6]}
             for t in tools]
    s_nodesc = [{"op": "strip_desc", "tool": t["name"]} for t in tools]
    s_shuffle = [{"op": "shuffle_params", "tool": t["name"], "seed": 13} for t in tools]

    return {"enum1": {"ops": enum1}, "enumfull": {"ops": enumfull},
            "mgroup_sem": {"ops": _merge_spec(sem)},
            "mgroup_anti": {"ops": _merge_spec(anti)},
            "mgroup_rand": {"ops": _merge_spec(rand)},
            "s_obf": {"ops": s_obf}, "s_nodesc": {"ops": s_nodesc},
            "s_shuffle": {"ops": s_shuffle}}
