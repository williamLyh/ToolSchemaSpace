"""Materialize 10 schema-granularity variants per dataset.

For each dataset under ``data/<name>/`` (which `loaders/` produced as
`tools.json` = grouped canonical schemas), we emit 10 variants spanning the
fine↔coarse spectrum and dump them to ``data/<name>/variants/variant_<k>.json``.

A *variant* is a granularity level applied **per group** (a tau2 domain, a BFCL
involved-class, an AppWorld app). Groups have different spectra lengths
(`max_level = N + M`), so we normalize: variant ``k ∈ 0..9`` maps each group to

    level = round(k / (K-1) * max_level)          # K = 10

so variant 0 is the coarse extreme (one `execute(operation, …)` per group) and
variant K-1 is the fine extreme (every enumerable field baked into names), with
the group's *native* level falling wherever ``N / (N+M)`` lands. The same
normalized fractions are used across datasets so variant indices are comparable.

Each dumped variant carries, per group, both the forward `tools` (what the model
sees) and a `call_map` (how to decode a variant call back to the native call),
produced by `schema_transform.group_variant_mapped`.

Run:  python schema_variants.py [synthetic|bfcl_multiturn|tau_bench|appworld|all]
"""

import json
import os
import sys

from .schema_adapter import variant_level
from .schema_transform import (
    _bake_plan, _enum_values, _gprops, decode_call, group_variant_mapped,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ["synthetic", "bfcl_multiturn", "tau_bench", "appworld"]
K = int(os.environ.get("SCHEMA_K", "10"))   # ladder length: v0 coarsest, v5 native, v9 finest


def _bake_count(tools):
    return len(_bake_plan(tools))


def _group_tools(dataset):
    """Read data/<dataset>/tools.json and bucket canonical schemas by group."""
    path = os.path.join(ROOT, "data", dataset, "tools.json")
    tools = json.load(open(path, encoding="utf-8"))
    groups = {}
    for t in tools:
        # strip loader-only keys; keep the canonical 3-key schema
        groups.setdefault(t.get("group"), []).append(
            {"name": t["name"], "description": t.get("description", ""),
             "parameters": t["parameters"]}
        )
    return groups


def _check_variant(native_tools, gv, group):
    """Structural soundness of one group-variant. Raises AssertionError on failure."""
    native_names = {t["name"] for t in native_tools}
    enums = {(t["name"], p): set(map(str, _enum_values(pd)))
             for t in native_tools for p, pd in _gprops(t).items() if _enum_values(pd)}
    cmap = gv["call_map"]

    # 1. every variant tool name is decodable
    for t in gv["tools"]:
        assert t["name"] in cmap, f"[{group}] tool {t['name']} missing from call_map"

    # 2. coverage: exactly the native operations are reachable, no more/less
    reachable = set()
    for name, e in cmap.items():
        if e.get("namespaced"):
            reachable |= set(e.get("ops", []))
        else:
            reachable.add(e["op"])
    assert reachable == native_names, (
        f"[{group}] reachable ops {reachable ^ native_names} mismatch native")

    # 3. baked fixed values must belong to that param's enum set
    for name, e in cmap.items():
        for p, v in e.get("fixed", {}).items():
            allowed = enums.get((e["op"], p))
            assert allowed and str(v) in allowed, (
                f"[{group}] baked {e['op']}.{p}={v!r} not in enum {allowed}")

    # 4. decode round-trip: each native op is recoverable from its encoding
    for name, e in cmap.items():
        if e.get("namespaced"):
            for op in e["ops"]:
                nm, _ = decode_call(cmap, name, {"operation": op})
                assert nm == op, f"[{group}] execute decode {op} -> {nm}"
        else:
            nm, na = decode_call(cmap, name, {})
            assert nm == e["op"], f"[{group}] named decode {name} -> {nm}"
            assert na == e["fixed"], f"[{group}] named decode args {na} != {e['fixed']}"


def build_dataset_variants(dataset):
    groups = _group_tools(dataset)
    out_dir = os.path.join(ROOT, "data", dataset, "variants")
    os.makedirs(out_dir, exist_ok=True)

    # per-group N (ops) and M (bakeable enum params); native (level N) is at v5
    NM = {g: (len(ts), _bake_count(ts)) for g, ts in groups.items()}
    max_level = {g: N + M for g, (N, M) in NM.items()}

    summary = []
    for k in range(K):
        payload = {"k": k, "dataset": dataset, "n_variants": K, "groups": {}}
        for g, ts in groups.items():
            N, M = NM[g]
            lvl = variant_level(k, N, M)
            gv = group_variant_mapped(ts, lvl)
            _check_variant(ts, gv, g)
            payload["groups"][g] = {
                "level": gv["level"], "max_level": gv["max_level"],
                "n_functions": len(gv["tools"]),
                "tools": gv["tools"], "call_map": gv["call_map"],
            }
        with open(os.path.join(out_dir, f"variant_{k}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        nfns = {g: payload["groups"][g]["n_functions"] for g in groups}
        summary.append((k, nfns))

    print(f"[{dataset}] wrote {K} variants to data/{dataset}/variants/ "
          f"({len(groups)} groups, max_level={max_level})")
    for k, nfns in summary:
        total = sum(nfns.values())
        print(f"  variant {k}: {total} fns total  " +
              "  ".join(f"{g}={n}" for g, n in sorted(nfns.items()))[:120])
    return summary


def main(argv):
    targets = DATASETS if (not argv or argv[0] == "all") else argv
    for ds in targets:
        if not os.path.exists(os.path.join(ROOT, "data", ds, "tools.json")):
            print(f"[{ds}] SKIP — no data/{ds}/tools.json")
            continue
        build_dataset_variants(ds)


if __name__ == "__main__":
    main(sys.argv[1:])
