"""Build the synthetic dataset artifacts from `domains.DOMAINS`.

Emits (under benchmarks/synthetic/data/generated/):
  - tools.json    : grouped canonical schemas (group = domain), the shape that
                    `toolschema.schema_adapter` and `toolschema.operators` take.
  - queries_templated.jsonl : one NL task per line, with programmatic GOLD calls. ~75%
                    single-call (one per tool x sampled enum combos) + ~25%
                    multi-call (domain scenarios).

Deterministic (fixed seed) so the gold is reproducible. The natural-language
phrasing here is the *base* form; generation/gen_complex.py and generation/harden.py
rewrite `query` into the released phrasing while keeping `gold_calls` fixed.

Run:  python -m benchmarks.synthetic.build      (from repo root)
"""

import json
import os
import random
import re

from .domains import DOMAINS
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")   # benchmarks/synthetic/data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(DATA, "generated")
SEED = 20260629
PER_TOOL = 12                                   # target single-call queries per tool
ROUTINE_SAMPLES = int(os.environ.get("ROUTINE_SAMPLES", "16"))  # instances per routine (pre-balance)
_TOK = re.compile(r"\{([^}]+)\}")
_INTERNAL = ("_range", "_samples", "_pool")


def clean_param(spec):
    """Canonical JSON-schema property: drop generator-internal keys (_range...)."""
    return {k: v for k, v in spec.items() if not k.startswith("_")}


def candidate_values(spec):
    """Sampling pool for a param when generating queries."""
    if "enum" in spec:
        return list(spec["enum"])
    t = spec.get("type")
    if t == "boolean":
        return [True, False]
    if t == "integer":
        if spec.get("_samples"):
            return list(spec["_samples"])
        lo, hi = spec.get("_range", (0, 10))
        step = max(1, (hi - lo) // 6)
        return list(range(lo, hi + 1, step))
    return list(spec.get("_pool", ["item"]))            # free string


def _phrase(rng, phrases, param, value):
    key = ("true" if value else "false") if isinstance(value, bool) else str(value)
    opts = phrases.get(param, {}).get(key)
    return rng.choice(opts) if opts else key


def render(rng, template, combo, phrases, params):
    """Fill a template: `{p}`->raw value, `{p_p}`/`{p_pN}`->surface phrase."""
    def sub(m):
        tok = m.group(1)
        pm = re.fullmatch(r"(.+?)_p\d*", tok)
        if pm and pm.group(1) in params:
            return _phrase(rng, phrases, pm.group(1), combo[pm.group(1)])
        if tok in params:
            v = combo[tok]
            return str(v).lower() if isinstance(v, bool) else str(v)
        return m.group(0)
    s = _TOK.sub(sub, template)
    return re.sub(r"\s+", " ", s).strip()


def _slot_phrase(rng, phrases, slot, value):
    key = ("true" if value else "false") if isinstance(value, bool) else str(value)
    opts = phrases.get(slot, {}).get(key)
    return rng.choice(opts) if opts else key


def render_routine(rng, template, combo, phrases):
    """Like `render`, but tokens resolve against a routine's sampled `combo`."""
    def sub(m):
        tok = m.group(1)
        pm = re.fullmatch(r"(.+?)_p\d*", tok)
        if pm and pm.group(1) in combo:
            return _slot_phrase(rng, phrases, pm.group(1), combo[pm.group(1)])
        if tok in combo:
            v = combo[tok]
            return str(v).lower() if isinstance(v, bool) else str(v)
        return m.group(0)
    s = _TOK.sub(sub, template)
    return re.sub(r"\s+", " ", s).strip()


def _resolve(v, combo):
    if isinstance(v, dict) and "__slot__" in v:
        return combo[v["__slot__"]]
    return v


def resolve_calls(calls, combo):
    return [{"name": n, "arguments": {k: _resolve(vv, combo) for k, vv in a.items()}}
            for n, a in calls]


def build_routines(rng, d):
    """Expand a domain's compound routines into compositional multi-call queries."""
    dom = d["name"]
    rows = []
    for rt in d.get("routines", []):
        slots = rt.get("slots", {})
        phrases = rt.get("phrases", {})
        n_target = ROUTINE_SAMPLES if slots else len(rt["templates"])
        seen, i, tries = set(), 0, 0
        while i < n_target and tries < n_target * 12:
            tries += 1
            combo = {s: rng.choice(vals) for s, vals in slots.items()}
            key = tuple(sorted((s, str(v)) for s, v in combo.items()))
            if slots and key in seen:
                continue
            seen.add(key)
            tmpl = rt["templates"][i % len(rt["templates"])]
            q = render_routine(rng, tmpl, combo, phrases)
            calls = resolve_calls(rt["calls"], combo)
            rows.append({
                "id": f"{dom}__{rt['name']}__{i}",
                "domain": dom, "system": d["system"], "type": rt.get("kind", "compound"),
                "query": q, "gold_calls": calls, "n_calls": len(calls),
            })
            i += 1
    return rows


def build_tools():
    tools = []
    for d in DOMAINS:
        for t in d["tools"]:
            tools.append({
                "uid": f"{d['name']}::{t['name']}",
                "group": d["name"],
                "name": t["name"],
                "description": t["description"],
                "parameters": {
                    "type": "object",
                    "properties": {p: clean_param(spec) for p, spec in t["params"].items()},
                    "required": list(t["params"].keys()),
                },
            })
    return tools


def build_queries():
    rng = random.Random(SEED)
    rows = []
    for d in DOMAINS:
        dom = d["name"]
        # ---- single-call: sample enum/param combos per tool --------------- #
        for t in d["tools"]:
            params = t["params"]
            pools = {p: candidate_values(spec) for p, spec in params.items()}
            seen, tries, i = set(), 0, 0
            while i < PER_TOOL and tries < PER_TOOL * 8:
                tries += 1
                combo = {p: rng.choice(vals) for p, vals in pools.items()}
                key = tuple(sorted((p, str(v)) for p, v in combo.items()))
                if key in seen:
                    continue
                seen.add(key)
                tmpl = t["templates"][i % len(t["templates"])]
                q = render(rng, tmpl, combo, t["phrases"], params)
                rows.append({
                    "id": f"{dom}__{t['name']}__{i}",
                    "domain": dom, "system": d["system"], "type": "single",
                    "query": q,
                    "gold_calls": [{"name": t["name"], "arguments": combo}],
                    "n_calls": 1,
                })
                i += 1
        # ---- multi-call scenarios ---------------------------------------- #
        for si, sc in enumerate(d.get("scenarios", [])):
            for ti, tmpl in enumerate(sc["templates"]):
                rows.append({
                    "id": f"{dom}__scenario{si}__{ti}",
                    "domain": dom, "system": d["system"], "type": "compound",
                    "query": tmpl,
                    "gold_calls": [{"name": n, "arguments": a} for n, a in sc["calls"]],
                    "n_calls": len(sc["calls"]),
                })
        # ---- compound + compositional routines (tau-bench-style, sampled) - #
        rows.extend(build_routines(rng, d))
    return balance(rows)


def balance(rows):
    """Trim the three complexity tiers (single / compound / compositional) to a
    common size so the subsets are comparable. Capped to the smallest tier (or
    SUBSET_SIZE if set and smaller); sampling is seeded for reproducibility."""
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)
    cap = min(len(v) for v in by.values())
    env = os.environ.get("SUBSET_SIZE")
    if env:
        cap = min(cap, int(env))
    rng = random.Random(SEED + 1)
    out = []
    for t in ("single", "compound", "compositional"):
        v = list(by.get(t, []))
        rng.shuffle(v)
        out.extend(v[:cap])
    return out


def queries_to_tasks(rows):
    """Convert queries.jsonl rows into the metrics/demo `tasks.json` shape
    ({id, group, query, tools, gold:{actions:[{name,arguments}]}})."""
    tasks = []
    for r in rows:
        tasks.append({
            "id": r["id"], "group": r["domain"], "query": r["query"],
            "tools": sorted({c["name"] for c in r["gold_calls"]}),
            "gold": {"actions": [{"name": c["name"], "arguments": c["arguments"]}
                                 for c in r["gold_calls"]]},
        })
    return tasks


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tools = build_tools()
    with open(os.path.join(OUT_DIR, "tools.json"), "w", encoding="utf-8") as f:
        json.dump(tools, f, ensure_ascii=False, indent=2)

    rows = build_queries()
    with open(os.path.join(OUT_DIR, "queries_templated.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT_DIR, "tasks.json"), "w", encoding="utf-8") as f:
        json.dump(queries_to_tasks(rows), f, ensure_ascii=False, indent=2)

    n_single = sum(1 for r in rows if r["type"] == "single")
    n_multi = len(rows) - n_single
    by_dom = {}
    for r in rows:
        by_dom[r["domain"]] = by_dom.get(r["domain"], 0) + 1
    print(f"tools: {len(tools)} across {len(DOMAINS)} domains")
    print(f"queries: {len(rows)}  (single={n_single}, multi={n_multi})")
    for dom in sorted(by_dom):
        print(f"  {dom}: {by_dom[dom]}")


if __name__ == "__main__":
    main()
