"""Surfaces used in the seven-variant mixed training data (paper Section 6).

The mixed data rotates over fully merged (v0), transaction, reference resolution,
namespaced names, fully split, schema discovery and native. Both new surfaces are built exactly as the eval builds them:
  fully split       SchemaAdapter(k=9).transform(native, group=domain)
                    (eval/run_agentic.py, SCHEMA_VARIANT=9)
  namespaced names  hierarchy.build_arm("qualified", tools.json)
                    (eval/run_hierarchy.py, HIER_ARM=qualified)

The k-ladder compiler has no oracle encoder, so `encode_split` searches the
variant tools for the one whose decoding reproduces each gold native call.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["SCHEMA_K"] = "10"          # synthetic ladder: v5 native, v9 finest
from toolschema.schema_adapter import SchemaAdapter
from toolschema.schema_transform import decode_call
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")   # benchmarks/synthetic/data

MIXED7 = ["v0", "curry", "indirect", "qualified", "v9", "disclose", "native"]

_SPLIT = SchemaAdapter(k=9, control="hard")


_CATALOG = json.load(open(os.path.join(DATA, "tools.json"), encoding="utf-8"))
_CACHE = {}


def split_surface(domain):
    """-> (OpenAI tools, call_map) of fully split for `domain`, from the same
    tools.json rows the eval feeds the adapter (eval_singleshot_synthetic.NATIVE)."""
    if domain not in _CACHE:
        nat = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                 "parameters": t["parameters"]}}
               for t in _CATALOG if t["group"] == domain]
        _CACHE[domain] = _SPLIT.transform(nat, group=domain)
    return _CACHE[domain]


def _norm(args):
    return json.dumps({k: v for k, v in args.items() if v not in ("", None)}, sort_keys=True)


def encode_split(tools, call_map, gold_calls):
    """Oracle call sequence on the fully split surface for native gold calls."""
    props = {t["function"]["name"]: set(t["function"]["parameters"].get("properties", {}))
             for t in tools}
    seq = []
    for g in gold_calls:
        want = (g["name"], _norm(g["arguments"]))
        hits = []
        for name, ps in props.items():
            args = {k: v for k, v in g["arguments"].items() if k in ps}
            n, a = decode_call(call_map, name, args)
            if (n, _norm(a)) == want:
                hits.append((name, args))
        assert len(hits) == 1, (g, [h[0] for h in hits])
        seq.append(hits[0])
    return seq
