"""Write the full tool schema of the native catalog and of every representative variant (Figure 2),
one JSON file per variant, to benchmarks/synthetic/data/schemas/.

Each file holds exactly what the model is shown, in OpenAI tool format:
  scope "per_domain"  -> {"domains": {domain: [tools]}}   the model sees only its task's domain
  scope "catalog"     -> {"tools": [tools]}               the model sees the whole 168-tool catalog
Every variant also records how it is built (see registry.json).

Run:  python -m benchmarks.synthetic.materialize_schemas
"""
import json
import os
import re
from collections import OrderedDict

from toolschema.hierarchy import build_arm
from toolschema.operators import compile_spec
from toolschema.schema_adapter import SchemaAdapter

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.join(DATA, "schemas")


def openai(tools):
    return [{"type": "function", "function": t} for t in tools]


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main():
    raw = json.load(open(os.path.join(DATA, "tools.json"), encoding="utf-8"))
    by_dom = OrderedDict()
    for t in raw:
        by_dom.setdefault(t["group"], []).append(
            {"name": t["name"], "description": t["description"], "parameters": t["parameters"]})
    registry = json.load(open(os.path.join(DATA, "registry.json"), encoding="utf-8"))["variants"]
    variants = [v for v in registry if v["figure"] == "Figure 2"]
    os.makedirs(OUT, exist_ok=True)
    index = []
    for i, v in enumerate(variants):
        rec = {"variant": v["name"], "operator": v["operator"], "method": v["method"],
               "construction": {"kind": v["kind"], "arg": v["arg"]}}
        if v["kind"] == "hierarchy":
            comp = build_arm(v["arg"], raw)
            rec["scope"] = "catalog"
            rec["tools"] = openai(comp.tools)
            if v["arg"] == "disclose":
                rec["note"] = ("list_methods(class) returns the native schemas of that class's methods "
                               "(the per-domain native catalog); invoke(class, method, arguments) executes one.")
            n = len(comp.tools)
        else:
            rec["scope"] = "per_domain"
            rec["domains"] = {}
            spec = (json.load(open(os.path.join(DATA, "specs", v["arg"]), encoding="utf-8"))
                    if v["kind"] == "spec" else None)
            for dom, tools in by_dom.items():
                if spec is not None:
                    rec["domains"][dom] = openai(compile_spec(tools, spec, group=dom).tools)
                else:
                    vt, _ = SchemaAdapter(k=int(v["arg"])).transform(openai(tools), group=dom)
                    rec["domains"][dom] = vt
            n = sum(len(x) for x in rec["domains"].values())
        fname = f"{i:02d}_{slug(v['name'])}.json"
        with open(os.path.join(OUT, fname), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        index.append({"file": fname, "variant": v["name"], "scope": rec["scope"], "n_tools": n})
        print(f"{fname:<32} {rec['scope']:<10} {n:>4} tools")
    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
