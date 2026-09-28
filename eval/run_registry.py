"""Run every synthetic-benchmark variant of a paper figure or table, as listed in registry.json.

    python -m eval.run_registry --figure "Figure 2" --out results/qwen3-4b
    python -m eval.run_registry --list                        # show all variants and their construction
    python -m eval.run_registry --figure "Figure 2" --dry-run  # print the commands only

Each variant is run by the right runner (run_agentic / run_hierarchy / run_class_grouping) with
SCHEMA_CONTROL=hard and writes <out>/<slug>.csv (aggregate) and <out>/<slug>.jsonl (per episode).
Model and endpoint come from the usual SYN_MODEL / REMOTE_OPENAI_BASE_URL environment variables.
"""
import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "benchmarks", "synthetic", "data")
CLASS_SPECS = re.compile(r"^(class_|random_neutral_|anti_class_neutral_)")


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower().replace("·", " ")).strip("_")


def command(v):
    """(env overrides, module) for one registry entry."""
    if v["kind"] == "ladder":
        return {"SCHEMA_VARIANT": str(v["arg"]), "SCHEMA_DATASET": "synthetic"}, "eval.run_agentic"
    if v["kind"] == "hierarchy":
        return {"HIER_ARM": v["arg"]}, "eval.run_hierarchy"
    spec = os.path.join(DATA, "specs", v["arg"])
    if CLASS_SPECS.match(v["arg"]):
        return {"SCHEMA_SPEC": spec}, "eval.run_class_grouping"
    return {"SCHEMA_SPEC": spec, "SCHEMA_DATASET": "synthetic"}, "eval.run_agentic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", default="", help="substring of the registry 'figure' field")
    ap.add_argument("--name", default="", help="substring of the variant name")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "synthetic"))
    ap.add_argument("--control", default="hard", choices=["hard", "loose"])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    variants = [v for v in json.load(open(os.path.join(DATA, "registry.json")))["variants"]
                if a.figure in v["figure"] and a.name in v["name"]]
    if a.list:
        for v in variants:
            print(f"{v['figure']:<48} {v['name']:<40} {v['kind']:<9} {v['arg']}")
        return
    os.makedirs(a.out, exist_ok=True)
    for v in variants:
        env, module = command(v)
        s = slug(v["name"])
        env.update(SCHEMA_CONTROL=a.control, OUT_CSV=os.path.join(a.out, f"{s}.csv"),
                   OUT_JSONL=os.path.join(a.out, f"{s}.jsonl"))
        print(" ".join(f"{k}={val}" for k, val in env.items()), "python -m", module, flush=True)
        if not a.dry_run:
            subprocess.run([sys.executable, "-m", module], cwd=ROOT, env={**os.environ, **env}, check=True)


if __name__ == "__main__":
    main()
