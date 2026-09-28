"""Regenerate every operator-spec file in benchmarks/synthetic/data/specs.

Operator specs are group-stamped and concatenated across the 12 domains: tool names
are not unique across domains, and operators ignore tools absent from a group's
catalog, so one global file applies cleanly per domain. Indirection tables embed
gold values from the query set (see opspecs.py).

  opspecs      pred, indirect, curry, pack, mixed
  opspecs2     enum1, enumfull, mgroup_{sem,anti,rand}, s_obf, s_nodesc, s_shuffle, ...
  opspecs3     pred2, pred4, curryind, currypart
  merge_encoding_specs   merge_full_*, merge_sem_*, mix_r0{3,5}_*
  class_grouping         class_*, random_neutral_r*, anti_class_neutral_r*

Run:  python -m benchmarks.synthetic.write_specs      then  python -m tests.test_spec_files
"""
import json
import os
import runpy
from collections import defaultdict

from toolschema import class_grouping
from . import merge_encoding_specs
from .opspecs import DOMAINS, specs_for
from .opspecs2 import specs2_for

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_DIR = os.path.join(DATA, "specs")


def dump(make):
    merged = defaultdict(list)
    for d in DOMAINS:
        for tag, spec in make(d).items():
            for o in spec["ops"]:
                o["group"] = d["name"]
            merged[tag].extend(spec["ops"])
    for tag, ops in merged.items():
        path = os.path.join(OUT_DIR, f"{tag}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ops": ops}, f, ensure_ascii=False)
        print(f"{path}: {len(ops)} ops")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    queries = defaultdict(list)
    for line in open(os.path.join(DATA, "queries.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        queries[r["domain"]].append(r)
    dump(lambda d: specs_for(d, queries[d["name"]]))
    dump(specs2_for)
    runpy.run_module("benchmarks.synthetic.opspecs3", run_name="__main__")
    merge_encoding_specs.write_merge_encodings()
    merge_encoding_specs.write_encoding_mixtures()
    class_grouping.write_specs()


if __name__ == "__main__":
    main()
