"""Assemble the released task file from the two generation branches.

  generated/queries_hard.jsonl     1,248 tasks: build.py -> gen_complex.py -> harden.py
  generated/queries_sampled.jsonl  1,500 tasks: gen_sampled.py (sampled tool combinations)

The release (data/queries.jsonl, 2,748 tasks) is the first file followed by the second.
Run:  python -m benchmarks.synthetic.generation.assemble [out.jsonl]
"""
import os
import sys

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, "generated", "queries.jsonl")
n = 0
with open(out, "w", encoding="utf-8") as f:
    for name in ("queries_hard.jsonl", "queries_sampled.jsonl"):
        for line in open(os.path.join(DATA, "generated", name), encoding="utf-8"):
            if line.strip():
                f.write(line if line.endswith("\n") else line + "\n")
                n += 1
print(f"{n} tasks -> {out}")
