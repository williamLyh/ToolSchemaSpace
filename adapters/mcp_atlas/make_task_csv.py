#!/usr/bin/env python3
"""Write the MCP-Atlas task subset used in the paper as a CSV that `run_eval.py --input` accepts.

The subset is the 89 tasks whose reference trajectory calls only tools that the key-free
sandbox exposes (all 20 servers online); their ids are listed in task_ids.txt.
    python adapters/mcp_atlas/make_task_csv.py tasks.csv
"""
import csv
import os
import sys

from datasets import load_dataset

ids = set(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "task_ids.txt")).read().split())
rows = [r for r in load_dataset("ScaleAI/MCP-Atlas", split="train") if r["TASK"] in ids]
with open(sys.argv[1], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)}/{len(ids)} tasks to {sys.argv[1]}")
