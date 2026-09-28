#!/usr/bin/env python3
"""Drop MCP-Atlas output rows that ended in an infrastructure error (e.g. the harness
timeout) so that `run_eval.py`, which skips task_ids already in the output, reruns them.
The dropped rows are kept in <out>.errors.csv for the record.
    python adapters/mcp_atlas/drop_error_rows.py outputs.csv
"""
import csv, sys
csv.field_size_limit(10 ** 9)
path = sys.argv[1]
rows = list(csv.DictReader(open(path, newline="")))
fields = list(rows[0].keys()) if rows else []
bad = [r for r in rows if str(r.get("response", "")).startswith("ERROR")]
good = [r for r in rows if r not in bad]
if bad:
    with open(path + ".errors.csv", "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writerows(bad)
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(good)
print(f"kept {len(good)}, dropped {len(bad)} error rows")
