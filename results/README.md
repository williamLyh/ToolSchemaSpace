# Results

This directory is reserved for released evaluation outputs: per-arm score tables, per-episode
records and the failure-mode profiles behind the paper's figures. The format is still being decided.

Runners write their outputs through the `OUT_CSV` / `OUT_JSONL` environment variables.
The adapter scripts write to `results/<benchmark>/<operator>`.
`eval/taxonomy.py` turns per-episode records (JSONL) into the seven failure modes.
