#!/usr/bin/env bash
# Run AutomationBench under one proxy operator.
#   OP=merge MODEL=Qwen/Qwen3.5-27B UPSTREAM=http://127.0.0.1:8000/v1 AB_DIR=/path/to/AutomationBench \
#     bash adapters/automationbench/run.sh
# OP=native runs the proxy as a pass-through (same code path, unchanged tools).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OP="${OP:-native}"; MODEL="${MODEL:?set MODEL}"; UPSTREAM="${UPSTREAM:-http://127.0.0.1:8000/v1}"
PORT="${PORT:-8100}"; AB_DIR="${AB_DIR:?set AB_DIR to your AutomationBench checkout}"
OUT="${OUT:-$ROOT/results/automationbench/$OP}"; mkdir -p "$(dirname "$OUT")"

cd "$ROOT"
python -m toolschema.proxy --op "$OP" --upstream "$UPSTREAM" --port "$PORT" --classes adapters/automationbench/classes.json \
  --temperature 0 --model "$MODEL" > "$OUT.proxy.log" 2>&1 &
PROXY=$!; trap 'kill $PROXY' EXIT
until curl -sf "http://127.0.0.1:$PORT/stats" > /dev/null; do sleep 1; done

cd "$AB_DIR"
OPENAI_API_KEY=EMPTY .venv/bin/auto-bench --model "$MODEL" --base-url "http://127.0.0.1:$PORT/v1" \
  --toolset limited_zapier --max-concurrent "${CONCURRENCY:-32}" --export-json "$OUT.json"
