#!/usr/bin/env bash
# Run MCP-Atlas (118-task subset) under one proxy operator. Needs the MCP-Atlas sandbox
# running (`make run-docker` in the MCP-Atlas checkout, ~15 min to start).
#   OP=merge MODEL=Qwen/Qwen3.5-27B UPSTREAM=http://127.0.0.1:8000/v1 MA_DIR=/path/to/mcp-atlas \
#     bash adapters/mcp_atlas/run.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OP="${OP:-native}"; MODEL="${MODEL:?set MODEL}"; UPSTREAM="${UPSTREAM:-http://127.0.0.1:8000/v1}"
PPORT="${PROXY_PORT:-8200}"; HPORT="${HARNESS_PORT:-3001}"; MA_DIR="${MA_DIR:?set MA_DIR to your mcp-atlas checkout}"
OUT="${OUT:-$ROOT/results/mcp_atlas/$OP}"; mkdir -p "$(dirname "$OUT")"
TASKS="$ROOT/results/mcp_atlas/tasks.csv"
[ -f "$TASKS" ] || python "$ROOT/adapters/mcp_atlas/make_task_csv.py" "$TASKS"

cd "$ROOT"
python -m toolschema.proxy --op "$OP" --upstream "$UPSTREAM" --port "$PPORT" --cuts adapters/mcp_atlas/cuts.json \
  --temperature 0 --model "$MODEL" > "$OUT.proxy.log" 2>&1 &
PIDS=$!; trap 'kill $PIDS' EXIT
until curl -sf "http://127.0.0.1:$PPORT/stats" > /dev/null; do sleep 1; done

# The harness appends /v1 itself, so it gets the proxy root.
(cd "$MA_DIR" && set -a && . ./.env && set +a && cd services/agent-harness && \
  PORT=$HPORT LLM_BASE_URL="http://127.0.0.1:$PPORT" LLM_API_KEY=EMPTY npx tsx src/index.ts) > "$OUT.harness.log" 2>&1 &
PIDS="$PIDS $!"; sleep 20

cd "$MA_DIR"
run() { HARNESS_URL="http://localhost:$HPORT" python run_eval.py --model "openai/$MODEL" --input "$TASKS" \
          --output "$OUT.csv" --concurrency "${CONCURRENCY:-8}" --timeout 10800 --skip-health-check "$@"; }
run
# rerun tasks that ended in a harness timeout, with a longer limit
python "$ROOT/adapters/mcp_atlas/drop_error_rows.py" "$OUT.csv" && run
