#!/usr/bin/env bash
# Run BFCL v3 multi_turn_base under one schema variant through the schema proxy.
# Needs bfcl_proxy_client.patch applied to BFCL (registry entry `schema-proxy-FC` + OPENAI_TIMEOUT).
#   OP=merge MODEL=Qwen/Qwen3.5-27B UPSTREAM=http://127.0.0.1:8000/v1 BFCL_DIR=/path/to/berkeley-function-call-leaderboard \
#     bash adapters/bfcl/run.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OP="${OP:-native}"; MODEL="${MODEL:?set MODEL}"; UPSTREAM="${UPSTREAM:-http://127.0.0.1:8000/v1}"
PORT="${PORT:-8300}"; BFCL_DIR="${BFCL_DIR:?set BFCL_DIR}"
OUT="${OUT:-$ROOT/results/bfcl/$OP}"; mkdir -p "$OUT"

cd "$ROOT"
python -m toolschema.proxy --op "$OP" --upstream "$UPSTREAM" --port "$PORT" --temperature 0 --model "$MODEL" \
  --classes adapters/bfcl/classes.json --cuts adapters/bfcl/cuts.json > "$OUT/proxy.log" 2>&1 &
PROXY=$!; trap 'kill $PROXY' EXIT
until curl -sf "http://127.0.0.1:$PORT/stats" > /dev/null; do sleep 1; done

cd "$BFCL_DIR"
# BFCL's .env is loaded with override=True: an empty OPENAI_BASE_URL= line there would blank this one.
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1" OPENAI_API_KEY=EMPTY OPENAI_TIMEOUT=10800
bfcl generate --model schema-proxy-FC --test-category multi_turn_base --num-threads "${CONCURRENCY:-16}" --result-dir "$OUT/result"
bfcl evaluate --model schema-proxy-FC --test-category multi_turn_base --result-dir "$OUT/result" --score-dir "$OUT/score"
