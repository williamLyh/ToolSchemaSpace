#!/usr/bin/env bash
# Run tau2-bench (one domain) under one schema variant through the schema proxy.
#   OP=merge DOMAIN=airline MODEL=Qwen/Qwen3.5-27B UPSTREAM=http://127.0.0.1:8000/v1 TAU2_DIR=/path/to/tau2-bench \
#     bash adapters/tau2/run.sh
# The agent talks to the proxy; the user simulator talks to the model server directly.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OP="${OP:-native}"; DOMAIN="${DOMAIN:-airline}"; MODEL="${MODEL:?set MODEL}"
UPSTREAM="${UPSTREAM:-http://127.0.0.1:8000/v1}"; PORT="${PORT:-8400}"; TAU2_DIR="${TAU2_DIR:?set TAU2_DIR}"
OUT="${OUT:-$ROOT/results/tau2/$OP/$DOMAIN}"; mkdir -p "$(dirname "$OUT")"

cd "$ROOT"
python -m toolschema.proxy --op "$OP" --upstream "$UPSTREAM" --port "$PORT" --temperature 0 --model "$MODEL" \
  --classes "adapters/tau2/classes_$DOMAIN.json" --cuts adapters/tau2/cuts.json > "$OUT.proxy.log" 2>&1 &
PROXY=$!; trap 'kill $PROXY' EXIT
until curl -sf "http://127.0.0.1:$PORT/stats" > /dev/null; do sleep 1; done

cd "$TAU2_DIR"
tau2 run --domain "$DOMAIN" \
  --agent-llm "openai/$MODEL" --agent-llm-args "{\"temperature\":0.0,\"api_base\":\"http://127.0.0.1:$PORT/v1\",\"api_key\":\"EMPTY\",\"timeout\":900}" \
  --user-llm  "openai/$MODEL" --user-llm-args  "{\"temperature\":0.0,\"api_base\":\"$UPSTREAM\",\"api_key\":\"EMPTY\",\"timeout\":300}" \
  --num-trials 1 --max-concurrency "${CONCURRENCY:-16}" --timeout 21600 --seed 42 --save-to "$OUT" --auto-resume
