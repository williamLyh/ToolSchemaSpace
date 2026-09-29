# BFCL adapter

Upstream: [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla), `berkeley-function-call-leaderboard/`
(Apache-2.0). We used BFCL v3 `multi_turn_base`, with all 200 entries. The patches were made against upstream `main`
as of 2026-06-26.

## Through the schema proxy (all 12 variants)

BFCL's generic OpenAI Chat Completions function-calling client talks to the proxy. `bfcl_proxy_client.patch` makes
two small changes:

- it registers that client under the name `schema-proxy-FC` (the proxy's `--model` sets the real model);
- it lets `OPENAI_TIMEOUT` raise the client's 600 s request timeout, because an agent turn through the proxy can
  include several model calls.

```bash
cd gorilla/berkeley-function-call-leaderboard
git apply /path/to/ToolSchemaSpace/adapters/bfcl/bfcl_proxy_client.patch && pip install -e .
OP=merge MODEL=Qwen/Qwen3.5-27B UPSTREAM=http://127.0.0.1:8000/v1 BFCL_DIR=$PWD bash /path/to/ToolSchemaSpace/adapters/bfcl/run.sh
```

- `classes.json` maps each function to its BFCL API class (GorillaFileSystem, TwitterAPI, …).
- `cuts.json` holds the interval-split cut for each required numeric argument: the median of the ground-truth values.
  Identifiers (names ending in "id") are not split.
- BFCL loads its `.env` with `override=True`. An empty `OPENAI_BASE_URL=` line there would replace the proxy URL, so
  remove it.

## Through the patch (earlier results)

`bfcl_schema_seam.patch` changes the open-model prompting handler (`local_inference/qwen_fc.py`). It rewrites the
entry's functions into the variant and decodes the model's calls before execution, using the same `SCHEMA_VARIANT` /
`SCHEMA_SPEC` / `SCHEMA_CONTROL` variables. It covers single-call variants only; multi-call protocols such as
transaction and schema discovery need the proxy.
