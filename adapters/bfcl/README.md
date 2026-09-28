# BFCL adapter

Upstream: [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla), `berkeley-function-call-leaderboard/` (Apache-2.0).
We used BFCL v3 `multi_turn_base`, with all 200 entries. The patch was made against upstream `main` as of 2026-06-26.

`bfcl_schema_seam.patch` changes the open-model prompting handler, `bfcl_eval/model_handler/local_inference/qwen_fc.py`:

- **Before the query**, it rewrites the union of the entry's `involved_classes` functions into the variant.
  It stores the call map in a thread-local, because each entry runs in one thread.
- **In `decode_execute`**, it decodes the structured `{name: args}` calls into native calls before
  `convert_to_function_call`. The state and response checkers therefore run unchanged.

```bash
git clone https://github.com/ShishirPatil/gorilla && cd gorilla/berkeley-function-call-leaderboard
git apply /path/to/ToolSchemaSpace/adapters/bfcl/bfcl_schema_seam.patch
pip install -e . && pip install -e /path/to/ToolSchemaSpace
export REMOTE_OPENAI_BASE_URL=http://localhost:8000/v1 REMOTE_OPENAI_API_KEY=EMPTY
SCHEMA_DATASET=bfcl_multiturn SCHEMA_VARIANT=0 bfcl generate --model <model>-FC \
    --test-category multi_turn_base --skip-server-setup
bfcl evaluate --model <model>-FC --test-category multi_turn_base
```

It uses the same `SCHEMA_VARIANT` / `SCHEMA_SPEC` / `SCHEMA_CONTROL` variables as the τ² adapter.
With `SCHEMA_CONTROL=hard`, an off-schema call is renamed to `<name>__OFF_SCHEMA`: it errors and does not change the state.
