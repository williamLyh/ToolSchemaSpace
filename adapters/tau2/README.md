# τ²-bench adapter

Upstream: [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) (MIT).
We used the airline and retail domains, with all tasks in each. The patch was made against upstream `main` as of 2026-06-26.

`tau2_schema_seam.patch` adds two hooks to `src/tau2/utils/llm_utils.py::generate()`. It applies them only to the
agent's calls (`call_name == "agent_response"`), so the user simulator and the scorer stay native. The hooks are:

1. Before the request, swap the agent's tool list for the variant.
2. After the response, decode every tool call back into native calls, before the calls are recorded.
   The action evaluator therefore scores native calls.

The patch also lets the NL-assertion judge run on an OpenAI-compatible endpoint (`TAU2_NL_JUDGE_BASE_URL`,
`TAU2_NL_JUDGE_MODEL`), and it retries a judge answer that is not valid JSON.

```bash
git clone https://github.com/sierra-research/tau2-bench && cd tau2-bench
git apply /path/to/ToolSchemaSpace/adapters/tau2/tau2_schema_seam.patch
pip install -e . && pip install -e /path/to/ToolSchemaSpace
```

Choose the variant with environment variables. When none of them is set, the hooks do nothing.

| variable | effect |
|---|---|
| `SCHEMA_VARIANT=0..9` | merge/split ladder (`SCHEMA_K=10`: 0 coarsest, 5 native, 9 finest) |
| `SCHEMA_SPEC=<file.json>` | any operator spec (see `toolschema/operators.py`) |
| `SCHEMA_TAU2_CURRY=1` | transaction protocol |
| `SCHEMA_TAU2_DISCLOSE=1` | schema discovery protocol |
| `SCHEMA_TAU2_INDIRECT=1` | reference resolution protocol (value tables come from `data/tasks.json`) |
| `SCHEMA_CONTROL=hard\|loose` | reject (`hard`) or pass through (`loose`) calls outside the variant |
| `SCHEMA_DATASET=tau_bench`, `SCHEMA_TAU2_DOMAIN=airline\|retail` | per-domain adapter cache |

`data/tasks.json` holds the τ²-bench tasks (MIT, © Sierra) in the normalized form that the
reference-resolution seam reads.
