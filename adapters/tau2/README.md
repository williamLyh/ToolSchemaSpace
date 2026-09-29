# τ²-bench adapter

Upstream: [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) (MIT). We used the airline (50)
and retail (114) domains, with all tasks.

## Through the schema proxy (all 12 variants)

The agent's model endpoint goes through `toolschema/proxy.py`, and τ²-bench is not modified. The user simulator talks
to the model server directly. This route supports every operator and is the one used for the 12-variant results.

```bash
OP=merge DOMAIN=airline MODEL=Qwen/Qwen3.5-27B UPSTREAM=http://127.0.0.1:8000/v1 TAU2_DIR=/path/to/tau2-bench \
    bash adapters/tau2/run.sh
```

- `classes_<domain>.json` maps each tool to the entity it acts on (reservation, user, flight, order, product,
  utility). These are the classes for class dispatch, namespaced names and schema discovery.
- `cuts.json` holds the interval-split cut for each required numeric argument: the median of the values in τ²'s gold
  actions. Retail has no numeric arguments, so interval split does not apply there.
- The per-task wall-clock limit is 6 h. It only guards against hangs; keep the server's queue short (see the main
  README) so that it never binds.

## Through the patch (earlier results)

`tau2_schema_seam.patch` hooks `src/tau2/utils/llm_utils.py::generate()` directly. It was made against upstream `main`
as of 2026-06-26, and the paper's earlier τ² table used it.

- **Controls:** the hook is driven by `SCHEMA_VARIANT` / `SCHEMA_SPEC`, plus `SCHEMA_TAU2_CURRY`,
  `SCHEMA_TAU2_DISCLOSE` and `SCHEMA_TAU2_INDIRECT` for the protocols, and `SCHEMA_CONTROL`.
- **Difference from the proxy:** under hard control, the hook **drops** an off-schema call without telling the model.
  The proxy (and the synthetic benchmark) answer such a call with a recoverable error.
- **Data:** `data/tasks.json` holds the τ²-bench tasks (MIT, © Sierra) in the form the reference-resolution hook
  reads.

```bash
git clone https://github.com/sierra-research/tau2-bench && cd tau2-bench
git apply /path/to/ToolSchemaSpace/adapters/tau2/tau2_schema_seam.patch
pip install -e . && pip install -e /path/to/ToolSchemaSpace
```
