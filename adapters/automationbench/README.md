# AutomationBench adapter

Upstream: [zapier/AutomationBench](https://github.com/zapier/AutomationBench) (MIT), `auto-bench` CLI. We used the `limited_zapier` toolset, with all 570 tasks.

AutomationBench is not patched. The model endpoint goes through the **schema proxy** (`toolschema/proxy.py`),
an OpenAI-compatible server that sits between the harness and the model. It:

- rewrites each request's native `tools` into the variant;
- decodes the model's calls back into native calls;
- answers protocol-internal calls itself, such as transaction begin and set;
- rejects off-schema calls with a recoverable error.

The harness's tools, executor and scorer stay native.

```bash
OP=merge MODEL=Qwen/Qwen3.5-27B UPSTREAM=http://127.0.0.1:8000/v1 AB_DIR=/path/to/AutomationBench \
    bash adapters/automationbench/run.sh
```

The operators are `native`, `merge`, `merge_app`, `split`, `nest`, `rename_opaque`, `rename_ns`, `strip`, `reorder`
and `transaction`. To check that every operator is lossless on this catalog, dump the tool catalog and the
per-task tool lists, then run `python -m tests.test_proxy catalog.json task_tools.json`.
