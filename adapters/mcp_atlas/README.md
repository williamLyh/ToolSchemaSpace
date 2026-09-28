# MCP-Atlas adapter

Upstream: [scaleapi/mcp-atlas](https://github.com/scaleapi/mcp-atlas) (MIT). The dataset is [`ScaleAI/MCP-Atlas`](https://huggingface.co/datasets/ScaleAI/MCP-Atlas).

We ran the 118 tasks whose MCP servers all work in the public sandbox without private credentials.
Their ids are in `task_ids.txt`, and `make_task_csv.py` rebuilds the CSV from the Hugging Face dataset.

The model endpoint goes through the schema proxy, as for AutomationBench (see `../automationbench/README.md`).
The MCP-Atlas agent harness appends `/v1` to its base URL, so point it at the proxy root.
The proxy also serves `/v1/v1/chat/completions`.

```bash
# in the MCP-Atlas checkout: fill .env (LLM_* can stay empty, run.sh sets them), then `make run-docker`
OP=merge MODEL=Qwen/Qwen3.5-27B UPSTREAM=http://127.0.0.1:8000/v1 MA_DIR=/path/to/mcp-atlas \
    bash adapters/mcp_atlas/run.sh
```

Rows that ended in a harness timeout (`ERROR: timeout`) are moved to `<out>.csv.errors.csv`
(`drop_error_rows.py`) and rerun once with a 3600 s limit. Score with MCP-Atlas's own judge script.
