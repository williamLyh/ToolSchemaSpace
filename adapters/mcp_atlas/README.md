# MCP-Atlas adapter

Upstream: [scaleapi/mcp-atlas](https://github.com/scaleapi/mcp-atlas) (MIT). The dataset is
[`ScaleAI/MCP-Atlas`](https://huggingface.co/datasets/ScaleAI/MCP-Atlas).

## Sandbox

The sandbox runs 20 MCP servers that need no API keys. Seven of them are Python servers started with `uvx`
(arxiv, calculator, cli-mcp-server, ddg-search, fetch, git, pubmed). They are pinned to versions that break on the
`mcp` 2.x API, which `uvx` now resolves. So start the sandbox with the `uvx` wrapper in `sandbox/`, which adds the
constraint `mcp<2`:

```bash
docker run --rm --name mcpatlas_sandbox -p 1984:1984 --env-file .env \
    -v $PWD/adapters/mcp_atlas/sandbox:/etc/uvconf:ro \
    -v $PWD/adapters/mcp_atlas/sandbox/uvx:/usr/local/bin/uvx:ro \
    agent-environment:latest
curl localhost:1984/enabled-servers          # expect "online": 20
```

The sandbox starts its servers with a clean environment, so `UV_*` variables do not reach them; the wrapper does.
Behind a restricted network you may also need a PyPI mirror (add `--default-index <mirror>` to the wrapper) and a
GitHub mirror for pubmed (a `url.<mirror>.insteadOf` rule in a mounted `/root/.gitconfig`).

## Tasks

We keep a task only if **every tool its reference trajectory calls is exposed by the sandbox**. That leaves 89 of
the 500 tasks; their ids are in `task_ids.txt`, and `make_task_csv.py` rebuilds the CSV from the Hugging Face
dataset. Selecting tasks by the tools *offered* to the model is not enough, because tasks list distractor tools
from servers that need keys.

## Running

```bash
OP=merge MODEL=Qwen/Qwen3.5-27B UPSTREAM=http://127.0.0.1:8000/v1 MA_DIR=/path/to/mcp-atlas \
    bash adapters/mcp_atlas/run.sh
```

- **Base URL:** the agent harness appends `/v1` to its base URL, so point it at the proxy root (the proxy also
  serves `/v1/v1/chat/completions`).
- **Cuts:** `cuts.json` holds the interval-split cuts, the medians of the numeric values in the reference
  trajectories. Classes are the server names, i.e. the prefix before the first `_`.
- **Timeouts:** rows that end in a harness timeout are moved to `<out>.csv.errors.csv` (`drop_error_rows.py`) and
  rerun once.
- **Scoring:** use MCP-Atlas's own judge script.
