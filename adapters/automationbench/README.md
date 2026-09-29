# AutomationBench adapter

Upstream: [zapier/AutomationBench](https://github.com/zapier/AutomationBench) (MIT), `auto-bench` CLI. We used the
`limited_zapier` toolset (per-task tool lists).

AutomationBench is not modified. Its model endpoint goes through the schema proxy (`toolschema/proxy.py`):

```bash
OP=merge MODEL=Qwen/Qwen3.5-27B UPSTREAM=http://127.0.0.1:8000/v1 AB_DIR=/path/to/AutomationBench \
    bash adapters/automationbench/run.sh
```

- **Classes:** `classes.json` maps each tool to its app, all 47 of them (from `automationbench/tools/zapier/<app>/`).
  Grouping by the prefix before the first underscore is wrong: it would put Google Sheets, Ads, Calendar and Drive into
  one class.
- **Tasks:** the dataset has **600 tasks**; the progress bar reports "570 groups". Task
  `hr.performance_feedback_logging` names a tool (`slack_find_user_by_id`) that `limited_zapier` does not provide,
  and AutomationBench aborts the whole run when it reaches that task. Exclude it with `--tasks`; our results cover
  the other 599.
- **Chunks:** AutomationBench keeps results in memory until the run ends. We ran each arm in chunks
  (`--skip`/`--num-examples`, or `--tasks`) and merged the exported JSON files.
- **Official toolsets:** the official `zapier` and `api` toolsets were run without the proxy as controls.
