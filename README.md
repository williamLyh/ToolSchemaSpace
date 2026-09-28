# ToolSchemaSpace

Code and data for **Action-Space Shaping for LLM Agents: Measuring and Mitigating Tool-Schema Bias**.

A tool schema is one interface to an agent's action space. The same executable action can be exposed
through many functionally equivalent tool definitions. This repository contains:

- `toolschema/`, a framework that rewrites a native tool schema into such variants. It decodes every
  call made in a variant back into the native call, so a benchmark's executor and scorer stay unchanged.
  Any change in success is therefore attributable to the interface alone.
- `benchmarks/synthetic/`, the synthetic benchmark.
- `adapters/`, adapters for four existing benchmarks.

```
toolschema/            transformation framework
  operators.py           operator-spec compiler: compile_spec(tools, spec) -> variant catalog + decoder
  schema_adapter.py      merge/split ladder (SCHEMA_VARIANT=0..9) and the env-var entry point
  merge_encodings.py     flat / nested / union argument structures for merged tools
  hierarchy.py           class-level arms over a whole catalog (flat, qualified, dispatch, disclose)
  class_grouping.py      whole-catalog class / random / anti-class groupings
  simenv.py              simulated environment: executes decoded calls, answers protocol calls
  curry_seam.py, indirect_seam.py, disclose_seam.py
                         transaction / reference resolution / schema discovery for real environments
  proxy.py               OpenAI-compatible schema proxy for unmodified agent harnesses
benchmarks/synthetic/  12 domains, 168 tools, 2,748 tasks (1-6 gold calls each)
  data/                  tools.json, queries.jsonl, specs/ (every operator spec), registry.json
  build.py, domains.py   catalog and task construction
  generation/            LLM rewriting of task queries
  write_specs.py         regenerates data/specs
eval/                  agentic runners and the failure-mode taxonomy
adapters/              tau2, bfcl, automationbench, mcp_atlas
tests/                 equivalence gates (a perfect model must score 1.0 on every variant)
results/               reserved for released outputs
```

## Install

```bash
pip install -e .                 # the toolschema package (openai, aiohttp)
```

Everything else runs from the repository root as modules (`python -m ...`).

## Operators

A variant is `S_i = O(m, S_0)`: operator `O`, applied with method `m`, to the native schema `S_0`.
A spec is a JSON list of operator applications. An operator that names a tool absent from a catalog is
ignored, so one spec file can cover all domains; entries carry `group` when tool names repeat across domains.

| operator | spec entry | what the model sees |
|---|---|---|
| merge | `{"op":"merge","tools":[...],"into":"execute","encoding":"flat\|nested\|union"}` | one dispatcher with an `operation` argument |
| split | `{"op":"split_enum","tool":t,"params":[p]}`, `{"op":"split_predicate","tool":t,"param":p,"cuts":[c]}` | one tool per enum value / numeric interval |
| nest / flatten | `{"op":"arg_lower","tool":t,"into":"options","params":[...]}`, `{"op":"arg_lift",...}` | arguments moved into (or out of) an object |
| rename | `{"op":"rename","tool":t,"to":"fn_d8abb4"}` | a different tool name |
| strip descriptions | `{"op":"strip_desc","tool":t}` | no description |
| reorder arguments | `{"op":"shuffle_params","tool":t,"seed":13}` | the same arguments in another order |
| transaction | `{"op":"curry","tool":t}` | `begin_*`, then `set_*` per argument, then `commit_*` |
| reference resolution | `{"op":"indirect","tool":t,"param":p,"values":[...]}` | a resolver that returns a handle to pass instead of the value |
| schema discovery | hierarchy arm `disclose` | `list_methods`, then `invoke` |

The merge/split **ladder** (`SCHEMA_VARIANT=0..9`, with `SCHEMA_K=10`) runs from one dispatcher per domain (0),
through native (5), to every enumerable argument split into names (9).

```python
from toolschema.operators import compile_spec

comp = compile_spec(tools, {"ops": [{"op": "merge", "tools": ["set_window", "set_seat_heat"],
                                     "into": "execute", "encoding": "nested"}]})
comp.tools                                   # variant catalog: [execute]
calls = comp.encode_episode(gold_calls)      # oracle: how a perfect model would act in this variant
native, flags = comp.decode_seq(calls)       # == gold_calls, flags == []
```

## Synthetic benchmark

`benchmarks/synthetic/data/registry.json` maps every variant reported in the paper to its construction:
a ladder step, a spec file, or a hierarchy arm. That covers the rows of Figure 2, the appendix operator
figure, and the merge-encoding and convention-mixture tables. The runners are:

```bash
export REMOTE_OPENAI_BASE_URL=http://localhost:8000/v1 SYN_MODEL=Qwen/Qwen3-4B-Instruct-2507
SCHEMA_VARIANT=0 SCHEMA_CONTROL=hard OUT_JSONL=v0.jsonl python -m eval.run_agentic            # ladder step
SCHEMA_SPEC=benchmarks/synthetic/data/specs/curry.json python -m eval.run_agentic               # operator spec
HIER_ARM=dispatch python -m eval.run_hierarchy                                                   # hierarchy arm
SCHEMA_SPEC=benchmarks/synthetic/data/specs/class_neutral.json python -m eval.run_class_grouping # class grouping
```

- Scoring is state-based: an episode succeeds when the multiset of executed native actions equals the gold.
- Under `SCHEMA_CONTROL=hard`, an off-schema call is rejected with a recoverable error.
- `eval/taxonomy.py` assigns each failed episode one of seven failure modes: livelock, transaction handle,
  rejected-and-stuck, wrong class, under-execution, over-execution, wrong arguments.

## Real benchmarks

| benchmark | source | how the variant is applied |
|---|---|---|
| τ²-bench (airline, retail) | [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) | patch: `adapters/tau2/` |
| BFCL v3 multi-turn | [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla) | patch: `adapters/bfcl/` |
| AutomationBench | [zapier/AutomationBench](https://github.com/zapier/AutomationBench) | schema proxy: `adapters/automationbench/` |
| MCP-Atlas (118-task subset) | [scaleapi/mcp-atlas](https://github.com/scaleapi/mcp-atlas) | schema proxy: `adapters/mcp_atlas/` |

**Patched adapters (τ² and BFCL).** The patch adds two hooks:

- **before each model call**, it swaps the agent's tool list for the variant;
- **after it**, it decodes the calls back to native.

**Proxy adapters (AutomationBench and MCP-Atlas).** These use `toolschema/proxy.py`, which works with any
OpenAI-compatible harness and needs no code change. For example, this starts a proxy on port 8100:
`python -m toolschema.proxy --op merge --upstream http://127.0.0.1:8000/v1 --port 8100`.

Each adapter's README gives the setup.

## Tests

Every variant must be lossless: encoding the gold with the variant's oracle and decoding it must give back
exactly the gold, under hard control, with no rejections.

```bash
python -m tests.test_ladder          # merge/split ladder, all 10 steps
python -m tests.test_ops             # operator specs, including call sequences
python -m tests.test_ops2            # enum splits, grouped merges, surface operators
python -m tests.test_spec_files      # every shipped spec file
python -m tests.test_merge_encodings # flat / nested / union structures and mixtures
python -m tests.test_hierarchy       # hierarchy arms
python -m tests.test_simenv          # simulated-environment semantics
python -m tests.test_proxy [catalog.json task_tools.json]   # schema proxy (default: synthetic catalog)
```

## Citation

```bibtex
@article{liu2026actionspace,
  title   = {Action-Space Shaping for LLM Agents: Measuring and Mitigating Tool-Schema Bias},
  author  = {Liu, Yinhong and Tan, Zhili and Wang, Zilin and Guo, Zhijiang},
  year    = {2026}
}
```

## License

[CC BY 4.0](LICENSE). The adapter patches modify third-party code and carry that code's license
(τ²-bench: MIT; BFCL: Apache-2.0). `adapters/tau2/data/tasks.json` is derived from τ²-bench (MIT).
