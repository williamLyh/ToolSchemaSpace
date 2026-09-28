<div align="center">

# ToolSchemaSpace

**Action-Space Shaping for LLM Agents: Measuring and Mitigating Tool-Schema Bias**

Yinhong Liu, Zhili Tan, Zilin Wang, Zhijiang Guo

University of Cambridge · Yinwang · Huawei · LARK, HKUST (GZ) · HKUST

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/equivalence%20tests-8%20suites-brightgreen.svg)](#tests)

[Overview](#overview) · [Installation](#installation) · [Quick start](#quick-start) · [Operators](#schema-operators) ·
[Synthetic benchmark](#synthetic-benchmark) · [Evaluation](#evaluating-a-model) · [Real benchmarks](#real-benchmarks) ·
[Reproduction](#reproducing-the-paper) · [Citation](#citation)

</div>

<p align="center">
  <img src="assets/figure1.png" alt="Figure 1: one action space, many schema spaces; schema bias across variants; training interventions" width="100%">
</p>

<p align="center"><sub><b>Figure 1.</b>
<b>(A)</b> Every schema variant S<sub>i</sub> is a different representation of the same action space Ω: a call in any of
them decodes to the same native action a* and reaches the same final state.
<b>(B)</b> Schema bias is large and model-specific: the same variant can match the native schema for one model and collapse to
zero for another.
<b>(C)</b> After SFT or RL of Qwen3-4B, native-only data leaves most of the bias in place, while RL on variant-mixture data
mitigates it and keeps the native score.</sub></p>

## Overview

LLM agents are usually evaluated with one fixed tool schema. A tool schema is not the agent's action space, though; it is
one interface to it. The same executable action can be exposed through many functionally equivalent tool definitions:
one dispatcher or many small tools, flat or nested arguments, a single call or a begin/set/commit transaction. An agent
that has learned the task should behave consistently across them. Current agents often do not, which we call
**schema bias**.

This repository provides:

- **An executable transformation framework** (`toolschema/`). Nine operators rewrite a native tool schema into variants,
  and every call made in a variant is decoded back into native actions. Tasks, executable actions and reachable states
  stay fixed, so a change in success is attributable to the interface alone.
- **A synthetic benchmark** (`benchmarks/synthetic/`): 12 domains, 168 tools and 2,748 tasks with programmatic gold calls,
  plus the spec file of every variant in the paper.
- **Adapters for four existing benchmarks** (`adapters/`): τ²-bench, BFCL v3 multi-turn, AutomationBench and MCP-Atlas.
  Two are patches; the other two use an OpenAI-compatible **schema proxy** that works with any agent harness unchanged.
- **Equivalence tests** (`tests/`). Every variant is proven lossless before use: a perfect model scores 1.0 on all of them.

### Findings

- **Schema bias is large and structured.** Across eleven LLMs, including two closed models, and up to 32 variants,
  success ranges from complete failure to 97% depending only on the schema. The hardest representation differs across
  model families, and neither the newest open models nor the closed models are immune.
- **Failures follow the variant.** A given schema tends to break different models in the same way, while how much it
  costs depends on the model.
- **Predicting difficulty needs the target queries.** Reliably ranking variants by difficulty requires running a small
  sample of the target queries; probes that use no task queries are not enough.
- **Training repairs what it sees.** A variant is repaired only when it appears in the training data. On-policy RL does
  so without the tax that SFT imposes on other variants, and the gains extend to new combinations of trained schema
  changes but not to new kinds of change.

## Installation

```bash
git clone https://github.com/williamLyh/ToolSchemaSpace.git
cd ToolSchemaSpace
pip install -e .            # installs the `toolschema` package (depends on openai, aiohttp)
```

Python 3.10 or newer is required. Run the benchmark, evaluation and test modules from the repository root
(`python -m ...`). Evaluation needs an OpenAI-compatible endpoint that supports tool calling, for example vLLM:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 --enable-auto-tool-choice --tool-call-parser hermes --port 8000
```

## Quick start

```python
from toolschema.operators import compile_spec

tools = [...]   # native catalog: [{"name", "description", "parameters"}]
spec = {"ops": [{"op": "merge", "tools": ["set_window", "set_seat_heat"],
                 "into": "execute", "encoding": "nested"}]}

comp = compile_spec(tools, spec)
comp.tools                                  # the variant catalog the model sees: [execute]
calls = comp.encode_episode(gold_calls)     # oracle: how a perfect model acts in this variant
native, flags = comp.decode_seq(calls)      # decoded native calls == gold_calls, flags == []
```

- [`examples/quickstart.py`](examples/quickstart.py) runs offline. It shows merge, split, transaction, rename and the
  merge/split ladder on a two-tool catalog.
- [`examples/agent_loop.py`](examples/agent_loop.py) puts a variant in front of your own agent loop. The model sees the
  variant tools, and `toolschema.simenv.SimEnv` decodes each call, answers protocol calls and rejects off-schema calls.
  Your environment then executes native actions only.

## Schema operators

A schema variant is `S_i = O(m, S_0)`: operator `O`, applied with method `m`, to the native schema `S_0`. A variant is
written as a JSON spec, a list of operator applications. One spec file can cover a whole multi-domain catalog:

- An operator that names a tool absent from a catalog is ignored.
- When tool names repeat across domains, entries carry a `group`.

| operator | spec entry | what the model sees instead of `set_window(position, state)` |
|---|---|---|
| **merge** | `{"op":"merge","tools":[…],"into":"execute","encoding":"flat\|nested\|union"}` | `execute(operation="set_window", …)` |
| **split** | `{"op":"split_enum","tool":t,"params":["state"]}` · `{"op":"split_predicate",…,"cuts":[…]}` | `set_window__state_vent(position)` |
| **nest / flatten** | `{"op":"arg_lower","tool":t,"into":"options","params":[…]}` · `{"op":"arg_lift",…}` | `set_window(options={position, state})` |
| **rename** | `{"op":"rename","tool":t,"to":"fn_d8abb4"}` | `fn_d8abb4(position, state)` |
| **strip descriptions** | `{"op":"strip_desc","tool":t}` | the same tool, without descriptions |
| **reorder arguments** | `{"op":"shuffle_params","tool":t,"seed":13}` | `set_window(state, position)` |
| **transaction** | `{"op":"curry","tool":t}` | `begin_set_window()` → `set_set_window__…(txn_id, …)` → `commit_set_window(txn_id)` |
| **reference resolution** | `{"op":"indirect","tool":t,"param":p,"values":[…]}` | a resolver returns a handle, which is passed instead of the value |
| **schema discovery** | hierarchy arm `disclose` | `list_methods(class)` → `invoke(class, method, arguments)` |

Operators compose within one spec. The **merge/split ladder** (`SCHEMA_VARIANT=0…9`) is a one-dimensional sweep:

- step 0: one dispatcher per domain;
- step 5: the native schema;
- step 9: every enumerable argument split into tool names.

**Class-level arms** operate on a whole catalog: `toolschema/hierarchy.py` (flat, namespaced, class dispatch, schema
discovery) and `toolschema/class_grouping.py` (class, random and anti-class groupings).

## Synthetic benchmark

The benchmark was built to span the variant space. It has many enumerable arguments (so the split operators have room
to act), tool names that repeat across domains (so class information matters), and programmatic gold calls (so every
task is exactly gradable under every variant).

| domains | native tools | tasks | task types | gold calls per task |
|:---:|:---:|:---:|:---:|:---:|
| 12 | 168 (261 enum arguments) | 2,748 | 585 single · 1,747 compound · 416 compositional | 1–6 |

The data card, file formats and construction pipeline are in [`benchmarks/synthetic/README.md`](benchmarks/synthetic/README.md).
[`registry.json`](benchmarks/synthetic/data/registry.json) maps every variant reported in the paper to its construction.
The full tool schema of native and of every representative variant is stored in
[`benchmarks/synthetic/data/schemas/`](benchmarks/synthetic/data/schemas), one file per variant.

## Evaluating a model

```bash
export REMOTE_OPENAI_BASE_URL=http://localhost:8000/v1 SYN_MODEL=Qwen/Qwen3-4B-Instruct-2507

# one ladder step, one operator spec, one hierarchy arm, one class grouping
SCHEMA_VARIANT=0 SCHEMA_DATASET=synthetic SCHEMA_CONTROL=hard OUT_JSONL=v0.jsonl python -m eval.run_agentic
SCHEMA_SPEC=benchmarks/synthetic/data/specs/curry.json SCHEMA_DATASET=synthetic python -m eval.run_agentic
HIER_ARM=dispatch python -m eval.run_hierarchy
SCHEMA_SPEC=benchmarks/synthetic/data/specs/class_neutral.json python -m eval.run_class_grouping
```

- **Episodes.** An episode runs up to 16 turns. The model's calls are executed by the simulated environment.
- **Scoring.** An episode succeeds when the multiset of executed native actions equals the gold calls.
- **Control modes.** `SCHEMA_CONTROL=hard` (the paper's setting) rejects off-schema calls with a recoverable error.
  `loose` executes them anyway.
- **Outputs.** `OUT_CSV` receives per-domain aggregates; `OUT_JSONL` receives one record per episode.

**Failure taxonomy.** `eval/taxonomy.py` assigns every failed episode one of seven modes: livelock, transaction handle,
rejected-and-stuck, wrong class, under-execution, over-execution or wrong arguments.

```python
import json
from eval.taxonomy import profile
rows = [json.loads(line) for line in open("v0.jsonl")]
profile(rows)   # {'n': .., 'success': .., 'livelock': .., 'txn_handle': .., ...}
```

Other environment variables: `SYN_WORKERS` (concurrency), `SYN_TEMP`, `SYN_SEED`, `SYN_TOOLCHOICE`,
`SYN_EXTRA_BODY` (for example `{"chat_template_kwargs":{"enable_thinking":false}}`), `MAX_TURNS`, and `SYN_API=responses`
for the OpenAI Responses API.

## Real benchmarks

| benchmark | tasks used | upstream | integration |
|---|---|---|---|
| τ²-bench | airline, retail (all tasks) | [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) | patch · [`adapters/tau2`](adapters/tau2) |
| BFCL v3 | `multi_turn_base` (200) | [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla) | patch · [`adapters/bfcl`](adapters/bfcl) |
| AutomationBench | `limited_zapier` (570) | [zapier/AutomationBench](https://github.com/zapier/AutomationBench) | schema proxy · [`adapters/automationbench`](adapters/automationbench) |
| MCP-Atlas | 118 tasks with public servers | [scaleapi/mcp-atlas](https://github.com/scaleapi/mcp-atlas) | schema proxy · [`adapters/mcp_atlas`](adapters/mcp_atlas) |

**Patches (τ², BFCL).** Each patch adds two hooks to the benchmark's model call:

- **before the call**, it swaps the agent's tool list for the variant;
- **after the call**, it decodes the model's calls back to native ones.

The benchmark's executor and scorer are untouched.

**Schema proxy (AutomationBench, MCP-Atlas).** The proxy is an OpenAI-compatible server between the harness and the
model; the harness needs no code change. It:

- rewrites each request's native `tools` into the variant;
- decodes every call the model makes back into native calls;
- answers protocol-internal calls itself, such as transaction begin and set;
- rejects off-schema calls with a recoverable error;
- keeps the model's conversation history in the variant vocabulary.

```bash
python -m toolschema.proxy --op merge --upstream http://127.0.0.1:8000/v1 --port 8100
# then point the harness's OpenAI base URL at http://127.0.0.1:8100/v1
```

The proxy supports `native`, `merge`, `merge_app`, `split`, `nest`, `rename_opaque`, `rename_ns`, `strip`, `reorder` and
`transaction`. Each adapter directory has setup instructions and a run script.

## Reproducing the paper

`eval/run_registry.py` runs every synthetic variant of a figure or table with the right runner and settings:

```bash
python -m eval.run_registry --list                                    # all variants and how each is built
python -m eval.run_registry --figure "Figure 2" --out results/<model> # the representative variants of Figure 2
python -m eval.run_registry --figure "Appendix" --out results/<model> # every appendix figure and table
```

| paper artefact | variants | how to run |
|---|---|---|
| Figure 1 (B), Figure 2: representative variants | 13 | `run_registry --figure "Figure 2"` |
| Appendix figure: remaining methods of each operator (`operator · method`) | 19 | `run_registry --figure "remaining methods"` |
| Appendix: argument structure of a merged tool | 6 | `run_registry --figure "argument structure"` |
| Appendix: convention-mixture test | 12 | `run_registry --figure "convention mixture"` |
| Real-benchmark results | 10 operators × 4 benchmarks | [`adapters/`](adapters) |

Model outputs and figure data will be released under [`results/`](results).

## Tests

A variant is admitted only if it is **lossless**. The gold calls of every task are encoded with the variant's oracle
and decoded back under hard control; the result must be exactly the gold, with no rejections. The suites below check
this for every construction the paper uses.

```bash
python -m tests.test_ladder            # merge/split ladder, 10 steps          (78,180 decodes)
python -m tests.test_ops               # operator specs incl. call sequences
python -m tests.test_ops2              # enum splits, grouped merges, surface operators
python -m tests.test_spec_files        # every shipped spec file              (137,400 episodes)
python -m tests.test_merge_encodings   # flat / nested / union structures and mixtures
python -m tests.test_hierarchy         # class-level arms
python -m tests.test_simenv            # the same, through the simulated environment
python -m tests.test_proxy             # the schema proxy, all 10 operators
```

`tests.test_proxy catalog.json task_tools.json` runs the same check on any real tool catalog. On AutomationBench it
covers 48,650 round trips.

## Repository structure

```
toolschema/                transformation framework (pip package)
├── operators.py             spec compiler: compile_spec → variant catalog, oracle encoder, sequence decoder
├── schema_adapter.py        merge/split ladder; SCHEMA_VARIANT / SCHEMA_SPEC entry point for benchmark hooks
├── merge_encodings.py       flat / nested / union argument structures
├── hierarchy.py             class-level arms over a whole catalog
├── class_grouping.py        class / random / anti-class groupings
├── simenv.py                simulated environment (decode, protocol calls, hard control)
├── curry_seam.py · indirect_seam.py · disclose_seam.py   multi-call protocols for real environments
└── proxy.py                 OpenAI-compatible schema proxy
benchmarks/synthetic/      the synthetic benchmark
├── data/                    tools.json · queries.jsonl · specs/ · registry.json · schemas/
├── domains.py · build.py    catalog and task construction
├── generation/              LLM rewriting of task queries
└── write_specs.py           regenerates data/specs
eval/                      runners, registry runner, failure taxonomy
adapters/                  tau2 · bfcl · automationbench · mcp_atlas
examples/                  quickstart.py · agent_loop.py
tests/                     equivalence suites
results/                   released outputs (reserved)
```

## Citation

If you use the framework or the benchmark, please cite:

```bibtex
@article{liu2026actionspace,
  title   = {Action-Space Shaping for {LLM} Agents: Measuring and Mitigating Tool-Schema Bias},
  author  = {Liu, Yinhong and Tan, Zhili and Wang, Zilin and Guo, Zhijiang},
  year    = {2026}
}
```

## License

This repository is released under [CC BY 4.0](LICENSE). Third-party material keeps its own license:

- the adapter patches modify τ²-bench (MIT) and BFCL (Apache-2.0);
- `adapters/tau2/data/tasks.json` is derived from τ²-bench (MIT);
- AutomationBench and MCP-Atlas (both MIT) are not redistributed and are fetched from their upstream repositories.

## Acknowledgements

We thank the authors of τ²-bench, BFCL, AutomationBench and MCP-Atlas for releasing their benchmarks, and the vLLM
team for the inference engine used throughout.
