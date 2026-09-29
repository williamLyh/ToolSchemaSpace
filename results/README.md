# Results

Evaluation outputs behind the paper. Everything here is regenerated from the raw per-episode records by
`analysis/export_open_results.py` in the research code base.

## Synthetic benchmark: all models × representative variants

2,748 tasks per cell, hard control, with the same queries and scorer as the paper. Rows follow Figure 2. Columns are 9 open-weight
models, then 2 closed models behind the vertical rule.

The **native** row gives each model's exact success rate, which is the baseline. Every other cell is the change from
that model's native score. Shading follows Table 3 of the paper:

- **green** is a gain and **red** a loss;
- a change smaller than 0.03 (about 2.5 standard errors of a difference) is not shaded;
- the shade deepens linearly up to a change of 0.45.

<p align="center"><img src="synthetic/representative_delta.png" alt="Change in success from native, all models x representative variants" width="100%"></p>

| variant | Qwen3-4B | Qwen3-30B-A3B | Qwen2.5-7B | Qwen2.5-14B | Qwen2.5-32B | Llama-3.1-8B | Gemma-4-12B | Qwen3.5-27B | Qwen3.5-35B-A3B | GLM-5.3-Flash | gpt-6-luna |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native (success) | **0.77** | **0.78** | **0.54** | **0.77** | **0.76** | **0.75** | **0.78** | **0.84** | **0.83** | **0.76** | **0.76** |
| fully merged | -0.41 | -0.71 | -0.19 | -0.23 | -0.01 | -0.04 | -0.78 | 0.00 | -0.04 | -0.05 | +0.01 |
| class dispatch | -0.11 | -0.11 | -0.41 | -0.24 | -0.03 | -0.15 | -0.75 | -0.06 | -0.08 | -0.15 | -0.13 |
| fully split | -0.01 | -0.01 | +0.06 | -0.01 | 0.00 | -0.05 | -0.01 | 0.00 | -0.01 | 0.00 | 0.00 |
| interval split | -0.01 | +0.01 | -0.14 | -0.01 | +0.01 | -0.25 | 0.00 | 0.00 | 0.00 | +0.01 | +0.01 |
| nested args | 0.00 | 0.00 | 0.00 | +0.01 | 0.00 | 0.00 | +0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| namespaced names | -0.08 | -0.07 | -0.03 | -0.05 | -0.01 | -0.08 | -0.10 | -0.04 | -0.03 | -0.12 | -0.06 |
| strip descriptions | +0.01 | -0.01 | +0.01 | 0.00 | -0.01 | -0.02 | -0.01 | 0.00 | 0.00 | -0.01 | -0.01 |
| reorder arguments | 0.00 | 0.00 | -0.02 | +0.01 | 0.00 | -0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| transaction | -0.49 | -0.06 | -0.29 | -0.67 | +0.01 | -0.75 | -0.18 | -0.08 | -0.05 | 0.00 | -0.02 |
| reference resolution | 0.00 | +0.08 | -0.15 | -0.16 | -0.13 | -0.21 | +0.14 | +0.12 | +0.12 | +0.17 | +0.19 |
| schema discovery | -0.70 | -0.66 | -0.51 | -0.72 | -0.47 | -0.71 | -0.01 | -0.05 | -0.08 | -0.09 | -0.04 |
| composed | -0.45 | -0.34 | -0.41 | -0.42 | -0.30 | -0.47 | -0.56 | +0.04 | +0.03 | +0.07 | +0.11 |

The appendix view covers the remaining methods of each operator (19 variants × the 9 open models) in the same
format: [`synthetic/appendix_delta.png`](synthetic/appendix_delta.png).

## Files

| file | contents |
|---|---|
| `synthetic/representative_long.csv` | one row per model × variant: task count, success, change from native, and the rate of each failure mode (livelock, transaction handle, rejected-and-stuck, wrong class, under-execution, over-execution, wrong arguments, other) |
| `synthetic/representative_success.csv` | success rate, variant × model |
| `synthetic/representative_delta.csv` | native success, then change from native, variant × model |
| `synthetic/representative_delta.{png,svg}` | the shaded table above |
| `synthetic/appendix_*` | the same for the 19 appendix variants (open models) |
| `synthetic/tables.md` | both tables as Markdown |

Failure-mode rates are shares of all tasks; together with the success rate they sum to 1. They are the segments of
Figure 2.

## Real benchmarks

τ²-bench, BFCL v3 multi-turn, AutomationBench and MCP-Atlas with Qwen3.5-27B under the 12 representative variants
(all but reference resolution) will be added here when the runs finish.
