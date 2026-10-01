# Results

Evaluation outputs behind the paper. The numbers for every figure are in [raw_tables.md](raw_tables.md), and the
CSVs sit next to each figure.

**How to read the figures.** The first row (**native**) is the baseline score. Every other cell is the change from
native for the same model and benchmark. Shading follows Table 3 of the paper:

- **green** is a gain and **red** a loss;
- a change smaller than 0.03 is not shaded;
- the shade deepens linearly up to a change of 0.45.

## Synthetic benchmark: all models × representative variants

2,748 tasks per cell, hard control, with the same queries and scorer as the paper. The rows are the variants of
Figure 2. The columns are 9 open-weight models and, behind the vertical rule, 2 closed models.

<p align="center"><img src="synthetic/representative_delta.png" alt="Synthetic benchmark: change in success from native, all models x representative variants" width="100%"></p>

## Synthetic benchmark: remaining methods of each operator

The variants of the paper's appendix operator figure, for the 9 open-weight models.

<p align="center"><img src="synthetic/appendix_delta.png" alt="Synthetic benchmark: change in success from native, remaining methods of each operator" width="85%"></p>

## Real benchmarks: Qwen3.5-27B × representative variants

Every benchmark runs through the schema proxy under hard control, with Qwen3.5-27B served by vLLM. Each column uses
the benchmark's own score:

| column | tasks | score |
|---|---|---|
| τ² airline | 50 | mean reward |
| τ² retail | 114 | mean reward |
| BFCL | 200 (v3 `multi_turn_base`) | accuracy |
| AutomationBench | 599 (`limited_zapier`) | mean partial credit |
| MCP-Atlas | 89 (key-free sandbox) | mean claim coverage, judged by Qwen3.8-27B |

- **n/a:** reference resolution is not run on real benchmarks, because it needs value tables drawn from the task
  data. Interval split does not apply to τ² retail, which has no numeric arguments.
- All cells are final.

<p align="center"><img src="real/real_delta.png" alt="Real benchmarks: change in score from native, Qwen3.5-27B x representative variants" width="75%"></p>

## Mitigation: can training remove schema bias?

This is Table 3 of the paper. Qwen3-4B on the synthetic benchmark, 2,748 tasks per cell, over the 12
representative variants. **Base** is the untrained model's success. Every other column is the change from Base on
the same variant:

| column | method |
|---|---|
| Instruction | a one-sentence description of the variant's calling convention, added to the prompt |
| Decoding | the first call is forced to be schema-valid by constrained decoding |
| SFT native | supervised fine-tuning on native-schema traces |
| SFT mixed | supervised fine-tuning on traces rotating over seven variants |
| RL native | on-policy RL (GRPO) on native-schema tasks |
| RL mixed | on-policy RL (GRPO) on tasks rotating over seven variants |

The SFT columns average several seeds, as do the RL mixed columns; RL native is a single run.

Shading is as above. A **red frame** marks a variant that is in that method's training data. The seven variants in
the mixed data are native, fully merged, fully split, namespaced names, transaction, reference resolution and
schema discovery.

<p align="center"><img src="mitigation/mitigation_delta.png" alt="Mitigation: change in success from the untrained model, Qwen3-4B" width="80%"></p>

## Files

| file | contents |
|---|---|
| `raw_tables.md` | all tables above as numbers |
| `synthetic/representative_long.csv` | one row per model × variant: task count, success, change from native, and the rate of each failure mode (livelock, transaction handle, rejected-and-stuck, wrong class, under-execution, over-execution, wrong arguments, other) |
| `synthetic/representative_{success,delta}.csv` | success, and change from native, as variant × model matrices |
| `synthetic/appendix_*` | the same for the appendix variants (open models) |
| `real/real_long.csv` | one row per variant × benchmark: metric, number of tasks, score, change from native, pass rate (AutomationBench) |
| `real/real_{score,delta}.csv` | scores, and change from native, as variant × benchmark matrices |
| `mitigation/mitigation_{score,delta}.csv` | Table 3: absolute success per method, and change from Base, with the methods whose training data contain each variant |
| `*/..._delta.{png,svg}` | the figures |

The failure-mode rates are shares of all tasks; with the success rate they sum to 1, and they are the segments of
Figure 2. All files are regenerated from the raw per-episode records by `analysis/export_open_results.py` and
`analysis/export_real_results.py` and `analysis/export_mitigation_results.py` in the research code base.
