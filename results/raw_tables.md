# Raw tables

The numbers behind the figures in [README.md](README.md). In each table the first row is the native score and every other row is the change from native. Blank = run not finished; n/a = not applicable.

## Synthetic benchmark: all models × representative variants (success rate)

| variant | Qwen3-4B | Qwen3-30B-A3B | Qwen2.5-7B | Qwen2.5-14B | Qwen2.5-32B | Llama-3.1-8B | Gemma-4-12B | Qwen3.5-27B | Qwen3.5-35B-A3B | GLM-5.3-Flash | gpt-6-luna |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native (success) | **0.773** | **0.779** | **0.539** | **0.770** | **0.761** | **0.750** | **0.785** | **0.844** | **0.833** | **0.755** | **0.761** |
| fully merged | -0.409 | -0.713 | -0.186 | -0.226 | -0.009 | -0.044 | -0.785 | -0.004 | -0.044 | -0.046 | +0.009 |
| class dispatch | -0.112 | -0.115 | -0.412 | -0.242 | -0.029 | -0.147 | -0.746 | -0.061 | -0.078 | -0.151 | -0.126 |
| fully split | -0.013 | -0.012 | +0.061 | -0.009 | +0.001 | -0.047 | -0.011 | -0.003 | -0.006 | -0.004 | -0.003 |
| interval split | -0.015 | +0.011 | -0.141 | -0.014 | +0.012 | -0.249 | +0.001 | -0.002 | -0.001 | +0.007 | +0.006 |
| nested args | -0.002 | +0.002 | -0.004 | +0.011 | +0.003 | -0.002 | +0.007 | -0.002 | +0.000 | -0.003 | +0.004 |
| namespaced names | -0.076 | -0.070 | -0.035 | -0.051 | -0.015 | -0.082 | -0.098 | -0.039 | -0.028 | -0.119 | -0.061 |
| strip descriptions | +0.010 | -0.014 | +0.013 | -0.001 | -0.009 | -0.024 | -0.007 | +0.001 | +0.003 | -0.011 | -0.014 |
| reorder arguments | +0.004 | +0.001 | -0.024 | +0.009 | -0.001 | -0.016 | -0.001 | +0.000 | +0.001 | -0.003 | +0.004 |
| transaction | -0.493 | -0.059 | -0.286 | -0.667 | +0.010 | -0.747 | -0.185 | -0.083 | -0.045 | +0.003 | -0.016 |
| reference resolution | +0.001 | +0.084 | -0.147 | -0.165 | -0.132 | -0.208 | +0.143 | +0.124 | +0.123 | +0.172 | +0.195 |
| schema discovery | -0.702 | -0.664 | -0.506 | -0.722 | -0.475 | -0.708 | -0.015 | -0.045 | -0.083 | -0.086 | -0.044 |
| composed | -0.447 | -0.344 | -0.412 | -0.417 | -0.299 | -0.474 | -0.563 | +0.041 | +0.030 | +0.074 | +0.108 |

## Synthetic benchmark: open models × remaining methods of each operator (success rate)

| variant | Qwen3-4B | Qwen3-30B-A3B | Qwen2.5-7B | Qwen2.5-14B | Qwen2.5-32B | Llama-3.1-8B | Gemma-4-12B | Qwen3.5-27B | Qwen3.5-35B-A3B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native (success) | **0.773** | **0.779** | **0.539** | **0.770** | **0.761** | **0.750** | **0.785** | **0.844** | **0.833** |
| merge · domain 0.29× | -0.354 | -0.300 | -0.199 | -0.144 | -0.133 | -0.258 | -0.748 | -0.007 | -0.026 |
| merge · domain 0.46× | -0.352 | -0.291 | -0.289 | -0.163 | -0.128 | -0.325 | -0.684 | -0.006 | -0.023 |
| merge · domain 0.69× | -0.284 | -0.230 | -0.280 | -0.118 | -0.096 | -0.268 | -0.570 | -0.006 | -0.006 |
| merge · domain 0.85× | -0.139 | -0.151 | -0.160 | -0.039 | -0.030 | -0.154 | -0.402 | -0.001 | -0.000 |
| merge · domain, nested | -0.450 | -0.734 | -0.198 | -0.409 | -0.331 | -0.296 | -0.553 | -0.643 | -0.605 |
| merge · domain, union | -0.737 | -0.747 | -0.364 | -0.381 | -0.214 | -0.253 | -0.033 | -0.021 | -0.056 |
| merge · semantic groups | -0.115 | -0.295 | -0.181 | -0.112 | -0.051 | -0.145 | -0.782 | -0.002 | -0.021 |
| merge · random groups | -0.130 | -0.354 | -0.258 | -0.144 | -0.071 | -0.255 | -0.773 | -0.008 | -0.018 |
| merge · anti-semantic groups | -0.209 | -0.254 | -0.287 | -0.158 | -0.089 | -0.327 | -0.743 | -0.010 | -0.014 |
| merge · catalog, neutral names | -0.119 | -0.096 | -0.440 | -0.310 | -0.020 | -0.147 | -0.780 | -0.066 | -0.071 |
| merge · catalog, random classes | -0.537 | -0.441 | -0.506 | -0.589 | -0.326 | -0.612 | -0.773 | -0.074 | -0.124 |
| merge · catalog, anti-classes | -0.576 | -0.492 | -0.500 | -0.602 | -0.305 | -0.626 | -0.761 | -0.094 | -0.207 |
| native · whole catalog | -0.067 | -0.059 | +0.015 | -0.048 | -0.013 | -0.087 | -0.105 | -0.037 | -0.030 |
| split · enum 1.5× | +0.005 | +0.003 | -0.086 | -0.004 | +0.003 | -0.005 | -0.001 | -0.004 | -0.004 |
| split · enum 2.9× | +0.009 | +0.011 | -0.088 | -0.001 | +0.002 | -0.020 | -0.008 | -0.005 | -0.006 |
| split · enum 5.4× | +0.012 | +0.006 | -0.021 | +0.003 | +0.009 | -0.004 | -0.006 | -0.000 | +0.001 |
| split · one enum | +0.011 | +0.001 | -0.009 | +0.000 | +0.003 | -0.004 | -0.010 | +0.004 | +0.003 |
| split · all enums | -0.004 | +0.003 | +0.020 | -0.002 | +0.005 | -0.043 | -0.014 | +0.000 | +0.002 |
| rename · opaque | -0.022 | -0.017 | -0.273 | -0.031 | +0.009 | -0.038 | -0.001 | -0.010 | -0.005 |

## Real benchmarks: Qwen3.5-27B × representative variants

| variant | τ² airline | τ² retail | BFCL | AutomationBench | MCP-Atlas |
|---|---:|---:|---:|---:|---:|
| native (score) | **0.780** | **0.474** | **0.730** | **0.532** | **0.538** |
| fully merged | -0.160 | -0.298 | -0.670 | -0.442 | -0.261 |
| class dispatch | -0.020 | -0.096 | -0.415 | -0.289 | -0.400 |
| fully split | +0.000 | +0.026 | -0.010 | -0.071 | -0.062 |
| interval split | -0.020 | n/a | -0.040 | -0.232 | -0.001 |
| nested args | +0.040 | +0.026 | -0.040 | -0.011 | -0.094 |
| namespaced names | +0.080 | -0.026 | -0.050 | -0.018 | -0.055 |
| strip descriptions | +0.020 | +0.053 | -0.230 | -0.110 | -0.075 |
| reorder arguments | -0.060 | -0.053 | -0.005 | -0.006 | -0.051 |
| transaction | -0.100 | -0.096 | -0.025 | -0.040 | +0.005 |
| reference resolution | n/a | n/a | n/a | n/a | n/a |
| schema discovery | -0.080 | -0.009 | +0.020 | -0.053 | -0.091 |
| composed | +0.020 | +0.035 | -0.315 | -0.268 | -0.264 |

## Real benchmarks: Qwen3-4B (zero-shot) × representative variants

| variant | τ² airline | τ² retail | BFCL | AutomationBench | MCP-Atlas |
|---|---:|---:|---:|---:|---:|
| native (score) | **0.260** | **0.254** | **0.330** | **0.166** | **0.172** |
| fully merged | +0.140 | -0.175 | -0.330 | -0.166 | -0.143 |
| class dispatch | +0.000 | -0.061 | -0.305 | -0.047 | -0.118 |
| fully split | -0.020 | +0.018 | -0.005 | -0.046 | -0.059 |
| interval split | +0.080 | n/a | -0.085 | -0.065 | -0.048 |
| nested args | +0.080 | +0.061 | -0.050 | -0.057 | -0.033 |
| namespaced names | +0.100 | -0.026 | -0.230 | -0.006 | -0.049 |
| strip descriptions | +0.020 | +0.009 | -0.095 | -0.082 | -0.070 |
| reorder arguments | +0.000 | +0.044 | -0.005 | -0.001 | -0.047 |
| transaction | -0.020 | -0.044 | -0.060 | -0.123 | -0.070 |
| reference resolution | n/a | n/a | n/a | n/a | n/a |
| schema discovery | +0.080 | -0.149 | -0.310 | -0.125 | -0.138 |
| composed | +0.140 | -0.175 | -0.305 | -0.135 | -0.124 |

## Real benchmarks: Qwen3-4B after RL mixed7 × representative variants

| variant | τ² airline | τ² retail | BFCL | AutomationBench | MCP-Atlas |
|---|---:|---:|---:|---:|---:|
| native (score) | **0.380** | **0.333** | **0.320** | **0.135** | **0.071** |
| fully merged | -0.040 | -0.044 | -0.315 | -0.018 | -0.028 |
| class dispatch | -0.020 | -0.114 | -0.280 | +0.011 | -0.043 |
| fully split | -0.060 | -0.009 | +0.000 | -0.012 | +0.008 |
| interval split | -0.060 | n/a | -0.055 | -0.083 | +0.012 |
| nested args | -0.100 | -0.018 | +0.005 | -0.020 | -0.017 |
| namespaced names | -0.020 | -0.018 | -0.200 | +0.002 | -0.005 |
| strip descriptions | -0.040 | -0.061 | -0.110 | -0.038 | +0.000 |
| reorder arguments | +0.040 | -0.018 | -0.030 | +0.008 | +0.014 |
| transaction | -0.080 | -0.123 | -0.065 | -0.053 | +0.006 |
| reference resolution | n/a | n/a | n/a | n/a | n/a |
| schema discovery | -0.040 | -0.079 | -0.225 | -0.062 | -0.021 |
| composed | +0.020 | -0.254 | -0.295 | -0.098 | -0.016 |

## Mitigation: Qwen3-4B, synthetic benchmark (paper Table 3)

Base = success of the untrained model; other columns = change from Base. † = the variant is in that method's training data; – = not applicable.

| variant | Base | Instruction | Decoding | SFT native | SFT mixed | RL native | RL mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native | 0.773 | – | +0.003 | +0.021 † | -0.075 † | +0.054 † | +0.042 † |
| Fully merged | 0.364 | +0.278 | +0.289 | +0.108 | +0.343 † | +0.111 | +0.435 † |
| Class dispatch | 0.661 | +0.008 | -0.018 | -0.260 | -0.005 | +0.049 | +0.055 |
| Fully split | 0.760 | +0.004 | -0.003 | +0.045 | -0.061 † | +0.060 | +0.047 † |
| Interval split | 0.759 | +0.005 | -0.043 | -0.090 | -0.073 | +0.053 | +0.048 |
| Nested args | 0.771 | +0.005 | +0.006 | +0.020 | -0.079 | +0.056 | +0.045 |
| Namespaced names | 0.697 | +0.002 | +0.001 | +0.031 | +0.008 † | +0.066 | +0.056 † |
| Strip descriptions | 0.783 | -0.001 | +0.001 | -0.003 | -0.098 | +0.039 | +0.028 |
| Reorder arguments | 0.777 | +0.008 | -0.003 | +0.017 | -0.080 | +0.056 | +0.043 |
| Transaction | 0.281 | +0.064 | -0.109 | -0.100 | +0.230 † | +0.018 | +0.336 † |
| Reference resolution | 0.775 | +0.026 | +0.067 | -0.135 | -0.041 † | +0.057 | +0.108 † |
| Schema discovery | 0.072 | +0.450 | +0.016 | -0.051 | +0.666 † | +0.062 | +0.683 † |

## Real benchmarks: RL mixed7 minus zero-shot, Qwen3-4B

RL score minus zero-shot score; † = the variant is in the mixed7 training data.

| variant | τ² airline | τ² retail | BFCL | AutomationBench | MCP-Atlas |
|---|---:|---:|---:|---:|---:|
| native † | +0.120 | +0.079 | -0.010 | -0.031 | -0.100 |
| fully merged † | -0.060 | +0.211 | +0.005 | +0.117 | +0.014 |
| class dispatch | +0.100 | +0.026 | +0.015 | +0.026 | -0.025 |
| fully split † | +0.080 | +0.053 | -0.005 | +0.002 | -0.033 |
| interval split | -0.020 | n/a | +0.020 | -0.049 | -0.041 |
| nested args | -0.060 | +0.000 | +0.045 | +0.005 | -0.085 |
| namespaced names † | +0.000 | +0.088 | +0.020 | -0.023 | -0.056 |
| strip descriptions | +0.060 | +0.009 | -0.025 | +0.012 | -0.031 |
| reorder arguments | +0.160 | +0.018 | -0.035 | -0.022 | -0.039 |
| transaction † | +0.060 | +0.000 | -0.015 | +0.039 | -0.024 |
| schema discovery † | +0.000 | +0.149 | +0.075 | +0.032 | +0.017 |
| composed | +0.000 | +0.000 | +0.000 | +0.005 | +0.007 |

## Real benchmarks: how well each estimator ranks a model's variants (Spearman ρ)

| model | benchmark | variants | split-half | compliance | exact call | likelihood | 36-task sample |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-27B | τ² airline | 12 | 0.333 | 0.000 | 0.000 | 0.731 | 0.308 |
| Qwen3.5-27B | τ² retail | 11 | 0.533 | 0.000 | 0.502 | 0.594 | 0.487 |
| Qwen3.5-27B | BFCL | 12 | 0.824 | 0.503 | 0.673 | 0.252 | 0.804 |
| Qwen3.5-27B | AutomationBench | 12 | 0.976 | 0.360 | 0.711 | 0.664 | 0.922 |
| Qwen3.5-27B | MCP-Atlas | 12 | 0.754 | 0.511 | 0.867 | 0.497 | 0.756 |
| Qwen3-4B | τ² airline | 12 | 0.330 | -0.420 | -0.420 | -0.164 | 0.263 |
| Qwen3-4B | τ² retail | 11 | 0.746 | 0.669 | 0.678 | 0.667 | 0.730 |
| Qwen3-4B | BFCL | 12 | 0.917 | 0.339 | 0.671 | 0.565 | 0.879 |
| Qwen3-4B | AutomationBench | 12 | 0.944 | 0.802 | 0.838 | 0.524 | 0.906 |
| Qwen3-4B | MCP-Atlas | 12 | 0.754 | 0.848 | 0.636 | 0.671 | 0.734 |
| Qwen3-4B RL | τ² airline | 12 | -0.165 | -0.397 | -0.397 | 0.307 | -0.171 |
| Qwen3-4B RL | τ² retail | 11 | 0.546 | 0.408 | 0.408 | 0.336 | 0.548 |
| Qwen3-4B RL | BFCL | 12 | 0.941 | 0.533 | 0.404 | -0.539 | 0.898 |
| Qwen3-4B RL | AutomationBench | 12 | 0.924 | 0.606 | 0.563 | -0.154 | 0.782 |
| Qwen3-4B RL | MCP-Atlas | 12 | 0.265 | 0.256 | -0.096 | 0.329 | 0.204 |
