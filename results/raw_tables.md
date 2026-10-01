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
| native (score) | **0.800** | **0.482** | **0.725** | **0.535** | **0.589** |
| fully merged | -0.040 | -0.333 | -0.515 | -0.219 | -0.220 |
| class dispatch | +0.000 | -0.053 | -0.305 | -0.199 | -0.157 |
| fully split | -0.060 | -0.079 | +0.005 | -0.082 | -0.059 |
| interval split | -0.060 | n/a | +0.015 | -0.049 | +0.008 |
| nested args | -0.060 | -0.053 | -0.060 | -0.020 | -0.034 |
| namespaced names | +0.040 | +0.018 | -0.010 | -0.011 | -0.027 |
| strip descriptions | -0.020 | -0.088 | -0.220 | -0.113 | -0.050 |
| reorder arguments | +0.000 | +0.000 | +0.010 | +0.000 | -0.054 |
| transaction | -0.080 | -0.009 | -0.020 | -0.032 | +0.002 |
| reference resolution | n/a | n/a | n/a | n/a | n/a |
| schema discovery | -0.040 | -0.044 | +0.020 | -0.043 | -0.130 |
| composed | -0.040 | -0.061 | -0.160 | -0.129 | -0.109 |

## Real benchmarks: Qwen3-4B (zero-shot) × representative variants

| variant | τ² airline | τ² retail | BFCL | AutomationBench | MCP-Atlas |
|---|---:|---:|---:|---:|---:|
| native (score) | **0.440** | **0.272** | **0.335** | **0.166** | **0.167** |
| fully merged | +0.020 | -0.219 | -0.335 | -0.165 | -0.139 |
| class dispatch | -0.080 | -0.105 | -0.300 | -0.028 | -0.112 |
| fully split | -0.100 | -0.061 | +0.000 | -0.041 | -0.013 |
| interval split | +0.000 | n/a | +0.000 | -0.013 | +0.010 |
| nested args | +0.040 | -0.035 | -0.020 | -0.059 | -0.013 |
| namespaced names | -0.120 | -0.009 | -0.195 | -0.003 | -0.002 |
| strip descriptions | -0.020 | -0.009 | -0.095 | -0.076 | -0.057 |
| reorder arguments | -0.080 | +0.009 | -0.010 | -0.004 | -0.016 |
| transaction | -0.040 | -0.079 | -0.050 | -0.100 | -0.012 |
| reference resolution | n/a | n/a | n/a | n/a | n/a |
| schema discovery | -0.020 | -0.184 | -0.310 | -0.125 | -0.130 |
| composed | -0.040 | -0.193 | -0.300 | -0.097 | -0.070 |

## Real benchmarks: Qwen3-4B after RL mixed7 × representative variants

| variant | τ² airline | τ² retail | BFCL | AutomationBench | MCP-Atlas |
|---|---:|---:|---:|---:|---:|
| native (score) | **0.420** | **0.184** | **0.305** | **0.132** | **0.111** |
| fully merged | -0.020 | +0.070 | -0.280 | -0.007 | -0.033 |
| class dispatch | -0.020 | -0.018 | -0.250 | +0.010 | -0.058 |
| fully split | +0.040 | +0.053 | +0.000 | -0.006 | -0.015 |
| interval split | -0.080 | n/a | -0.015 | -0.022 | -0.008 |
| nested args | +0.000 | +0.070 | +0.000 | -0.011 | +0.003 |
| namespaced names | -0.060 | +0.114 | -0.155 | +0.009 | -0.029 |
| strip descriptions | -0.020 | +0.061 | -0.115 | -0.031 | -0.003 |
| reorder arguments | +0.000 | +0.096 | +0.015 | +0.018 | -0.023 |
| transaction | +0.060 | +0.105 | -0.050 | -0.037 | +0.036 |
| reference resolution | n/a | n/a | n/a | n/a | n/a |
| schema discovery | +0.000 | -0.035 | -0.210 | -0.067 | -0.011 |
| composed | -0.080 | -0.132 | -0.285 | -0.060 | -0.057 |

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
| native † | -0.020 | -0.088 | -0.030 | -0.034 | -0.056 |
| fully merged † | -0.060 | +0.202 | +0.025 | +0.124 | +0.050 |
| class dispatch | +0.040 | +0.000 | +0.020 | +0.003 | -0.002 |
| fully split † | +0.120 | +0.026 | -0.030 | +0.000 | -0.058 |
| interval split | -0.100 | n/a | -0.045 | -0.043 | -0.074 |
| nested args | -0.060 | +0.018 | -0.010 | +0.013 | -0.040 |
| namespaced names † | +0.040 | +0.035 | +0.010 | -0.022 | -0.083 |
| strip descriptions | -0.020 | -0.018 | -0.050 | +0.011 | -0.002 |
| reorder arguments | +0.060 | +0.000 | -0.005 | -0.012 | -0.063 |
| transaction † | +0.080 | +0.096 | -0.030 | +0.029 | -0.007 |
| schema discovery † | +0.000 | +0.061 | +0.070 | +0.024 | +0.062 |
| composed | -0.060 | -0.026 | -0.015 | +0.002 | -0.043 |

## Real benchmarks: how well each estimator ranks a model's variants (Spearman ρ)

| model | benchmark | variants | split-half | compliance | exact call | likelihood | 36-task sample |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-27B | τ² airline | 12 | -0.346 | 0.000 | 0.000 | 0.243 | -0.204 |
| Qwen3.5-27B | τ² retail | 11 | 0.354 | 0.000 | 0.502 | 0.338 | 0.306 |
| Qwen3.5-27B | BFCL | 12 | 0.809 | 0.195 | 0.616 | 0.196 | 0.814 |
| Qwen3.5-27B | AutomationBench | 12 | 0.968 | 0.430 | 0.751 | 0.636 | 0.868 |
| Qwen3.5-27B | MCP-Atlas | 12 | 0.697 | 0.532 | 0.807 | 0.545 | 0.709 |
| Qwen3-4B | τ² airline | 12 | 0.234 | -0.186 | -0.186 | -0.067 | 0.218 |
| Qwen3-4B | τ² retail | 11 | 0.741 | 0.676 | 0.684 | 0.662 | 0.736 |
| Qwen3-4B | BFCL | 12 | 0.853 | 0.282 | 0.630 | 0.589 | 0.848 |
| Qwen3-4B | AutomationBench | 12 | 0.954 | 0.802 | 0.845 | 0.622 | 0.869 |
| Qwen3-4B | MCP-Atlas | 12 | 0.648 | 0.529 | 0.750 | 0.699 | 0.634 |
| Qwen3-4B RL | τ² airline | 12 | -0.024 | 0.449 | 0.449 | -0.560 | -0.054 |
| Qwen3-4B RL | τ² retail | 11 | 0.642 | 0.304 | 0.304 | 0.288 | 0.589 |
| Qwen3-4B RL | BFCL | 12 | 0.889 | 0.527 | 0.341 | -0.570 | 0.872 |
| Qwen3-4B RL | AutomationBench | 12 | 0.879 | 0.527 | 0.516 | -0.196 | 0.685 |
| Qwen3-4B RL | MCP-Atlas | 12 | 0.397 | 0.353 | 0.177 | -0.035 | 0.352 |
