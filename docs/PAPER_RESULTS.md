# Paper results (HRI 2026)

> **These are the paper's numbers, unchanged.** They come from Table 5 of the HRI 2026 paper
> ([`SCOPE_HRI26.pdf`](../SCOPE_HRI26.pdf)). They are not the current benchmark results; for
> those see the [README](../README.md#results) and
> [`RESULTS_V2.md`](RESULTS_V2.md).

**How the paper scored them.** Answers generated in September 2025 on dataset v1, the 536 tasks
now archived as [`benchmark/archive/scope_536_v1.csv`](../benchmark/archive/scope_536_v1.csv).
A single judge, gpt-oss-120B, graded every answer (paper, section 5: "We use gpt-oss-120B as
the sole LLM-as-Judge for all experiments"). Earlier versions of this repository said the paper
was GPT-4o-judged; that was wrong. "Average accuracy" is the paper's unweighted mean of the
eight category accuracies (Table 5 caption), not a pass rate over all 536 tasks.

Do not compare these values with a run on dataset v2 or on the 541-row file SCOPE 0.1.0 shipped
as `benchmark/scope_536.csv`: the tasks, the grading and (for v2) the models served all differ.
[`RESULTS_V2.md`](RESULTS_V2.md#reading-these-numbers) explains the difference for the paper's
best pairing.

## Selected view

The paper evaluates 19 planner–perception pairings. This selected view makes the architectural comparison legible; the full matrix follows below.

| Published configuration | Planner type | Average accuracy (536 tasks) | Reading it |
| --- | --- | :---: | --- |
| Qwen3-30B-A3B + Moondream3 | MoE | **73.8%** | Best reported configuration |
| Qwen3-Next-80B-A3B + Moondream2 | MoE | 70.6% | Second-highest published configuration |
| Qwen3-30B-A3B-FP8 + Moondream3 | MoE | 69.1% | Quantized counterpart of the best planner family |
| Qwen3-32B + Moondream3 | Dense | 68.8% | Highest published dense-planner configuration |
| Qwen3-Next-80B-A3B + Qwen2.5-VL-7B | MoE | 68.3% | Cross-family vision comparison |

`MoE` and `Dense` are the planner architectures reported in the paper. The paper’s interpretation is not that one model answers every PTZ question by itself: stronger planners improve tool selection and sequencing, while perception becomes the dominant remaining limitation for stronger planner pairings.

## All 19 pairings

| Planner | Planner type | Precision | Vision model | Average accuracy |
| --- | --- | --- | --- | :---: |
| Qwen3-4B | Dense | FP16 | Moondream2 | 59.7% |
| Qwen3-4B-FP8 | Dense | FP8 | Moondream2 | 60.3% |
| Qwen3-30B-A3B | MoE | FP16 | Moondream2 | 67.3% |
| Qwen3-30B-A3B-FP8 | MoE | FP8 | Moondream2 | 69.6% |
| Qwen3-32B | Dense | FP16 | Moondream2 | 68.0% |
| Qwen3-Next-80B-A3B | MoE | FP16 | Moondream2 | 70.6% |
| Qwen3-4B | Dense | FP16 | Moondream2-4bit | 62.3% |
| Qwen3-4B-FP8 | Dense | FP8 | Moondream2-4bit | 61.3% |
| Qwen3-30B-A3B | MoE | FP16 | Moondream2-4bit | 66.9% |
| Qwen3-30B-A3B-FP8 | MoE | FP8 | Moondream2-4bit | 65.7% |
| Qwen3-32B | Dense | FP16 | Moondream2-4bit | 65.9% |
| Qwen3-Next-80B-A3B | MoE | FP16 | Moondream2-4bit | 66.8% |
| Qwen3-4B | Dense | FP16 | Moondream3 | 61.4% |
| Qwen3-4B-FP8 | Dense | FP8 | Moondream3 | 62.6% |
| Qwen3-30B-A3B | MoE | FP16 | Moondream3 | **73.8%** |
| Qwen3-30B-A3B-FP8 | MoE | FP8 | Moondream3 | 69.1% |
| Qwen3-32B | Dense | FP16 | Moondream3 | 68.8% |
| Qwen3-Next-80B-A3B | MoE | FP16 | Moondream3 | 69.3% |
| Qwen3-Next-80B-A3B | MoE | FP16 | Qwen2.5-VL-7B | 68.3% |

The paper finds that stronger planners reduce tool-routing and sequencing errors;
for stronger planner pairings, visual perception accounts for most of the
remaining end-to-end failures. Reported latency also depends on the serving path
and evaluation hardware, so this table is an accuracy reference rather than a
deployment promise.
