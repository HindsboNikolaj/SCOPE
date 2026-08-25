# Published Results — 19 Planner–Perception Pairings

The HRI paper reports the following average accuracies on its GPT-4o-judged
**536-task published subset** (Table 5 in [`SCOPE_HRI26.pdf`](../SCOPE_HRI26.pdf)).
The repository now ships a 541-row CSV under the historical
[`benchmark/scope_536.csv`](../benchmark/scope_536.csv) filename; five shipped
rows fall outside the published score subset. Public Git history begins with the
541-row CSV, so it does not establish task-by-task provenance for that difference.

Do not compare a fresh 541-task run directly with the 536-task paper values.
The curated reading guide is in the [README](../README.md#published-results).

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
