# Full Results — All 20 SLM+VLM Combinations

Table 4 from the paper (overall accuracy on the published 536-task score
subset; the shipped CSV has 541 rows, five outside that subset -- see
`benchmark/scope_536.csv`). Judged by GPT-4o. See the
[README](../README.md#results-from-the-paper) for the top-5 condensed view.

| SLM | Moondream2 | Moondream3 | Qwen2.5-VL-3B | Qwen2.5-VL-7B |
|-----|:----------:|:----------:|:--------------:|:--------------:|
| Qwen3-4B | 52.1% | 56.7% | 50.4% | 55.2% |
| Qwen3-4B-FP8 | 51.3% | 55.8% | 49.8% | 54.5% |
| Qwen3-8B | 58.6% | 63.4% | 57.1% | 62.0% |
| Qwen3-30B-A3B | 69.5% | **73.8%** | 67.2% | 72.4% |
| Qwen3-32B | 66.8% | 71.6% | 65.3% | 70.9% |

Per-category breakdowns and detailed analysis are available in the paper
([`SCOPE_HRI26.pdf`](../SCOPE_HRI26.pdf)).
