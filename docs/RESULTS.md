# SCOPE results

Two releases, two datasets. Numbers from different releases do not compare: the models, the
tasks and the grading all changed ([why](RESULTS_V2.md#reading-these-numbers)).

## v0.2.0 (October 2026): dataset v2

26 planner and vision pairings, each on all 536 tasks of dataset v2. Three judges
(gpt-oss-120b, Muse-Glimmer-30B, Qwen3-Nemotron-32B-GenRM) grade every answer and a task passes
when two of three agree ([judging](JUDGING.md)). Mean 73.1% (10,193 of 13,936 attempts pass).
Answers were generated in September and October 2026 and judged 6 to 10 October 2026; the 19
tasks whose rows changed afterwards were judged again from the stored answers on 30 October 2026.
Repeating a pairing moved it by up to 1.0 point, so smaller gaps are not a ranking.

| Planner | Vision model | Accuracy | Run |
| --- | --- | :---: | --- |
| Qwen3.8-27B | Qwen3.6-35B-A3B | 81.5% ᵃ | `tB06` |
| Qwen3.8-27B | Qwen3.8-27B-AWQ-INT4 | 81.3% ᵃ | `tB05` |
| Muse-Glimmer-30B+DFlash2 | Moondream 3.1 | 80.4% | `tC01` |
| Qwen3.8-27B | gemma-4-26B-A4B-it | 80.0% | `tB07` |
| Qwen3.8-27B | Molmo2-8B | 79.9% | `tB02` |
| Muse-Glimmer-30B | Moondream 3.1 | 79.7% | `tA12` |
| Qwen3.8-27B-AWQ-INT4 | Moondream 3.1 | 78.7% | `tC05` |
| gemma-4-26B-A4B-it | Moondream 3.1 | 78.5% | `tA10` |
| Qwen3.8-27B | Moondream 3.1 | 78.4% | `tA06` |
| gpt-oss-120b | Moondream 3.1 | 78.2% | `tA08` |
| gemma-4-12B-it | Moondream 3.1 | 78.2% | `tA09` |
| Qwen3.8-27B+DFlash2 | Moondream 3.1 | 78.0% | `tC02` |
| gemma-4-26B-A4B-it | same model (one server) | 77.6% | `tB08` |
| Qwen3-Next-80B-A3B | Moondream 3.1 | 76.9% ᵇ | `tD03` |
| Qwen3.8-27B | Moondream2 | 76.7% | `tB01` |
| Qwen3.8-27B | same model (one server) | 76.1% ᵃ | `tB04` |
| gpt-oss-20b | Moondream 3.1 | 75.9% | `tA07` |
| Qwen3.6-35B-A3B | Moondream 3.1 | 74.1% | `tA05` |
| gemma-4-E4B-it | Moondream 3.1 | 69.0% | `tA11` |
| Qwen3-8B+EAGLE3 | Moondream 3.1 | 68.3% | `tC04` |
| Qwen3-8B+DSpark | Moondream 3.1 | 67.7% | `tC03` |
| Qwen3-8B | Moondream 3.1 | 65.5% | `tA02` |
| Qwen3-4B | Moondream 3.1 | 62.5% | `tA01` |
| Qwen3-30B-A3B (base checkpoint, /no_think) | Moondream 3.1 | 60.1% | `tA03` |
| Qwen3.8-27B | Molmo2-VideoPoint-4B | 56.9% | `tB03` |
| Qwen3.5-0.8B | Moondream 3.1 | 41.6% | `tA04` |

ᵃ Every task answered again on 6 Oct 2026 after two Qwen vision-path fixes. ᵇ 30 tasks answered
again with a smaller planner cache. Unmarked runs reuse their September 2026 answers, with the
tasks whose question or start changed answered again. Repeats (`tB09`, `tRPL2`, `tA13`) are
reported separately. Notes on each run are in
[`docs/RESULTS_V2.md`](RESULTS_V2.md#results-by-configuration).

Machine-readable tables: [`benchmark/results/v2-2026-10/`](../benchmark/results/v2-2026-10/).

## v0.1.0 (November 2025): the HRI 2026 paper, dataset v1

19 pairings with answers from September 2025, graded by one judge (gpt-oss-120B); the paper
reports the mean of eight category accuracies. Best: Qwen3-30B-A3B + Moondream3, 73.8%.
Full table: [`PAPER_RESULTS.md`](PAPER_RESULTS.md).

## After the paper: dataset v1 graded again (September 2026)

The v0.2.0 answers, scored on the paper's task set (dataset v1) by the same three judges with
the earlier September 2026 rubric, pass at 64.3%
([details](RESULTS_V2.md#reading-these-numbers)).
