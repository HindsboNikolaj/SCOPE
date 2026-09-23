# Preliminary results, September 2026

> **PRELIMINARY.** These runs answered the paper's task set
> ([`scope_536_paper.csv`](../benchmark/scope_536_paper.csv)) and were graded *before* the audit
> corrections and the C1/C2 criteria changes. Every run needs 2 rows re-run and 154 rows
> re-graded before its number means the same thing as a v1.1 number. They are not comparable
> with the paper's GPT-4o numbers either: the judge is a three-model panel.

Data: [`benchmark/results/preliminary-2026-09/`](../benchmark/results/preliminary-2026-09/).
`pairings.json` holds each run's models, panel and accuracy; `verdicts.csv` holds the panel's
majority verdict (0/1) and yes-vote count per question per run. Every number on this page is
computed from those two files, and the table below is the output of
`python scripts/14_rerun_status.py`.

## What each run needs

The classification is mechanical, per row, from [`corrections.json`](../benchmark/corrections.json):

| Row status | Rule | Rows per run |
|---|---|---:|
| **re-run** | question text changed (Q_014, Q_236), so the recorded answer is to a different question | 2 |
| **re-grade** | answer exists and input is unchanged, but grading changed: a new expected answer, or the row is `comparative_relational` (C1) or `descriptor` (C2) | 154 |
| **valid** | nothing that affects it changed | 380 |

The 154 re-grade rows are all 62 `comparative_relational` rows except Q_014, all 89
`descriptor` rows, and 4 answer changes elsewhere (Q_098, Q_099, Q_412, Q_523). No run needs a
full re-run: re-run the 2 rows, re-judge the 156, and every run is on v1.1.

"Accuracy on valid rows" is the one figure readable today without re-judging. It is higher
than the 536-row figure because the rows set aside are the hardest categories, so compare it
only between runs, never with a 536-row number.

| Tag | Pairing | Status | Valid rows | Re-grade rows | Re-run rows | Accuracy as graded (536) | Accuracy on valid rows |
|---|---|---|---:|---:|---:|---:|---:|
| tB07 | Qwen3.8-27B x gemma-4-26B-A4B-it | needs re-run | 380 | 154 | 2 | 73.9% | 81.8% |
| tC01 | Muse-Glimmer-30B+DFlash2 x Moondream 3.1 | needs re-run | 380 | 154 | 2 | 73.7% | 80.8% |
| tRPL2 | gemma-4-26B-A4B-it x same server | needs re-run | 380 | 154 | 2 | 73.5% | 82.9% |
| tB01 | Qwen3.8-27B x Moondream2 | needs re-run | 380 | 154 | 2 | 72.9% | 80.8% |
| tB08 | gemma-4-26B-A4B-it x same server | needs re-run | 380 | 154 | 2 | 72.8% | 81.1% |
| tA12 | Muse-Glimmer-30B x Moondream 3.1 | needs re-run | 380 | 154 | 2 | 71.1% | 79.2% |
| tA10 | gemma-4-26B-A4B-it x Moondream 3.1 | needs re-run | 380 | 154 | 2 | 70.7% | 77.6% |
| tB02 | Qwen3.8-27B x Molmo2-8B | needs re-run | 380 | 154 | 2 | 70.5% | 76.1% |
| tA08 | gpt-oss-120b x Moondream 3.1 | needs re-run | 380 | 154 | 2 | 70.3% | 77.9% |
| tA13 | Qwen3.8-27B (thinking off) x Moondream 3.1 | needs re-run | 380 | 154 | 2 | 70.1% | 76.8% |
| tA09 | gemma-4-12B-it x Moondream 3.1 | needs re-run | 380 | 154 | 2 | 69.8% | 76.1% |
| tC05 | Qwen3.8-27B-AWQ-INT4 x Moondream 3.1 | needs re-run | 380 | 154 | 2 | 69.6% | 76.8% |
| tC02 | Qwen3.8-27B+DFlash2 x Moondream 3.1 | needs re-run | 380 | 154 | 2 | 69.2% | 75.0% |
| tB05 | Qwen3.8-27B x Qwen3.8-27B-AWQ-INT4 | needs re-run | 380 | 154 | 2 | 69.0% | 78.2% |
| tB09 | Qwen3.8-27B x same server | needs re-run | 380 | 154 | 2 | 68.3% | 74.7% |
| tA06 | Qwen3.8-27B x Moondream 3.1 | needs re-run | 380 | 154 | 2 | 67.9% | 76.1% |
| tB04 | Qwen3.8-27B x Qwen3.8-27B (same server) | needs re-run | 380 | 154 | 2 | 67.5% | 75.3% |
| tA07 | gpt-oss-20b x Moondream 3.1 | needs re-run | 380 | 154 | 2 | 67.2% | 73.2% |
| tD03 | Qwen3-Next-80B-A3B x Moondream 3.1 | needs re-run | 380 | 154 | 2 | 64.9% | 72.6% |
| tA05 | Qwen3.6-35B-A3B x Moondream 3.1 | needs re-run | 380 | 154 | 2 | 64.2% | 74.2% |
| tA11 | gemma-4-E4B-it x Moondream 3.1 | needs re-run | 380 | 154 | 2 | 62.5% | 71.8% |
| tB06 | Qwen3.8-27B x Qwen3.6-35B-A3B | needs re-run | 380 | 154 | 2 | 62.5% | 69.2% |
| tC04 | Qwen3-8B+EAGLE3 x Moondream 3.1 | needs re-run | 380 | 154 | 2 | 56.7% | 66.1% |
| tA02 | Qwen3-8B x Moondream 3.1 | needs re-run | 380 | 154 | 2 | 56.5% | 66.8% |
| tA01 | Qwen3-4B x Moondream 3.1 | needs re-run | 380 | 154 | 2 | 55.4% | 65.3% |
| tB03 | Qwen3.8-27B x Molmo2-VideoPoint-4B | needs re-run | 380 | 154 | 2 | 54.7% | 63.4% |
| tA03 | Qwen3-30B-A3B x Moondream 3.1 | needs re-run | 380 | 154 | 2 | 53.5% | 65.8% |
| tC03 | Qwen3-8B+DSpark x Moondream 3.1 | needs re-run | 380 | 154 | 2 | 49.1% | 54.7% |
| tA04 | Qwen3.5-0.8B x Moondream 3.1 | needs re-run | 380 | 154 | 2 | 36.6% | 44.7% |

`tRPL2` repeats the `tB08` configuration to capture per-question frames for the audit, and is a
second noise measurement rather than a separate pairing.

### Other result sets

| Result set | Dataset | Status | Why |
|---|---|---|---|
| The paper's 19 pairings (GPT-4o judge) | paper set | needs re-run (2 rows) + re-grade (154 rows) | Same rows as above. Raw answers are not in this repository |
| Any run on the old 541-row `scope_536.csv` | 541 lineage | **needs full re-run** | Ids were renumbered and 44 rows removed, 39 added, 75 edited ([`changelog_541_to_536.csv`](../benchmark/changelog_541_to_536.csv)). Matching old answers to new rows by text is possible but not supported |
| Early campaign runs judged by an earlier panel (Qwen3-32B-FP8 and ERNIE-4.5-VL seats) | paper set | superseded, not published | Judged by a panel that was replaced; the runs above re-cover the same configurations |

## The corrected rows, as the models saw them

Majority-verdict pass rate across the 28 pairings (tRPL2 excluded), before correction:

| Id | Category | Change | Pass rate |
|---|---|---|---:|
| Q_001 | comparative_relational | answer | 57% |
| Q_002 | comparative_relational | answer | 0% |
| Q_006 | comparative_relational | answer | 0% |
| Q_007 | comparative_relational | answer | 11% |
| Q_010 | comparative_relational | answer | 43% |
| Q_014 | comparative_relational | question | 50% |
| Q_098 | multi_step_reasoning | answer | 39% |
| Q_099 | multi_step_reasoning | answer | 4% |
| Q_236 | multi_step_command | question | 0% |
| Q_315 | comparative_relational | answer | 0% |
| Q_356 | descriptor | answer | 0% |
| Q_412 | ocr_identification | answer | 7% |
| Q_462 | descriptor | answer | 0% |
| Q_465 | descriptor | answer | 0% |
| Q_523 | ocr_identification | answer | 0% |

Eleven of fifteen sit at or below 11%: the reference was wrong and nearly every model was marked
down for describing the scene correctly.

## What the preliminary numbers already say

Read these as directions, not rankings.

- **The top ten are within 4.1 points** (73.9% to 69.8%), and that is inside the noise:
  `tB09` repeats `tB04` with nothing changed, and 88 of 536 verdicts flip between the two
  (67.5% against 68.3%).
- **Vision model size does not predict accuracy.** At a fixed Qwen3.8-27B planner, Moondream2
  (72.9%) beats Qwen3.6-35B-A3B as the vision model (62.5%).
- **One model can serve both roles.** gemma-4-26B-A4B-it as planner and vision from one
  instance scores 72.8%, against 73.9% for the best split pairing.
- **Speculative decoding and INT4 cost no accuracy here**: Qwen3.8-27B 67.9%, with DFlash2
  69.2%, AWQ-INT4 69.6%.
- **The panel's third seat pulls every run down**, most at the top; see
  [`JUDGING.md`](JUDGING.md#3-a-judge-panel).

## To finish

1. Re-run Q_014 and Q_236 for every pairing (2 rows x 29 runs).
2. Re-judge the 156 affected rows per run with the v1.1 prompts, same panel.
3. Regenerate `pairings.json` and `verdicts.csv`, then drop the PRELIMINARY label.
