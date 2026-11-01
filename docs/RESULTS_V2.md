# SCOPE v2 results (October 2026)

The v2 round, scored on 6 October 2026 (tool-server failures answered again and the home_action answers answered again with the per-task home reset, judged again on 10 October 2026; the 19 tasks whose v2 rows changed since, judged again on 30 October 2026): 26 distinct planner and vision configurations, each on
all 536 tasks of the corrected benchmark, graded by a panel of three judges. Three more runs
repeat a configuration and are shown only under [Repeatability](#repeatability). The paper's
results (v1) are in [`PAPER_RESULTS.md`](PAPER_RESULTS.md); the two are not directly comparable (see
[Reading these numbers](#reading-these-numbers)).

**Mean over the 26 configurations: 73.1%** (10,193 of 13,936 task attempts pass).

## How v2 is scored

| | |
| --- | --- |
| Tasks | [`benchmark/scope_v2.csv`](../benchmark/scope_v2.csv): the paper's 536 tasks with 187 rows corrected ([`benchmark/CHANGES_v2.md`](../benchmark/CHANGES_v2.md)); judged on exactly this dataset (re-judged 30 October 2026 for 19 tasks, see [below](#dataset-behind-these-numbers)) |
| Judges | gpt-oss-120b, Muse-Glimmer-30B and Qwen3-Nemotron-32B-GenRM; a task passes when two of the three say it is correct ([`scripts/13_judge_panel.py`](../scripts/13_judge_panel.py), [`JUDGING.md`](JUDGING.md#3-the-v2-judge-panel)) |
| Rubric | criteria C1 to C3 and the October fixes J1 to J6 ([`JUDGING.md`](JUDGING.md)) |
| What the judge sees | the final answer (reasoning removed) and the tool-call trajectory; a reply with no final answer, or a camera action the run claims but did not perform, fails in code |
| Verdicts | every record has all three verdicts; a missing verdict is asked again, never counted as a fail |

## Dataset behind these numbers

These numbers are for shipped v2 exactly ([`scope_v2.csv`](../benchmark/scope_v2.csv), sha256
`b45c601b…`). The answers were first judged between 6 and 10 October 2026 against the corrected
task set as it stood then, which differs from shipped v2 in 19 tasks: Q_051 and Q_099 have
reworded references (Q_099 also its alternates), Q_259 a different alternate and expected tool
calls, and 16 descriptive tasks different lists of accepted alternates (Q_189, Q_343, Q_350,
Q_356, Q_461 to Q_463, Q_465, Q_468, Q_470 to Q_472, Q_474, Q_476 to Q_478). Those changes are
planned for v2.1 ([`CHANGES_v2.md`](../benchmark/CHANGES_v2.md#planned-for-v21)). No question
wording or starting preset differs.

On 30 October 2026 those 19 tasks were judged again for every run, 26 configurations and 3
repeats: 551 stored answers, each by all three judges, with the same weights, prompts, client
settings, retries and 2-of-3 rule. Nothing was answered again. The other 517 tasks of each run keep
their October verdicts; what the judges saw for them is unchanged. The re-judge moved
configurations by -0.6 to +0.6 points (a repeat by -0.7) and the mean from 73.3% (10,212 passes) to
73.1% (10,193); of the 551 records, 44 went from pass to fail and 20 from fail to pass.

gpt-oss-120b was served on the October serving image. Muse-Glimmer-30B and
Qwen3-Nemotron-32B-GenRM were served on a different machine with a newer vLLM build (0.28.1).
Before use, each judge graded 100 stored answers from unchanged tasks again (all 29 runs, all eight
categories) and was compared with its own October verdicts: Muse-Glimmer-30B agreed on 99 of 100
and Qwen3-Nemotron-32B-GenRM on 100 of 100. gpt-oss-120b on the October image agreed on 96 of 100,
which is its run-to-run variation: on the other machine it agreed on 97. What matters for the scores is the panel: on the same 100 answers, the 2-of-3 verdict from the
three new votes matched October's stored panel verdict on 100 of 100.

| Results | Tasks | Judge |
| --- | --- | --- |
| HRI 2026 paper ([`PAPER_RESULTS.md`](PAPER_RESULTS.md)) | v1, answers from September 2025 | gpt-oss-120B alone, the published category prompts |
| September 2026 re-run, graded the September way (64.3% figure below) | v1 | the three-judge panel, earlier rubric |
| This page (73.1%) | v2 exactly as shipped (judged 6 to 10 October 2026; 19 tasks re-judged on 30 October 2026) | the three-judge panel, 2 of 3, criteria C1 to C3 and fixes J1 to J6 |

## Results by configuration

The runs did not all have the same setup, so they are grouped by how they were answered and
ranked only within a group. No single overall winner is claimed.

### Standard runs (22, mean 72.1%)

September 2026 answers; tasks whose question or starting preset changed were answered again (6 per run on 30 September and 6 on 6 October).

| Rank in group | Planner | Vision model | Planner precision | Planner serving | Accuracy (536 tasks) | Run | Notes |
| :---: | --- | --- | --- | --- | :---: | --- | :---: |
| 1 | Muse-Glimmer-30B+DFlash2 | Moondream 3.1 | BF16 | vLLM nightly (6 Sep 2026 build) | 80.4% | `tC01` |  |
| 2 | Qwen3.8-27B | gemma-4-26B-A4B-it | BF16 | vLLM 0.27.1 | 80.0% | `tB07` | d |
| 3 | Qwen3.8-27B | Molmo2-8B | BF16 | vLLM 0.27.1 | 79.9% | `tB02` | e |
| 4 | Muse-Glimmer-30B | Moondream 3.1 | BF16 | vLLM nightly (6 Sep 2026 build) | 79.7% | `tA12` |  |
| 5 | Qwen3.8-27B-AWQ-INT4 | Moondream 3.1 | AWQ-INT4 | vLLM 0.27.1 | 78.7% | `tC05` | d |
| 6 | gemma-4-26B-A4B-it | Moondream 3.1 | BF16 | vLLM nightly (6 Sep 2026 build) | 78.5% | `tA10` |  |
| 7 | Qwen3.8-27B | Moondream 3.1 | BF16 | vLLM 0.27.1 | 78.4% | `tA06` |  |
| 8 | gpt-oss-120b | Moondream 3.1 | MXFP4 | vLLM 0.27.1 | 78.2% | `tA08` | c |
| 8 | gemma-4-12B-it | Moondream 3.1 | BF16 | vLLM nightly (6 Sep 2026 build) | 78.2% | `tA09` | d |
| 10 | Qwen3.8-27B+DFlash2 | Moondream 3.1 | BF16 | vLLM nightly (6 Sep 2026 build) | 78.0% | `tC02` |  |
| 11 | gemma-4-26B-A4B-it | same model (one server) | BF16 | vLLM nightly (6 Sep 2026 build) | 77.6% | `tB08` | d |
| 12 | Qwen3.8-27B | Moondream2 | BF16 | vLLM 0.27.1 | 76.7% | `tB01` |  |
| 13 | gpt-oss-20b | Moondream 3.1 | MXFP4 | vLLM 0.27.1 | 75.9% | `tA07` | cd |
| 14 | Qwen3.6-35B-A3B | Moondream 3.1 | BF16 | vLLM 0.27.1 | 74.1% | `tA05` |  |
| 15 | gemma-4-E4B-it | Moondream 3.1 | BF16 | vLLM nightly (6 Sep 2026 build) | 69.0% | `tA11` |  |
| 16 | Qwen3-8B+EAGLE3 | Moondream 3.1 | BF16 | vLLM 0.27.1 | 68.3% | `tC04` | d |
| 17 | Qwen3-8B+DSpark | Moondream 3.1 | BF16 | vLLM development build (PR 51655) | 67.7% | `tC03` | d |
| 18 | Qwen3-8B | Moondream 3.1 | BF16 | vLLM 0.27.1 | 65.5% | `tA02` |  |
| 19 | Qwen3-4B | Moondream 3.1 | BF16 | vLLM 0.27.1 | 62.5% | `tA01` |  |
| 20 | Qwen3-30B-A3B (base checkpoint, /no_think) | Moondream 3.1 | BF16 | vLLM 0.27.1 | 60.1% | `tA03` |  |
| 21 | Qwen3.8-27B | Molmo2-VideoPoint-4B | BF16 | vLLM 0.27.1 | 56.9% | `tB03` | e |
| 22 | Qwen3.5-0.8B | Moondream 3.1 | BF16 | vLLM 0.27.1 | 41.6% | `tA04` |  |

### Qwen vision runs, answered again (3, mean 79.7%)

Every task answered again on 6 October after the Qwen vision fixes (note a).

| Rank in group | Planner | Vision model | Planner precision | Planner serving | Accuracy (536 tasks) | Run | Notes |
| :---: | --- | --- | --- | --- | :---: | --- | :---: |
| 1 | Qwen3.8-27B | Qwen3.6-35B-A3B | BF16 | vLLM 0.27.1 | 81.5% | `tB06` | a |
| 2 | Qwen3.8-27B | Qwen3.8-27B-AWQ-INT4 | BF16 | vLLM 0.27.1 | 81.3% | `tB05` | a |
| 3 | Qwen3.8-27B | same model (one server) | BF16 | vLLM 0.27.1 | 76.1% | `tB04` | a |

### Large planner, smaller cache for 30 tasks (1, mean 76.9%)

As the standard runs, but its 30 October re-answers ran with a smaller planner cache (note b).

| Rank in group | Planner | Vision model | Planner precision | Planner serving | Accuracy (536 tasks) | Run | Notes |
| :---: | --- | --- | --- | --- | :---: | --- | :---: |
| 1 | Qwen3-Next-80B-A3B | Moondream 3.1 | BF16 | vLLM nightly (6 Sep 2026 build) | 76.9% | `tD03` | b |

Notes:

- **a.** Answered again on 6 October after two fixes in the Qwen vision path (native `point_2d`
  and `bbox_2d` output is parsed, as pointing counts had returned 0; the vision model answers with
  thinking off, as a 128-token cap had cut its replies) and with the per-task home reset
  (`home_action` returns to the task's own start view).
- **b.** The 30 tasks answered again in October (the six reworded on 5 October and the 24 that
  called `home_action`) were answered with the planner's GPU memory share at 0.55 instead of 0.70,
  so that the vision model and renderer fit on the same four cards. This
  changes only the size of the planner's cache, not its model, settings or sampling.
- **c.** The answers regenerated after the dataset corrections (30 September and 6 October) were
  served with vLLM 0.10.2; the run's other answers with vLLM 0.27.1.
- Gemma vision runs (`tB07`, `tB08`, `tRPL2`): Gemma 4 points and draws boxes in its own format
  (`box_2d` as [y1, x1, y2, x2] and `point_2d` as [y, x], on a 0 to 1000 grid). The September
  answers, most of each run, were made with the old reader: a `box_2d` reply was not read and
  counted as 0, a `point_2d` reply was not read either (and from 3 October would have been read x
  before y). A reader for Gemma's format was added on 10 October and only answers made after that
  use it. Values on the 0 to 1000 grid given in SCOPE's own JSON format are still read as pixels.
  Raw vision replies were not saved, so the effect on these runs' scores cannot be measured.
- **d.** The tasks answered again in October for `tA07`, `tA09`, `tB07`, `tB08`, `tC03`, `tC04`, `tC05`, `tRPL2` were served on fewer cards (planner, vision model and renderer on one card; for `tB07` the vision model and renderer on one card), with a GPU memory share lowered (planner 0.55 to 0.45; planner 0.60 to 0.45; planner 0.85 to 0.80; vision 0.85 to 0.80). This changes only the size of that server's cache, not its model, settings or sampling.
- **e.** Molmo2 runs (`tB02`, `tB03`): before 10 October 2026 every `zoom_bounding` call failed. The vision server rejected the detection request (its chat template takes no system message), and the tool reported that it could not find the target without trying its pointing fallback. Molmo2 now zooms from its points; the 85 records whose answer had a failed zoom (tB02 44, tB03 41) were answered again.
- Model names are as served. "+DFlash2", "+DSpark" and "+EAGLE3" are speculative-decoding
  draft models added to the same planner; "same model (one server)" means the planner also
  answers the vision tools. Vision models are served with Moondream Photon (Moondream) or vLLM.

## Repeatability

`tB04` 76.1% and its repeat `tB09` 75.4%; `tB08` 77.6% and its repeat `tRPL2` 78.0%; `tA06` 78.4% and its repeat `tA13` 77.4%. `tA13` was recorded as a "thinking off" variant of `tA06`, but both runs sent `/no_think`
on every task and neither produced a reasoning block, so it is a repeat of the same settings.

## By task category

| Category | Records | Accuracy |
| --- | ---: | :---: |
| `single_call` | 1,872 | 97.3% |
| `multi_step_command` | 1,482 | 86.7% |
| `ocr_identification` | 1,404 | 72.6% |
| `descriptor` | 2,314 | 72.5% |
| `location_spatial` | 1,378 | 70.8% |
| `multi_step_reasoning` | 1,404 | 68.6% |
| `counting` | 2,470 | 60.4% |
| `comparative_relational` | 1,612 | 59.6% |

## Duplicate tasks

127 of the 536 tasks repeat another task's scene, starting preset, question and answer (64 groups; 409 distinct tasks). The primary score keeps all 536, as the paper did. Counting each group once (the mean of its rows' verdicts) moves a configuration by -5.1 to +0.0 points; the mean over the 26 configurations moves from 73.1% to 70.2%.

| Run | Group | 536 tasks | 409 distinct tasks | Change |
| --- | --- | :---: | :---: | :---: |
| `tB07` | Standard runs | 80.0% | 80.0% | +0.0 |
| `tB08` | Standard runs | 77.6% | 77.5% | -0.1 |
| `tB04` | Qwen vision runs, answered again | 76.1% | 74.9% | -1.2 |
| `tB01` | Standard runs | 76.7% | 75.1% | -1.6 |
| `tB03` | Standard runs | 56.9% | 55.3% | -1.6 |
| `tA03` | Standard runs | 60.1% | 58.3% | -1.8 |
| `tB06` | Qwen vision runs, answered again | 81.5% | 79.5% | -2.0 |
| `tB05` | Qwen vision runs, answered again | 81.3% | 78.9% | -2.4 |
| `tB02` | Standard runs | 79.9% | 77.4% | -2.5 |
| `tA10` | Standard runs | 78.5% | 75.8% | -2.7 |
| `tA12` | Standard runs | 79.7% | 76.9% | -2.8 |
| `tA09` | Standard runs | 78.2% | 75.0% | -3.2 |
| `tA02` | Standard runs | 65.5% | 62.2% | -3.3 |
| `tA06` | Standard runs | 78.4% | 75.1% | -3.3 |
| `tA07` | Standard runs | 75.9% | 72.5% | -3.4 |
| `tA08` | Standard runs | 78.2% | 74.7% | -3.5 |
| `tD03` | Large planner, smaller cache for 30 tasks | 76.9% | 73.2% | -3.7 |
| `tA11` | Standard runs | 69.0% | 65.3% | -3.7 |
| `tC01` | Standard runs | 80.4% | 76.7% | -3.7 |
| `tC04` | Standard runs | 68.3% | 64.6% | -3.7 |
| `tC03` | Standard runs | 67.7% | 63.8% | -3.9 |
| `tC05` | Standard runs | 78.7% | 74.8% | -3.9 |
| `tC02` | Standard runs | 78.0% | 74.0% | -4.0 |
| `tA04` | Standard runs | 41.6% | 37.4% | -4.2 |
| `tA05` | Standard runs | 74.1% | 69.7% | -4.4 |
| `tA01` | Standard runs | 62.5% | 57.4% | -5.1 |

## The judges

| Judge | Passes it gives |
| --- | :---: |
| gpt-oss-120b | 71.6% |
| Muse-Glimmer-30B | 76.5% |
| Qwen3-Nemotron-32B-GenRM | 67.4% |

All three agree on 87.8% of records. Qwen3-Nemotron-32B-GenRM is the strictest seat; it
runs with thinking off and a 3,072-token reply cap. The code checks fail
170 records for a camera action the run did not carry out and
48 for having no final answer. Tallied apart: 9 records whose
vision reply could not be read (scored as failures, maintainer's rule of 9 October 2026); a record
whose tool server failed is answered again instead.

## Reading these numbers

- **Why v1 and v2 differ.** v1 numbers (the HRI 2026 paper) are unweighted means of eight category accuracies, graded by a single gpt-oss-120b judge on answers generated in September 2025. v2 numbers are pass rates over all 536 tasks, graded by a three-judge panel with the October rubric on the corrected dataset (187 of 536 rows changed), on answers regenerated with the current harness and the models now served (for example Moondream 3.1 instead of Moondream3-preview, and the base Qwen3-30B-A3B checkpoint instead of Qwen3-30B-A3B-Instruct-2507), so most of the difference for a given pairing comes from different answers, not from different grading. Compare configurations within one version; [`PAPER_RESULTS.md`](PAPER_RESULTS.md) keeps the paper's numbers unchanged.
- **Same answers, older grading.** Scored with the September 2026 campaign's grading (the same
  three judges with the earlier rubric, on the paper's task set), the same answers pass at
  64.3%. Between the two, 1,818 attempts went from fail to pass and
  591 from pass to fail.
- **No bridge to the paper, by design.** v2 is a new version of the benchmark; no paper answer
  was re-judged and no paper setup re-run. `tA03` runs the base Qwen3-30B-A3B checkpoint with
  `/no_think`, not the paper's Qwen3-30B-A3B-Instruct-2507, with Moondream 3.1 instead of
  Moondream3-preview, so it is not the paper's best configuration re-scored.
- **What the saved data supports for that pairing, and what it does not.** The paper's 73.8% is a
  mean of eight category rates; the same verdicts counted per task are 72.4%. With the judge held
  fixed (one gpt-oss-120b judge, the same rules and tasks), the answers regenerated in September
  2026 pass at 55.6%, 16.8 points lower. v2's grading then moves those answers up by 4.2 points
  net (the panel -2.1, the corrections and rubric +6.3), to 59.9% (60.1% once its nine `home_action`
  tasks were answered again with the per-task home reset). **Unexplained remainder:** the
  16.8 points cannot be divided among their causes (planner checkpoint, Moondream version, system
  prompt, stored panoramas, the frame size limit, serving engine) from the saved votes.

## Files

[`benchmark/results/v2-2026-10/`](../benchmark/results/v2-2026-10/): `summary.json` (these
figures), `models.csv` (every run, including the repeats, the earlier scorings and the flips),
`primitives.csv` (per category, all runs), `per-task.csv` (each run's panel verdict, pass or fail,
on each of the 536 tasks; 29 runs, recomputed from the stored votes by `13_judge_panel.py vote`)
and `configs.json` (every run's planner and vision
model, precision, serving engine and server settings; `tool_call_args` are the tool-calling flags
each planner model was launched with in the October 2026 server records). Ready-to-run serving
recipes for the recommended pairings are in [`CONFIGURATION.md`](CONFIGURATION.md). Per-task answers, tool traces and individual judge votes are not
published (41 records' votes are a test fixture, `tests/fixtures/judge_panel/`, with synthetic conversations
in place of the answers).

To grade new answers the v2 way, use the judge panel, `scripts/13_judge_panel.py`
([`JUDGING.md`](JUDGING.md#3-the-v2-judge-panel)). From the stored votes it reproduces the v2
scores in `summary.json` and `models.csv` exactly.
