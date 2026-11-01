# Judging

> **What this release includes.** Dataset v2 (`benchmark/scope_v2.csv`), its corrections, the v2 results and the October judging code that produced them: the October rubric prompts, structured verdicts, the code checks for missing answers and camera actions not carried out, and the 2-of-3 panel (`scripts/13_judge_panel.py`, [section 3](#3-the-v2-judge-panel)).

The paper graded every answer with a single gpt-oss-120b judge. v2 is scored by a three-judge
panel (gpt-oss-120b, Muse-Glimmer-30B, Qwen3-Nemotron-32B-GenRM): each judge grades the run, and
a task passes when 2 of 3 judges pass it. `scripts/13_judge_panel.py` runs the panel;
`scripts/11_judge.py` runs one judge against any OpenAI-compatible endpoint. This page covers what
the rubric grades, how the judge script behaves on a long run, and the panel.

## 1. Three criteria changes in the rubric

All three change published numbers. They, and the October fixes below, are applied in the v2
results (`RESULTS_V2.md`, data in
`benchmark/results/v2-2026-10/`). The September campaign
(also a three-judge panel, under the old rubric; 64.3% passing over the 26 configurations) predates them.

**C1, `comparative_relational`: grade only the claim the question asks for.**
[`prompts/judge_category_comparative_relational.md`](../prompts/judge_category_comparative_relational.md)
now decides on the comparison itself (which side, view or object has more). Wrong details the
answer volunteers alongside a correct comparison, typically a count, no longer fail it. The
old rubric validated every count it saw first, so "the left side has more trees (3 vs 1)" was
failed when the counts were off even though the question asked only which side. In the
September runs 80% of wrong `comparative_relational` answers already carried the reference's
comparison. Grounding still applies: a comparison the tool outputs contradict is a
hallucination.

**C2, `descriptor`: any fair description is correct.**
[`prompts/judge_category_descriptor.md`](../prompts/judge_category_descriptor.md) now treats the
expected answer as one fair description rather than the required phrasing. A grounded answer
that describes the same thing differently, or names other true details, passes unless it
contradicts the reference's core attribute. The `also_accept` list beside an expected answer lists
alternates (Q_356 uses it). Scope is deliberate: `ocr_identification` stays exact, because a
sign has one right reading, and `location_spatial` stays tight.

C2 removes the rubric's pull toward one phrasing, but open answers are only fully fair once
each reference carries its alternates. Since 6 October 2026 alternates are their own column, `also_accept` (a JSON list), not text inside `expected_answer`; 116 tasks of v2 carry one.

**C3, every category: `also_accept` alternates count.** The judge sees them as an "Also acceptable
(any of these also passes)" list under the expected answer, and every category prompt says each is a
correct answer on its own. Alternates were added outside the
descriptor category too (location questions, multi-step answers, absence answers), so the line
has to be in every prompt. It does not loosen counting or sign reading: an alternate there is
only what the view really supports, and grounding still applies.

### What the judge sees, and what is checked in code (October 2026)

An independent audit of the relaxed judging (30 September 2026) found the judges were
checking words, not actions, and the judge read the whole conversation, reasoning and leaked
planning notes included. The maintainer approved six fixes on 2 October 2026:

| Fix | What changed | Where |
|---|---|---|
| J1 | Camera commands are checked in code before any judge call. A record fails when the question explicitly commands a pan, tilt, zoom, preset or return home and no recorded tool call carries it out in that direction (zoom out needs `zoom_factor < 1`, `zoom_percent < 0` or `zoom_full="out"`; pan right needs `pan_deg > 0`; a half turn may go either way). Conditional or optional moves are not required. | `src/scope/eval/action_check.py`, `judge.precheck` |
| J2 | Comparisons: a tie, a hedge or a comparison of different things (the full panorama against the current view) is not the expected comparison; a before/after comparison needs both views observed. | `judge_category_comparative_relational.md` |
| J3 | Counts stay exact inside C2: a number in a description must match. | `judge_category_descriptor.md` |
| J4 | Invented objects fail: things named in neither the tool outputs nor the expected answer are hallucination, even when plausible. | `judge_general.md`, `judge_category_descriptor.md` |
| J5 | Multi-step reasoning: a summarize or describe step passes with any fair, grounded summary; the reference's details are examples. Counts, readings and comparisons stay exact. | `judge_category_multi_step_reasoning_*.md` |
| J6 | Small reference edits: brown-orange leaves accepted for Q_202/Q_204. | `benchmark/corrections.json` |

**Final answer plus trajectory.** The judge now grades only the final answer: the text of the
last assistant message with reasoning blocks, channel markers and leaked planning notes removed
(Muse-Glimmer writes planning before `<|eom|><|start|>assistant to=user<|message|>`; only what
follows counts). In place of the conversation it reads the trajectory: each tool call with its
arguments and what the tool returned. A reply with neither final text nor a final tool call
fails in code (`no_final_answer`). `JUDGE_INPUT=full` restores the old input to reproduce
September grades. `src/scope/eval/final_answer.py`

### The verdict and its error modes (rubric of 5 October 2026)

Decided by the maintainer on 5 October 2026 after the first October judging pass; every v2 run
is judged once under this rubric in the 6 October round (the pass before it is kept aside, not
used).

**Every verdict is schema-checked JSON.** Each seat answers through structured output
(`judge.verdict_schema`): `is_correct` (true/false), `reason` (one line, at most 400
characters), `error_mode` (from the task's allowed list), `all_modes` and `first_break`.
`JUDGE_MAX_TOKENS` caps a reply. A row without a verdict is asked again, then reported; it is
never counted as a failing vote and never decided by two seats. Nemotron answers with thinking
off (`JUDGE_DISABLE_THINKING=1`; with it on, the final-answer prompt sends it into a thinking
loop) and a 3,072-token reply cap. `JUDGE_INPUT=full` still reproduces the September grades
without any of this.

**The primary mode is the one that best explains the failure**, not the first that applies.
`all_modes` lists every mode that applies; `first_break` says where the chain first went wrong.

| Mode | Meaning |
|---|---|
| hallucination | the final answer states content (objects, details, values) that neither the tool outputs nor the expected answer support |
| claimed_unperformed_action | new: the answer says the agent did, checked or looked at something, but the trajectory has no matching tool call or result |
| admitted_failure | new: the answer plainly says the agent could not do, find or answer the task and claims nothing unsupported ("I could not find a red car in this view."); incorrect, no partial credit; if it also invents something it is hallucination |
| lack_of_tool_call | answered without the tool call(s) the task needed |
| tool_routing | the wrong tool for the task |
| tool_args | the right tool with wrong arguments |
| vision_request | new: the planner asked the vision model the wrong thing (wrong object, vague query, wrong view); it could have used the vision model better |
| vision_counting | the vision model miscounted on a reasonable request |
| VLM_query | the vision model misread, misdescribed or mislocated something on a reasonable request |
| Reasoning | the tool outputs held the right facts and the agent drew the wrong conclusion |
| view_scope, order, coverage | only when the question cues them |
| no_final_answer, action_contradiction | checked in code before any judge call (unchanged) |

| first_break | Meaning |
|---|---|
| vision_output | a vision tool's own output already disagrees with the expected answer, or is too vague to support it |
| planner | the tool outputs were right (or no vision tool was needed) and the agent's choices or wording went wrong, including inventing details over a correct vision output |
| harness | a tool or the environment failed (an error, an empty result, a camera that did not move) |
| none | the answer is correct |

**September-style label.** For continuity, the label September's first-match rubric would
give is derived in code from `all_modes` with September's order (hallucination, then
admitted_failure, lack_of_tool_call, tool_routing, tool_args, vision_counting, VLM_query,
Reasoning, view_scope, order, coverage); no extra judging. claimed_unperformed_action counts as
hallucination and vision_request as VLM_query; the label is marked approximate when it rests on
a new mode. Columns: `judge_all_modes`, `judge_first_break`, `judge_mode_september`,
`judge_mode_september_approx`.

**Reporting.** Verdicts stay two of three seats, Nemotron included. Failure-mode shares are
pooled over the incorrect votes of gpt-oss-120b and Muse-Glimmer (up to 536 x 2 labels per
run, N shown). Nemotron (Qwen3-Nemotron-32B-GenRM, a reward model) votes like the others but
labels differently: before this rubric it called a much larger share of its wrong votes
hallucination than the other two seats, so its labels are shown separately.

### Correction rules

Every change in [`corrections.json`](../benchmark/corrections.json) names its source: a
decision the benchmark owner recorded, or one of these rules, which were derived from those
decisions.

| Rule | What it says |
|---|---|
| R1 | Open-ended questions accept any fair, grounded description; the task gets an `also_accept` list written from the scene and the fair model answers. |
| R2 | A comparative reference states only the comparison; volunteered counts are dropped (C1). |
| R3 | The same claim in other words passes; where judges failed a paraphrase, it goes into `also_accept`. |
| R4 | A reference that contradicts the view is replaced with what the view shows, checked against the frames rather than the model majority. |
| R5 | The reference answers the question asked: a type for a type question, the reading for a reading question, the result for a command that asks for one. |
| R6 | A question points at a view where it can be answered, or names the part it means. |
| R7 | Counting stays exact; a count changes only when the frame shows a different number. |
| R8 | Text reading stays exact, against what the sign actually says. |
| R9 | When the thing asked about is absent, the reference says so and fair ways of saying it pass. |
| R10 | A question that repeats a decided one (same scene, preset and question) takes the same ruling. |
| R11 | A reference that matches the scene stays; low pass rates there are real model failure. |

## 2. The judge script no longer loses work silently

All in [`src/scope/eval/judge.py`](../src/scope/eval/judge.py). Each fix exists because a run
reported success while grading nothing or grading twice.

| Before | Now | Found by |
|---|---|---|
| JSON taken with a greedy `\{.*\}`, so a verdict followed by an explanation failed to parse and the row was scored wrong | First complete JSON object via `raw_decode` | 36% of one judge's rows failed on formatting alone |
| Resume skipped any row present in the output | Skips only rows with a verdict; ungraded rows are retried | One seat held 536 rows, 385 of them errors, and every rerun said "done" |
| Resume appended | Output rewritten whole through a temp file, swapped atomically, checkpointed every 25 rows | 792 rows out of a 541-row input after three restarts |
| One over-long transcript raised and killed the tag | Optional head-and-tail transcript cap (`JUDGE_TRANSCRIPT_CAP`, for example 24,000 chars; off by default); a refused row records `judge_error` and the tag continues | A 47,879-character transcript stalled a tag for two days |
| Serial calls | `--concurrency N` (`JUDGE_CONCURRENCY`; default 1, serial), written back in row order | 40,000 judge calls in a sweep |

Three of these change what a run measures, so record them with its results. Concurrency changes
only scheduling, but it does not make scores from different serving setups equal. The
transcript cap changes the evidence the judge sees; it is off unless `JUDGE_TRANSCRIPT_CAP` is
set (the panel sets 24,000, as October did). The parser can turn rows that were formatting failures into verdicts. Keep the raw judge
replies. Resume keys a row by question id and repeat index, so repeats stay separate.

## 3. The v2 judge panel

[`scripts/13_judge_panel.py`](../scripts/13_judge_panel.py) grades runs the way v2 was graded:
each of three judges grades every record with the October rubric above, the code checks fail
records first, and a record passes when at least two judges pass it. A record whose vision reply
could not be read as points also fails, and is tallied apart (the maintainer's rule of 9 October
2026).

**Serve the three judges** behind OpenAI-compatible endpoints. In October each had one 80 GB
A100, `--max-model-len 16384`, and the weights below; the GPU memory share and cache size change
speed, not verdicts.

| Seat (served name) | Model, revision | vLLM in October | Extra flags | Endpoint variable |
| --- | --- | --- | --- | --- |
| `gpt-oss-120b` | `openai/gpt-oss-120b`, `b5c939de` | `vllm/vllm-openai:v0.10.2` | `--max-num-seqs 64` | `PANEL_GPT_OSS_URL` |
| `muse-glimmer-judge` | `meta-models/Muse-Glimmer-30B`, `a4e59da5` | development build (vLLM PR 51655) | | `PANEL_MUSE_URL` |
| `qwen3-nemotron-genrm` | `nvidia/Qwen3-Nemotron-32B-GenRM-Principle`, `9fd58dd6` | `vllm/vllm-openai:v0.10.2` | | `PANEL_NEMOTRON_URL` |

The 30 October re-judge of 19 tasks served Muse-Glimmer and Nemotron with vLLM 0.28.1 instead;
how closely each agreed with its October verdicts is in
[`RESULTS_V2.md`](RESULTS_V2.md#dataset-behind-these-numbers).

**Run it.** Set the three endpoints in `.env` (see `.env.example`; `PANEL_<SEAT>_MODEL`,
`_API_KEY` and `_CONCURRENCY` are optional) or pass `--config panel.json`, then:

```bash
python scripts/13_judge_panel.py seats      # the endpoints and client settings it will use
python scripts/13_judge_panel.py judge -i results/my-pairing/raw_results.csv --run my-pairing -o results/panel
python scripts/13_judge_panel.py vote results/panel -o results/panel
```

`judge` writes one file per run and seat (`<run>__<seat>.csv`); `vote` writes
`panel_records.csv` (each record's three votes, code check and verdict), `panel_runs.csv` (per
run) and `panel_summary.json` (seat pass rates, agreement, categories). Several runs can share
one folder.

**Client settings, as in October.** Every seat grades the final answer and trajectory
(`JUDGE_INPUT=final+trajectory`) at temperature 0, answers in schema-checked JSON
(`JUDGE_STRUCTURED=1`) with replies capped at 8,192 tokens, and reads at most 24,000 characters
of trajectory (`JUDGE_TRANSCRIPT_CAP`, head and tail kept). Nemotron answers with thinking off and
a 3,072-token cap. Concurrency changes speed only.

**Resume and retries.** Re-run the same `judge` command to resume: rows with a verdict are kept;
a saved verdict whose task or answer has changed is dropped and judged again. A reply that is not
a full verdict is recorded without one and asked again, up to twice more; the last try lifts the
reply cap and bounds the reason and mode list in the grammar. Rows still without a verdict are
listed and the command exits non-zero. `vote` refuses to score while any verdict is missing: a
missing verdict is never counted as a fail.

**Reproducing the v2 tables.** `vote --check-against benchmark/results/v2-2026-10` compares each
run's passes and code-check counts with `models.csv`, and the overall v2 figures (passes, seat
pass rates, agreement, code checks, categories, group means) with `summary.json`. Run on the stored October votes (29 runs x 536 records x 3 seats; not published),
it matches every run and the summary exactly: 10,193 of 13,936 records pass over the 26
configurations, the code checks fail 218 (170 camera actions not carried out, 48 with no final
answer) and 9 vision replies could not be read. `tests/test_judge_panel.py` runs the same vote on
41 stored records (`tests/fixtures/judge_panel/`: the real October verdicts, with short synthetic
conversations in place of the agents' answers) and checks each verdict against the October
scoring code's. On 2,680 stored records the shipped judge builds the same prompts, verdict schema
and code-check results as the October runtime.

`scripts/11_judge.py` runs one judge with the same rubric against any OpenAI-compatible endpoint
(`JUDGE_*`); one seat's grades are not the panel's.

Why a panel of *different* models: one judge repeated three times at temperature 0 changed 1
verdict in 50, while two different judges on the same answers differed by 6.9 points. The
variance is between judges, so repeating one does not reduce it (Verga et al. 2024, arXiv 2404.18796). Why majority
and not unanimity: requiring all seats to agree moves the operating point toward "wrong"; on the
same answers the all-agree and any-agree rules were 25.6 points apart on one run.

The v2 panel uses the same three seats as the September campaign. Seat pass rates over the 26
v2 configurations (from `benchmark/results/v2-2026-10/summary.json`):

| Seat | Pass rate |
|---|---:|
| gpt-oss-120b | 71.6% |
| Muse-Glimmer-30B | 76.5% |
| Qwen3-Nemotron-32B-GenRM | 67.4% |

All three agree on 87.8% of records (79.8% under the September campaign's rubric). Nemotron sits below the
other two seats. The panel keeps it; whether to replace that seat is an open question.

## Known open issues, not fixed here

- `expected_tool_args_json` asks for `view_type: "current"` on some rows while
  `src/scope/tools/schema.json` marks `view_type` optional for `count_pointing` and
  `query_answer`, so a model that omits it can be marked wrong.
- The multi-step rubrics allow a `tool_args` error mode, but `judge.py` never passes
  `expected_tool_args_json` to the judge, so the judge infers the expectation. Camera
  directions are now checked in code (J1); other arguments are still inferred.
