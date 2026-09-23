# Judging

The paper graded every answer with a single GPT-4o judge. This release keeps that path
(`scripts/11_judge.py` against any OpenAI-compatible endpoint) and changes three things around
it: what the rubric grades, how the judge script behaves on a long run, and a panel option.

## 1. Two criteria changes in the rubric

Both change published numbers. Neither was applied to the preliminary results in
[`PRELIMINARY_RESULTS.md`](PRELIMINARY_RESULTS.md), which is why those rows are marked for
re-grading.

**C1, `comparative_relational`: grade only the claim the question asks for.**
[`prompts/judge_category_comparative_relational.md`](../prompts/judge_category_comparative_relational.md)
now decides on the comparison itself (which side, view or object has more). Wrong details the
answer volunteers alongside a correct comparison, typically a count, no longer fail it. The
old rubric validated every count it saw first, so "the left side has more trees (3 vs 1)" was
failed when the counts were off even though the question asked only which side. In the
preliminary runs 80% of wrong `comparative_relational` answers already carried the reference's
comparison. Grounding still applies: a comparison the tool outputs contradict is a
hallucination.

**C2, `descriptor`: any fair description is correct.**
[`prompts/judge_category_descriptor.md`](../prompts/judge_category_descriptor.md) now treats the
expected answer as one fair description rather than the required phrasing. A grounded answer
that describes the same thing differently, or names other true details, passes unless it
contradicts the reference's core attribute. `Also accept: X` in an expected answer lists
alternates (Q_356 uses it). Scope is deliberate: `ocr_identification` stays exact, because a
sign has one right reading, and `location_spatial` stays tight.

C2 is the weaker of the two. It removes the rubric's pull toward one phrasing, but open answers
are only fully fair once each descriptor reference carries its alternates, and only one does
today.

## 2. The judge script no longer loses work silently

All in [`src/scope/eval/judge.py`](../src/scope/eval/judge.py). Each fix exists because a run
reported success while grading nothing or grading twice.

| Before | Now | Found by |
|---|---|---|
| JSON taken with a greedy `\{.*\}`, so a verdict followed by an explanation failed to parse and the row was scored wrong | First complete JSON object via `raw_decode` | 36% of one judge's rows failed on formatting alone |
| Resume skipped any row present in the output | Skips only rows with a verdict; ungraded rows are retried | One seat held 536 rows, 385 of them errors, and every rerun said "done" |
| Resume appended | Output rewritten whole through a temp file, swapped atomically, checkpointed every 25 rows | 792 rows out of a 541-row input after three restarts |
| One over-long transcript raised and killed the tag | Transcript capped head-and-tail (`JUDGE_TRANSCRIPT_CAP`, 24,000 chars); a refused row records `judge_error` and the tag continues | A 47,879-character transcript stalled a tag for two days |
| Serial calls | `--concurrency N` (`JUDGE_CONCURRENCY`, default 8), written back in row order | 40,000 judge calls in a sweep |

## 3. A judge panel

[`scripts/13_judge_panel.py`](../scripts/13_judge_panel.py) majority-votes several judged CSVs
of the same run:

```bash
python scripts/13_judge_panel.py judged/run__gpt-oss-120b.csv judged/run__muse.csv \
    judged/run__genrm.csv -o judged/run__panel.csv
```

It writes the voted CSV plus a JSON summary with the rule, per-seat accuracy, disagreement rate
and the all-agree to any-agree band. A seat below 98% of the fullest seat's graded rows is
dropped with a warning instead of shrinking the vote to its coverage.

Why a panel of *different* models: one judge repeated three times at temperature 0 changed 1
verdict in 50 ([`judge_repeatability.json`](../benchmark/results/preliminary-2026-09/judge_repeatability.json)),
while two different judges on the same answers differed by 6.9 points. The variance is between
judges, so repeating one does not reduce it (Verga et al. 2024, arXiv 2404.18796). Why majority
and not unanimity: requiring all seats to agree moves the operating point toward "wrong"; on the
same answers the all-agree and any-agree rules were 25.6 points apart on one run.

The preliminary panel was gpt-oss-120b, Muse-Glimmer-30B and Qwen3-Nemotron-32B-GenRM. Its
third seat is the weak one. From
[`pairings.json`](../benchmark/results/preliminary-2026-09/pairings.json), across 28 pairings it
scores on average 11.0 points below the majority (6.5 to 29.5), and the gap widens as pairings
get better (r = -0.48), so it compresses the top of the table. The other two sit at +0.1 and
+3.1. Treat the panel as provisional; replacing that seat is open.

## Known open issues, not fixed here

- `expected_tool_args_json` asks for `view_type: "current"` on some rows while
  `src/scope/tools/schema.json` marks `view_type` optional for `count_pointing` and
  `query_answer`, so a model that omits it can be marked wrong.
- The multi-step rubrics allow a `tool_args` error mode, but `judge.py` never passes
  `expected_tool_args_json` to the judge, so the judge infers the expectation.
