# Benchmark changelog

## v1.1 (September 2026 pre-release)

Three changes, in order: the dataset lineage, the audit corrections, and two judging criteria. Each has its own file so it can be checked on its own.

| File | What it is | Rows |
|---|---|---:|
| [`scope_536_paper.csv`](scope_536_paper.csv) | The task set the HRI '26 numbers were computed on | 536 |
| [`scope_536.csv`](scope_536.csv) | The paper set with the audit corrections applied. **The default.** | 536 |
| [`corrections.json`](corrections.json) | Every adjudicated change, with the text it replaces and why | 29 entries |
| [`changelog_541_to_536.csv`](changelog_541_to_536.csv) | Row-by-row map from the old 541-row file to the paper set | 580 rows |

`scope_536.csv` is generated, not edited: `python scripts/apply_corrections.py` rebuilds it
and `--check` fails if the committed file is stale.

### 1. The shipped file was the wrong lineage (541 rows -> 536)

Until this release `scope_536.csv` held **541 rows** from the pre-correction branch of the
dataset. The paper was scored on a different, corrected 536-row set. That set was recovered and
checked against the paper two ways:

- Re-running published pairings on it reproduces the paper's table to 0.5 points mean error
  across 18 of 19 pairings, several exactly (Qwen3-4B x Moondream2-4bit 62.3 against 62.3).
  Zero is not claimed: the judge and serving stack differ.
- Its per-category counts match the counts implied by the rounding in the paper's own table,
  predicted before the file was found: counting 95, descriptor 89, single_call 72,
  comparative_relational 62, multi_step_command 57, multi_step_reasoning 54,
  ocr_identification 54, location_spatial 53.

It is not "541 minus five". Matched by scene and question text:

| Change | Rows |
|---|---:|
| Unchanged | 422 |
| Edited (answer, question wording, start preset or step mode) | 75 |
| Removed | 44 |
| Added | 39 |

Edits by field: expected answer 62, start preset 15, question wording 13, multi-step mode 5,
expected tool order 2 (a row can change several). Removed rows by category: location_spatial 9,
comparative_relational 8, ocr_identification 8, multi_step_reasoning 6, descriptor 5,
multi_step_command 5, counting 3. Added: comparative_relational 9, counting 9,
location_spatial 7, multi_step_command 5, descriptor 4, ocr_identification 3,
multi_step_reasoning 2. Scene mix moved too: book-nook 168 -> 170, whitechapel 133 -> 137,
postwar-city 120 -> 109, city-street 120 -> 120.

**Question ids were renumbered.** The ids in this release are the paper set's ids; `Q_001` in
the old file and `Q_001` here are different questions. Use the changelog CSV to map old ids.

Per-row reasons for this lineage change were not recorded when the paper's correction pass was
made, so the changelog says *what* moved, not *why*. The matching is by content: rows whose
question was reworded are paired by similarity (>= 0.85 within a scene), so a heavily
rewritten question shows as one removal plus one addition.

### 2. Audit corrections (15 rows changed, 0 dropped)

A September 2026 audit examined 29 low-scoring questions closely, using the answers of 27
model pairings. 15 were faults in the benchmark; 14 were sound and their low scores are real
model failure (recorded as `keep` in `corrections.json`).

The dominant fault: **whitechapel has buildings on one side of the street only**, so a
180-degree pan faces open sky. Eight questions asked about that view with references asserting
street lamps, buildings and a wet road that are not there. Two were the same question with
opposite answers, both scoring near 40%.

**Answer changed (13):** Q_001, Q_002, Q_006, Q_007, Q_010, Q_098, Q_099, Q_315, Q_356,
Q_412, Q_462, Q_465, Q_523.

**Question changed (2):** Q_014 (now asks about windows; it duplicated Q_001), Q_236 (it
instructed the model to read a bank sign and then graded "no sign visible" as correct; it now
asks for the bank's name, which requires checking both store-front views).

`corrections.json` records the exact old and new text and the rationale for each. The audit's
display text was abbreviated for four rows (Q_412, Q_462, Q_465, Q_485); the file stores the
CSV's real text so `apply_corrections.py` can verify it before changing anything.

### 3. Judging criteria (no row changes, grading changes)

| | Rule | Scope |
|---|---|---|
| C1 | Grade only the claim the question asks for; wrong volunteered extras do not count against a correct answer | `comparative_relational` (62 rows) |
| C2 | Any fair, grounded description is correct, not one reference phrasing; `Also accept:` lists alternates | `descriptor` (89 rows). Not `ocr_identification`, not `location_spatial` |

Why C1: 80% of wrong `comparative_relational` answers in the preliminary runs already contained
the reference's comparison and were failed for a wrong count given alongside it. Details in
[`docs/JUDGING.md`](../docs/JUDGING.md).

### Not changed, and open

- **Duplicates.** 130 rows repeat another row's scene, start preset, question and answer
  exactly (406 distinct tasks by that definition). The audit measured that de-duplicating moves
  pairings by up to 2.3 points and swaps 17 pairs in the ranking. They are kept in this release
  because the paper counted them; whether v1.2 drops them is an open decision.
- The Hugging Face dataset (`HindsboNikolaj/scope-benchmark`) still ships `scope_541.csv`.

## v1.0 (HRI '26 release)

541-row `scope_536.csv`, superseded by v1.1 above.
