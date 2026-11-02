# Archived benchmark files

The active benchmark is [`../scope_v2.csv`](../scope_v2.csv). Dataset v1 is kept so the paper
stays reproducible; select it with `scripts/run_fast.py --dataset v1`, or `SCOPE_DATASET=v1` for
`scripts/run_eval_pipeline.sh`.

SCOPE 0.1.0 shipped a 541-row file named `benchmark/scope_536.csv` (SHA-256
`84d34d28b9b626e7d8799ab4fa167eb3e3baf5e27c394fc889d35065c5835dcc`); it is removed, and
[`lineage_541_to_536.csv`](lineage_541_to_536.csv) lists its 23 removed and 18 inserted rows.

| File | What it is | Rows | SHA-256 |
|---|---|---:|---|
| `scope_536_v1.csv` | **Dataset v1.** The 536 tasks the HRI 2026 paper was scored on. v2 is this file plus [`../corrections.json`](../corrections.json). | 536 | `2003c5f40fdac248a58a4b7aca61dc3594a099a6c116befe877c01d0afe2affd` |
| `lineage_541_to_536.csv` | The correction pass that turned the 541-row file into v1: the 23 rows it removed and the 18 it inserted, each with its v1 counterpart. `row_in_541_file_json` holds each removed row whole, all 25 columns. | 41 | `dcad0915ff58e8cd5ff5ea3394d1cef0e3158c017b1b24d06fb770cc96505691` |
| `changelog_541_to_536.csv` | Row-by-row id map from the 541-row file to v1 (`change`, fields changed, old id, new id, scene, categories, question). Old and new answers: the 541-row file in git history (`benchmark/scope_536.csv` before 0.2.0) and `scope_536_v1.csv`. | 580 | `c9b781c230ced56f5abb13534f68ada95303cf7d7e00b40b249acb520ff9c468` |

## How v1 relates to the 541-row file

**Why 541 and not 536.** The paper's correction pass removed 23 rows from the 541-row file and
inserted 18: 541 - 23 + 18 = 536. Both lists survive from that pass and are in
[`lineage_541_to_536.csv`](lineage_541_to_536.csv), which carries every removed row exactly as
the 541-row file had it. The recorded reason for every removal is
"manifest"; nothing more specific was written down.

- **18 of the 23 removed rows are repeats.** Each asks a question that v1 still asks at the same
  scene and start preset (for example four copies of "Describe the storefront area in the
  current view." at book-nook store-front2, and four of "What is the name of the bank visible
  here?" at road1).
- **5 removed rows have no counterpart in v1 at all**, so they are the five tasks the 541-row
  file has and the paper never asked:

  | Id in the 541-row file | Scene, start | Question | Reference |
  |---|---|---|---|
  | Q_225 | book-nook, store-front2 | Which preset shows more bikes, store-front or road1? | The store-front preset shows more bikes (about six vs one). |
  | Q_229 | book-nook, road1 | (same as Q_225) | (same) |
  | Q_539 | postwar-city, street-view-1 | Go to street-view-2, tilt down a little, then tell me how many barricades are visible. | I moved to 'street-view-2', tilted down slightly, and there are several barricades visible. |
  | Q_540 | postwar-city, street-view-2 | (same as Q_539) | (same) |
  | Q_541 | postwar-city, diff-view-1 | (same as Q_539) | (same) |

  Each is one question repeated from several starts, and Q_539 to Q_541 ask for a count but
  accept "several". That reading is ours; the record says only "manifest".
- **17 of the 18 inserted rows are not in the 541-row file**; they are variants of existing
  questions (for example "Which side has more trees?" at whitechapel sor-viewpoint). The
  remaining one is in both files and was reworded later.
- **The other 518 rows carry edits and new ids.** Ids were renumbered by scene and category, so
  `Q_001` in the 541-row file and `Q_001` in v1 are different tasks. Matched by scene and
  question text ([`changelog_541_to_536.csv`](changelog_541_to_536.csv)): 422 unchanged, 75 edited
  (expected answer 62, start preset 15, question wording 13, multi-step mode 5, expected tool
  order 2; a row can change several fields), and 21 reworded enough that text matching pairs
  them as one removal plus one addition. That is why the text match reports 44 removed
  (23 + 21) and 39 added (18 + 21).

How v1 is known to be the paper's set:

- It is byte-identical to the question set the September 2026 re-run campaign used for all 29
  of its runs (same SHA-256 as above).
- Its per-category counts (counting 95, descriptor 89, single_call 72, comparative_relational
  62, multi_step_command 57, multi_step_reasoning 54, ocr_identification 54, location_spatial 53)
  match the counts implied by the rounding in the paper's own results table.
- The dataset audit reports that re-scoring published pairings against it reproduced the
  paper's table to 0.5 points mean error on 18 of 19 pairings. How that re-scoring was run is
  not recorded, so treat this as supporting evidence rather than proof.

## Duplicates

127 rows of v2 (130 of v1) repeat another row's scene, start preset, question and answer.
They are kept because the paper counted them; see [`../../docs/RESULTS_V2.md`](../../docs/RESULTS_V2.md#duplicate-tasks)
for what removing them would change.
