---
name: scope-authoring-rows
description: Use when writing new benchmark questions for SCOPE, or reviewing existing ones. Covers the CSV columns, choosing answer_view, specifying expected tool calls, and what makes a question gradeable rather than merely plausible.
---

# Writing benchmark rows

A row is one question asked of a model at one camera position, plus everything needed to grade
the answer. The CSV has 25 columns; most rows use about ten of them.

## The columns that matter

| column | what it does |
|---|---|
| `question_id` | unique, e.g. `Q_217` |
| `file_location` | `scenes/<scene>/<scene>.blend` |
| `preset_start` | the camera position the question starts from. Must exist, or the row is answered from wherever the previous row left the camera |
| `presets_available` | JSON list the agent may move between |
| `question` | what is asked |
| `expected_answer` | the graded answer, written as a person would say it |
| `question_type` | `QA` or `Action` |
| `eval_category` | `counting`, `descriptor`, `ocr_identification`, `comparative_relational`, `multi_step_reasoning`, `single_call` |
| `answer_view` | `current` or `full` — see below |
| `expected_tool_order_json` | the tool calls that should happen, in order |
| `expected_tool_args_json` | the arguments they should carry |
| `multi_step_mode` | `none`, `unordered`, `strict` |
| `difficulty` | `Easy`, `Medium`, `Hard` |
| `evaluation_notes` | anything a grader needs to know |

## Choosing `answer_view`

`current` means answerable from the starting frame. `full` means the agent has to look around
first. Roughly one row in six is `full`.

**Only mark a row `full` when it genuinely needs it.** A full view is a stored panorama or wide
shot, and marking a row `full` unnecessarily tests the looking-around machinery rather than the
question.

Before writing a `full` row, look at that viewpoint's full view in
`docs/VISUAL_SMOKE_TEST.md` and check the thing you are asking about is actually in it. Some
scenes are modelled only on one side, so "how many X in the whole scene" may have a different
answer than you expect, and some viewpoints are served by a wide shot rather than a 360&deg;
sweep.

## Writing a gradeable question

- **Ask about something that is unambiguously there.** Signage, counts of a distinct object,
  a material, a spatial relation. Check the picture first.
- **Zero is a good answer.** Several shipped rows ask about objects that are absent, which
  tests whether a model will invent one.
- **Write `expected_answer` as a sentence**, not a bare token: "There are zero tanks."
- **Prefer counts you can verify by looking.** If you cannot count it confidently in the
  reference picture, a grader cannot either.
- **Avoid questions about the sky.** Captures do not draw it.
- **Avoid questions whose answer depends on lighting mode.** Anything behind glass reads
  differently in Solid shading than in Material Preview.

## Specifying tool calls

`expected_tool_order_json` is a list of `{"name": ..., "args": {...}}`. Use it when the *route*
matters, not only the answer. A counting question over the whole scene, for instance, should
use `view_type="full"`, and a row can require that:

```json
[{"name": "count_pointing", "args": {"instruction": "tanks", "view_type": "full"}}]
```

`multi_step_mode` controls how strictly the order is enforced: `strict` for a genuine sequence,
`unordered` when the calls can happen in any order, `none` when only the answer is graded.

## Before committing a batch

1. **Check every `preset_start` exists** for that scene. A missing preset does not stop the run;
   the camera stays where the previous row left it and the row is graded on the wrong picture.
   The runner warns and records `preset_applied` on the row, so check that field in the results.
2. **Check the `full` rows against the shipped full views.**
3. **Run a few end to end** and read the transcripts, not just the scores.

## The failure worth knowing about

A row can be answered confidently from a completely wrong picture. A capture that came back
blank, a preset that did not apply, a panorama of a different viewpoint: none of these raise,
and the model will answer whatever it was shown. Results carry a `panorama` provenance record
naming the exact image that answered each full-view row, and a `preset_applied` flag. Use them
when a score looks surprising, before assuming the model is at fault.
