# Dataset v2: every change from v1

Generated from `benchmark/archive/scope_536_v1.csv` (v1, the paper's 536 tasks), `benchmark/scope_v2.csv` (v2) and `benchmark/corrections.json` (the reasons). Regenerate v2 with `python scripts/apply_corrections.py`; `--check` fails if the committed file has drifted. Nothing in v2 was edited by hand.

**187 of 536 rows changed, none added or removed, question ids unchanged.** `corrections.json` has 190 entries; 3 are `keep` (examined, nothing changed) and are listed at the end.

## What v2 contains, and what it leaves out

v2 applies exactly the corrections recorded as applied in the dataset audit's decision record (snapshot of 5 October 2026, covering the full audit of 27 September, the follow-ups of 30 September and 5 October, and the answer to Q_340 on 6 October), with accepted alternates moved verbatim into their own `also_accept` column, plus the maintainer's decision of 29 October 2026 on Q_347, Q_473 and Q_484. Where that record flagged the audit's release as going beyond a decision, v2 follows the decision:

- **Q_465, Q_468, Q_471, Q_474, Q_477**: the audit's duplicate rule (R10) had replaced the reference with Q_462's text. The decision was to keep the reference and accept more descriptions, so v2 keeps the v1 reference and lists the descriptions as `also_accept`.
- **Q_347, Q_473, Q_484**: the audit's rules (R1, R3, R10) added accepted descriptions that no decision covered (the decision was Keep for Q_473 and Q_484; there was none for Q_347). On 29 October 2026 the maintainer accepted the extra descriptions. v2 keeps the v1 references and takes the `also_accept` lists as pruned by the independent review of 6 October 2026 (row by row below).

The v2 results ([`docs/RESULTS_V2.md`](../docs/RESULTS_V2.md)) were judged on exactly this file.

## Planned for v2.1

Changes made after the decision record are **not** in v2 (maintainer's decision of 29 October 2026); they are planned for v2.1. 19 tasks:

- The independent review of 6 October 2026 of 30 reference rows: the references of Q_051 and Q_099 reworded to the maintainer's own text (Q_099's `also_accept` list with it), and the `also_accept` lists of 16 descriptive tasks pruned or merged (Q_189, Q_343, Q_350, Q_356, Q_461 to Q_463, Q_465, Q_468, Q_470 to Q_472, Q_474, Q_476 to Q_478). Its lists for Q_347, Q_473 and Q_484 are already in v2 (above).
- Q_259: later decisions (9 and 10 October 2026) made the zoom optional and changed its accepted alternate and expected tool calls.

Judging criteria C1 to C3 and the judging fixes change how answers are graded, not what any task says, so they are not in the CSV; they are defined in [`docs/JUDGING.md`](../docs/JUDGING.md), as are rules R1 to R11. "Maintainer decision" means the row was decided individually during the audit; a rule means the change follows that rule. Reasons ending in `…` were abbreviated in the audit record itself; the field values in each table are exact.

## Fields changed

| Field | Rows |
|---|---:|
| `also_accept` | 116 |
| `expected_answer` | 98 |
| `expected_tool_order_json` | 32 |
| `expected_tool_args_json` | 23 |
| `answer_view` | 13 |
| `evaluation_notes` | 10 |
| `question` | 10 |
| `scene_objects_mentioned` | 10 |
| `scope_requirement` | 6 |
| `expected_tool_order_signature` | 4 |
| `args_check` | 4 |
| `gt_evidence_json` | 2 |
| `preset_start` | 1 |

## Row by row

### Q_149 · book-nook · road1 · `comparative_relational`

| Field | v1 | v2 |
|---|---|---|
| `expected_tool_order_json` | [{"name": "go_to_preset", "args": {"name": "store-front"}}, {"name": "count_pointing", "args": {"instruction": "bike", "view_type": "current"}}, {"name": "go_to_preset", "args": {"name": "road1"}}, {"name": "count_pointing", "args": {"instruction": "bike", "view_type": "current"}}] | [{"name": "go_to_preset", "args": {"name": "store-front2"}}, {"name": "count_pointing", "args": {"instruction": "bike", "view_type": "current"}}, {"name": "go_to_preset", "args": {"name": "road1"}}, {"name": "count_pointing", "args": {"instruction": "bike", "view_type": "current"}}] |
| `expected_tool_args_json` | {"go_to_preset": {"name": "store-front"}, "count_pointing": {"instruction": "bike"}} | {"go_to_preset": {"name": "store-front2"}, "count_pointing": {"instruction": "bike", "view_type": "current"}} |
| `gt_evidence_json` | {"store-front": {"bike": 6}, "road1": {"bike": 1}} | {"store-front2": {"bike": 1}, "road1": {"bike": 1}} |

**Why** (Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): Expected tool calls had been copied from another question.

### Q_153 · book-nook · store-front · `comparative_relational`

| Field | v1 | v2 |
|---|---|---|
| `expected_tool_order_json` | [{"name": "go_to_preset", "args": {"name": "store-front"}}, {"name": "count_pointing", "args": {"instruction": "bike", "view_type": "current"}}, {"name": "go_to_preset", "args": {"name": "road1"}}, {"name": "count_pointing", "args": {"instruction": "bike", "view_type": "current"}}] | [{"name": "go_to_preset", "args": {"name": "store-front"}}, {"name": "count_pointing", "args": {"instruction": "bike", "view_type": "current"}}, {"name": "go_to_preset", "args": {"name": "store-front2"}}, {"name": "count_pointing", "args": {"instruction": "bike", "view_type": "current"}}] |
| `expected_tool_args_json` | {"go_to_preset": {"name": "store-front"}, "count_pointing": {"instruction": "bike"}} | {"go_to_preset": {"name": "store-front"}, "count_pointing": {"instruction": "bike", "view_type": "current"}} |
| `gt_evidence_json` | {"store-front": {"bike": 6}, "road1": {"bike": 1}} | {"store-front": {"bike": 6}, "store-front2": {"bike": 1}} |

**Why** (Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): Expected tool calls had been copied from another question.

### Q_155 · book-nook · road1 · `comparative_relational`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | There's one bike and one tree—an equal number. | There are an equal number of trees and bikes in the current view. |

**Why** (applied by rule R2 R10): Start view (Q_155_0_001217_raw) shows one prominent tree and one yellow bike, so 'equal' is right, but the reference leads with the counts; a second tree is faintly visible as a trunk sliver, so an answer with a different volunteered count but the right comparison should not fail. Stating only the comparison matches twin Q_143.

### Q_157 · book-nook · road1 · `comparative_relational`

| Field | v1 | v2 |
|---|---|---|
| `expected_tool_order_json` | [{"name": "query_answer", "args": {"instruction": "Which half has more bikes, the left or the right?", "view_type": "current"}}] | [{"name": "query_answer", "args": {"instruction": "Is the tree in the left or right half of the current view?", "view_type": "current"}}] |

**Why** (Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): Expected tool calls had been copied from another question.

### Q_161 · book-nook · store-front · `counting`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | There are about six bikes. | There are six bikes. |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_185 option A (exactly six everywhere)): 5 Oct: Same six bikes as Q_185; 'about' dropped so 5 and 7 no longer pass.

### Q_163 · book-nook · store-front2 · `counting`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | There is one tree. | I don't see any trees in the left half. |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_163 option A): 5 Oct: The left half of the store-front2 view holds the bank and a brick column; the one tree is at the right edge. Same wording as its verbatim twin Q_183.

### Q_165 · book-nook · store-front · `counting`

| Field | v1 | v2 |
|---|---|---|
| `expected_tool_order_json` | [{"name": "ptz_adjust"}, {"name": "count_pointing", "args": {"instruction": "bike", "view_type": "full"}}] | [{"name": "zoom_bounding", "args": {"instruction": "bike racks"}}, {"name": "count_pointing", "args": {"instruction": "bike", "view_type": "current"}}] |
| `expected_tool_args_json` | {"count_pointing": {"instruction": "bike", "view_type": "full"}} | {"count_pointing": {"instruction": "bike", "view_type": "current"}} |
| `answer_view` | full | current |
| `evaluation_notes` | Must use full 360° view. |  |
| `scope_requirement` | must_full | any |

**Why** (Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): Sound. 4% is genuine counting failure.

### Q_166 · book-nook · store-front · `counting`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I don't see any. | After zooming in 2×, I see 3 bikes in the current view. |
| `expected_tool_order_json` | [{"name": "ptz_adjust"}, {"name": "count_pointing", "args": {"instruction": "bike", "view_type": "full"}}] | [{"name": "ptz_adjust"}, {"name": "count_pointing", "args": {"instruction": "bike", "view_type": "current"}}] |
| `expected_tool_args_json` | {"count_pointing": {"instruction": "bike", "view_type": "full"}} | {"count_pointing": {"instruction": "bike", "view_type": "current"}} |
| `answer_view` | full | current |
| `evaluation_notes` | Must use full 360° view. |  |
| `scope_requirement` | must_full | any |

**Why** (Maintainer decision; rules R4 R7; Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): a captured frame and trace Q_166_0_001512_raw (2x zoom) show the upper halves of three bikes (red, teal, red) at the bottom edge beside the bench, so 'I don't see any' is wrong.

### Q_169 · book-nook · store-front · `counting`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | There are about six bikes in the left half | There are six bikes to the left. |
| `expected_tool_order_json` | [{"name": "count_pointing", "args": {"instruction": "bike", "view_type": "full"}}] | [{"name": "count_pointing", "args": {"instruction": "bikes in the left half of the frame", "view_type": "current"}}] |
| `answer_view` | full | current |
| `evaluation_notes` | Must use full 360° view. |  |
| `scope_requirement` | must_full | any |

**Why** (Maintainer decision; rules R7; Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): a captured frame (store-front start) and the render show all six bikes left of the frame centre; 'about six' becomes exact six. The existing corrections row wrongly lists preset store-front2, where the left half has no bikes; runs and paper say store-front.

### Q_170 · book-nook · store-front2 · `counting`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | There is one tree. | No, there is only one tree in view. |

**Why** (applied by rule R5): Count unchanged (one tree); the reference now answers the yes/no asked (R5).

### Q_171 · book-nook · road1 · `counting`

| Field | v1 | v2 |
|---|---|---|
| `expected_tool_order_json` | [{"name": "ptz_adjust"}, {"name": "count_pointing", "args": {"instruction": "fire hydrant", "view_type": "full"}}] | [{"name": "ptz_adjust"}, {"name": "count_pointing", "args": {"instruction": "fire hydrant", "view_type": "current"}}] |
| `expected_tool_args_json` | {"count_pointing": {"instruction": "fire hydrant", "view_type": "full"}} | {"count_pointing": {"instruction": "fire hydrant", "view_type": "current"}} |
| `answer_view` | full | current |
| `evaluation_notes` | Must use full 360° view. |  |

**Why** (Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): Expected tool calls had been copied from another question.

### Q_172 · book-nook · road1 · `counting`

| Field | v1 | v2 |
|---|---|---|
| `expected_tool_order_json` | [{"name": "ptz_adjust"}, {"name": "count_pointing", "args": {"instruction": "fire hydrant", "view_type": "full"}}] | [{"name": "zoom_bounding", "args": {"instruction": "fire hydrant"}}, {"name": "count_pointing", "args": {"instruction": "tree", "view_type": "current"}}] |
| `expected_tool_args_json` | {"count_pointing": {"instruction": "fire hydrant", "view_type": "full"}} | {"count_pointing": {"instruction": "tree", "view_type": "current"}} |
| `answer_view` | full | current |
| `evaluation_notes` | Must use full 360° view. |  |

**Why** (Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): Expected tool calls had been copied from another question.

### Q_173 · book-nook · road1 · `counting`

| Field | v1 | v2 |
|---|---|---|
| `expected_tool_order_json` | [{"name": "ptz_adjust"}, {"name": "count_pointing", "args": {"instruction": "fire hydrant", "view_type": "full"}}] | [{"name": "ptz_adjust"}, {"name": "count_pointing", "args": {"instruction": "fire hydrant", "view_type": "current"}}] |
| `expected_tool_args_json` | {"count_pointing": {"instruction": "fire hydrant", "view_type": "full"}} | {"count_pointing": {"instruction": "fire hydrant", "view_type": "current"}} |
| `answer_view` | full | current |
| `evaluation_notes` | Must use full 360° view. |  |

**Why** (Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): Expected tool calls had been copied from another question.

### Q_174 · book-nook · store-front · `counting`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | There are about six bikes. | There are six bikes. |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_185 option A (exactly six everywhere)): 5 Oct: Same six bikes as Q_185; 'about' dropped so 5 and 7 no longer pass.

### Q_176 · book-nook · road1 · `counting`

| Field | v1 | v2 |
|---|---|---|
| `expected_tool_order_json` | [{"name": "count_pointing", "args": {"instruction": "fire hydrant", "view_type": "full"}}] | [{"name": "count_pointing", "args": {"instruction": "fire hydrant in the left half of the frame", "view_type": "current"}}] |
| `answer_view` | full | current |
| `evaluation_notes` | Must use full 360° view. |  |

**Why** (Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): Expected tool calls had been copied from another question.

### Q_180 · book-nook · store-front2 · `counting`

| Field | v1 | v2 |
|---|---|---|
| `question` | After zooming in, how many trees do you see? | After zooming in 3x, how many trees do you see? |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_180 option A): 5 Oct: The tree stands at the right edge of the store-front2 view: still in frame after a 1.5x zoom, gone after 2x. Naming the amount makes 'none' the only right count.

### Q_185 · book-nook · store-front · `counting`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I count about six bikes. | I count six bikes. |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_185 option A (exactly six everywhere)): 5 Oct: Five bikes in the rack and a yellow one by the corner; Q_169 and Q_188 already say six.

### Q_188 · book-nook · store-front · `counting`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | After panning 20 degrees to the left, I see zero bikes. | After panning 20° to the left, I see 6 bikes. |

**Why** (Maintainer decision; rules R4 R7 R10): After 20 degrees left (Q_188_0_002337_ptz, Q_185_0_002255_ptz) six bikes are still in view, so 'zero bikes' is wrong.

### Q_189 · book-nook · store-front2 · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | The area shows a small shopfront, bikes, a bench, and nearby trees. | The Book Nook storefront: a black shopfront under a black-and-cream striped awning, display windows full of books and ornaments, and a yellow bike parked at the right. |
| `also_accept` | — | ["A black storefront with a striped awning and large windows displaying books", "a yellow bicycle is parked to the right", "A dark bookshop front with black trim, a black door and display windows of books and decorative items", "The Book Nook with book-filled windows, a chalkboard A-frame sign and a vertical BOOKS sign on the sidewalk", "A red-brick building with a fire escape above a black bookstore front, with an autumn tree at the right", "A bookshop with hanging plant baskets, a yellow curb and a red fire hydrant on the sidewalk to the left"] |

**Why** (applied by rule R4 R1): Start frame (render; Q_189_0_002351_raw.jpg) shows one yellow bike, one tree and no bench in front of the black Book Nook shopfront; the paper reference (bikes, a bench, trees) describes the store-front preset facade (bike rack and bench, a captured frame not this view. The qwen3 judge fails every fair description.

### Q_191 · book-nook · store-front · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A brick building with The Book Nook storefront and striped awning, autumn trees, a sidewalk and a street with parking lines", "The Book Nook bookshop with a rack of parked bicycles, a bench and a green trash can out front", "Red brick and white buildings, graffiti on the brick wall and shutters, two autumn trees and a street with a crosswalk", "A bookstore with book-filled display windows, bicycles, payphones, a fire hydrant, trees, a sidewalk and a road"] |

**Why** (applied by rule R1): The render book-nook__store-front.png shows The Book Nook (black storefront, striped awning), a five-bike rack plus a yellow bike, a bench, green trash can, two autumn trees, graffiti on the brick wall and neighbouring shutters, payphones, hydrant, crosswalk and parking lines. Judges fail fair lists that omit bikes, as with most failures here.

### Q_192 · book-nook · store-front2 · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | The area shows a small shopfront, bikes, a bench, and nearby trees. | The Book Nook storefront: a black shopfront under a black-and-cream striped awning, display windows full of books and ornaments, and a yellow bike parked at the right. |
| `also_accept` | — | ["A black storefront with a striped awning and large windows displaying books", "a yellow bicycle is parked to the right", "A dark bookshop front with black trim, a black door and display windows of books and decorative items", "The Book Nook with book-filled windows, a chalkboard A-frame sign and a vertical BOOKS sign on the sidewalk", "A red-brick building with a fire escape above a black bookstore front, with an autumn tree at the right", "A bookshop with hanging plant baskets, a yellow curb and a red fire hydrant on the sidewalk to the left"] |

**Why** (applied by rule R4 R1 R10): Start frame (render; Q_192_0_002423_screenshot.jpg) shows one yellow bike, one tree and no bench in front of the black Book Nook shopfront; the paper reference (bikes, a bench, trees) describes the store-front preset facade (bike rack and bench, a captured frame not this view. The qwen3 judge fails every fair description.

### Q_194 · book-nook · store-front · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A brick building with The Book Nook storefront and striped awning, autumn trees, a sidewalk and a street with parking lines", "The Book Nook bookshop with a rack of parked bicycles, a bench and a green trash can out front", "Red brick and white buildings, graffiti on the brick wall and shutters, two autumn trees and a street with a crosswalk", "A bookstore with book-filled display windows, bicycles, payphones, a fire hydrant, trees, a sidewalk and a road"] |

**Why** (applied by rule R1 R10): Verbatim repeat of Q_191; same view and same failure pattern (fair descriptions failed for omitting bikes). Takes the Q_191 ruling.

### Q_195 · book-nook · store-front2 · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | The area shows a small shopfront, bikes, a bench, and nearby trees. | The Book Nook storefront: a black shopfront under a black-and-cream striped awning, display windows full of books and ornaments, and a yellow bike parked at the right. |
| `also_accept` | — | ["A black storefront with a striped awning and large windows displaying books", "a yellow bicycle is parked to the right", "A dark bookshop front with black trim, a black door and display windows of books and decorative items", "The Book Nook with book-filled windows, a chalkboard A-frame sign and a vertical BOOKS sign on the sidewalk", "A red-brick building with a fire escape above a black bookstore front, with an autumn tree at the right", "A bookshop with hanging plant baskets, a yellow curb and a red fire hydrant on the sidewalk to the left"] |

**Why** (applied by rule R4 R1 R10): Start frame (render; Q_195_0_002457_raw.jpg) shows one yellow bike, one tree and no bench in front of the black Book Nook shopfront; the paper reference (bikes, a bench, trees) describes the store-front preset facade (bike rack and bench, a captured frame not this view. The qwen3 judge fails every fair description.

### Q_199 · book-nook · store-front · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A brick building with The Book Nook storefront and striped awning, autumn trees, a sidewalk and a street with parking lines", "The Book Nook bookshop with a rack of parked bicycles, a bench and a green trash can out front", "Red brick and white buildings, graffiti on the brick wall and shutters, two autumn trees and a street with a crosswalk", "A bookstore with book-filled display windows, bicycles, payphones, a fire hydrant, trees, a sidewalk and a road"] |

**Why** (applied by rule R1 R10): Verbatim repeat of Q_191; same view and same failure pattern (fair descriptions failed for omitting bikes). Takes the Q_191 ruling.

### Q_201 · book-nook · store-front · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A brick building with The Book Nook storefront and striped awning, autumn trees, a sidewalk and a street with parking lines", "The Book Nook bookshop with a rack of parked bicycles, a bench and a green trash can out front", "Red brick and white buildings, graffiti on the brick wall and shutters, two autumn trees and a street with a crosswalk", "A bookstore with book-filled display windows, bicycles, payphones, a fire hydrant, trees, a sidewalk and a road"] |

**Why** (applied by rule R1 R10): Verbatim repeat of Q_191; same view and same failure pattern (fair descriptions failed for omitting bikes). Takes the Q_191 ruling.

### Q_202 · book-nook · road1 · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `question` | What color is the nearest bike? | What color are the leaves on the tree? |
| `expected_answer` | The nearest bike is yellow. | The tree's leaves are yellow. |
| `also_accept` | — | ["yellow-orange autumn leaves", "golden or golden-yellow leaves", "a mix of yellow and orange autumn colours", "brown-orange or orange-brown autumn leaves, alone or with yellow (for example a mix of brown, orange and yellow)"] |
| `expected_tool_order_json` | [{"name": "query_answer", "args": {"instruction": "What color is the nearest bike?", "view_type": "current"}}] | [{"name": "query_answer", "args": {"instruction": "What color are the leaves on the tree?", "view_type": "current"}}] |
| `scene_objects_mentioned` | bike, book store | tree |

**Why** (Maintainer decision; rules R6 R10 R3; judging audit fix 6 (approved 2026-10-02); Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): The large corner tree in the start view (render crop) has golden-yellow autumn leaves with orange-brown tones (median leaf hue about 36 degrees), so yellow is right and yellow-orange/golden are the same claim; plain 'orange' is borderline and not added. This copy of the bike question is repurposed as the maintainer asked; the wording differs from its twin so the pair is not verbatim. October 2026: the judging audit found three answers failed for 'brown-orange with yellow' although the leaves carry orange-brown tones; brown-orange and orange-brown are added. Plain 'brown' alone is still not accepted.

### Q_203 · book-nook · store-front · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A brick building with The Book Nook storefront and striped awning, autumn trees, a sidewalk and a street with parking lines", "The Book Nook bookshop with a rack of parked bicycles, a bench and a green trash can out front", "Red brick and white buildings, graffiti on the brick wall and shutters, two autumn trees and a street with a crosswalk", "A bookstore with book-filled display windows, bicycles, payphones, a fire hydrant, trees, a sidewalk and a road"] |

**Why** (applied by rule R1 R10): Verbatim repeat of Q_191; same view and same failure pattern (fair descriptions failed for omitting bikes). Takes the Q_191 ruling.

### Q_204 · book-nook · road1 · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `question` | What color is the nearest bike? | What color are the tree's leaves in the current view? |
| `expected_answer` | The nearest bike is yellow. | The tree's leaves are yellow. |
| `also_accept` | — | ["yellow-orange autumn leaves", "golden or golden-yellow leaves", "a mix of yellow and orange autumn colours", "brown-orange or orange-brown autumn leaves, alone or with yellow (for example a mix of brown, orange and yellow)"] |
| `expected_tool_order_json` | [{"name": "query_answer", "args": {"instruction": "What color is the nearest bike?", "view_type": "current"}}] | [{"name": "query_answer", "args": {"instruction": "What color are the tree's leaves in the current view?", "view_type": "current"}}] |
| `scene_objects_mentioned` | bike, book store | tree |

**Why** (Maintainer decision; rules R6 R10 R3; judging audit fix 6 (approved 2026-10-02); Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): The large corner tree in the start view (render crop) has golden-yellow autumn leaves with orange-brown tones (median leaf hue about 36 degrees), so yellow is right and yellow-orange/golden are the same claim; plain 'orange' is borderline and not added. This copy of the bike question is repurposed as the maintainer asked; the wording differs from its twin so the pair is not verbatim. October 2026: the judging audit found three answers failed for 'brown-orange with yellow' although the leaves carry orange-brown tones; brown-orange and orange-brown are added. Plain 'brown' alone is still not accepted.

### Q_205 · book-nook · store-front · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A brick building with The Book Nook storefront and striped awning, autumn trees, a sidewalk and a street with parking lines", "The Book Nook bookshop with a rack of parked bicycles, a bench and a green trash can out front", "Red brick and white buildings, graffiti on the brick wall and shutters, two autumn trees and a street with a crosswalk", "A bookstore with book-filled display windows, bicycles, payphones, a fire hydrant, trees, a sidewalk and a road"] |

**Why** (applied by rule R1 R10): Verbatim repeat of Q_191; same view and same failure pattern (fair descriptions failed for omitting bikes). Takes the Q_191 ruling.

### Q_207 · book-nook · store-front · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["The tree is near the storefront", "The tree stands on the sidewalk by the curb, in front of the building next to The Book Nook", "One tree is at the left corner and one in front of the neighbouring building, both near the storefront"] |

**Why** (applied by rule R3 R10): 'Near' is one of the question's own options and is true of the scene (R3); all four copies alike (R10).

### Q_209 · book-nook · road1 · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["near the storefront", "in front of and near the storefront", "on the sidewalk strip just outside the store, left of its corner"] |

**Why** (applied by rule R3 R10): 'Near' is one of the question's own options and is true of the scene (R3); all four copies alike (R10).

### Q_212 · book-nook · store-front2 · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | The tree is near and in front of the storefront. Other acceptable: to the right of the store. | The tree is near and in front of the storefront. |
| `also_accept` | — | ["to the right of the storefront", "at the right end of the storefront, its leaves overlapping the sign and awning", "on the sidewalk beside the store, to its right"] |

**Why** (applied by rule R1 R3): The tree stands on the sidewalk at the right end of the storefront with its leaves overlapping the sign and awning (render), so 'near and in front' and 'to the right' are both fair; the existing 'Other acceptable' clause is restated in Also accept form.

### Q_213 · book-nook · road1 · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `expected_tool_order_json` | [{"name": "query_answer", "args": {"instruction": "Is the bike left or right of the fire hydrant?", "view_type": "current"}}] | [{"name": "query_answer", "args": {"instruction": "Is the fire hydrant on the left or right side of the road?", "view_type": "current"}}] |
| `expected_tool_args_json` | {"query_answer": {"instruction": "Is the bike left or right of the fire hydrant?", "view_type": "current"}} | {"query_answer": {"instruction": "Is the fire hydrant on the left or right side of the road?", "view_type": "current"}} |

**Why** (Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): Expected tool calls had been copied from another question.

### Q_214 · book-nook · store-front · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["The tree is near the storefront", "The tree stands on the sidewalk by the curb, in front of the building next to The Book Nook", "One tree is at the left corner and one in front of the neighbouring building, both near the storefront"] |

**Why** (applied by rule R3 R10): 'Near' is one of the question's own options and is true of the scene (R3); all four copies alike (R10).

### Q_217 · book-nook · store-front · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `expected_tool_order_json` | [{"name": "query_answer", "args": {"instruction": "Is the bike left or right of the fire hydrant?", "view_type": "current"}}] | [{"name": "query_answer", "args": {"instruction": "Is the graffiti to the right or left of the bikes?", "view_type": "current"}}] |
| `scene_objects_mentioned` | bike, fire hydrant | graffiti, bike |

**Why** (Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): Expected tool calls had been copied from another question.

### Q_219 · book-nook · road1 · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | The bike is left of the fire hydrant. | The bike is to the right of the fire hydrant. |

**Why** (applied by rule R4 R10): The start view (render; Q_219_0_002902_raw) has the hydrant at x about 0.84 and the yellow bike at about 0.94, so the bike is right of the hydrant; 'left' contradicts the frame and twin Q_206. Only 5/29 pass, the ones that echo 'left'.

### Q_220 · book-nook · store-front · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["The tree is near the storefront", "The tree stands on the sidewalk by the curb, in front of the building next to The Book Nook", "One tree is at the left corner and one in front of the neighbouring building, both near the storefront"] |

**Why** (applied by rule R3 R10): 'Near' is one of the question's own options and is true of the scene (R3); all four copies alike (R10).

### Q_222 · book-nook · store-front2 · `multi_step_command`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have completed the requested actions. | I zoomed in slightly and counted one bike. |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_222 option A): 5 Oct: At the 10-25% zooms every run used, only the yellow bike stays in view at store-front2.

### Q_225 · book-nook · store-front2 · `multi_step_command`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have completed the requested actions. | I zoomed in slightly and counted one bike. |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_222 option A): 5 Oct: At the 10-25% zooms every run used, only the yellow bike stays in view at store-front2.

### Q_228 · book-nook · store-front2 · `multi_step_command`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have completed the requested actions. | I zoomed in slightly and counted one bike. |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_222 option A): 5 Oct: At the 10-25% zooms every run used, only the yellow bike stays in view at store-front2.

### Q_231 · book-nook · store-front2 · `multi_step_command`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have completed the requested actions. | I zoomed in slightly and counted one bike. |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_222 option A): 5 Oct: At the 10-25% zooms every run used, only the yellow bike stays in view at store-front2.

### Q_234 · book-nook · store-front2 · `multi_step_command`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have completed the requested actions. | I zoomed in slightly and counted one bike. |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_222 option A): 5 Oct: At the 10-25% zooms every run used, only the yellow bike stays in view at store-front2.

### Q_236 · book-nook · store-front · `multi_step_command`

| Field | v1 | v2 |
|---|---|---|
| `question` | Go to the store-front preset, then zoom in on the bank sign, then read it out loud. | Go to the store-front2 preset, then zoom in on the bank sign, then read it out loud. |
| `expected_answer` | I don't see a bank sign in this view. | Banque Strechnitz. |
| `expected_tool_order_json` | [{"name": "go_to_preset", "args": {"name": "store-front"}}, {"name": "zoom_bounding", "args": {"instruction": "bank sign"}}, {"name": "query_answer", "args": {"instruction": "Read the bank name", "view_type": "current"}}] | [{"name": "go_to_preset", "args": {"name": "store-front2"}}, {"name": "zoom_bounding", "args": {"instruction": "bank sign"}}, {"name": "query_answer", "args": {"instruction": "Read the bank name", "view_type": "current"}}] |
| `scene_objects_mentioned` |  | bank, sign |
| `expected_tool_order_signature` | [{"name": "go_to_preset", "args": {"name": "store-front"}}, {"name": "zoom_bounding", "args": {"instruction": "bank sign"}}, {"name": "query_answer", "args": {"instruction": "Read the bank name", "view_type": "current"}}] | [{"name": "go_to_preset", "args": {"name": "store-front2"}}, {"name": "zoom_bounding", "args": {"instruction": "bank sign"}}, {"name": "query_answer", "args": {"instruction": "Read the bank name", "view_type": "current"}}] |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_236 option B (store-front2 wording); Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): As written it instructs the model to zoom in on a sign and read it, then grades it correct only for reporting the sign is absent. 9% pass across 163 gradings. The instruction and the answer are pulling opposite ways, which is why almost everything fails it. The rewritten open-navigation question does not prescribe the old go-to-store-front-and-zoom route, so its obsolete expected tool sequence is removed; its category stays unchanged. 5 Oct: the maintainer chose the store-front2 wording although Q_242 asks nearly the same; expected tools now go to store-front2, where the bank sign is.

### Q_239 · book-nook · store-front · `multi_step_command`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have executed the steps in order. | The bank is Banque Strechnitz. |
| `expected_tool_order_json` | [{"name": "go_to_preset", "args": {"name": "store-front"}}, {"name": "zoom_bounding", "args": {"instruction": "bank sign"}}, {"name": "query_answer", "args": {"instruction": "Read the bank name", "view_type": "current"}}] | [{"name": "go_to_preset", "args": {"name": "store-front2"}}, {"name": "zoom_bounding", "args": {"instruction": "bank sign"}}, {"name": "query_answer", "args": {"instruction": "Read the bank name", "view_type": "current"}}] |
| `scene_objects_mentioned` |  | bank, sign |
| `expected_tool_order_signature` | [{"name": "go_to_preset", "args": {"name": "store-front"}}, {"name": "zoom_bounding", "args": {"instruction": "bank sign"}}, {"name": "query_answer", "args": {"instruction": "Read the bank name", "view_type": "current"}}] | [{"name": "go_to_preset", "args": {"name": "store-front2"}}, {"name": "zoom_bounding", "args": {"instruction": "bank sign"}}, {"name": "query_answer", "args": {"instruction": "Read the bank name", "view_type": "current"}}] |

**Why** (Maintainer decision; rules R5 R8; Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): 'I have executed the steps in order' does not give the reading. No bank sign is visible from store-front (render); store-front2 frames (Q_239_0_001052_raw) show 'BANQUE STRECHNITZ', so the model must find it there.

### Q_242 · book-nook · store-front · `multi_step_command`

| Field | v1 | v2 |
|---|---|---|
| `expected_tool_order_json` | [{"name": "go_to_preset", "args": {"name": "store-front"}}, {"name": "zoom_bounding", "args": {"instruction": "bank sign"}}, {"name": "query_answer", "args": {"instruction": "Read the bank name", "view_type": "current"}}] | [{"name": "go_to_preset", "args": {"name": "store-front2"}}, {"name": "zoom_bounding", "args": {"instruction": "bank sign"}}, {"name": "query_answer", "args": {"instruction": "Read the bank name", "view_type": "current"}}] |
| `scene_objects_mentioned` |  | bank, sign |
| `expected_tool_order_signature` | [{"name": "go_to_preset", "args": {"name": "store-front"}}, {"name": "zoom_bounding", "args": {"instruction": "bank sign"}}, {"name": "query_answer", "args": {"instruction": "Read the bank name", "view_type": "current"}}] | [{"name": "go_to_preset", "args": {"name": "store-front2"}}, {"name": "zoom_bounding", "args": {"instruction": "bank sign"}}, {"name": "query_answer", "args": {"instruction": "Read the bank name", "view_type": "current"}}] |

**Why** (Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): Expected tool calls had been copied from another question.

### Q_245 · book-nook · store-front · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have compared the requested views and provided the result. | The count before the slight zoom is higher or the same; zooming in only narrows the view. |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_245 option A (grade only the comparison)): 5 Oct: Six bikes before and five after a 1.5x zoom; only the comparison is graded (C1).

### Q_246 · book-nook · store-front2 · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have compared the requested views and provided the result. | Start view: the stone BANQUE STRECHNITZ bank and the brick Book Nook bookshop with a striped awning, a red fire hydrant, a yellow bike and an autumn tree. road1: a street corner with a crosswalk, the autumn tree on the corner sidewalk and the Book Nook building at the right edge. |
| `also_accept` | — | ["First a bank and The Book Nook bookstore with a hydrant, bike and tree", "then road1: a corner with a tree and crosswalk", "Initial view: The Book Nook's black storefront beside a stone bank", "road1: an empty street and crosswalk with a yellow-leaved tree by the building", "A bookstore street scene with hydrant, yellow bike and autumn tree, then a road1 corner view with crosswalk, tree and brick building"] |

**Why** (applied by rule R5 R1): The reference 'I have compared the requested views and provided the result' states no summary. Start frame is the bank + Book Nook view (render); road1 frames (e.g. Q_246_0_001257_raw.jpg) show a wide street, a white crosswalk, the autumn tree on the corner with a small red hydrant, and the brick Book Nook building at the right edge.

### Q_247 · book-nook · road1 · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have compared the requested views and provided the result. | Yes, a bike is present, and the bookstore sign reads "The Book Nook". |

**Why** (applied by rule R5 R8): The reference is a comparison placeholder that states no result, so judges passed wrong readings ('The Booth', 'The Book Store', 'Bookends', 'illegible'). The start view has the yellow bike and the sign's first letters 'Th' at the right edge; turned toward that facade (Q_140_0_000653_raw) the sign reads 'The Book Nook'. This task's own frames share timestamps with store-front2 tasks and are not reliable.

### Q_248 · book-nook · store-front · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have compared the requested views and provided the result. | The count before the slight zoom is higher or the same; zooming in only narrows the view. |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_245 option A (grade only the comparison)): 5 Oct: Six bikes before and five after a 1.5x zoom; only the comparison is graded (C1).

### Q_249 · book-nook · store-front2 · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have compared the requested views and provided the result. | Start view: the stone BANQUE STRECHNITZ bank and the brick Book Nook bookshop with a striped awning, a red fire hydrant, a yellow bike and an autumn tree. road1: a street corner with a crosswalk, the autumn tree on the corner sidewalk and the Book Nook building at the right edge. |
| `also_accept` | — | ["First a bank and The Book Nook bookstore with a hydrant, bike and tree", "then road1: a corner with a tree and crosswalk", "Initial view: The Book Nook's black storefront beside a stone bank", "road1: an empty street and crosswalk with a yellow-leaved tree by the building", "A bookstore street scene with hydrant, yellow bike and autumn tree, then a road1 corner view with crosswalk, tree and brick building"] |

**Why** (applied by rule R5 R1 R10): The reference 'I have compared the requested views and provided the result' states no summary. Start frame is the bank + Book Nook view (render); road1 frames (e.g. Q_246_0_001257_raw.jpg) show a wide street, a white crosswalk, the autumn tree on the corner with a small red hydrant, and the brick Book Nook building at the right edge.

### Q_250 · book-nook · road1 · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have compared the requested views and provided the result. | Yes, a bike is present, and the bookstore sign reads "The Book Nook". |

**Why** (applied by rule R5 R8 R10): Verbatim repeat of Q_247 with the same placeholder reference; takes the Q_247 ruling. The reference is a comparison placeholder that states no result, so judges passed wrong readings ('The Booth', 'The Book Store', 'Bookends', 'illegible'). The start view has the yellow bike and the sign's first letters 'Th' at the right edge; turned toward that facade (Q_140_0_000653_raw) the sign reads 'The Book Nook'.

### Q_251 · book-nook · store-front · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have compared the requested views and provided the result. | The count before the slight zoom is higher or the same; zooming in only narrows the view. |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_245 option A (grade only the comparison)): 5 Oct: Six bikes before and five after a 1.5x zoom; only the comparison is graded (C1).

### Q_252 · book-nook · store-front2 · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have compared the requested views and provided the result. | Start view: the stone BANQUE STRECHNITZ bank and the brick Book Nook bookshop with a striped awning, a red fire hydrant, a yellow bike and an autumn tree. road1: a street corner with a crosswalk, the autumn tree on the corner sidewalk and the Book Nook building at the right edge. |
| `also_accept` | — | ["First a bank and The Book Nook bookstore with a hydrant, bike and tree", "then road1: a corner with a tree and crosswalk", "Initial view: The Book Nook's black storefront beside a stone bank", "road1: an empty street and crosswalk with a yellow-leaved tree by the building", "A bookstore street scene with hydrant, yellow bike and autumn tree, then a road1 corner view with crosswalk, tree and brick building"] |

**Why** (applied by rule R5 R1 R10): The reference 'I have compared the requested views and provided the result' states no summary. Start frame is the bank + Book Nook view (render); road1 frames (e.g. Q_246_0_001257_raw.jpg) show a wide street, a white crosswalk, the autumn tree on the corner with a small red hydrant, and the brick Book Nook building at the right edge.

### Q_253 · book-nook · road1 · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have compared the requested views and provided the result. | Yes, a bike is present, and the bookstore sign reads "The Book Nook". |

**Why** (applied by rule R5 R8 R10): Verbatim repeat of Q_247 with the same placeholder reference; takes the Q_247 ruling. The reference is a comparison placeholder that states no result, so judges passed wrong readings ('The Booth', 'The Book Store', 'Bookends', 'illegible'). The start view has the yellow bike and the sign's first letters 'Th' at the right edge; turned toward that facade (Q_140_0_000653_raw) the sign reads 'The Book Nook'.

### Q_254 · book-nook · store-front · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have compared the requested views and provided the result. | The count before the slight zoom is higher or the same; zooming in only narrows the view. |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_245 option A (grade only the comparison)): 5 Oct: Six bikes before and five after a 1.5x zoom; only the comparison is graded (C1).

### Q_255 · book-nook · store-front2 · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have compared the requested views and provided the result. | Start view: the stone BANQUE STRECHNITZ bank and the brick Book Nook bookshop with a striped awning, a red fire hydrant, a yellow bike and an autumn tree. road1: a street corner with a crosswalk, the autumn tree on the corner sidewalk and the Book Nook building at the right edge. |
| `also_accept` | — | ["First a bank and The Book Nook bookstore with a hydrant, bike and tree", "then road1: a corner with a tree and crosswalk", "Initial view: The Book Nook's black storefront beside a stone bank", "road1: an empty street and crosswalk with a yellow-leaved tree by the building", "A bookstore street scene with hydrant, yellow bike and autumn tree, then a road1 corner view with crosswalk, tree and brick building"] |

**Why** (applied by rule R5 R1 R10): The reference 'I have compared the requested views and provided the result' states no summary. Start frame is the bank + Book Nook view (render); road1 frames (e.g. Q_246_0_001257_raw.jpg) show a wide street, a white crosswalk, the autumn tree on the corner with a small red hydrant, and the brick Book Nook building at the right edge.

### Q_256 · book-nook · road1 · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have compared the requested views and provided the result. | Yes, a bike is present, and the bookstore sign reads "The Book Nook". |

**Why** (applied by rule R5 R8 R10): Verbatim repeat of Q_247 with the same placeholder reference; takes the Q_247 ruling. The reference is a comparison placeholder that states no result, so judges passed wrong readings ('The Booth', 'The Book Store', 'Bookends', 'illegible'). The start view has the yellow bike and the sign's first letters 'Th' at the right edge; turned toward that facade (Q_140_0_000653_raw) the sign reads 'The Book Nook'.

### Q_258 · book-nook · store-front2 · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have followed the steps and provided the result. | I went to road1, panned right 51.5 degrees and read the bookstore sign: 'The Book Nook'. |

**Why** (applied by rule R5 R8): After road1 + 51.5 degrees right (Q_258_0_002228_ptz.jpg) the Book Nook facade faces the camera and its sign reads 'The Book Nook'; the reference 'I have followed the steps and provided the result' omits the reading asked for.

### Q_259 · book-nook · road1 · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have followed the steps and provided the result. | The storefront windows are not visible from this view, so what is inside cannot be described from here. |
| `also_accept` | — | ["books on shelves and a few ornaments (after moving to a view of the windows)"] |
| `args_check` | True | False |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_259 option B (keep road1, accept 'not visible from here'); Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): 5 Oct: From road1 the Book Nook's windows are out of frame. Interpretation: an agent that moves and describes the books still passes; expected tools no longer require the zoom.

### Q_262 · book-nook · road1 · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have followed the steps and provided the result. | The bookstore sign reads "The Book Nook". |

**Why** (applied by rule R5 R8): The placeholder states no name. road1 panned right about 50 degrees (Q_140_0_000653_raw) frames the Book Nook main facade with the sign 'The Book Nook' clearly legible, and 51.5 degrees is the same view; most models read it correctly.

### Q_263 · book-nook · store-front · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have followed the steps and provided the result. | Inside the windows are wooden bookshelves filled with books, along with decorative items such as a globe, a small cactus and ornaments. |
| `also_accept` | — | ["Wooden bookshelves filled with books of various colours", "Shelves of books with some decorative objects and plants", "A bookshop display of books on shelves with a few ornaments"] |

**Why** (applied by rule R5 R1): 'I have followed the steps' is not a description, and fair answers ('bookshelves filled with books') fail against it. The render book-nook__store-front.png windows show wooden shelves with books, a globe, a small cactus, a picture frame and ornaments.

### Q_265 · book-nook · road1 · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have followed the steps and provided the result. | There are six bikes at store-front, all on the left side, so the left side has more. |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_376 option A (write the counts in)): 5 Oct: store-front: five bikes in the rack and a yellow one by the corner, all left of centre.

### Q_266 · book-nook · store-front · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have followed the steps and provided the result. | The bookstore sign reads 'The Book Nook'. |

**Why** (applied by rule R5 R8): Reading 'The Book Nook' is established by his Q_271/Q_276 decisions; same change as Q_258 and Q_262 (R5, R8, R10).

### Q_267 · book-nook · store-front2 · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have followed the steps and provided the result. | I zoomed to the storefront windows; inside are wooden bookshelves holding books and small decorative items such as potted plants, a globe and ornaments. |
| `also_accept` | — | ["The windows display a wooden bookshelf filled with books and decorative items", "Shelves of books, upright and stacked, with small potted plants, vases and ornaments", "Books on display shelves inside the bookshop windows"] |

**Why** (applied by rule R5 R1): The reference gives no description. The render shows both display windows as wooden shelves of books with small ornaments (mushroom figures, a globe, vases, potted plants and a succulent); the traced zoom frames belong to other tasks.

### Q_271 · book-nook · store-front · `ocr_identification`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | The sign reads 'Books'. | The sign says "The Book Nook." |
| `also_accept` | — | ["There is no sign reading just 'Books' in this view, the storefront sign reads 'The Book Nook'"] |

**Why** (Maintainer decision; rules R4 R8 R9): From store-front the only readable storefront sign is 'The Book Nook' (render, a captured frame the corner A-frame shows a poster, not text. A white vertical 'BOOKS' sign exists by the entrance but only in store-front2 frames (Q_147_0_000919_raw). The paper's 'Books' just echoes the question.

### Q_272 · book-nook · store-front · `ocr_identification`

| Field | v1 | v2 |
|---|---|---|
| `expected_tool_order_json` | [{"name": "zoom_bounding", "args": {"instruction": "storefront sign"}}, {"name": "query_answer", "args": {"instruction": "Read the sign that says 'Books' near the storefront.", "view_type": "full"}}] | [{"name": "query_answer", "args": {"instruction": "Is the graffiti on the Book Nook store, to the left, or right of the store?", "view_type": "current"}}] |
| `args_check` | True | False |
| `expected_tool_args_json` | {"query_answer": {"instruction": "Read the sign that says 'Books' near the storefront.", "view_type": "current"}} | {} |
| `answer_view` | full | current |
| `evaluation_notes` | Must use full 360° view.; Headline/name match can be case/punctuation/spacing normalized. |  |
| `scene_objects_mentioned` | book store, bank, sign | book store, graffiti |
| `scope_requirement` | must_full | any |

**Why** (Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): Expected tool calls had been copied from another question.

### Q_273 · book-nook · store-front · `ocr_identification`

| Field | v1 | v2 |
|---|---|---|
| `expected_tool_order_json` | [{"name": "zoom_bounding", "args": {"instruction": "storefront sign"}}, {"name": "query_answer", "args": {"instruction": "Read the sign that says 'Books' near the storefront.", "view_type": "full"}}] | [{"name": "query_answer", "args": {"instruction": "Is there any graffiti on the black-painted part of the Book Nook store?", "view_type": "current"}}] |
| `args_check` | True | False |
| `expected_tool_args_json` | {"query_answer": {"instruction": "Read the sign that says 'Books' near the storefront.", "view_type": "current"}} | {} |
| `answer_view` | full | current |
| `evaluation_notes` | Must use full 360° view.; Headline/name match can be case/punctuation/spacing normalized. |  |
| `scene_objects_mentioned` | book store, bank, sign | book store, graffiti |
| `scope_requirement` | must_full | any |

**Why** (Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): Expected tool calls had been copied from another question.

### Q_274 · book-nook · store-front · `ocr_identification`

| Field | v1 | v2 |
|---|---|---|
| `question` | Is there any graffiti on the Book Nook store? | Is there any graffiti on the black part of the Book Nook store? |
| `expected_answer` | No, I don't see any graffiti on the store. | No, I don't see any graffiti on the black part of the Book Nook store. |
| `also_accept` | — | ["No, the black storefront is clean and the graffiti is on the red brick wall to its right and the neighbouring building"] |
| `expected_tool_order_json` | [{"name": "zoom_bounding", "args": {"instruction": "storefront sign"}}, {"name": "query_answer", "args": {"instruction": "Read the sign that says 'Books' near the storefront.", "view_type": "full"}}] | [{"name": "query_answer", "args": {"instruction": "Is there any graffiti on the black part of the Book Nook store?", "view_type": "current"}}] |
| `args_check` | True | False |
| `expected_tool_args_json` | {"query_answer": {"instruction": "Read the sign that says 'Books' near the storefront.", "view_type": "current"}} | {} |
| `answer_view` | full | current |
| `evaluation_notes` | Must use full 360° view.; Headline/name match can be case/punctuation/spacing normalized. |  |
| `scene_objects_mentioned` | book store, bank, sign | book store, graffiti |
| `scope_requirement` | must_full | any |

**Why** (Maintainer decision; rules R6; Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): The Book Nook occupies the black ground floor of a red brick building whose wall right of the shop carries graffiti (render book-nook__store-front.png), so 'on the Book Nook store' is ambiguous and 'yes' is defensible (1/29 pass).

### Q_276 · book-nook · store-front · `ocr_identification`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | The sign reads 'Books'. | The sign says "The Book Nook." |
| `also_accept` | — | ["There is no sign reading just 'Books' in this view, the storefront sign reads 'The Book Nook'"] |

**Why** (Maintainer decision; rules R4 R8 R10): Same question as Q_271; from store-front only 'The Book Nook' is readable, the 'BOOKS' vertical sign is only in the store-front2 view.

### Q_308 · city-street · mailbox · `comparative_relational`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | There are the same number of mailboxes and trees (2 vs 2). | There are the same number of mailboxes and trees. |

**Why** (Maintainer decision; rules R2): start frame shows exactly two dark mailboxes (left of the red stoop, between the stoops) and two trees; the light-grey cylinder by the right stoop is a trash can, not a mailbox. The reference is right and the failures are real (runs answer 'more mailboxes' from a count of 3); the change only drops the parenthetical count, per R2 and his decision.

### Q_309 · city-street · hotel-m · `comparative_relational`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | There are an equal number of garbage cans and trees (2 vs 2). | There are an equal number of garbage cans and trees. |
| `answer_view` | full | current |

**Why** (applied by rule R2; Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): The hotel-m start frame (Q_330_0_001501_raw.jpg) shows two garbage cans (a beige stone one and a dark grey one) at the curb and two trees (the bare tree left of the entrance and the tree with dried leaves at right); only stray branches reach in at the far left edge. Failing runs report 3 cans, apparently counting the mailbox by the lamp. The first-audit frame Q_309.jpg shows the brownstone (mailbox) side, not this…

### Q_314 · city-street · mailbox · `comparative_relational`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | There are more mailboxes than fire hydrants (3 vs 2). | There are more mailboxes than fire hydrants. |

**Why** (Maintainer decision; rules R2): The shipped full-view panorama benchmark/panoramas/city-street__preset-mailbox.png (byte-identical to the audit's own GPU-stitched panorama) shows 3 mailboxes (two at the brownstones, one by the hotel's lamp post) and 2 red fire hydrants (one at each far street corner, pano frames 002 and 006). So 'more mailboxes' is right. Nearly every run gets that comparison right but…

### Q_315 · city-street · hotel-m · `comparative_relational`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | There are more trees than street lamps (8 vs 4). | There are more trees than street lamps. |

**Why** (Maintainer decision; rules R2): The full 360 from this camera position (the audit's stitched render, matching the benchmark pan frames in Q_396/Q_419) has about 8 trees and 3-4 street lamps. Every run said "more trees" and was failed on its volunteered counts (7 vs 2).

### Q_321 · city-street · hotel-m · `counting`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | There are two fire hydrants. | There is 1 fire hydrant visible in the full scene. |

**Why** (Maintainer decision of 2026-10-05, task card Q_321 (new reference)): 5 Oct: His count of the full scene.

### Q_330 · city-street · hotel-m · `counting`

| Field | v1 | v2 |
|---|---|---|
| `expected_tool_args_json` | {"count_pointing": {"instruction": "garbage can", "view_type": "full"}} | {"count_pointing": {"instruction": "garbage can", "view_type": "current"}} |
| `answer_view` | full | current |

**Why** (Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): Expected tool calls had been copied from another question.

### Q_331 · city-street · hotel-m · `counting`

| Field | v1 | v2 |
|---|---|---|
| `expected_tool_args_json` | {"count_pointing": {"instruction": "tree", "view_type": "full"}} | {"count_pointing": {"instruction": "tree", "view_type": "current"}} |
| `answer_view` | full | current |

**Why** (Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): Expected tool calls had been copied from another question.

### Q_337 · city-street · hotel-m · `counting`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | There are at least four floors. | There are four floors. |
| `also_accept` | — | ["five floors"] |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_337 note ('Accept 4-5 stories')): 5 Oct: Four floors plus a sliver of a fifth; 'at least four' let 11 or 21 pass. Q_340 on the same card has no decision and stays as is.

### Q_340 · city-street · hotel-m · `counting`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | There are many windows on the hotel facade. | There are about 15 windows on the hotel facade; any count from 14 to 18 is correct. |

**Why** (Maintainer decision of 2026-10-05, the maintainer's answer, 2026-10-06): 5 Oct: The frame shows about 14-18 windows; 'many windows' accepted any number.

### Q_342 · city-street · mailbox · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | A quiet city street with two mailboxes, trees, and brick sidewalks is visible. | A quiet city street with two mailboxes, trees, and grey stone-slab sidewalks is visible. |
| `also_accept` | — | ["A row of brown and tan brownstone townhouses with stoops, mailboxes and two sparse-leaved trees on a paved sidewalk", "A quiet 3D-rendered street of brownstones with stairs to their doors, bare trees and mailboxes", "An empty urban street lined with brownstones and a brick corner building with a shop", "Multi-storey brown buildings with iron railings and stoops behind a grey stone sidewalk with trees and mailboxes"] |

**Why** (applied by rule R4 R1): The start frame shows two brownstones with stoops, two dark mailboxes, two sparse-leaved trees and a sidewalk of grey rectangular stone slabs. The sidewalk is not brick (the maintainer's own Q_356 ruling), so 'brick sidewalks' contradicts the view; 8/29 pass.

### Q_343 · city-street · mailbox · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A brown multi-storey building with rows of windows, stairs with metal railings, bare-branched trees on either side and grey stone paving", "Brownstone townhouses with stoops and railings, two mailboxes on the sidewalk and two sparse-leaved trees", "A row of brown and tan buildings and a corner shop with arched wooden doors and a partial yellow sign", "Many windows, two entrance stairways, two dark mailboxes, trees and a grey stone-slab sidewalk"] |

**Why** (Maintainer decision; rules R1): The start frame shows two mailboxes, two trees, many windows, two stoops with railings, a grey stone-slab sidewalk and the partial Frondy shop. The reference is fair, but grounded descriptions that skip the mailboxes or the storefront fail (5/29).

### Q_347 · city-street · hotel-m · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A stone-paved sidewalk with two trees, a street lamp and trash cans, and the street beyond the curb", "Trees, a lamppost and a pair of trash cans on the sidewalk near the entrance steps"] |

**Why** (Maintainer decision of 29 October 2026, accept the extra descriptions; rules R1; list as pruned by the independent review of 6 October 2026): The full audit (27 September, rule R1) proposed three accepted descriptions of the hotel-m start view, which shows the mailbox, a street lamp, two trees, two garbage cans, the entrance steps and the stone sidewalk; fair answers failed for leaving out the mailbox. The audit's decision record flagged them as undecided, so v2 first left the task as in v1; on 29 October 2026 the maintainer accepted the extra descriptions. The independent review of 6 October 2026 dropped one as a near-duplicate of the other two ('A sidewalk lined with trees, a streetlight and garbage cans in front of the Moonlight entrance'); v2 uses the reviewed list.

### Q_350 · city-street · hotel-m · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A grey stone building with black-framed windows and a \"Moonlight\" sign stands beside a wet street with bare trees and a street lamp", "The Moonlight building's entrance, with trees, a lamppost and trash cans on the sidewalk in front", "A street-level view of the grey Moonlight hotel beside a wet road, with trees and a streetlight"] |

**Why** (applied by rule R1): The hotel-m start frame (Q_330_0_001501_raw.jpg) matches the reference: the grey stone Moonlight hotel dominates, with two trees, the lamp, the mailbox and two cans in front. Failing runs described parking lots, cars or pedestrians that are not there.

### Q_354 · city-street · hotel-m · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | The hotel sign is clearly legible. | The hotel sign, white "Moonlight" lettering on the black canopy above the entrance, is intact and clearly legible. |
| `also_accept` | — | ["The \"Moonlight\" sign is clean, undamaged and easy to read", "The sign is in good condition, with no missing letters or visible damage", "White \"Moonlight\" letters on a black sign over the door, clearly visible and intact"] |

**Why** (applied by rule R1): Zoomed sign frames (Q_399_0_002345_postzoom.jpg) show white 3D "Moonlight" letters on the black canopy, intact and fully legible. The reference is right but so generic that a run calling it "a blue sign on a metal pole" passed; naming the look in the reference lets a wrong colour fail (R1).

### Q_355 · city-street · mailbox · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A storefront with arched wooden doors and windows", "Part of a shop with a cut-off yellow 'Fron…' sign", "A bare tree in front of a corner store with arched orange doors", "The ground-floor shop of a brick corner building"] |

**Why** (applied by rule R1): The far right of the start frame shows the ground-floor shop of the brick corner building (arched orange wooden doors and windows, a yellow 'Fron…' sign cut off by the frame) with the bare right tree in front. The reference is right (24/29); R1 alternates added.

### Q_356 · city-street · mailbox · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | The sidewalks appear to be made of brick and are well maintained. | Grey rectangular stone slabs. |
| `also_accept` | — | ["paved with concrete or similar grey slabs", "grey stone slabs bordered by a dark curb", "grey rectangular paving stones laid in a grid pattern", "clean, flat grey stone paving along the building fronts and the street"] |

**Why** (Maintainer decision; rules R4 R1): The start frame and the pan frame a captured frame show sidewalks of large grey rectangular stone slabs in a grid with a dark curb; no brick. The current release already has the stone-slab answer. corrections.json labels this task preset 'hotel-m', but the paper row and the trace both start at 'mailbox'.

### Q_357 · city-street · hotel-m · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | There is nothing occluding the sign from being read as “Moonlight”, but the trees are partially obstructing the view. | A tree is occluding the sign, but the sign can be fully read. |
| `also_accept` | — | ["Thin tree branches cross in front of the sign, but \"Moonlight\" is fully legible"] |

**Why** (Maintainer decision; rules R4 R3): Zoomed frames Q_399_0_002345_postzoom.jpg and Q_407_0_002600_postzoom.jpg show thin bare branches crossing the upper left of the sign and canopy, with every letter of "Moonlight" readable. The paper reference contradicts itself ("nothing occluding ... but the trees are partially obstructing"). Answers about a letter "s" occluding the name are wrong.

### Q_359 · city-street · mailbox · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["Smooth brown building walls, rough tree bark and grey stone paving slabs", "Red brickwork on the corner building, glass windows and the stone-slab sidewalk", "Tree bark, black iron railings and paved stone sidewalk", "Brownstone facades with glass window panes and textured grey paving stones"] |

**Why** (applied by rule R1): The start frame (also trace Q_359_0_113007_raw) shows smooth brownstone facades, red-and-white brickwork on the corner building's upper floors, glass windows, grey stone paving slabs, tree bark and iron railings. The reference is a fair but narrow pick; 12/29 pass, and the distinct answers shown all pass with split judges.

### Q_360 · city-street · hotel-m · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | Another building stands to the right of the hotel. | Another building, with an orange-brown brick facade, stands to the right of the hotel. |
| `also_accept` | — | ["A reddish-brown or terracotta building with windows and a black railing in front", "A brown brick building right next to the hotel", "The neighbouring orange-brown building, partly behind the tree with dried leaves"] |

**Why** (applied by rule R1 R4): The right edge of the hotel-m start frame (Q_330_0_001501_raw.jpg) shows an orange-brown brick building with light-framed windows and a black railing, partly behind the tree with dried leaves. Runs saying "Coffee Buzz" or "a coffee shop" passed though Coffee Buzz is across the intersection, not beside the hotel; the generic "Another building" lets them through.

### Q_364 · city-street · mailbox · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["On the sidewalk in front of the buildings, near the base of the entrance stairs", "On the pavement beside the stoops, in front of the iron railings", "Between the building fronts and the curb, next to the stairs"] |

**Why** (applied by rule R1 R3): In the start frame one mailbox stands left of the red brownstone's stoop and one between the stoops, both on the sidewalk in front of the railings. The reference is right; paraphrases like 'on the sidewalk near the base of the stairs' pass only 2 of 3 judges (17/29).

### Q_366 · city-street · hotel-m · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["On the sidewalk near the hotel entrance, by the curb", "On the sidewalk next to the hotel, to the right of the entrance steps", "Near the entrance steps, on the sidewalk in front of the building"] |

**Why** (applied by rule R3): In the hotel-m start frame (Q_330_0_001501_raw.jpg) both cans stand on the sidewalk at the curb, in front of the hotel just right of the entrance steps. Runs saying "near the stairs leading to the hotel entrance" or "on the sidewalk next to the hotel" failed on wording while identical phrasings passed elsewhere.

### Q_368 · city-street · hotel-m · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A wet, reflective street surface", "The paved street, with the curb and part of the stone sidewalk at the bottom right", "A paved street or sidewalk area"] |

**Why** (applied by rule R1 R3): The bottom edge of the hotel-m start frame (Q_330_0_001501_raw.jpg) is the wet light-grey road across the left two thirds, then the curb and stone sidewalk with a tree pit at the right. Runs answering "a paved sidewalk or street" or "a wet street" failed although both are there.

### Q_369 · city-street · mailbox · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | After panning right, the partial storefront area becomes more centered. | After panning right 20°, a building with stairs leading to a doorway is centered in the frame. |
| `also_accept` | — | ["A building entrance with stairs and a door", "The stoop and front door of the tan brownstone, beside the Frondy storefront", "Stairs with railings leading up to a doorway"] |

**Why** (Maintainer decision; rules R4 R1 R3): The 20-degree-right frame has the tan brownstone's stoop and doorway at the centre, with the Frondy storefront starting just right of centre. 'The partial storefront becomes more centred' misstates what is centred; runs describing the stairs and doorway failed (1/29).

### Q_370 · city-street · hotel-m · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | Tilting up reveals more of the hotel’s upper floors. | Tilting up reveals more of the hotel's upper floors. |
| `also_accept` | — | ["The full vertical \"HOTEL\" sign on the building's corner comes into view", "More rows of dark-framed windows on the upper part of the grey stone building appear", "The upper stories of the hotel and its vertical HOTEL sign become visible"] |

**Why** (applied by rule R1 R3): After tilting up 10 degrees (Q_370_0_000756_ptz.jpg, Q_423_0_162906_ptz.jpg) the complete vertical "HOTEL" blade sign and more upper window rows come into view. Runs saying the HOTEL sign appears, or that the upper stories are now visible, failed; "no visible change" answers fail correctly.

### Q_376 · city-street · hotel-m · `multi_step_command`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have tracked the hotel sign briefly and counted the garbage cans. | I tracked the hotel sign briefly and counted two garbage cans. |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_376 option A (write the counts in)): 5 Oct: hotel-m shows two garbage cans.

### Q_377 · city-street · hotel-m · `multi_step_command`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have zoomed to the entrance and read any posted text. | The only text at the hotel entrance is the sign reading 'Moonlight'. |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_380 option A): 5 Oct: The hotel entrance carries only the 'Moonlight' name sign.

### Q_378 · city-street · mailbox · `multi_step_command`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have counted the mailboxes and the trees. | There are two mailboxes and two trees. |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_376 option A (write the counts in)): 5 Oct: The mailbox view shows two mailboxes and two trees.

### Q_379 · city-street · hotel-m · `multi_step_command`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have taken a screenshot and counted the trees. | I took a screenshot and counted two trees. |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_376 option A (write the counts in)): 5 Oct: hotel-m shows two trees.

### Q_380 · city-street · mailbox · `multi_step_command`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have panned and read the partial store sign. | I panned right; the partial store sign reads 'Frondy'. |
| `also_accept` | — | ["'rondy' (the F hidden by the tree trunk)"] |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_380 option A): 5 Oct: The corner shop at the mailbox preset reads 'Frondy' (scene file); a trunk can hide the F. Readings stay exact otherwise (rule R8).

### Q_387 · city-street · mailbox · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have counted before and after the pan and summarized the change. | There are two mailboxes before panning 20° right and one after, so one fewer after the pan. |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_376 option A (write the counts in)): 5 Oct: Two mailboxes in the mailbox view, one after the 20 degree pan.

### Q_388 · city-street · hotel-m · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have described the scene before and after a slight zoom. | I described the scene, zoomed in 1.5x and described it again: first the Moonlight hotel, a grey stone building with black-framed windows and a black entrance canopy, with a street lamp and bare tree on the left, a tree with dried leaves on the right, two garbage cans on the sidewalk and a wet road; then a closer view of the same entrance, "Moonlight" sign, steps and garbage cans. |
| `also_accept` | — | ["Two descriptions of the grey \"Moonlight\" building with trees, a lamppost and trash cans, the second one closer", "Before: the Moonlight hotel on a wet street with trees and a street lamp", "after 1.5x: a tighter view of its entrance and sign", "A wide then a closer description of the hotel entrance, its \"Moonlight\" sign, the trees and the two trash cans"] |

**Why** (applied by rule R5 R1): Q_388_0_001355_raw.jpg is the hotel-m start view; Q_388_0_001405_ptz.jpg after 1.5x shows a closer view of the entrance with the sign, steps, bare tree, lamp, mailbox and both cans. The reference only says the descriptions were given (R5); failing runs gave only the second description.

### Q_389 · city-street · hotel-m · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have summarized both views. | This view shows the Moonlight hotel, a grey stone building with black-framed windows and a black entrance canopy, with trees, a street lamp and two garbage cans on the sidewalk beside a wet road; the "mailbox" preset shows two brick townhouses with front stoops and railings, two black mailboxes on the sidewalk, trees, and the corner of the "Frondy" shop. |
| `also_accept` | — | ["Hotel view: the grey \"Moonlight\" building on a sidewalk with trees and a streetlight", "mailbox view: brown brick townhouses with stairs and black mailboxes", "A grey stone hotel with a \"Moonlight\" sign, and a brick residential street front with stoops and mailboxes", "Current: the Moonlight hotel entrance with trees and trash cans", "mailbox preset: townhouses with steps, mailboxes and trees"] |

**Why** (applied by rule R5 R1): Q_389_0_001428_raw.jpg is the hotel-m view; Q_389_0_001438_raw.jpg (after go_to "mailbox") shows two brick townhouses (orange-red and tan) with front stoops and railings, two black mailboxes on the sidewalk, a grey can by the second stoop, trees left and right, and the edge of the "Frondy" shop. The reference only says both were summarized (R5).

### Q_390 · city-street · mailbox · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have counted the trees and read the street sign. | There are two trees, and no street sign is visible. |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_376 option A (write the counts in)): 5 Oct: Two trees in the mailbox view and no street sign in view.

### Q_391 · city-street · hotel-m · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have compared window counts before and after zooming out. | More windows are visible after zooming out 50% than before. |

**Why** (applied by rule R5 R2): After the -50% zoom (Q_391_0_001538_ptz.jpg, Q_424_0_002922_ptz.jpg) the whole hotel facade and the brownstone's windows are in frame, so more windows are visible than in the start view; passing runs report about 18 before and 26-28 after. The reference only says a comparison was made (R5); volunteered counts should not decide the grade (R2).

### Q_392 · city-street · mailbox · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have checked and verified the partial store name. | The store name on the right is 'Frondy'. |
| `also_accept` | — | ["'rondy' (the F hidden by the tree trunk)"] |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_380 option A): 5 Oct: Same sign as Q_380.

### Q_394 · city-street · hotel-m · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | The store name is “Frondy”. (Also accept “Trondy”.) | The store name is “Frondy”. |
| `also_accept` | — | ["“Trondy”", "“rondy” (the F hidden by the tree trunk)"] |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_380 option A): 5 Oct: Same sign as Q_380; the existing 'Trondy' alternate stays.

### Q_397 · city-street · mailbox · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `question` | Go to ‘hotel-m’, compare mailbox count here with the current view, then explain briefly. | Count the mailboxes in the current view, then go to hotel-m, count them there, and briefly explain the difference. |
| `expected_answer` | I have compared the counts: two mailboxes here vs one at the hotel view. | There is one mailbox at the hotel-m preset, and two in the current view. |

**Why** (Maintainer decision of 2026-10-05, task card Q_397 (new reference) and follow-up Q_397 option A): 5 Oct: The old wording was ambiguous once the camera was at hotel-m; the counts are two at the mailbox view and one at hotel-m.

### Q_398 · city-street · hotel-m · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have done a full sweep and read a store name such as ‘Coffee Buzz’. | I have done a full sweep and read a store name such as "Coffee Buzz". |
| `also_accept` | — | ["The store name \"Frondy\", read during the sweep", "\"Coffee Buzz\", also naming other signs seen such as \"Moonlight\" or \"Frondy\""] |

**Why** (applied by rule R3 R8): The mailbox preset shares the hotel-m camera position (identical audit panoramas); its 360 shows "Frondy" beside the brownstones (Q_394_0_001828_ptz.jpg) and "Coffee Buzz" across the corner (audit render). Failed runs include "Frondy" and "Coffee Buzz (Moonlight also seen)", both fair reports of a store name.

### Q_400 · city-street · mailbox · `ocr_identification`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | The store name begins with ‘Fron…’. | The store name reads 'Frondy'. |
| `also_accept` | — | ["'rondy' (the F hidden by the tree trunk)"] |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_380 option A): 5 Oct: Same sign as Q_380.

### Q_401 · city-street · hotel-m · `ocr_identification`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | No hours are clearly posted; the name ‘Moonlight’ is visible. | No hours are clearly posted; the name "Moonlight" is visible. |
| `also_accept` | — | ["The only text near the entrance is the \"Moonlight\" sign above the doors", "The sign above the entrance reads \"Moonlight\", with no hours posted"] |

**Why** (applied by rule R3 R8): Reading unchanged (Moonlight); alternates only (R3).

### Q_402 · city-street · mailbox · `ocr_identification`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | The store name starts ‘Fron…’. | The store name is 'Frondy'. |
| `also_accept` | — | ["'rondy' (the F hidden by the tree trunk)"] |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_380 option A): 5 Oct: Same sign as Q_380.

### Q_405 · city-street · hotel-m · `ocr_identification`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | The bookstore is called Coffee Buzz. | The coffee shop is called "Coffee Buzz". |

**Why** (applied by rule R4 R5): Reading unchanged (Coffee Buzz); 'bookstore' corrected to 'coffee shop' (R5).

### Q_408 · city-street · hotel-m · `ocr_identification`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | An LED sign is visible across the street. | There is no LED sign across from the store in the full scene. |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_408 option A (absence answers)): 5 Oct: No frame, panorama or scene object shows an LED sign (rule R9).

### Q_411 · city-street · mailbox · `ocr_identification`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["After the 1.5× zoom the store sign is out of frame, so no letters can be read", "No store letters are visible in the zoomed view"] |

**Why** (applied by rule R9 R3): The 1.5x centre-zoom frame (trace Q_334_0_001603_ptz, also labelled for Q_411) ends at the shop's plain wall: the 'Fron…' sign sits in the outer right sixth of the start frame and leaves the zoomed frame. 'Could not read any letters' is right, but 'no store letters are visible' passes only 2 of 3 judges.

### Q_412 · city-street · hotel-m · `ocr_identification`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | No brand logo is clearly visible near the entrance. | Yes - the sign above the entrance reads “Moonlight”. |

**Why**: The sign is legible in the render. Models that said yes were marked wrong.

### Q_413 · city-street · mailbox · `ocr_identification`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | No poster text is clearly readable in this view. | There is no poster or notice in this view, so there is no text to read. |
| `also_accept` | — | ["No poster text is readable in this view, because there is no poster"] |

**Why** (applied by rule R9 R4): Round 2: the old wording implied a poster exists, so invented illegible posters passed; the view has none.

### Q_415 · city-street · mailbox · `ocr_identification`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | The coffee shop sign reads ‘OPEN’. | There are no OPEN or CLOSED signs; the blue sign by the coffee shop reads COFFEE SHOP. |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_408 option A (absence answers)): 5 Oct: The blue sign models read as 'OPEN' is a neon 'COFFEE SHOP' blade sign (checked in the scene file); no OPEN/CLOSED sign exists (rule R9).

### Q_001 · whitechapel · eor-viewpoint · `comparative_relational`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | The view after panning left has more street lamps. | There are more street lamps after panning left 180°. |
| `also_accept` | — | ["The view after panning left has more street lamps", "The after view has more street lamps"] |

**Why** (Maintainer decision; rules R4 R3): Q_001_0_000712_ptz.jpg (180 deg left) shows a long wet cobbled street with at least five street lamps along both sidewalks, versus two lamps in the preset render, so the current-release 'original view has more' (built on the empty-sky claim) is wrong and the paper reference is right. The two blank frames are flat RGB 43,43,43 (the backdrop colour) and are most likely failed captures, which fits the 27B runs that…

### Q_002 · whitechapel · sor-viewpoint · `comparative_relational`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | The view after the half-turn shows more buildings. | The original view has more street lamps. |
| `also_accept` | — | ["The view before the turn shows more street lamps", "Before, because after the turn the camera faces open sky with no lamps"] |
| `expected_tool_order_json` | [{"name": "query_answer", "args": {"instruction": "After a half-turn, which view shows more buildings—before or after?", "view_type": "current"}}] | [{"name": "query_answer", "args": {"instruction": "After a 180 degree turn to the left, which view shows more street lamps - before or after?", "view_type": "current"}}] |

**Why** (Maintainer decision; rules R5 R3; Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): The paper reference answers about buildings, not lamps. All tB09 frames after the 180 degree left turn (Q_002_0_185828_ptz.jpg and the four query frames) are flat grey, the open sky from round 0; the start view has about nine lamps along both sidewalks.

### Q_003 · whitechapel · eor-viewpoint · `comparative_relational`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["there is no shrubbery, only moss between the cobblestones"] |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_003 option A): 5 Oct: At eor-viewpoint the only greenery is moss between the cobblestones, mostly in the left half; Q_068 at this preset already says there is no shrubbery.

### Q_004 · whitechapel · sor-viewpoint · `comparative_relational`

| Field | v1 | v2 |
|---|---|---|
| `expected_tool_order_json` | [{"name": "ptz_adjust", "args": {"pan_deg": -180}}, {"name": "query_answer", "args": {"instruction": "Which side has more trees, left or right?", "view_type": "current"}}] | [{"name": "query_answer", "args": {"instruction": "Which side has more trees, left or right?", "view_type": "current"}}] |
| `expected_tool_args_json` | {"ptz_adjust": {"pan_deg": -180}} | {} |

**Why** (Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): Correct; 43% against the contradicting pair's 11%.

### Q_005 · whitechapel · sor-viewpoint · `comparative_relational`

| Field | v1 | v2 |
|---|---|---|
| `expected_tool_order_json` | [{"name": "ptz_adjust", "args": {"pan_deg": -180}}, {"name": "query_answer", "args": {"instruction": "Which side has more trees, left or right?", "view_type": "current"}}] | [{"name": "query_answer", "args": {"instruction": "Which side has more trees, left or right?", "view_type": "current"}}] |
| `expected_tool_args_json` | {"ptz_adjust": {"pan_deg": -180}} | {} |

**Why** (Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): Correct; same cluster as Q_004.

### Q_006 · whitechapel · sor-viewpoint · `comparative_relational`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | There are no trees on either side of the road | There are more trees on the left side of the road. |
| `expected_tool_order_json` | [{"name": "ptz_adjust", "args": {"pan_deg": -180}}, {"name": "query_answer", "args": {"instruction": "Which side has more trees, left or right?", "view_type": "current"}}] | [{"name": "query_answer", "args": {"instruction": "Which side has more trees, left or right?", "view_type": "current"}}] |
| `expected_tool_args_json` | {"ptz_adjust": {"pan_deg": -180}} | {} |

**Why** (Maintainer decision; rules R4 R10; Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): Start frames (Q_006_0_000812_raw.jpg, a captured frame show two trees on the left sidewalk and none on the right; 'no trees' only holds after the 180 degree pan in the expected tools, which the question never asks for. Verbatim copy of Q_004.

### Q_007 · whitechapel · eor-viewpoint · `comparative_relational`

| Field | v1 | v2 |
|---|---|---|
| `preset_start` | sor-viewpoint | eor-viewpoint |
| `expected_answer` | There are no trees on either side | There are no trees on either side. |
| `expected_tool_order_json` | [{"name": "ptz_adjust", "args": {"pan_deg": -180}}, {"name": "query_answer", "args": {"instruction": "Which side has more trees, left or right?", "view_type": "current"}}] | [{"name": "query_answer", "args": {"instruction": "Which side has more trees, left or right?", "view_type": "current"}}] |
| `expected_tool_args_json` | {"ptz_adjust": {"pan_deg": -180}} | {"query_answer": {"instruction": "Which side has more trees, left or right?", "view_type": "current"}} |

**Why** (the maintainer's saved follow-up note, approved for execution by the maintainer's instruction; rules R4 R6; Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): Use the eor-viewpoint start requested in the follow-up so the no-tree reference is correct. Remove the unrelated half-turn from the expected tool metadata.

### Q_008 · whitechapel · eor-viewpoint · `comparative_relational`

| Field | v1 | v2 |
|---|---|---|
| `question` | Which view shows a clearer intersection—before or after panning right 20°? | Which view shows more of the open cobblestone square—before or after panning right 20°? |
| `expected_answer` | The view after panning right 20° shows a clearer intersection. | The view before panning right 20° shows more of the open cobblestone square. |
| `expected_tool_order_json` | [{"name": "query_answer", "args": {"instruction": "Which view shows a clearer intersection—before or after panning right 20°?", "view_type": "current"}}] | [{"name": "ptz_adjust", "args": {"pan_deg": 20}}, {"name": "query_answer", "args": {"instruction": "Which view shows more of the open cobblestone square, before or after panning right 20°?", "view_type": "current"}}] |
| `expected_tool_args_json` | {} | {"ptz_adjust": {"pan_deg": 20}} |
| `scene_objects_mentioned` | intersection | square |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_008 option A; Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): 5 Oct: 'Clearer intersection' is subjective; after the pan the near red-brick building fills the right half, so the square is more open before.

### Q_010 · whitechapel · eor-viewpoint · `comparative_relational`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["After the half-turn, the view shows more buildings", "After a half-turn (180 degrees), the view after the turn shows more buildings"] |

**Why** (Maintainer decision; rules R4 R3): Q_010_0_000843_ptz.jpg (180 deg turn) shows a long street lined with roughly eight or more facades versus about six in the preset render, so the paper reference is right and the current-release 'before' is wrong. The two failed answers the maintainer cites (Qwen3-4B, Qwen3-8B) failed because they only panned and made the claim without any vision call, not because of wording.

### Q_011 · whitechapel · sor-viewpoint · `comparative_relational`

| Field | v1 | v2 |
|---|---|---|
| `question` | Which half of the frame has more shrubbery, left or right? | Which side of the street has more shrubbery, left or right? |
| `expected_answer` | The left half has more shrubbery. | The left side of the street has more shrubbery. |
| `expected_tool_order_json` | [{"name": "query_answer", "args": {"instruction": "Which half of the frame has more shrubbery, left or right?", "view_type": "current"}}] | [{"name": "query_answer", "args": {"instruction": "Which side of the street has more shrubbery, left or right?", "view_type": "current"}}] |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_011 option A; Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): 5 Oct: By frame halves it is too close to call (planters left, trees and tufts right); by side of the street the railing planters make the left side greener.

### Q_013 · whitechapel · sor-viewpoint · `comparative_relational`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | The view after panning right 20° shows a clearer intersection. | Neither view shows a clear intersection; both show one long street. |
| `expected_tool_order_json` | [{"name": "ptz_adjust", "args": {"pan_deg": -180}}, {"name": "query_answer", "args": {"instruction": "Which view shows a clearer intersection—before or after panning right 20°?", "view_type": "current"}}] | [{"name": "ptz_adjust", "args": {"pan_deg": 20}}, {"name": "query_answer", "args": {"instruction": "Which view shows a clearer intersection—before or after panning right 20°?", "view_type": "current"}}] |
| `expected_tool_args_json` | {"ptz_adjust": {"pan_deg": -180}} | {"ptz_adjust": {"pan_deg": 20}} |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_013 option A; Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): 5 Oct: At sor-viewpoint the start view and the view after panning right 20 degrees both show one long street with no cross street.

### Q_014 · whitechapel · eor-viewpoint · `comparative_relational`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["There are more street lamps after panning left 180°.", "The after view has more street lamps"] |
| `expected_tool_order_json` | [{"name": "query_answer", "args": {"instruction": "After panning left 180°, which view has more street lamps—before or after?", "view_type": "current"}}] | [{"name": "ptz_adjust", "args": {"pan_deg": -180}}, {"name": "query_answer", "args": {"instruction": "After panning left 180°, which view has more street lamps—before or after?", "view_type": "current"}}] |

**Why** (applied by rule R4 R10; Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended; reconciliation of 2026-10-06 (maintainer approval): original reference restored, descriptions kept as also_accept): The windows rewrite rested on the empty-sky claim; back to the paper question, same answer as its twin Q_001. 6 Oct: he chose Keep; the Q_001-style wording had replaced the reference.

### Q_016 · whitechapel · eor-viewpoint · `comparative_relational`

| Field | v1 | v2 |
|---|---|---|
| `question` | Which half of the frame has more shrubbery, left or right? | Pan left 180 degrees, then tell me which half of the frame has more shrubbery. |
| `also_accept` | — | ["there is no shrubbery, only moss between the cobblestones"] |
| `expected_tool_order_json` | [{"name": "ptz_adjust", "args": {"pan_deg": -180}}, {"name": "query_answer", "args": {"instruction": "Which half of the frame has more shrubbery, left or right?", "view_type": "current"}}] | [{"name": "ptz_adjust", "args": {"pan_deg": -180}}, {"name": "query_answer", "args": {"instruction": "Pan left 180 degrees, then tell me which half of the frame has more shrubbery.", "view_type": "current"}}] |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_003 option A): 5 Oct: Same question as Q_003, but its expected tools pan 180 degrees first; the pan is now in the wording.

### Q_040 · whitechapel · eor-viewpoint · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["The cobblestones are worn and uneven, with patches of moss and dirt", "The street is damp, with puddles and dark wet patches", "An old, neglected cobblestone street with discoloured, mossy patches"] |

**Why** (applied by rule R1): The preset render shows worn cobblestones with dark glossy puddles and moss patches; fair answers describing worn, mossy, dirty cobblestones failed only for not saying 'wet'.

### Q_041 · whitechapel · sor-viewpoint · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["An elevated view of a wet cobblestone street lined with old brick buildings", "A deserted, run-down street of weathered multi-storey buildings with puddles", "A narrow cobbled street between aged brick and stone buildings under a dark sky", "An empty old street with street lamps, puddles and small autumn trees"] |

**Why** (applied by rule R1): The start view is an elevated view down an empty, wet cobblestone street lined with weathered brick and stone buildings, lamps and two small autumn trees. Failures are mostly hallucinated scenes (traffic, lakeside, stop sign). The alternates are added per R1. Q_071 is a verbatim copy.

### Q_044 · whitechapel · eor-viewpoint · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A cobblestone street or square lined with old multi-storey brick and stone buildings", "Weathered buildings with many windows, a balcony and a red-orange door", "Worn cobblestones with dark damp patches and moss, and two street lamps"] |

**Why** (applied by rule R1): The preset render shows a cobblestone square with puddles and moss, weathered brick and stone buildings, one balcony, a red-orange door and two lamps; fair feature lists failed only for omitting puddles.

### Q_049 · whitechapel · sor-viewpoint · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["The sky is dark and overcast", "The sky is a flat, featureless dark grey", "A uniform dark charcoal-grey sky with no sun visible"] |

**Why** (applied by rule R1): The sky in the render is a flat, uniform dark grey (RGB about 43,43,43) with no sun or cloud detail. Accurate answers like 'a solid, featureless expanse of dark charcoal gray' failed; 'clear blue sky' answers rightly fail.

### Q_051 · whitechapel · eor-viewpoint · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | Two street lamps flank the intersection near the buildings. | The intersection is paved with cobblestones with dark patches that resemble water or mud, surrounded by multi-storey buildings with stained, weathered walls, and a street lamp on each side. |
| `also_accept` | — | ["Two street lamps flank the intersection near the buildings", "A cobblestone square with wet, dark patches, surrounded by old multi-storey brick and stone buildings", "A quiet, empty cobblestone junction between weathered buildings, with moss or water patches on the paving", "Worn cobblestones with moss and puddles and a street lamp on each side"] |

**Why** (Maintainer decision; rules R1; Maintainer decision of 2026-10-05, follow-up Q_051 option A): Round 2: his description becomes an accepted answer, with 'broken windows and peeling paint' (not visible) replaced by 'stained, weathered walls'; making it the main reference waits for his confirmation. 5 Oct: His description becomes the main reference, with 'stained, weathered walls' as the frames show (no broken windows or peeling paint); the paper's sentence stays accepted.

### Q_053 · whitechapel · sor-viewpoint · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["small grass or weed tufts grow between the cobblestones (not real shrubs)"] |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_053 option A): 5 Oct: The 'shrubbery' is small yellow grass tufts on the cobbles; answers that describe exactly those tufts pass.

### Q_054 · whitechapel · eor-viewpoint · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["Shades of brown, grey and red brick, with some white accents", "Earthy tones: tan and red brick, grey stone and white trim", "Muted browns, greys and reddish brick"] |

**Why** (applied by rule R1): The render shows tan brick (left), mottled pink, dark grey-brown and grey stone, red brick and white trim; 'brown, grey and red brick' answers describe it fairly but failed with gpt-oss and qwen3 judges.

### Q_055 · whitechapel · sor-viewpoint · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["Worn cobblestones with puddles, mud and scattered debris", "The cobbled street is worn and dirty, with dark wet patches and grass tufts", "A neglected, muddy cobblestone street with standing water"] |

**Why** (applied by rule R1): The render shows worn cobbles with dark puddles, muddy patches and scattered grass tufts. Fair answers ('worn cobblestones with patches of dirt and scattered debris') failed because they did not say 'wet'; the 'dry clean asphalt' answer rightly fails.

### Q_056 · whitechapel · eor-viewpoint · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A cobblestone courtyard surrounded by aged brick and stone buildings under a dark sky", "An empty, wet cobblestone square framed by old, weathered multi-storey buildings", "A quiet old-town square with worn cobblestones, puddles and moss patches"] |

**Why** (applied by rule R1): The render is an empty cobblestone square with puddles and moss framed by old buildings under a dark sky; answers like 'a cobblestone courtyard surrounded by aged brick buildings under a dark sky' failed because they did not say 'street after rain'.

### Q_059 · whitechapel · sor-viewpoint · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["Old multi-storey brick and stone buildings along a wet cobblestone street", "Puddles and wet patches on the cobbled road with scattered yellow grass tufts", "Street lamps along both sidewalks and small autumn trees on the left", "Iron railings with planted tufts in front of the left-hand buildings"] |

**Why** (applied by rule R1): The render shows weathered brick and stone buildings, a wet cobbled road with puddles and white patches, yellow grass tufts, lamps on both sidewalks, two small autumn trees and iron railings on the left. Many failing feature lists are grounded (buildings, cobblestones, weathered brick). Some add unverifiable extras (lit windows, fire damage) and can still fail.

### Q_064 · whitechapel · eor-viewpoint · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["The sky is a flat, featureless dark grey", "A uniform dark charcoal-grey sky with no sun", "The sky is dark and gloomy"] |

**Why** (applied by rule R1): The sky in the render is a flat dark charcoal grey (the untextured backdrop, about RGB 43,43,43); 'dark, featureless grey' answers describe it accurately but failed with two of three judges, while 'blue sky with clouds' answers rightly fail.

### Q_066 · whitechapel · eor-viewpoint · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["They stand on posts at the edges of the cobblestone square, one on each side, near the buildings", "On short poles beside the cobblestone street, near the buildings on either side"] |

**Why** (applied by rule R1): In the render one lamp stands at the left edge of the square by the curb and the other on the right sidewalk corner, both on short posts near buildings; the reference is fair at 24/29.

### Q_069 · whitechapel · sor-viewpoint · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["Mostly shades of brown, red and grey with white trim", "Primarily red and brown brick with grey and white stone", "A muted, earthy palette of brick reds, browns, greys and off-whites"] |

**Why** (applied by rule R1): Facades in the render are pinkish-red brick, dark brown brick, red brick, tan and cream stone and grey-white render. About 20 fair palette answers ('brown, red and gray with white trim', 'red brick, gray stone, white plaster') failed; 'The red ball' is junk and should fail.

### Q_070 · whitechapel · eor-viewpoint · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["The cobblestones are worn and uneven, with patches of moss and dirt", "The street is damp, with puddles and dark wet patches", "An old, neglected cobblestone street with discoloured, mossy patches"] |

**Why** (applied by rule R1 R10): Verbatim twin of Q_040; same frames and the same failure pattern (worn, mossy cobblestone descriptions failing for not saying 'wet').

### Q_071 · whitechapel · sor-viewpoint · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["An elevated view of a wet cobblestone street lined with old brick buildings", "A deserted, run-down street of weathered multi-storey buildings with puddles", "A narrow cobbled street between aged brick and stone buildings under a dark sky", "An empty old street with street lamps, puddles and small autumn trees"] |

**Why** (applied by rule R10 R1): Verbatim copy of Q_041 with the same reference; it takes Q_041's alternates. Failures are hallucinated scenes or mention unverified snow and caution tape.

### Q_074 · whitechapel · sor-viewpoint · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | (same spatial relation as Q_058 rewrite) | The two trees are on the left sidewalk. |
| `also_accept` | — | ["Both trees stand on the left side of the street, along the sidewalk in front of the buildings"] |

**Why** (applied by rule R4 R5): The reference is a placeholder ('same spatial relation as Q_058 rewrite'), and Q_058 at eor-viewpoint is an unrelated street-light question. That is why answers saying 'right sidewalk' or 'in the road' pass today. The render shows both small autumn trees standing on the left sidewalk.

### Q_076 · whitechapel · eor-viewpoint · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["Most puddles are in the centre of the cobblestone square", "On the cobblestones in the middle and foreground of the view", "Near the bottom centre of the image, on the paved ground"] |

**Why** (applied by rule R1 R3): In the render the dark glossy puddles lie on the cobblestones across the centre and bottom centre of the square; 'centre of the courtyard' and 'bottom centre of the image' answers are accurate but failed for not saying 'middle of the road'.

### Q_079 · whitechapel · sor-viewpoint · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["Yes, a nearby brick wall on the right edge blocks part of the view", "Yes, the close right-hand building obstructs the right side of the frame"] |

**Why** (applied by rule R3): In the render a near tan-brick building fills the right edge and hides the right-hand row behind it. Answers saying 'yes, a brick wall on the right blocks the view' state the same fact and still failed (four cases).

### Q_080 · whitechapel · eor-viewpoint · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A snow-covered roof", "The sloped, weathered grey roof of the red-brick building on the right", "The snowy roof of the right-hand building, with dark sky above it"] |

**Why** (applied by rule R1 R3): The top-right of the render is the mottled grey-white, snowy-looking sloped roof of the right red-brick building with dark sky above; 'a snow-covered roof' failed with most judges only for omitting 'gray'.

### Q_085 · whitechapel · sor-viewpoint · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["There is no clear intersection; the lamps line both sides of the street"] |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_085 option A): 5 Oct: Lamps line both sidewalks; the street meets a cross street only at its far end.

### Q_086 · whitechapel · eor-viewpoint · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["Yes, the red-brick building in the right foreground blocks part of the view", "Yes, a white porch canopy at the lower right partly covers the building behind it"] |

**Why** (applied by rule R3): The right side of the render is a near red-brick building with a white porch canopy that blocks the view behind it; 'yes' answers naming the canopy as the occluder failed with muse-g and qwen3.

### Q_096 · whitechapel · eor-viewpoint · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["Yes, a blimp (airship) is visible in the sky", "Yes, a small white oblong object is in the sky", "A blip is present in the sky"] |
| `expected_tool_args_json` | {"ptz_adjust": {"tilt_deg": 50}, "query_answer": {"instruction": "Is there a blip visible in the sky?", "view_type": "current"}} | {"ptz_adjust": {"tilt_deg": 50}} |

**Why** (applied by rule R3; Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): Q_096_0_001747_ptz.jpg (tilt up 50 deg) shows a small white blimp against the dark sky, so the reference is right; about 17 of 23 failures report the object correctly but fail because their query_answer instruction differs from 'Is there a blip visible in the sky?' under exact arg matching.

### Q_098 · whitechapel · eor-viewpoint · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["The road behind me is also wet", "Yes, the cobbled street behind is wet, with puddles"] |
| `expected_tool_args_json` | {"ptz_adjust": {"pan_deg": 180}, "query_answer": {"instruction": "Is the road also wet?", "view_type": "current"}} | {"ptz_adjust": {"pan_deg": 180}} |

**Why** (Maintainer decision; rules R4 R3; Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): Q_098_0_001839_ptz.jpg (180 deg right) shows a long cobbled street with pale reflective puddles and damp patches, so the current-release 'open sky' answer is wrong and the paper reference is right. Most failures give that answer but fail on the exact query_answer instruction 'Is the road also wet?'.

### Q_099 · whitechapel · sor-viewpoint · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | Yes, the road behind also appears wet. | No. After turning around the view is open sky, so there is no wet road behind. |
| `also_accept` | — | ["No, the road behind is not wet", "There is no road behind, only open sky", "No road or wet surface is visible after the turn"] |

**Why** (Maintainer decision; rules R9 R4): The frames after the 180 degree pan (Q_099_0_001908_ptz.jpg, Q_099_0_001918_raw.jpg) are flat grey open sky with no geometry, matching round 0. The paper answer 'yes, also wet' describes a road that is not there.

### Q_133 · whitechapel · sor-viewpoint · `single_call`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I have zoomed out 0.5×. | The camera has been zoomed out 2×. |
| `also_accept` | — | ["Zoomed out by a factor of 2 (zoom factor 0.5)"] |

**Why** (Maintainer decision; rules R5 R3): The paper reference 'zoomed out 0.5×' misstates the command. The frame after zoom factor 0.5 (Q_133_0_002326_ptz.jpg) shows a wider street view with empty margins, i.e. zoomed out. Correct answers like 'zoom factor set to 0.5 (zoomed out 2×)' failed.

### Q_433 · postwar-city · diff-view-1 · `comparative_relational`

| Field | v1 | v2 |
|---|---|---|
| `question` | Which half has more rocks right now, the left half or the right half? | Go to street-view-1; which half has more rocks, the left half or the right half? |
| `expected_answer` | It’s not exactly apparent which half has more rocks | There are more rocks on the left side. |
| `expected_tool_order_json` | [{"name": "query_answer", "args": {"instruction": "Which half has more rocks, left or right?", "view_type": "current"}}] | [{"name": "go_to_preset", "args": {"name": "street-view-1"}}, {"name": "query_answer", "args": {"instruction": "Which half has more rocks, left or right?", "view_type": "current"}}] |
| `expected_tool_args_json` | {"query_answer": {"instruction": "Which half has more rocks, left or right?", "view_type": "current"}} | {"go_to_preset": {"name": "street-view-1"}, "query_answer": {"instruction": "Which half has more rocks, left or right?", "view_type": "current"}} |
| `expected_tool_order_signature` | [] | [{"name": "go_to_preset", "args": {"name": "street-view-1"}}, {"name": "query_answer", "args": {"instruction": "Which half has more rocks, left or right?", "view_type": "current"}}] |

**Why** (the maintainer's saved follow-up option C; rules R4 R6): Keep the requested left-side reference by making the move to street-view-1 explicit. The aerial start view cannot resolve the rocks.

### Q_461 · postwar-city · street-view-2 · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["Weathered concrete walls, cracked pavement and tall aged buildings in the background", "the scene is quiet and desolate", "A segmented grey concrete wall or barrier in the foreground, with multi-storey buildings of varied heights behind it", "Cracked dark asphalt in front of a damaged concrete barrier, with weeds growing at its base", "A tall stone-faced building with a plain beige side wall beside a pale, ornate apartment block on the right", "Urban decay: a partly broken concrete barricade with a corrugated-metal section, a cracked road and old buildings"] |

**Why** (Maintainer decision; rules R1): preset render postwar-city__street-view-2.png and the matching real frame (e.g. Q_461_0_000935_raw.jpg) show a three-section concrete slab wall with a corrugated middle part, cracked asphalt with weed tufts, and aged multi-storey buildings; "rocks" are only a few tiny pebbles at the wall base. Fair answers naming concrete walls, cracked pavement and old buildings failed.

### Q_462 · postwar-city · diff-view-1 · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | It is a medium-sized town with low-rise buildings, visible barricades, and overgrown areas. | A dense urban area of mixed heights and modern styles - taller buildings with balconies alongside shorter boxy structures, ranging from low-rise blocks to taller towers. Shrubbery and barricades are present. |
| `also_accept` | — | ["a dense cluster of buildings of varying heights and styles, from low-rise residential blocks to taller towers", "a high-angle view of clustered buildings of varying heights around a central open space with greenery and scattered debris", "a high overview of a paved area with several buildings of varying heights and architectural styles", "older, weathered apartment blocks and a few tall modern towers around an overgrown, rubble-strewn central lot", "a compact, run-down town block of multi-story buildings around an overgrown lot with low concrete walls and wrecked cars"] |

**Why** (Maintainer decision; rules R1 R4 R10): Render and tRPL2 frames show a ~15-storey tower with balconies, a white slab tower, 4-7 storey older blocks and plain boxy blocks around an overgrown lot with low concrete-block walls and two wrecked cars, so 'low-rise' is wrong; nearly every model answer describes varied heights fairly and fails on the old reference.

### Q_463 · postwar-city · street-view-1 · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A cracked, worn cobblestone street with patches of grass and moss, lined by tall, aged buildings, with small plants along the walls and ground", "Overgrown vegetation on the buildings, a cracked and worn road, and scattered debris and plants along the street", "Weathered, neglected buildings, rubble and concrete blocks in the street, and weeds pushing through cracked pavement", "Old, worn buildings with grass and small plants along the street and scattered small yellow and white panels", "A purple door on the left, a rusted wrecked car, and a vine-covered tree against a building"] |

**Why** (Maintainer decision; rules R1 R10): Start view (render; Q_463_0_000950_raw) shows low concrete barricades, scattered rocks on a cracked moss-patched street, grass clumps and plant cards, weathered buildings, a purple panel/door at left, a rusted car and an ivy-covered tree. Reference is fair, but 0-3/29 pass: grounded answers (cracked street, moss, aged buildings, overgrowth) fail for not naming barricades/rocks.

### Q_464 · postwar-city · street-view-2 · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["Weathered concrete walls, cracked pavement and tall aged buildings in the background", "the scene is quiet and desolate", "A segmented grey concrete wall or barrier in the foreground, with multi-storey buildings of varied heights behind it", "Cracked dark asphalt in front of a damaged concrete barrier, with weeds growing at its base", "A tall stone-faced building with a plain beige side wall beside a pale, ornate apartment block on the right", "Urban decay: a partly broken concrete barricade with a corrugated-metal section, a cracked road and old buildings"] |

**Why** (applied by rule R1 R10): preset render postwar-city__street-view-2.png and the matching real frame (e.g. Q_461_0_000935_raw.jpg) show a three-section concrete slab wall with a corrugated middle part, cracked asphalt with weed tufts, and aged multi-storey buildings; "rocks" are only a few tiny pebbles at the wall base. Fair answers naming concrete walls, cracked pavement and old buildings failed.

### Q_465 · postwar-city · diff-view-1 · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A dense urban area of mixed heights and modern styles - taller buildings with balconies alongside shorter boxy structures, ranging from low-rise blocks to taller towers, with shrubbery and barricades", "a dense cluster of buildings of varying heights and styles, from low-rise residential blocks to taller towers", "a high-angle view of clustered buildings of varying heights around a central open space with greenery and scattered debris", "a high overview of a paved area with several buildings of varying heights and architectural styles", "older, weathered apartment blocks and a few tall modern towers around an overgrown, rubble-strewn central lot", "a compact, run-down town block of multi-story buildings around an overgrown lot with low concrete walls and wrecked cars"] |

**Why** (Maintainer decision; rules R1 R4 R10; Maintainer decision of 2026-10-05, the task-card decision of 2026-09-27 ('fine as is; just add more descriptors'), restored in the reconciliation of 2026-10-06; reconciliation of 2026-10-06 (maintainer approval): original reference restored, descriptions kept as also_accept): Render and tRPL2 frames show a ~15-storey tower with balconies, a white slab tower, 4-7 storey older blocks and plain boxy blocks around an overgrown lot with low concrete-block walls and two wrecked cars, so 'low-rise' is wrong; nearly every model answer describes varied heights fairly and fails on the old reference. 5 Oct: He kept the reference and asked for more accepted descriptions; the release had replaced it with his Q_462 text, which is now an accepted alternate. 6 Oct: he chose 'fine as is, add more descriptors' (task card, 27 Sep).

### Q_466 · postwar-city · street-view-1 · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A cracked, worn cobblestone street with patches of grass and moss, lined by tall, aged buildings, with small plants along the walls and ground", "Overgrown vegetation on the buildings, a cracked and worn road, and scattered debris and plants along the street", "Weathered, neglected buildings, rubble and concrete blocks in the street, and weeds pushing through cracked pavement", "Old, worn buildings with grass and small plants along the street and scattered small yellow and white panels", "A purple door on the left, a rusted wrecked car, and a vine-covered tree against a building"] |

**Why** (Maintainer decision; rules R1 R10): Start view (render; Q_463_0_000950_raw) shows low concrete barricades, scattered rocks on a cracked moss-patched street, grass clumps and plant cards, weathered buildings, a purple panel/door at left, a rusted car and an ivy-covered tree. Reference is fair, but 0-3/29 pass: grounded answers (cracked street, moss, aged buildings, overgrowth) fail for not naming barricades/rocks.

### Q_467 · postwar-city · street-view-2 · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["Weathered concrete walls, cracked pavement and tall aged buildings in the background", "the scene is quiet and desolate", "A segmented grey concrete wall or barrier in the foreground, with multi-storey buildings of varied heights behind it", "Cracked dark asphalt in front of a damaged concrete barrier, with weeds growing at its base", "A tall stone-faced building with a plain beige side wall beside a pale, ornate apartment block on the right", "Urban decay: a partly broken concrete barricade with a corrugated-metal section, a cracked road and old buildings"] |

**Why** (Maintainer decision; rules R1): preset render postwar-city__street-view-2.png and the matching real frame (e.g. Q_461_0_000935_raw.jpg) show a three-section concrete slab wall with a corrugated middle part, cracked asphalt with weed tufts, and aged multi-storey buildings; "rocks" are only a few tiny pebbles at the wall base. Fair answers naming concrete walls, cracked pavement and old buildings failed.

### Q_468 · postwar-city · diff-view-1 · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A dense urban area of mixed heights and modern styles - taller buildings with balconies alongside shorter boxy structures, ranging from low-rise blocks to taller towers. Shrubbery and barricades are present.", "a dense cluster of buildings of varying heights and styles, from low-rise residential blocks to taller towers", "a high-angle view of clustered buildings of varying heights around a central open space with greenery and scattered debris", "a high overview of a paved area with several buildings of varying heights and architectural styles", "older, weathered apartment blocks and a few tall modern towers around an overgrown, rubble-strewn central lot", "a compact, run-down town block of multi-story buildings around an overgrown lot with low concrete walls and wrecked cars"] |

**Why** (Maintainer decision; rules R1 R4 R10; reconciliation of 2026-10-06 (maintainer approval): original reference restored, descriptions kept as also_accept): Render and tRPL2 frames show a ~15-storey tower with balconies, a white slab tower, 4-7 storey older blocks and plain boxy blocks around an overgrown lot with low concrete-block walls and two wrecked cars, so 'low-rise' is wrong; nearly every model answer describes varied heights fairly and fails on the old reference. 6 Oct: his 27 Sep decision kept the reference; the duplicate rule had replaced it.

### Q_469 · postwar-city · street-view-1 · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A cracked, worn cobblestone street with patches of grass and moss, lined by tall, aged buildings, with small plants along the walls and ground", "Overgrown vegetation on the buildings, a cracked and worn road, and scattered debris and plants along the street", "Weathered, neglected buildings, rubble and concrete blocks in the street, and weeds pushing through cracked pavement", "Old, worn buildings with grass and small plants along the street and scattered small yellow and white panels", "A purple door on the left, a rusted wrecked car, and a vine-covered tree against a building"] |

**Why** (Maintainer decision; rules R1 R10): Start view (render; Q_463_0_000950_raw) shows low concrete barricades, scattered rocks on a cracked moss-patched street, grass clumps and plant cards, weathered buildings, a purple panel/door at left, a rusted car and an ivy-covered tree. Reference is fair, but 0-3/29 pass: grounded answers (cracked street, moss, aged buildings, overgrowth) fail for not naming barricades/rocks.

### Q_470 · postwar-city · street-view-2 · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["Weathered concrete walls, cracked pavement and tall aged buildings in the background", "the scene is quiet and desolate", "A segmented grey concrete wall or barrier in the foreground, with multi-storey buildings of varied heights behind it", "Cracked dark asphalt in front of a damaged concrete barrier, with weeds growing at its base", "A tall stone-faced building with a plain beige side wall beside a pale, ornate apartment block on the right", "Urban decay: a partly broken concrete barricade with a corrugated-metal section, a cracked road and old buildings"] |

**Why** (applied by rule R1 R10): preset render postwar-city__street-view-2.png and the matching real frame (e.g. Q_461_0_000935_raw.jpg) show a three-section concrete slab wall with a corrugated middle part, cracked asphalt with weed tufts, and aged multi-storey buildings; "rocks" are only a few tiny pebbles at the wall base. Fair answers naming concrete walls, cracked pavement and old buildings failed.

### Q_471 · postwar-city · diff-view-1 · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A dense urban area of mixed heights and modern styles - taller buildings with balconies alongside shorter boxy structures, ranging from low-rise blocks to taller towers. Shrubbery and barricades are present.", "a dense cluster of buildings of varying heights and styles, from low-rise residential blocks to taller towers", "a high-angle view of clustered buildings of varying heights around a central open space with greenery and scattered debris", "a high overview of a paved area with several buildings of varying heights and architectural styles", "older, weathered apartment blocks and a few tall modern towers around an overgrown, rubble-strewn central lot", "a compact, run-down town block of multi-story buildings around an overgrown lot with low concrete walls and wrecked cars"] |

**Why** (Maintainer decision; rules R1 R4 R10; reconciliation of 2026-10-06 (maintainer approval): original reference restored, descriptions kept as also_accept): Render and tRPL2 frames show a ~15-storey tower with balconies, a white slab tower, 4-7 storey older blocks and plain boxy blocks around an overgrown lot with low concrete-block walls and two wrecked cars, so 'low-rise' is wrong; nearly every model answer describes varied heights fairly and fails on the old reference. 6 Oct: his 27 Sep decision kept the reference; the duplicate rule had replaced it.

### Q_472 · postwar-city · street-view-1 · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A cracked, worn cobblestone street with patches of grass and moss, lined by tall, aged buildings, with small plants along the walls and ground", "Overgrown vegetation on the buildings, a cracked and worn road, and scattered debris and plants along the street", "Weathered, neglected buildings, rubble and concrete blocks in the street, and weeds pushing through cracked pavement", "Old, worn buildings with grass and small plants along the street and scattered small yellow and white panels", "A purple door on the left, a rusted wrecked car, and a vine-covered tree against a building"] |

**Why** (Maintainer decision; rules R1 R10): Start view (render; Q_463_0_000950_raw) shows low concrete barricades, scattered rocks on a cracked moss-patched street, grass clumps and plant cards, weathered buildings, a purple panel/door at left, a rusted car and an ivy-covered tree. Reference is fair, but 0-3/29 pass: grounded answers (cracked street, moss, aged buildings, overgrowth) fail for not naming barricades/rocks. The first-audit frame for this task…

### Q_473 · postwar-city · street-view-2 · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["Weathered concrete walls, cracked pavement and tall aged buildings in the background", "A segmented grey concrete wall or barrier in the foreground, with multi-storey buildings of varied heights behind it", "Cracked dark asphalt in front of a damaged concrete barrier, with weeds growing at its base", "A tall stone-faced building with a plain beige side wall beside a pale, ornate apartment block on the right", "A cracked, worn concrete surface with buildings of varying styles and colours on both sides"] |

**Why** (Maintainer decision of 29 October 2026, accept the extra descriptions; rules R1 R10; list as pruned by the independent review of 6 October 2026): The full audit (27 September, rules R1 and R10) proposed accepted descriptions of the street-view-2 start view: a three-section concrete wall, cracked asphalt with weeds at its base and aged multi-storey buildings ('rocks' are a few pebbles). The decision was Keep, so v2 first left the task as in v1; on 29 October 2026 the maintainer accepted the extra descriptions. v2 uses the list from the independent review of 6 October 2026, which dropped 'the scene is quiet and desolate' (a fragment that alone passes any empty-scene answer) and a near-duplicate, and added the description accepted on the identical Q_476 (same question, preset and frame as Q_461, Q_470 and Q_476, which share this list, rule R10).

### Q_474 · postwar-city · diff-view-1 · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A dense urban area of mixed heights and modern styles - taller buildings with balconies alongside shorter boxy structures, ranging from low-rise blocks to taller towers. Shrubbery and barricades are present.", "a dense cluster of buildings of varying heights and styles, from low-rise residential blocks to taller towers", "a high-angle view of clustered buildings of varying heights around a central open space with greenery and scattered debris", "a high overview of a paved area with several buildings of varying heights and architectural styles", "older, weathered apartment blocks and a few tall modern towers around an overgrown, rubble-strewn central lot", "a compact, run-down town block of multi-story buildings around an overgrown lot with low concrete walls and wrecked cars", "a dense urban area mixing residential and commercial buildings of varying designs, some with flat roofs and some sloped, around a central open area"] |

**Why** (applied by rule R1 R4 R10; Maintainer decision of 2026-10-05, task card Q_474 (accept); reconciliation of 2026-10-06 (maintainer approval): original reference restored, descriptions kept as also_accept): Render and tRPL2 frames show a ~15-storey tower with balconies, a white slab tower, 4-7 storey older blocks and plain boxy blocks around an overgrown lot with low concrete-block walls and two wrecked cars, so 'low-rise' is wrong; nearly every model answer describes varied heights fairly and fails on the old reference. 5 Oct: Adds his accepted description. 6 Oct: his 27 Sep decision kept the reference; the duplicate rule had replaced it.

### Q_475 · postwar-city · street-view-1 · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A cracked, worn cobblestone street with patches of grass and moss, lined by tall, aged buildings, with small plants along the walls and ground", "Overgrown vegetation on the buildings, a cracked and worn road, and scattered debris and plants along the street", "Weathered, neglected buildings, rubble and concrete blocks in the street, and weeds pushing through cracked pavement", "Old, worn buildings with grass and small plants along the street and scattered small yellow and white panels", "A purple door on the left, a rusted wrecked car, and a vine-covered tree against a building", "a cracked cobblestone street with moss and grass growing between the stones", "numerous small plants or weeds growing along the walls and throughout the scene"] |

**Why** (applied by rule R1 R10; Maintainer decision of 2026-10-05, task card Q_475 (accept)): Start view (render; Q_463_0_000950_raw) shows low concrete barricades, scattered rocks on a cracked moss-patched street, grass clumps and plant cards, weathered buildings, a purple panel/door at left, a rusted car and an ivy-covered tree. Reference is fair, but 0-3/29 pass: grounded answers (cracked street, moss, aged buildings, overgrowth) fail for not naming barricades/rocks. No decision recorded; follows the… 5 Oct: Adds his accepted descriptions.

### Q_476 · postwar-city · street-view-2 · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["Weathered concrete walls, cracked pavement and tall aged buildings in the background", "the scene is quiet and desolate", "A segmented grey concrete wall or barrier in the foreground, with multi-storey buildings of varied heights behind it", "Cracked dark asphalt in front of a damaged concrete barrier, with weeds growing at its base", "A tall stone-faced building with a plain beige side wall beside a pale, ornate apartment block on the right", "Urban decay: a partly broken concrete barricade with a corrugated-metal section, a cracked road and old buildings", "a cracked, worn concrete surface with buildings of varying styles and colours on both sides"] |

**Why** (applied by rule R1 R10; Maintainer decision of 2026-10-05, task card Q_476 (accept)): preset render postwar-city__street-view-2.png and the matching real frame (e.g. Q_461_0_000935_raw.jpg) show a three-section concrete slab wall with a corrugated middle part, cracked asphalt with weed tufts, and aged multi-storey buildings; "rocks" are only a few tiny pebbles at the wall base. Fair answers naming concrete walls, cracked pavement and old buildings failed. 5 Oct: Adds his accepted description.

### Q_477 · postwar-city · diff-view-1 · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A dense urban area of mixed heights and modern styles - taller buildings with balconies alongside shorter boxy structures, ranging from low-rise blocks to taller towers. Shrubbery and barricades are present.", "a dense cluster of buildings of varying heights and styles, from low-rise residential blocks to taller towers", "a high-angle view of clustered buildings of varying heights around a central open space with greenery and scattered debris", "a high overview of a paved area with several buildings of varying heights and architectural styles", "older, weathered apartment blocks and a few tall modern towers around an overgrown, rubble-strewn central lot", "a compact, run-down town block of multi-story buildings around an overgrown lot with low concrete walls and wrecked cars", "a dense cluster of multi-storey buildings of varying heights, mostly grey and beige with darker accents, around a central courtyard with greenery and scattered trees"] |

**Why** (applied by rule R1 R4 R10; Maintainer decision of 2026-10-05, task card Q_477 (accept); reconciliation of 2026-10-06 (maintainer approval): original reference restored, descriptions kept as also_accept): Same question with a full-view tool; the full panorama shows the same town block on a flat cracked ground plate with grey background around it. The reference's 'low-rise' contradicts the tall towers, and fair varied-height descriptions fail. 5 Oct: Adds his accepted description. 6 Oct: his 27 Sep decision kept the reference; the duplicate rule had replaced it.

### Q_478 · postwar-city · street-view-1 · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A cracked, worn cobblestone street with patches of grass and moss, lined by tall, aged buildings, with small plants along the walls and ground", "Overgrown vegetation on the buildings, a cracked and worn road, and scattered debris and plants along the street", "Weathered, neglected buildings, rubble and concrete blocks in the street, and weeds pushing through cracked pavement", "Old, worn buildings with grass and small plants along the street and scattered small yellow and white panels", "A purple door on the left, a rusted wrecked car, and a vine-covered tree against a building", "tall, weathered buildings lining a narrow, cracked cobblestone street, with grass, moss and small plants in the cracks and on the walls"] |

**Why** (applied by rule R1 R10; Maintainer decision of 2026-10-05, task card Q_478 (accept)): Start view (render; Q_463_0_000950_raw) shows low concrete barricades, scattered rocks on a cracked moss-patched street, grass clumps and plant cards, weathered buildings, a purple panel/door at left, a rusted car and an ivy-covered tree. Reference is fair, but 0-3/29 pass: grounded answers (cracked street, moss, aged buildings, overgrowth) fail for not naming barricades/rocks. No decision recorded; follows the… 5 Oct: Adds his accepted description.

### Q_479 · postwar-city · street-view-2 · `descriptor`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["Weathered concrete walls, cracked pavement and tall aged buildings in the background", "the scene is quiet and desolate", "A segmented grey concrete wall or barrier in the foreground, with multi-storey buildings of varied heights behind it", "Cracked dark asphalt in front of a damaged concrete barrier, with weeds growing at its base", "A tall stone-faced building with a plain beige side wall beside a pale, ornate apartment block on the right", "Urban decay: a partly broken concrete barricade with a corrugated-metal section, a cracked road and old buildings"] |

**Why** (applied by rule R1 R10): preset render postwar-city__street-view-2.png and the matching real frame (e.g. Q_461_0_000935_raw.jpg) show a three-section concrete slab wall with a corrugated middle part, cracked asphalt with weed tufts, and aged multi-storey buildings; "rocks" are only a few tiny pebbles at the wall base. Fair answers naming concrete walls, cracked pavement and old buildings failed.

### Q_480 · postwar-city · diff-view-1 · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["a cluster of multi-story buildings with gray roofs", "rows of older multi-story apartment buildings with many windows along a paved street", "several buildings of varying heights and architectural styles, some with flat roofs"] |

**Why** (applied by rule R1): Left third of the render and tRPL2 001209: rows of 4-7 storey older apartment blocks with flat roofs, a boxy dark-grey-roofed block with a pink door, and a small paved square. Reference is fair; the common answer 'cluster of multi-story buildings with gray roofs' passes only 2/3 (qwen judge fails it).

### Q_482 · postwar-city · street-view-2 · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A row of multi-storey buildings extends to the right of the barricade", "A large, light-coloured multi-storey apartment building with many windows", "The street opens onto a row of pale, ornate residential buildings"] |

**Why** (applied by rule R1 R3): In preset render postwar-city__street-view-2.png (and Q_482_0_001222_raw.jpg) the wall ends at about x=760 and the right third is a pale, ornate white/grey apartment block beside the street. "A row of buildings" answers failed only for leaving out "white".

### Q_483 · postwar-city · street-view-1 · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A slender tree with green foliage growing up the front of a tan building", "Green ivy or vines climbing up a building facade near the centre", "A small green tree growing in front of the buildings amid the overgrown ruins"] |

**Why** (applied by rule R1 R3): After the 20 deg right pan (Q_483_0_001226_ptz) the ivy-covered tree against the tan building sits just left of centre, with the red-awning storefront right of centre. Reference is right but narrow; the fair "vertical structure covered in green vines" answer failed, while the "street sign" answer is a hallucination and should keep failing.

### Q_484 · postwar-city · street-view-2 · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A large, pale multi-storey apartment building with many windows is near the centre", "A row of white, ornate multi-storey buildings", "A white building with a dark shadow across part of its facade"] |

**Why** (Maintainer decision of 29 October 2026, accept the extra descriptions; rules R1 R3; list as pruned by the independent review of 6 October 2026): The full audit (27 September, rules R1 and R3) proposed accepted descriptions of the 20-degree-right view, which centres on a pale, ornate apartment block. The decision was Keep, so v2 first left the task as in v1; on 29 October 2026 the maintainer accepted the extra descriptions. v2 uses the list from the independent review of 6 October 2026, which reworded 'A white building whose side wall is in dark shadow' to '... dark shadow across part of its facade' (the dark band is a cast shadow across the front, not a side wall).

### Q_485 · postwar-city · diff-view-1 · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["a tall slender light-coloured building with many windows", "a tall rectangular high-rise"] |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_485 option A): Sound. 5 Oct: Two high-rises sit near the centre after the pan; the failing answers describe the high-rise the reference names.

### Q_486 · postwar-city · street-view-1 · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A storefront with a reddish-brown awning and a red door", "A row of multi-storey buildings with a red-awning storefront, with rubble, concrete blocks, grass and small plants in front", "Tall grass clumps and a low concrete wall on the cracked pavement in front of a red storefront", "The right side of the frame shows a building with a reddish-brown awning"] |
| `expected_tool_order_json` | [{"name": "query_answer", "args": {"instruction": "What is at the left side of the frame right now?", "view_type": "current"}}] | [{"name": "query_answer", "args": {"instruction": "What is on the right side of the frame right now?", "view_type": "current"}}] |
| `expected_tool_args_json` | {"query_answer": {"instruction": "What is at the left side of the frame right now?", "view_type": "current"}} | {"query_answer": {"instruction": "What is on the right side of the frame right now?", "view_type": "current"}} |

**Why** (applied by rule R1; Maintainer decision of 2026-10-05, task card Q_486 (accept); Maintainer decision of 2026-10-05, follow-up 'Tools (36)' option A: expected tool calls fixed as the full audit recommended): Right third of the start view (render) shows grass clumps, plant cards, a low concrete barricade, rocks on the pavement and a red-walled storefront with a reddish-brown awning. The reference is right, but 8 of 9 answers ("a building with a reddish-brown awning") are correct and fail. The expected query_answer instruction asks about the LEFT side, contradicting the question; fixed in set_fields. 5 Oct: Adds his accepted description.

### Q_487 · postwar-city · diff-view-1 · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["a cluster of multi-story buildings with gray roofs", "rows of older multi-story apartment buildings with many windows along a paved street", "several buildings of varying heights and architectural styles, some with flat roofs"] |

**Why** (applied by rule R1 R10): Same question and preset as Q_480 with an equally fair reference; tRPL2 001255 is the same aerial, so it takes the same alternates.

### Q_488 · postwar-city · street-view-1 · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A purple door or panel on a dark, weathered building", "Weathered buildings with a purple door, the sidewalk, and a rusted wrecked car beside a rubble pile", "A dark old building with a bright purple panel, the raised sidewalk, and rocks scattered on the ground"] |

**Why** (applied by rule R1): Left third of the start view (render; Q_488_0_103744_postzoom zoomed on the left) shows a dark weathered building with a bright purple panel/door, the raised sidewalk, a rubble pile and the rusted car, and rocks on the ground; shrubs there are sparse. Reference fair and "purple door" answers pass 2/3; alternates added because the task is open-ended.

### Q_489 · postwar-city · street-view-2 · `location_spatial`

| Field | v1 | v2 |
|---|---|---|
| `also_accept` | — | ["A large grey concrete wall in the foreground with a tall white building behind it", "A segmented grey concrete wall or barrier", "A tall, light-coloured multi-storey building above the concrete wall"] |

**Why** (applied by rule R1): The left half of preset render postwar-city__street-view-2.png is taken up by the tall concrete slab wall in front, with a tall white tower block and a lower white building above it at the top left. Answers naming only one of the two (for example "a tall, light-coloured building") failed.

### Q_503 · postwar-city · street-view-2 · `multi_step_reasoning`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I captured a screenshot and there is one phone booth in the full view. | I took a screenshot, and yes, a phone booth exists in the full 360° view. |
| `also_accept` | — | ["Screenshot taken; a red phone booth is present in the panorama", "Screenshot captured, and there is one phone booth in the full view"] |

**Why** (applied by rule R5 R3): The stitched 360 panorama of postwar-city street-view-2 shows one red phone booth, so "yes" is right. At least 5 runs that took the screenshot and said a phone booth exists failed: judges objected that the query_answer instruction was worded differently (args_check is False) or that the answer gave no count.

### Q_516 · postwar-city · street-view-2 · `ocr_identification`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | It appears to be a barricade. | It is a concrete barricade: a wall of tall, weathered vertical concrete slabs. |
| `also_accept` | — | ["A concrete barrier or concrete wall", "A concrete slab fence made of vertical slabs", "A partly broken concrete barricade with a corrugated-metal section"] |

**Why** (applied by rule R5 R3): Identification, not text: 'a barricade' restated as the concrete barricade it is, with fair alternates (R5, R3).

### Q_519 · postwar-city · street-view-2 · `ocr_identification`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | It appears to be a barricade. | It is a concrete barricade: a wall of tall, weathered vertical concrete slabs. |
| `also_accept` | — | ["A concrete barrier or concrete wall", "A concrete slab fence made of vertical slabs", "A partly broken concrete barricade with a corrugated-metal section"] |

**Why** (applied by rule R5 R3 R10): Twin of Q_516 (R10).

### Q_523 · postwar-city · diff-view-1 · `ocr_identification`

| Field | v1 | v2 |
|---|---|---|
| `expected_answer` | I see two cars. | The vehicles are cars; there are two. |
| `also_accept` | — | ["There are 2 vehicles visible in the current view"] |

**Why** (Maintainer decision of 2026-10-05, follow-up Q_523 option B (keep the count-only alternate)): Names the type, which is what the question asks for. 5 Oct: Count-only answers pass, as entered.

## Examined and kept unchanged

- **Q_009**: Correct as written. 39% is genuine model failure, not a grading fault.
- **Q_015**: Correct as written. 32% is genuine model failure.
- **Q_323**: Sound as written. 2% across 163 gradings is genuine model failure: counting over a full 360 view is simply hard, and the question is fair.
