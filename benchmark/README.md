# SCOPE Benchmark

The SCOPE benchmark is **536 tasks**, dataset v2, in [`scope_v2.csv`](scope_v2.csv):
the HRI '26 paper's tasks ([`archive/scope_536_v1.csv`](archive/scope_536_v1.csv)) with 187
rows corrected. Every change and its reason is in [`CHANGES_v2.md`](CHANGES_v2.md); v2 is
rebuilt from v1 and [`corrections.json`](corrections.json) by `scripts/apply_corrections.py`.
v2 adds one column, `also_accept`: a JSON list of alternates that also pass. SCOPE 0.1.0 shipped
a 541-row file named `scope_536.csv`; it is removed, and
[`archive/lineage_541_to_536.csv`](archive/lineage_541_to_536.csv) lists its 23 removed and 18
inserted rows. Each row
specifies a natural-language question, the Blender scene to load, the expected
answer, and metadata used for automated evaluation.

---

## CSV Schema

| Column | Type | Description |
|--------|------|-------------|
| `question_id` | string | Unique identifier (e.g., `Q_001`) |
| `file_location` | string | Relative path to the `.blend` scene file under `benchmark/` |
| `preset_start` | string | Camera preset to apply before posing the question |
| `presets_available` | JSON array (string) | List of preset names available in the scene |
| `question` | string | Natural-language question posed to the agent |
| `expected_answer` | string | Ground-truth answer used for evaluation |
| `question_type` | string | High-level question type (e.g., `QA`) |
| `eval_category` | string | Benchmark category (see below) |
| `multi_step_mode` | string | `none` for single-step; describes chaining mode for multi-step tasks |
| `required_tools_policy` | string | Policy for which tools must be used (optional) |
| `expected_tool_order_json` | JSON array (string) | Ordered list of expected tool calls with arguments |
| `args_check` | boolean | Whether tool argument matching is enforced |
| `expected_tool_args_json` | JSON object (string) | Expected arguments per tool for argument-level evaluation |
| `answer_view` | string | `current` or `full` -- whether the answer requires panorama |
| `evaluation_notes` | string | Human-written notes for the evaluator/judge |
| `difficulty` | string | `Easy`, `Medium`, or `Hard` |
| `scene_objects_mentioned` | string | Key objects referenced in the question |
| `tool_schema_version` | string | Schema version the task was written against (e.g., `v1.0`) |
| `nl_required` | string | Whether natural-language output is required beyond tool results |
| `text_normalization` | string | Normalization rules for answer comparison |
| `expected_tools_multiset` | string | Unordered multiset of expected tools (for flexible matching) |
| `expected_tool_order_signature` | string | Compact signature of the expected tool sequence |
| `scope_requirement` | string | Additional scope/constraint notes |
| `arg_match_policy` | string | How strictly tool arguments are compared (`exact`, `fuzzy`, etc.) |
| `gt_evidence_json` | string | Ground-truth evidence supporting the expected answer |

---

## Evaluation Categories

| Category | `eval_category` value | Count | Description |
|----------|-----------------------|------:|-------------|
| Counting | `counting` | 95 | Count objects matching a natural-language description. Requires `count_pointing` with correct `view_type`. |
| Descriptor | `descriptor` | 89 | Identify or describe visual attributes (color, shape, material) of objects in the scene. |
| Location / Spatial | `location_spatial` | 53 | Reason about spatial relationships between objects (left of, above, behind, etc.). |
| OCR Identification | `ocr_identification` | 54 | Read and report text visible on signs, labels, or surfaces in the scene. |
| Single Call | `single_call` | 72 | Tasks solvable with exactly one tool call. Tests basic tool selection. |
| Multi-step Command | `multi_step_command` | 57 | Tasks requiring a sequence of tool calls executed in order (e.g., go to preset, then count). |
| Multi-step Reasoning | `multi_step_reasoning` | 54 | Multi-step chains where intermediate results inform subsequent tool calls. |
| Comparative Relational | `comparative_relational` | 62 | Compare attributes or counts across multiple objects or viewpoints. |

---

## Scenes

Blender scene files (`.blend`) are stored in `benchmark/scenes/`. Each scene
corresponds to a 3D environment with pre-defined camera presets. The
`file_location` column in the CSV gives the relative path from the `benchmark/`
directory.

Scene files are not included in this repository due to size constraints.
Download them separately following the instructions in the project root README,
or create your own scenes following the guide in
[`docs/creating_scenes.md`](../docs/creating_scenes.md).

---

## Running the Evaluation

### Full Benchmark

```bash
bash scripts/run_eval_pipeline.sh configs/agent_config.yaml results/   # one Blender process
python3 scripts/run_fast.py --gpu 0 --out results/my-pairing --judge    # GPU server, parallel shards
```

Both read `benchmark/scope_v2.csv` by default; see [`docs/RUN_FAST.md`](../docs/RUN_FAST.md) for
the parallel runner.

### Single Category

Write that category's rows to their own CSV and pass it as the questions file:

```bash
mkdir -p results/counting
python3 -c "import pandas as pd; d = pd.read_csv('benchmark/scope_v2.csv'); d[d.eval_category == 'counting'].to_csv('results/counting/questions.csv', index=False)"
QUESTIONS_CSV=results/counting/questions.csv bash scripts/run_eval_pipeline.sh configs/agent_config.yaml results/counting/
```

`run_fast.py --questions results/counting/questions.csv` does the same on a GPU server.

### Evaluation Pipeline

1. **Scene loading** -- The harness opens the Blender scene specified by
   `file_location` and applies the `preset_start` camera preset.
2. **Agent execution** -- The question is passed to the `AgentClient`, which
   runs its tool-calling loop until a final text response is produced.
3. **Tool-call scoring** -- The agent's tool-call trace is compared against
   `expected_tool_order_json` and `expected_tool_args_json` using the policy
   specified by `required_tools_policy` and `arg_match_policy`.
4. **Answer scoring** -- The agent's final text answer is compared against
   `expected_answer` using an LLM-as-Judge (configured in `evaluation.judge`).
5. **Results** -- Per-task and aggregate scores are written to the output
   directory.

---

## Adding New Tasks

To add tasks to the benchmark:

1. Create or select a Blender scene with appropriate camera presets.
2. Add one row per task to a new CSV file (keep `scope_v2.csv` fixed so results stay comparable) following the
   schema above.
3. Ensure `expected_tool_order_json` accurately reflects the ideal tool-call
   sequence.
4. Set `difficulty` based on the number of reasoning steps and tool calls
   required.
5. Run the evaluation to verify that a known-good agent configuration produces
   the expected answer.
