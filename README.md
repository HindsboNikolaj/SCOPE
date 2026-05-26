# SCOPE: Simulation and Camera Operations for Perception and Evaluation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![HRI '26](https://img.shields.io/badge/HRI%20'26-Edinburgh-green.svg)](https://doi.org/10.1145/3757279.3785641)
[![Benchmark: 536 tasks](https://img.shields.io/badge/Benchmark-536%20tasks-orange.svg)](benchmark/scope_536.csv)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)

**SCOPE is a modular multimodal agentic system for natural-language PTZ camera control.** A Small Language Model planner orchestrates a fixed action space — a set of `skills` (camera-control and perception workflows) exposed through an OpenAI-compatible JSON tool schema identical on Blender simulation and a physical AXIS PTZ. A Vision-Language Model handles perception as a callable skill. The repo ships the agent loop, the 9-skill schema, the 536-task HRI '26 benchmark, and an LM-as-Judge eval harness.

**Paper:** [`paper/SCOPE_HRI26.pdf`](paper/SCOPE_HRI26.pdf) |
**DOI:** [10.1145/3757279.3785641](https://doi.org/10.1145/3757279.3785641)

---

## Architecture

```
                          +------------------+
                          |   User / Eval    |
                          |   Harness        |
                          +--------+---------+
                                   |
                              natural language
                                   |
                          +--------v---------+
                          |   SLM Planner    |
                          |  (Qwen3, etc.)   |
                          +--------+---------+
                                   |
                            tool calls (JSON)
                                   |
                     +-------------+-------------+
                     |                           |
              +------v------+           +--------v--------+
              |  PTZ Tools  |           | Perception Tools|
              |  (Blender)  |           |     (VLM)       |
              +------+------+           +--------+--------+
                     |                           |
              Blender scene             caption / VQA /
              manipulation              detect / point
                     |                           |
              +------v---------------------------v--------+
              |          Blender 3D Scene                  |
              |   (camera, presets, rendered frames)       |
              +-------------------------------------------+
```

---

## Quick Start

```bash
pip install -e .
cp .env.example .env  # fill MOONDREAM_API_KEY, AGENT_API_BASE, AGENT_MODEL_ID, OPENAI_API_KEY
bash scripts/01_install.sh
bash scripts/02_pull_models.sh qwen3:30b-a3b
bash scripts/03_download_scenes.sh                                # ~GB of .blend scenes; required before any benchmark step
blender --background --python scripts/04_install_presets.py       # uses bpy; must run inside Blender
```

## Quick Start for AI agents

If you're using Claude Code or Codex, paste the contents of [`AGENT_INSTRUCTIONS.md`](AGENT_INSTRUCTIONS.md) into your session and the agent will set the project up end-to-end.

---

## Skills

| Skill | Type | Description |
|------|------|-------------|
| `ptz_adjust` | Camera | Adjust pan, tilt, and zoom numerically (relative moves) |
| `go_to_preset` | Camera | Move camera to a named preset position |
| `home_action` | Camera | Return camera to its home position |
| `get_presets` | Camera | List all available camera preset names |
| `take_image` | Camera | Capture the current camera frame to disk |
| `count_pointing` | Perception | Count objects matching a description via VLM pointing |
| `query_answer` | Perception | Answer a visual question about the scene via VLM VQA |
| `zoom_bounding` | Perception | Zoom to fill the frame with a described object via VLM detection |
| `track_object` | Camera | Track a described object for a specified duration |

The benchmark covers 8 task categories (counting, descriptor, location/spatial, OCR identification, single-call, multi-step command, multi-step reasoning, comparative/relational); see paper §4 for the breakdown.

---

## Top-5 Results

| Rank | SLM | VLM | Overall Accuracy |
|-----:|-----|-----|:----------------:|
| 1 | Qwen3-30B-A3B | Moondream3 | **73.8%** |
| 2 | Qwen3-30B-A3B | Qwen2.5-VL-7B | 72.4% |
| 3 | Qwen3-32B | Moondream3 | 71.6% |
| 4 | Qwen3-32B | Qwen2.5-VL-7B | 70.9% |
| 5 | Qwen3-30B-A3B | Moondream2 | 69.5% |

See [`docs/RESULTS_FULL.md`](docs/RESULTS_FULL.md) for the full 20-configuration matrix.

---

## Run the Benchmark

The canonical entry point is the end-to-end pipeline wrapper. It runs
`scope.eval.runner` inside Blender (which writes the rich, judge-compatible
CSV schema), then the LLM-as-Judge, then the metrics report.

```bash
./scripts/run_eval_pipeline.sh configs/agent_config.yaml results/
```

Knobs (all optional, all env-var driven):

| Var | Default | Notes |
| --- | --- | --- |
| `QUESTIONS_CSV` | `benchmark/scope_536.csv` | Input question set |
| `OUT_CSV` | `results/run_<ts>/raw_results.csv` | Pre-set to resume into a specific file |
| `REPEATS` | `1` | Repeats per question |
| `SCOPE_RESUME` | `1` | Skip qids that already have a non-empty `final_answer`; `0` = fresh run |
| `BENCH_LIMIT` | `0` (all) | Cap to first N rows -- handy for smoke tests |
| `BLENDER_BIN` | `blender` | Full path if Blender isn't on `PATH` |
| `JUDGE_API_BASE` / `JUDGE_MODEL_ID` / `JUDGE_API_KEY` | OpenAI / `gpt-4o` / `$OPENAI_API_KEY` | Judge config |

If you want to invoke the three stages manually, the equivalents are:

```bash
# 1. Runner (canonical, writes the rich schema).
#    NOTE: do NOT pass --background -- the runner needs a real Blender
#    GUI window so screenshot capture (helper_funcs.screenshot_camera_view)
#    has 3D viewport context. The launcher opens the row's .blend itself,
#    and the runner re-opens per row as the benchmark walks through scenes.
QUESTIONS_CSV=benchmark/scope_536.csv \
OUT_CSV=results/raw_results.csv \
blender benchmark/scenes/after-the-rain-vr-sound/Whitechapel.blend \
        --python src/scope/eval/runner.py

# 2. Judge.
python scripts/11_judge.py -i results/raw_results.csv \
                           -o results/judged_results.csv

# 3. Report.
python scripts/12_metrics.py report -i results/judged_results.csv
```

---

## Documentation

- [`AGENT_INSTRUCTIONS.md`](AGENT_INSTRUCTIONS.md) — setup brief for Claude Code / Codex
- [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) — model hosting, env vars, full YAML reference
- [`docs/RESULTS_FULL.md`](docs/RESULTS_FULL.md) — all 20 SLM+VLM combinations
- [`docs/architecture.md`](docs/architecture.md) — agent loop, tool dispatch, project layout
- [`docs/tool_reference.md`](docs/tool_reference.md) — per-skill parameters and return types
- [`docs/creating_scenes.md`](docs/creating_scenes.md) — adding new Blender scenes
- [`paper/SCOPE_HRI26.pdf`](paper/SCOPE_HRI26.pdf) — HRI '26 paper

---

## Citation

```bibtex
@inproceedings{Armada2026SCOPE,
  title     = {SCOPE: A Real-Time Natural Language Camera Agent at the Edge:
               A Sim-to-Real Benchmark and Analysis of Open-Source Vision
               and Language Agents for PTZ Camera Tasks},
  author    = {Hindsbo, Nikolaj and Ehsani, Sina and Mishra, Pragyana},
  booktitle = {Proceedings of the ACM/IEEE International Conference on
               Human-Robot Interaction (HRI '26)},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/3757279.3785641},
}
```

---

## License

MIT. See [LICENSE](LICENSE).
