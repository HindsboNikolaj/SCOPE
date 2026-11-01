# AGENT_INSTRUCTIONS.md

Paste this file into a Claude Code or Codex session to have it set up SCOPE
end-to-end. The agent should follow these steps in order and stop if any
step fails.

## 1. What this is

SCOPE is a modular multimodal agentic system for natural-language PTZ camera
control: an SLM planner + a VLM perception backend + a 9-skill OpenAI-compatible
tool schema + a 536-task benchmark (dataset v2) + an LLM-as-Judge harness.

## 2. Prerequisites

Install or verify these before running setup:

- **Blender 4.0+**, for the serial pipeline only (the GPU Docker path in
  [`docs/RUN_FAST.md`](docs/RUN_FAST.md) runs Blender in its image) — must be on `PATH` as
  `blender`, or set `BLENDER_BIN` to the full binary path
  (macOS example: `/Applications/Blender.app/Contents/MacOS/Blender`). If you prefer a symlink:
  `ln -s /Applications/Blender.app/Contents/MacOS/Blender /usr/local/bin/blender`.
- **Python 3.10+** — `python3 --version` should report ≥ 3.10.
- **A planner and a vision model** behind OpenAI-compatible endpoints. The default is one vLLM
  server running gemma-4-26B-A4B-it for both (v2: 77.6%). Pick a recipe by hardware from
  [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md#measured-recipes-dataset-v2) and use its exact
  serving command: vLLM needs `--enable-auto-tool-choice --tool-call-parser <parser>` or it
  rejects SCOPE's tool calls. Ollama, Moondream Cloud and hosted APIs work but are not
  measured in v2.
- **`OPENAI_API_KEY`** (or any OpenAI-compatible endpoint via `JUDGE_*`) for the LLM-as-Judge.

## 3. Setup sequence

Run these from the repo root, in this order. Don't continue past a failure.

```bash
python3 -m venv .venv
source .venv/bin/activate
bash scripts/01_install.sh                     # SCOPE (+ huggingface_hub) into the venv, the runner's
                                               # packages into Blender's Python, .env from .env.example
python scripts/install_blender_deps.py --check # must print OK; without it Blender lacks openai/yaml
bash scripts/02_pull_models.sh                 # default recipe's weights and vLLM image (on the model server)
# Start the model server(s) with the recipe's command, then edit .env (see below).
set -a; source .env; set +a                    # load .env into the shell for the commands below
bash scripts/03_download_scenes.sh             # ~1.2 GB of .blend scenes; required before any benchmark step
"${BLENDER_BIN:-blender}" --background --python scripts/04_install_presets.py    # uses bpy; writes Blender camera preset .py files into the user scripts/presets/camera/ dir
# NOTE: the eval pipeline below intentionally launches Blender WITHOUT --background --
# screenshot_camera_view() requires a real 3D viewport. --background is only safe here
# for 04_install_presets.py (no UI needed).
# 24-task end-to-end smoke test (all four worlds, every category): runner, judge and metrics.
# Compare with the default setup's 21 of 24 in v2 (README "Run the benchmark").
QUESTIONS_CSV=benchmark/scope_v2_smoke.csv SCOPE_RESUME=0 bash scripts/run_eval_pipeline.sh
```

### Serial pipeline options

`bash scripts/run_eval_pipeline.sh [config] [results_dir]` runs one Blender process; on a GPU
server `scripts/run_fast.py` is faster ([`docs/RUN_FAST.md`](docs/RUN_FAST.md)).

| Variable | Default | Use |
| --- | --- | --- |
| `BENCH_LIMIT` | `0` (all) | First N rows only; they are all book-nook comparisons, so use `benchmark/scope_v2_smoke.csv` for a smoke test |
| `SCOPE_RESUME` | `1` | `0` for a clean run |
| `REPEATS` | `1` | Repeated executions per task |
| `SCOPE_DATASET` | `v2` | `v1` for the paper's tasks |
| `QUESTIONS_CSV` | from `SCOPE_DATASET` | Any other questions file |

## 4. Required env vars

Read `.env.example` first, then fill at minimum:

| Var | Used by | Notes |
| --- | --- | --- |
| `AGENT_API_BASE` | Planner | default `http://localhost:8000/v1` (the Gemma server) |
| `AGENT_MODEL_ID` | Planner | the served model name, default `gemma4-26b-a4b` |
| `AGENT_API_KEY` | Planner | `EMPTY` for vLLM; `ollama` for Ollama; real key for hosted APIs |
| `VLM_MODEL` | Vision | `gemma` (any non-Moondream name = OpenAI-compatible vision server), `Qwen`, or `Moondream2` |
| `VLM_MODEL_URL` | Vision | default `http://localhost:8000/v1`; `http://localhost:2020` for `scripts/serve_moondream.py`; blank for Moondream Cloud. `VLM_BASE_URL` is an alias |
| `VLM_MODEL_ID` | Vision | served model name; `gemma`, `qwen3` or `molmo` in it selects that model's point/box reader |
| `MOONDREAM_API_KEY` | Vision (cloud) | only if using Moondream Cloud |
| `OPENAI_API_KEY` | Judge | or any OpenAI-compatible endpoint via `JUDGE_API_BASE` |
| `BLENDER_BIN` | Setup / runner | Required when `blender` is not on `PATH`; macOS app path shown above |
| `PYTHON_BIN` | Judge / metrics | Optional; defaults to `python` in `scripts/run_eval_pipeline.sh` |

## 5. Validation

After step 3, the smoke run should:
- Print 5 task rows from `[runner]` (one per question) ending with the runner
  shutting Blender down cleanly.
- Write a raw CSV at `results/run_<ts>/raw_results.csv` containing the rich
  schema (`question_id`, `llm_raw`, `llm_readable`, `final_answer`,
  `actual_tool_calls_json`, etc.) -- this is what the judge consumes.
- Run the LLM-as-Judge on that CSV and emit `judged_results.csv`.
- Print a numeric accuracy report broken down by category.

**Then check the pictures, not just the exit codes.** A run can complete, produce a full set of
numbers and a category breakdown, and be graded entirely on blank or wrong images: a capture
that came back black raises nothing, and the model will answer whatever it was shown. Compare
one capture against [`docs/VISUAL_SMOKE_TEST.md`](docs/VISUAL_SMOKE_TEST.md), which shows what
every camera position should look like. The `scope-verify-capture` skill in `.claude/skills/`
walks through it and lists the symptoms.

If the runner errors out because `benchmark/scenes/` is empty, re-run
`bash scripts/03_download_scenes.sh`. The primary mirror is Hugging Face Hub
(`HindsboNikolaj/scope-benchmark`); Google Drive is a fallback.

## 6. Common failure modes

- **`blender: command not found`** → install Blender, or set `BLENDER_BIN=/full/path/to/blender`.
- **`ModuleNotFoundError` inside Blender for `yaml`, `openai`, `requests` or `PIL`** →
  Blender runs the runner with its own Python. Run `python scripts/install_blender_deps.py`
  (it asks Blender for `sys.executable` and installs what is missing; add `--moondream` for
  Moondream Cloud or Photon), then `python scripts/install_blender_deps.py --check`.
- **Judge fails after `pip install -e .` with `ModuleNotFoundError: scope`** → `python3`
  and `pip` may point at different interpreters. Re-run install as `python -m pip install -e .`
  for the interpreter shown by `python --version`, or set `PYTHON_BIN=/path/to/python` in `.env`.
- **vLLM rejects tool calls with "auto tool choice requires --enable-auto-tool-choice"** →
  re-launch vLLM with `--enable-auto-tool-choice --tool-call-parser <parser>`: `gemma4` for
  Gemma 4, `qwen3_xml` for Qwen3.5/3.6/3.8, `hermes` for Qwen3, `openai` for gpt-oss
  ([`docs/CONFIGURATION.md`](docs/CONFIGURATION.md#other-planners)).
- **Serial pipeline stops after the runner** → the runner now exits Blender itself after the
  last row when no window manager drives Blender's timers; status 1 means a fatal error ended
  the batch early (see the `[runner]` lines).
- **Moondream server port collision on 2020** → stop the conflicting process or start
  `scripts/serve_moondream.py --port <other>` and update `VLM_MODEL_URL`.
- **`scene_not_found` errors** → `03_download_scenes.sh` likely failed. Try
  `hf download HindsboNikolaj/scope-benchmark --repo-type dataset --include "scenes/*"
   --local-dir benchmark/` manually (only `scenes/`: the benchmark CSV comes from this repository), or fall back to the Google Drive mirror
  with `gdown`. See `docs/MISSING_TEXTURES.md` for the texture caveats.
- **CUDA OOM when serving** → the recipes' `--gpu-memory-utilization` values are for an 80 GB
  card; lower it, or pick a smaller recipe (gemma-4-12B or E4B with Moondream 3.1).

## 7. Action space reference

The 9 skills and their JSON schema live at:
- `prompts/agent_system_prompt.md` — the live system prompt the SLM sees
- `src/scope/tools/schema.json` — OpenAI function-calling schema for all 9 skills

Implementations: `src/scope/tools/blender_tools.py` (camera) and
`src/scope/tools/vlm_clients.py` (perception).

## 8. Where the pieces live

| you want | look at |
| --- | --- |
| set up and run end to end | this file |
| check the captures are correct | `.claude/skills/scope-verify-capture/`, `docs/VISUAL_SMOKE_TEST.md` |
| verify tools, or use an LLM/VLM as a judge | `.claude/skills/scope-smoke-test/` |
| add a scene, or author rows | `docs/creating_scenes.md` |
| what the agent is told, and how answers are graded | `prompts/` |

`prompts/` is the benchmark's measured apparatus rather than configuration. The system prompt
and the per-category judge rubrics are inputs to the measurement, so changing one changes what
the numbers mean. Version them deliberately; never edit one to make a run pass.

## 9. If anything fails

1. Read [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) for the hosting-specific config that matches the SLM/VLM you're using.
2. Surface the failing command and the last 20 lines of stderr verbatim.
3. Do not retry blindly; diagnose first.
