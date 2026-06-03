# AGENT_INSTRUCTIONS.md

Paste this file into a Claude Code or Codex session to have it set up SCOPE
end-to-end. The agent should follow these steps in order and stop if any
step fails.

## 1. What this is

SCOPE is a modular multimodal agentic system for natural-language PTZ camera
control: an SLM planner + a VLM perception backend + a 9-skill OpenAI-compatible
tool schema + a 541-task benchmark + an LLM-as-Judge harness.

## 2. Prerequisites

Install or verify these before running setup:

- **Blender 4.0+** — must be on `PATH` as `blender`, or set `BLENDER_BIN` to the full binary path
  (macOS example: `/Applications/Blender.app/Contents/MacOS/Blender`). If you prefer a symlink:
  `ln -s /Applications/Blender.app/Contents/MacOS/Blender /usr/local/bin/blender`.
- **Python 3.10+** — `python3 --version` should report ≥ 3.10.
- **An OpenAI-compatible inference endpoint for the SLM**, one of:
  - [Ollama](https://ollama.com) (`ollama serve` on port 11434)
  - [vLLM](https://docs.vllm.ai) (`vllm serve <model>` on port 8000+)
  - any hosted OpenAI-compatible API
- **A VLM**, one of:
  - Moondream Cloud (`MOONDREAM_API_KEY`)
  - `moondream-station` running locally on port 2020
  - Qwen2.5-VL served via vLLM
- **`OPENAI_API_KEY`** (or any OpenAI-compatible endpoint) for the LLM-as-Judge.

## 3. Setup sequence

Run these from the repo root, in this order. Don't continue past a failure.

```bash
python -m pip install -e .  # use the same Python that should run judge/metrics
cp .env.example .env        # then edit to fill the required env vars (see below)
# If .env contains BLENDER_BIN, AGENT_*, VLM_*, or JUDGE_* values needed for
# setup commands outside run_eval_pipeline.sh, load them into the shell:
set -a; source .env; set +a
bash scripts/01_install.sh
bash scripts/02_pull_models.sh qwen3:30b-a3b   # Ollama path; skip if using vLLM/API
bash scripts/03_download_scenes.sh             # ~GB of .blend scenes; required before any benchmark step
"${BLENDER_BIN:-blender}" --background --python scripts/04_install_presets.py    # uses bpy; writes Blender camera preset .py files into the user scripts/presets/camera/ dir
# NOTE: the eval pipeline below intentionally launches Blender WITHOUT --background --
# screenshot_camera_view() requires a real 3D viewport. --background is only safe here
# for 04_install_presets.py (no UI needed).
# 5-task end-to-end smoke test: runs the runner AND the judge AND the metrics report.
BENCH_LIMIT=5 SCOPE_RESUME=0 bash scripts/run_eval_pipeline.sh
```

## 4. Required env vars

Read `.env.example` first, then fill at minimum:

| Var | Used by | Notes |
| --- | --- | --- |
| `AGENT_API_BASE` | SLM client | e.g. `http://localhost:11434/v1` for Ollama |
| `AGENT_MODEL_ID` | SLM client | e.g. `qwen3:30b-a3b` |
| `AGENT_API_KEY` | SLM client | `ollama` for Ollama; `EMPTY` for vLLM; real key for hosted APIs |
| `MOONDREAM_API_KEY` | VLM (cloud) | only if using Moondream Cloud |
| `VLM_BASE_URL` | VLM (local) | e.g. `http://localhost:2020` for moondream-station |
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

If the runner errors out because `benchmark/scenes/` is empty, re-run
`bash scripts/03_download_scenes.sh`. The primary mirror is Hugging Face Hub
(`HindsboNikolaj/scope-benchmark`); Google Drive is a fallback.

## 6. Common failure modes

- **`blender: command not found`** → install Blender, or set `BLENDER_BIN=/full/path/to/blender`.
- **`ModuleNotFoundError` inside Blender for `yaml`, `openai`, `requests`, `PIL`, or `pandas`** →
  install deps into Blender's bundled Python, not just your shell Python. On macOS Blender 4.4:
  `/Applications/Blender.app/Contents/Resources/4.4/python/bin/python3.11 -m pip install pyyaml openai requests Pillow pandas python-dotenv`.
- **Judge fails after `pip install -e .` with `ModuleNotFoundError: scope`** → `python3`
  and `pip` may point at different interpreters. Re-run install as `python -m pip install -e .`
  for the interpreter shown by `python --version`, or set `PYTHON_BIN=/path/to/python` in `.env`.
- **vLLM rejects tool calls with "auto tool choice requires --enable-auto-tool-choice"** → re-launch vLLM with `--enable-auto-tool-choice --tool-call-parser hermes`.
- **`moondream-station` port collision on 2020** → kill the conflicting process or change the station port and update `VLM_BASE_URL`.
- **`scene_not_found` errors** → `03_download_scenes.sh` likely failed. Try
  `huggingface-cli download HindsboNikolaj/scope-benchmark --repo-type dataset
   --local-dir benchmark/` manually, or fall back to the Google Drive mirror
  with `gdown`. See `docs/MISSING_TEXTURES.md` for the texture caveats.
- **CUDA OOM on Qwen3-30B-A3B via vLLM** → reduce `--max-model-len` (try 8000), lower `--gpu-memory-utilization` (0.7), or fall back to `qwen3:4b` via Ollama.

## 7. Action space reference

The 9 skills and their JSON schema live at:
- `prompts/agent_system_prompt.md` — the live system prompt the SLM sees
- `src/scope/tools/schema.json` — OpenAI function-calling schema for all 9 skills

Implementations: `src/scope/tools/blender_tools.py` (camera) and
`src/scope/tools/vlm_clients.py` (perception).

## 8. If anything fails

1. Read [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) for the hosting-specific config that matches the SLM/VLM you're using.
2. Surface the failing command and the last 20 lines of stderr verbatim.
3. Do not retry blindly; diagnose first.
