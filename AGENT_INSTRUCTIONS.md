# AGENT_INSTRUCTIONS.md

Paste this file into a Claude Code or Codex session to have it set up SCOPE
end-to-end. The agent should follow these steps in order and stop if any
step fails.

## 1. What this is

SCOPE is a modular multimodal agentic system for natural-language PTZ camera
control: an SLM planner + a VLM perception backend + a 9-skill OpenAI-compatible
tool schema + a 536-task benchmark + an LLM-as-Judge harness.

## 2. Prerequisites

Install or verify these before running setup:

- **Blender 4.0+** — must be on `PATH` as `blender`, or set `BLENDER_BIN` to the full binary path
  (macOS example: `/Applications/Blender.app/Contents/MacOS/Blender`).
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
pip install -e .
cp .env.example .env  # then edit to fill the required env vars (see below)
bash scripts/01_install.sh
bash scripts/02_pull_models.sh qwen3:30b-a3b   # Ollama path; skip if using vLLM/API
bash scripts/03_download_scenes.sh
python scripts/04_install_presets.py            # must run via Blender; see script header
python -m scope.eval.run --limit 5             # 5-task smoke test
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

## 5. Validation

After step 3, the smoke run should print 5 task rows ending in `[BENCHMARK]
Complete: 5 ok, 0 errors`. A judged metrics report should show numeric accuracy
per category. If any row reports `error: scene_not_found`, re-run step
`03_download_scenes.sh`.

## 6. Common failure modes

- **`blender: command not found`** → install Blender, or set `BLENDER_BIN=/full/path/to/blender`.
- **vLLM rejects tool calls with "auto tool choice requires --enable-auto-tool-choice"** → re-launch vLLM with `--enable-auto-tool-choice --tool-call-parser hermes`.
- **`moondream-station` port collision on 2020** → kill the conflicting process or change the station port and update `VLM_BASE_URL`.
- **`scene_not_found` errors** → `03_download_scenes.sh` likely failed (Google Drive rate limit); re-run, or download manually per the script's printed instructions.
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
