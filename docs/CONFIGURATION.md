# Configuration

SCOPE needs a planner (an LLM with tool calling) and a vision model. Both are reached over
OpenAI-compatible HTTP endpoints, set in `.env` (copy `.env.example`). The judge is a third
endpoint (`JUDGE_*`, default `gpt-4o`); grading like v2 takes three (`PANEL_*`, see
[`JUDGING.md`](JUDGING.md#3-the-v2-judge-panel)).

## Measured recipes (dataset v2)

Pick by hardware. Each recipe is the serving setup a v2 run used, taken from
[`benchmark/results/v2-2026-10/configs.json`](../benchmark/results/v2-2026-10/configs.json) and
the October 2026 server launch records; scores are from [`RESULTS_V2.md`](RESULTS_V2.md) (536
tasks, three-judge panel). Grade with the same panel (`scripts/13_judge_panel.py`) to compare your
number with them; a single judge's number is not directly comparable.

| Hardware (as measured, A100 80 GB) | Planner + vision | v2 | Run | Pull |
| --- | --- | :---: | --- | --- |
| One GPU, one server (**default**) | gemma-4-26B-A4B-it, same server for vision | 77.6% | `tB08` | `bash scripts/02_pull_models.sh` |
| One GPU, two servers | gemma-4-12B-it + Moondream 3.1 | 78.2% | `tA09` | `... gemma-12b` |
| Smallest planner tested well | gemma-4-E4B-it + Moondream 3.1 | 69.0% | `tA11` | `... gemma-e4b` |
| Two GPUs, best measured | Qwen3.8-27B + Qwen3.6-35B-A3B | 81.5% | `tB06` | `... qwen-top` |

The server commands mount `HF_HOME`, where `scripts/02_pull_models.sh` puts the weights
(default `~/.cache/huggingface`). If you use a custom cache, export `HF_HOME` before pulling and
serving. The Docker volume must hold the real cache files: links whose targets are outside the
mounted folder fail when the server starts.

The GPU memory shares below are what the runs used on an 80 GB card; on a different card,
lower or raise `--gpu-memory-utilization` to fit. That changes only the cache size. The v2
runs also passed `--enable-sleep-mode` (for their scheduler; not needed here).

### Default: gemma-4-26B-A4B-it, one server (77.6%, `tB08`)

```bash
docker run -d --name scope-gemma --gpus '"device=0"' --ipc=host -p 127.0.0.1:8000:8000 \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  vllm/vllm-openai@sha256:41d42cfabd3289f40fdca71b4a0fe290474880d20e9534a90c698eb78e3eecee \
  --model google/gemma-4-26B-A4B-it --served-model-name gemma4-26b-a4b \
  --gpu-memory-utilization 0.85 --max-model-len 24576 --max-num-seqs 32 \
  --enable-auto-tool-choice --tool-call-parser gemma4
```

The image is the vLLM nightly build of 6 September 2026 that the run used. `.env`
(as in `.env.example`):

```
AGENT_API_BASE=http://localhost:8000/v1
AGENT_MODEL_ID=gemma4-26b-a4b
AGENT_API_KEY=EMPTY
VLM_MODEL=gemma
VLM_MODEL_URL=http://localhost:8000/v1
VLM_MODEL_ID=gemma4-26b-a4b
SCOPE_VLM_MAX_PIXELS=2073600
SCOPE_VLM_COORD_SPACE=auto
```

How this compares with the run: `tB08` was answered mostly in September 2026, before SCOPE read
Gemma's own box and point format (`box_2d` and `point_2d`, y before x); its October re-answers
and this release read it (see the Gemma note in [`RESULTS_V2.md`](RESULTS_V2.md)). Expect a
score near 77.6%, not exactly it; this release has not yet been run end to end on this recipe.
Keep `gemma` in the served model name: it selects Gemma's reader.

One difference from the October 2026 runtime applies to every recipe: it filled a run's
`final_answer` column more strictly (only an explicit final answer counted; an empty one stayed
empty), where this release's runner falls back to the last assistant message. The judge does not
grade that column: as in v2, it takes the final answer from the saved conversation and fails a
reply with no final answer in code.

### gemma-4-12B-it or gemma-4-E4B-it + Moondream 3.1 (78.2%, 69.0%)

Planner, same image as above:

```bash
docker run -d --name scope-gemma12 --gpus '"device=0"' --ipc=host -p 127.0.0.1:8000:8000 \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  vllm/vllm-openai@sha256:41d42cfabd3289f40fdca71b4a0fe290474880d20e9534a90c698eb78e3eecee \
  --model google/gemma-4-12B-it --served-model-name gemma4-12b \
  --gpu-memory-utilization 0.55 --max-model-len 24576 --max-num-seqs 32 \
  --enable-auto-tool-choice --tool-call-parser gemma4
```

For E4B use `--model google/gemma-4-E4B-it --served-model-name gemma4-e4b
--gpu-memory-utilization 0.45` with the same other flags.

Vision: Moondream 3.1 on Photon, Moondream's own runtime, behind the small HTTP server the
runs used (needs `pip install moondream`, CUDA and a torch build that matches the driver):

```bash
python scripts/serve_moondream.py --host 127.0.0.1 --port 2020 --backend photon --model moondream3.1-9B-A2B
```

```
AGENT_API_BASE=http://localhost:8000/v1
AGENT_MODEL_ID=gemma4-12b
AGENT_API_KEY=EMPTY
VLM_MODEL=Moondream2
VLM_MODEL_URL=http://localhost:2020
SCOPE_VLM_MAX_PIXELS=2073600
```

`VLM_MODEL=Moondream2` selects the Moondream HTTP client; the server decides the model.

### Qwen3.8-27B + Qwen3.6-35B-A3B, two GPUs (81.5%, `tB06`)

```bash
docker run -d --name scope-qwen-planner --gpus '"device=0"' --ipc=host -p 127.0.0.1:8000:8000 \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" vllm/vllm-openai:v0.27.1 \
  --model Qwen/Qwen3.8-27B --served-model-name qwen38-27b \
  --gpu-memory-utilization 0.85 --max-model-len 24576 --max-num-seqs 32 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
docker run -d --name scope-qwen-vision --gpus '"device=1"' --ipc=host -p 127.0.0.1:8001:8000 \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" vllm/vllm-openai:v0.27.1 \
  --model Qwen/Qwen3.6-35B-A3B --served-model-name qwen36-35b-v \
  --gpu-memory-utilization 0.9 --max-model-len 16384 --max-num-seqs 32
```

```
AGENT_API_BASE=http://localhost:8000/v1
AGENT_MODEL_ID=qwen38-27b
AGENT_API_KEY=EMPTY
VLM_MODEL=Qwen
VLM_MODEL_URL=http://localhost:8001/v1
VLM_MODEL_ID=qwen36-35b-v
SCOPE_VLM_MAX_PIXELS=2073600
SCOPE_VLM_COORD_SPACE=auto
```

Every task in `tB06` was answered with the Qwen vision fixes this release carries (native
`point_2d`/`bbox_2d`, vision thinking off). Keep `qwen3` in the vision model's served name: it
selects those.

### Other planners

Every v2 planner's flags are in `configs.json`. The tool-call parser follows the model family:
`hermes` for Qwen3 (4B, 8B, 30B-A3B, Next-80B), `qwen3_xml` for Qwen3.5/3.6/3.8, `gemma4` for
Gemma 4, `openai` for gpt-oss, `muse_glimmer` for Muse-Glimmer. Without
`--enable-auto-tool-choice --tool-call-parser ...` vLLM rejects SCOPE's tool calls with HTTP 400.

### Not measured in v2

Ollama, Moondream Cloud and any other endpoint work (any OpenAI-compatible server with tool
calling), but have no v2 score. For Ollama:

```bash
bash scripts/02_pull_models.sh --ollama qwen3:8b
# .env: AGENT_API_BASE=http://localhost:11434/v1  AGENT_API_KEY=ollama  AGENT_MODEL_ID=qwen3:8b
```

For Moondream Cloud: `VLM_MODEL=Moondream2`, `VLM_MODEL_URL=` (blank), `MOONDREAM_API_KEY=...`,
and `python scripts/install_blender_deps.py --moondream`. The paper's best (73.8% on v1) was
Qwen3-30B-A3B-Instruct-2507 with Moondream3-preview; no Ollama tag reproduces that setup.

---

## Configuration Reference

The YAML config (`configs/agent_config.yaml`) supports `${ENV_VAR}` references
that are resolved at load time from the shell environment or a `.env` file.

```yaml
agent:
  slm:
    backend: "vllm"                    # vllm | ollama | openai-compatible
    model_id: "gemma4-26b-a4b"         # the served model name
    base_url: "http://localhost:8000/v1"
    api_key: "EMPTY"                   # "EMPTY" for vLLM; "ollama" for Ollama; real key for hosted APIs
    temperature: 0.7
    max_tokens: 2048
    thinking: "toggle"                 # never | toggle | always | levels

  vlm:                                 # documentation only: the runner reads VLM_* from .env
    backend: "openai-compatible"
    model_id: "gemma4-26b-a4b"
    base_url: "http://localhost:8000/v1"
    api_key: "${MOONDREAM_API_KEY}"    # only for Moondream Cloud

evaluation:
  benchmark: "benchmark/scope_v2.csv"
  output_dir: "results/"
  judge:
    model: "${JUDGE_MODEL_ID}"
    base_url: "${JUDGE_API_BASE}"
    api_key: "${JUDGE_API_KEY}"

blender:
  scenes_dir: "benchmark/scenes/"
  screenshots_dir: "output/screenshots/"
  render_resolution: [1920, 1080]
```

### Configuration Fields

**Local / Self-hosted** (for Ollama or vLLM):

| Field | Default | Description |
|-------|---------|-------------|
| `agent.slm.backend` | — | SLM serving backend (`ollama`, `vllm`, `openai-compatible`) |
| `agent.slm.model_id` | — | Model name or path as recognized by the backend |
| `agent.slm.base_url` | `http://localhost:11434/v1` | OpenAI-compatible `/v1` endpoint |
| `agent.slm.api_key` | `ollama` | `"ollama"` for Ollama; `"EMPTY"` for vLLM |
| `agent.slm.temperature` | — | Sampling temperature for the planner |
| `agent.slm.thinking` | `toggle` | Thinking mode: `never` / `toggle` / `always` / `levels` |
| `agent.vlm.backend` | — | VLM backend identifier |
| `agent.vlm.model_id` | — | HuggingFace model ID for local VLM loading |
| `agent.vlm.base_url` | — | Documentation only; the runner reads `VLM_MODEL_URL` |
| `evaluation.benchmark` | — | Path to the benchmark CSV |
| `evaluation.judge.model` | — | Model used for LLM-as-Judge scoring |
| `blender.scenes_dir` | — | Directory containing `.blend` scene files |
| `blender.render_resolution` | — | Render resolution as `[width, height]` |

**API / Cloud-based** (for hosted services):

| Field | Description |
|-------|-------------|
| `agent.slm.api_key` | API key for your hosted LLM |
| `agent.vlm.api_key` | `MOONDREAM_API_KEY` for Moondream Cloud |
| `evaluation.judge.api_key` | `OPENAI_API_KEY` for GPT-4o; `JUDGE_API_KEY` for local judge |
