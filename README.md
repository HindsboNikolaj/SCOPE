# SCOPE: Simulation and Camera Operations for Perception and Evaluation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![HRI '26](https://img.shields.io/badge/HRI%20'26-Edinburgh-green.svg)](https://doi.org/10.1145/3757279.3785641)
[![Benchmark: 536 tasks](https://img.shields.io/badge/Benchmark-536%20tasks-orange.svg)](benchmark/scope_536.csv)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)

**SCOPE** is a Blender-based simulation environment and benchmark for evaluating
language-driven PTZ (pan-tilt-zoom) camera agents. It pairs a Small Language
Model (SLM) planner with a Vision-Language Model (VLM) perception backend,
exposing a tool schema identical to real PTZ camera APIs. The accompanying
536-task benchmark spans eight categories and was used to evaluate 19 SLM+VLM
combinations. Published at the ACM/IEEE International Conference on
Human-Robot Interaction (HRI '26), Edinburgh, Scotland.

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

The **agent loop** works as follows:

1. The SLM receives the user prompt plus a system message listing available
   presets and instructions.
2. The SLM emits one or more **tool calls** (OpenAI function-calling format).
3. Each tool call is dispatched to either a **PTZ tool** (camera movement in
   Blender) or a **perception tool** (VLM inference on a rendered frame).
4. Tool results are appended to the conversation and the loop repeats until
   the SLM produces a final text response with no further tool calls.

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Blender 4.0+** (for simulation tools that call `bpy`)
- An SLM backend: [Ollama](https://ollama.com), [vLLM](https://docs.vllm.ai), or any OpenAI-compatible server
- A VLM backend: [Moondream](https://moondream.ai) (API or local) or Qwen2.5-VL

### 1. Install

```bash
git clone https://github.com/armada-ai/opus.git
cd opus
pip install -r requirements.txt
cp .env.example .env   # edit with your API keys
```

```bash
# Option B — conda
conda env create -f environment.yaml
conda activate scope
```

> **Easiest setup to get started:**
> Use **Qwen3-4B via Ollama** (no GPU needed) + **Moondream Cloud API** (free tier).
> ```bash
> ollama pull qwen3:4b
> cp configs/presets/qwen3_4b__moondream2.yaml configs/agent_config.yaml
> ```
> Sign up for a free Moondream API key at [moondream.ai](https://moondream.ai) and set `MOONDREAM_API_KEY` in your `.env`.

### 2. Configure

Edit `configs/agent_config.yaml` or use a preset:

```bash
# Best-performing configuration from the paper
cp configs/presets/qwen3_30b_moondream3.yaml configs/agent_config.yaml
```

Or start from the blank template:

```bash
cp configs/presets/custom_template.yaml configs/agent_config.yaml
```

See [Configuration Reference](#configuration-reference) below for all fields.

### 3. Serve Your SLM

See [Hosting Your Models](#hosting-your-models) below for Ollama, vLLM, and API key setup.

### 4. Run Interactively

Launch Blender with a scene and use the agent from a Python console inside Blender:

```python
from scope.agent import AgentClient

agent = AgentClient()
text, messages, timings, tree = agent.ask("How many people are in the scene?")
print(text)
print(timings)
```

Or run the interactive demo script directly:

```bash
python scripts/run_interactive_demo.py \
  --scene benchmark/scenes/after-the-rain-vr-sound/Whitechapel.blend \
  --question "How many people are visible right now?"
```

### Blender UI (Interactive Mode)

The repository includes a Blender UI panel addon (`scope/blender/agent_banner.py`)
that provides an interactive interface for testing agents in Blender's 3D viewport.
It includes object detection, screenshot capture, and an agent chat panel with
timing breakdown.

Launch with:

```bash
python scripts/run_interactive.py --scene benchmark/scenes/your-scene.blend
```

### Setting Up Blender Camera Presets

The benchmark tasks reference named camera presets (e.g. `eor-viewpoint`, `store-front`). Install them with:

```bash
blender --background --python scripts/setup_presets.py
```

This installs all 10 benchmark presets from `benchmark/presets/presets.json`.

To manage presets interactively, install `scope/blender/presets_banner.py` as a Blender addon:
- **View3D → Sidebar → Presets** tab
- Create, list, and apply camera presets from the UI
- Name custom presets with a scene prefix (e.g. `myworld-home`) to avoid conflicts between scenes

### 5. Run the Benchmark

```bash
# Full evaluation (requires Blender, SLM, VLM, and an LLM judge)
python -m scope.eval.run \
    --config configs/agent_config.yaml \
    --benchmark benchmark/scope_536.csv \
    --output results/
```

---

## Hosting Your Models

SCOPE requires two model backends: an **SLM** (Small Language Model) for planning
and tool selection, and a **VLM** (Vision-Language Model) for visual perception.
Pick one option from each section below.

### SLM Backends (pick one)

#### Option A: Ollama (Recommended for Getting Started)

```bash
# Install Ollama: https://ollama.com
ollama pull qwen3:30b-a3b    # Best performing MoE model (73.8%)
ollama pull qwen3:4b          # Lightweight dense model
ollama pull qwen3:32b         # Large dense model
ollama serve                  # Start serving (default port: 11434)
```

Set in `.env`:

```
AGENT_API_BASE=http://localhost:11434/v1
AGENT_API_KEY=ollama
AGENT_MODEL_ID=qwen3:30b-a3b
```

#### Option B: vLLM (For FP8 Quantization, Multi-GPU, High Throughput)

```bash
pip install vllm>=0.9

# Serve a standard model:
vllm serve Qwen/Qwen3-30B-A3B --port 8005

# Serve an FP8-quantized model:
vllm serve Qwen/Qwen3-4B-FP8 --port 8005

# Enable reasoning parser (vLLM 0.9+):
vllm serve Qwen/Qwen3-30B-A3B --port 8005 --reasoning-parser qwen3

# Serve Qwen2.5-VL (vision model, used as VLM backend):
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8004 \
    --limit-mm-per-prompt '{"image":2,"video":0}'
```

Set in `.env`:

```
AGENT_API_BASE=http://localhost:8005/v1
AGENT_API_KEY=EMPTY
AGENT_MODEL_ID=Qwen/Qwen3-30B-A3B
```

**Docker (recommended for production):**

```bash
# Qwen3-4B FP8 — lightweight, single consumer GPU
docker run --gpus all --name qwen3-4b \
  -v ~/.cache/huggingface:/cache -p 8005:8000 --ipc=host \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-4B-FP8 --download-dir /cache \
  --enable-auto-tool-choice --tool-call-parser hermes

# Qwen3-30B-A3B — best accuracy from the paper
docker run --gpus all --name qwen3-30b-a3b \
  -v ~/.cache/huggingface:/cache -p 8005:8000 --ipc=host \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-30B-A3B --download-dir /cache \
  --max-model-len 16000 --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8_e5m2 \
  --enable-auto-tool-choice --tool-call-parser hermes

# Restart a stopped container
docker start qwen3-30b-a3b
```

See [Qwen3-30B-A3B on HuggingFace](https://huggingface.co/Qwen/Qwen3-30B-A3B) and [Qwen2.5-VL-7B on HuggingFace](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) for model cards and additional serving tips.

#### Option C: Any OpenAI-Compatible API

```
AGENT_API_BASE=https://your-endpoint.com/v1
AGENT_API_KEY=sk-your-key
AGENT_MODEL_ID=your-model-name
```

### VLM Backends (pick one)

#### Moondream Cloud API (Easiest)

```bash
pip install moondream
```

Get API key from [https://moondream.ai](https://moondream.ai) (includes $5 free monthly credits).
Set `MOONDREAM_API_KEY=your-key` in `.env`.

#### Moondream Local (moondream-station)

```bash
pip install moondream-station
moondream-station              # Starts on port 2020
```

Set `VLM_BASE_URL=http://localhost:2020` in `.env`.

#### Qwen2.5-VL via vLLM

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8004 \
    --limit-mm-per-prompt '{"image":2,"video":0}'
```

Set in `.env`:

```
VLM_BASE_URL=http://localhost:8004/v1
VLM_MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct
```

---

## Benchmark

The SCOPE benchmark contains **536 tasks** across **8 categories**, each
requiring the agent to select and execute the correct tools to answer a
question about a 3D scene.

| Category                | Code           | Count | Description |
|-------------------------|----------------|------:|-------------|
| Counting                | `counting`     |    95 | Count objects matching a description |
| Descriptor              | `descriptor`   |    89 | Describe visual attributes of objects |
| Location / Spatial      | `location`     |    53 | Reason about spatial relationships |
| OCR Identification      | `ocr`          |    54 | Read text visible in the scene |
| Single Call             | `single_call`  |    72 | Tasks requiring exactly one tool call |
| Multi-step Command      | `multi_cmd`    |    57 | Sequential tool-use chains |
| Multi-step Reasoning    | `multi_reason` |    54 | Multi-step chains with intermediate reasoning |
| Comparative Relational  | `comparative`  |    62 | Compare attributes across objects |

Tasks are defined in [`benchmark/scope_536.csv`](benchmark/scope_536.csv).
See [`benchmark/README.md`](benchmark/README.md) for the full schema.

---

## Running the Evaluation

The full evaluation pipeline has three stages: benchmark execution, LLM-as-Judge
scoring, and metric computation.

### Step 1: Run the Benchmark

```bash
blender --background --python scripts/run_benchmark.py -- \
    --config configs/presets/qwen3_30b_a3b__moondream3.yaml \
    --output results/benchmark_results.csv
```

### Step 2: Judge Results with LLM-as-Judge

```bash
# Using OpenAI API:
python -m scope.eval.judge -i results/benchmark_results.csv -o results/judged.csv

# Using a local judge (e.g., gpt-oss-120b via vLLM):
python -m scope.eval.judge \
    -i results/benchmark_results.csv \
    -o results/judged.csv \
    --base-url http://localhost:9000/v1 \
    --model gpt-oss-120b \
    --api-key EMPTY
```

### Step 3: Compute Metrics

```bash
python -m scope.eval.metrics report -i results/judged.csv
```

### One-Step (with --judge flag)

```bash
blender --background --python scripts/run_benchmark.py -- \
    --config configs/agent_config.yaml \
    --output results/run1.csv \
    --judge
```

### Convenience Pipeline

```bash
./scripts/run_eval_pipeline.sh configs/agent_config.yaml results/
```

---

## Configuration Reference

The YAML config (`configs/agent_config.yaml`) supports `${ENV_VAR}` references
that are resolved at load time from the shell environment or a `.env` file.

```yaml
agent:
  slm:
    backend: "ollama"                  # ollama | vllm | openai-compatible
    model_id: "qwen3:30b-a3b"         # Model identifier for your backend
    base_url: "http://localhost:11434/v1"
    api_key: "ollama"                  # "ollama" for Ollama; real key for vLLM/OpenAI
    temperature: 0.7
    max_tokens: 2048
    thinking: "toggle"                 # never | toggle | always | levels

  vlm:
    backend: "moondream"               # moondream-api | moondream-local | qwen2.5-vl
    api_key: "${MOONDREAM_API_KEY}"
    model_id: "vikhyatk/moondream2"    # For local loading
    # base_url: ""                     # Set for self-hosted VLM servers

evaluation:
  benchmark: "benchmark/scope_536.csv"
  output_dir: "results/"
  judge:
    model: "gpt-4o"                    # LLM-as-Judge model
    api_key: "${OPENAI_API_KEY}"

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
| `agent.vlm.base_url` | — | VLM server URL (moondream-station or Qwen2.5-VL) |
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

---

## Tool Reference

SCOPE exposes nine tools through an OpenAI function-calling schema
(`scope/tools/schema.json`). The schema is identical to the one used with
real PTZ cameras, enabling sim-to-real transfer.

| Tool | Schema Name | Type | Description |
|------|-------------|------|-------------|
| ADJUST_PTZ | `ptz_adjust` | PTZ | Adjust pan, tilt, and zoom numerically (relative moves) |
| GO_TO_PRESET | `go_to_preset` | PTZ | Move camera to a named preset position |
| GO_HOME | `home_action` | PTZ | Return camera to its home position |
| GET_PRESETS | `get_presets` | PTZ | List all available camera preset names |
| TAKE_IMAGE | `take_image` | PTZ | Capture the current camera frame to disk |
| COUNT_OBJECTS | `count_pointing` | Perception | Count objects matching a description via VLM pointing |
| QUERY_ANSWER | `query_answer` | Perception | Answer a visual question about the scene via VLM VQA |
| ZOOM_TO_OBJECT | `zoom_bounding` | Perception | Zoom to fill the frame with a described object via VLM detection |
| TRACK_OBJECT | `track_object` | PTZ | Track a described object for a specified duration |

See [`docs/tool_reference.md`](docs/tool_reference.md) for full parameter
details and return types.

---

## Results

Across 19 SLM+VLM combinations evaluated in the paper, mixture-of-experts
architectures with thinking-mode support consistently outperformed dense models
of similar size. The top 5 configurations by overall accuracy:

| Rank | SLM | VLM | Overall Accuracy |
|-----:|-----|-----|:----------------:|
| 1 | Qwen3-30B-A3B | Moondream3 | **73.8%** |
| 2 | Qwen3-30B-A3B | Qwen2.5-VL-7B | 72.4% |
| 3 | Qwen3-32B | Moondream3 | 71.6% |
| 4 | Qwen3-32B | Qwen2.5-VL-7B | 70.9% |
| 5 | Qwen3-30B-A3B | Moondream2 | 69.5% |

**Full results (Table 4 from the paper) -- all 19 SLM+VLM combinations:**

| SLM | Moondream2 | Moondream3 | Qwen2.5-VL-3B | Qwen2.5-VL-7B |
|-----|:----------:|:----------:|:--------------:|:--------------:|
| Qwen3-4B | 52.1% | 56.7% | 50.4% | 55.2% |
| Qwen3-4B-FP8 | 51.3% | 55.8% | 49.8% | 54.5% |
| Qwen3-8B | 58.6% | 63.4% | 57.1% | 62.0% |
| Qwen3-30B-A3B | 69.5% | **73.8%** | 67.2% | 72.4% |
| Qwen3-32B | 66.8% | 71.6% | 65.3% | 70.9% |

Per-category breakdowns and detailed analysis are available in the paper
([`paper/SCOPE_HRI26.pdf`](paper/SCOPE_HRI26.pdf)).

---

## Extending SCOPE

### Adding Models

SCOPE supports any OpenAI-compatible SLM backend and ships with VLM adapters
for Moondream (API, local, REST) and Qwen2.5-VL (server, local). To add a new
backend, see [`docs/adding_models.md`](docs/adding_models.md).

### Creating Scenes

New Blender scenes can be added to the benchmark by placing `.blend` files in
`benchmark/scenes/` and defining corresponding rows in the benchmark CSV. See
[`docs/creating_scenes.md`](docs/creating_scenes.md).

### Adding Tools

New tools can be added by:

1. Defining the function in `scope/tools/blender_tools.py`
2. Adding the OpenAI function schema to `scope/tools/schema.json`
3. The agent client auto-discovers tools from the schema at startup

---

## Project Structure

```
scope-release/
  benchmark/
    scope_536.csv           # 536-task benchmark
    scenes/                 # Blender scene files (.blend)
  configs/
    agent_config.yaml       # Main configuration
    presets/                # 21 pre-built SLM+VLM configs (19 paper combos + extras)
  docs/                    # Extended documentation
  paper/
    SCOPE_HRI26.pdf        # Published paper
  scope/
    agent/
      client.py            # AgentClient with tool-execution loop
      thinking.py          # Thinking-mode helpers per model
    blender/
      helper_funcs.py      # Camera screenshot, zoom, panorama
      preset_helpers.py    # Blender camera preset management
      agent_banner.py      # Blender UI panel addon
    tools/
      blender_tools.py     # All 9 tool implementations
      vlm_clients.py       # VLM adapters (Moondream, Qwen)
      schema.json          # OpenAI function-calling tool schema
    utils/
      config.py            # YAML loader with ${ENV} resolution
    eval/
      runner.py            # Benchmark batch runner (Blender state machine)
      judge.py             # LLM-as-Judge with 10 category templates
      metrics.py           # Accuracy computation and reporting
  scripts/
    run_eval_pipeline.sh   # Full pipeline: benchmark -> judge -> metrics
  .env.example             # Environment variable template
  requirements.txt
  LICENSE                  # MIT
  CITATION.cff
```

---

## Citation

If you use SCOPE in your research, please cite:

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

SCOPE is released under the [MIT License](LICENSE).

Copyright (c) 2025, Armada AI.
