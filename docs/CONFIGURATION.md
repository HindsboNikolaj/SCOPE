# Configuration

SCOPE needs an SLM (planner) and a VLM (perception). Pick **Local /
Self-hosted** for reproducibility and air-gapped runs; pick **API / Cloud**
when you don't have the GPU budget. The judge always uses an OpenAI-compatible
endpoint (defaults to `gpt-4o`, but any local OpenAI-compatible server works
too).

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
