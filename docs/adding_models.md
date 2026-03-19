# Adding Models to SCOPE

SCOPE is designed to work with any combination of SLM (planner) and VLM
(perception) backends. This guide explains how to add new models of each type.

---

## Adding a New SLM Backend

The `AgentClient` communicates with the SLM through an **OpenAI-compatible
chat completions API**. Any server that exposes a `/v1/chat/completions`
endpoint with function-calling support will work.

### Option A: Use an Existing OpenAI-Compatible Server

If your model is served by Ollama, vLLM, or another OpenAI-compatible server,
no code changes are needed. Configure it in `configs/agent_config.yaml`:

```yaml
agent:
  slm:
    backend: "vllm"                           # or "ollama", "openai-compatible"
    model_id: "your-org/your-model"
    base_url: "http://localhost:8000/v1"
    api_key: "your-api-key"
    temperature: 0.7
    thinking: "never"                         # adjust based on model capability
```

Or set environment variables:

```bash
export AGENT_API_BASE="http://localhost:8000/v1"
export AGENT_MODEL_ID="your-org/your-model"
export AGENT_API_KEY="your-api-key"
```

### Option B: Register a Thinking Mode

If your model supports thinking/reasoning toggles, add it to the
`MODEL_CATALOG` in `scope/agent/thinking.py`:

```python
MODEL_CATALOG = {
    # ... existing entries ...
    "your-org/your-model": {"thinking": ThinkingMode.TOGGLE},
}
```

Available thinking modes:

| Mode | Behavior | Example Models |
|------|----------|---------------|
| `NEVER` | No thinking directive | Default for unrecognized models |
| `TOGGLE` | `/think` or `/no_think` injected | Qwen3-8B, Qwen3-30B-A3B, Qwen3-32B |
| `ALWAYS` | Always injects `/think` | Models that only work with CoT |
| `LEVELS` | Injects `Reasoning: low\|medium\|high` | GPT-oss-20B, GPT-oss-120B |

The model ID matching is case-insensitive and supports substring matching,
so `"Qwen/Qwen3-8B"` will match a served model ID like
`"Qwen/Qwen3-8B-Instruct"`.

### Verifying SLM Integration

After configuration, verify that the agent can connect:

```python
from scope.agent import AgentClient

agent = AgentClient()
print(f"Connected to: {agent.model_id}")
print(f"Base URL: {agent.base_url}")

# Quick test (does not require Blender or VLM)
agent.warmup_once()
```

---

## Adding a New VLM Backend

VLM backends implement the `VLMClient` interface defined in
`scope/tools/vlm_clients.py`. A VLM must support one or more of four
capabilities: `caption`, `vqa`, `detect`, and `point`.

### Step 1: Implement the VLMClient Interface

Create a new class that extends `VLMClient`:

```python
from scope.tools.vlm_clients import VLMClient, VLMCaps, _to_pil

class MyVLMBackend(VLMClient):
    def __init__(self, base_url: str, api_key: str = ""):
        self.name = "MyVLM"
        self.label = "MyVLM(server)"
        self.caps = VLMCaps(
            caption=True,
            vqa=True,
            detect=True,   # set False if not supported
            point=True,    # set False if not supported
        )
        self.base_url = base_url
        self.api_key = api_key

    def caption(self, image):
        pil = _to_pil(image)
        # ... call your VLM API ...
        return {"caption": "A description of the image."}

    def query(self, image, question: str):
        pil = _to_pil(image)
        # ... call your VLM API with the question ...
        return {"answer": "The answer to the question."}

    def detect(self, image, instruction: str):
        pil = _to_pil(image)
        # ... run detection ...
        # Return normalized bounding boxes (0..1)
        return {"objects": [
            {"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.8}
        ]}

    def point(self, image, instruction: str):
        pil = _to_pil(image)
        # ... run pointing ...
        # Return normalized point coordinates (0..1)
        return {"points": [
            {"x": 0.3, "y": 0.5}
        ]}
```

### Step 2: Register in the Factory

Add your backend to the `create_vlm()` factory function in
`scope/tools/vlm_clients.py`:

```python
def create_vlm(kind, base_url=None, model_id=None, mode_hint=None, api_key=None):
    k = (kind or "").strip().lower()
    url = (base_url or "").strip()

    # ... existing backends ...

    if "myvlm" in k:
        return MyVLMBackend(base_url=url, api_key=api_key or "")

    raise ValueError(f"Unsupported VLM kind: {kind!r}")
```

### Step 3: Configure

Set environment variables or update the YAML config:

```bash
export VLM_MODEL="myvlm"
export VLM_MODEL_URL="http://localhost:9000"
export VLM_API_KEY="your-key"
```

Or in `configs/agent_config.yaml`:

```yaml
agent:
  vlm:
    backend: "myvlm"
    base_url: "http://localhost:9000"
    api_key: "${MY_VLM_API_KEY}"
```

### VLM Capability Requirements

Not all tools require all VLM capabilities. The minimum requirements are:

| Tool | Required Capabilities |
|------|-----------------------|
| `query_answer` | `vqa` |
| `count_pointing` | `point` |
| `zoom_bounding` | `detect` (with `point` as fallback) |

If your VLM does not support `detect`, `zoom_bounding` will fall back to
using `point` to estimate a bounding box from point coordinates.

### Return Value Formats

**`caption(image)`** must return:
```python
{"caption": "string describing the image"}
```

**`query(image, question)`** must return:
```python
{"answer": "string answering the question"}
```

**`detect(image, instruction)`** must return:
```python
{"objects": [
    {"x_min": float, "y_min": float, "x_max": float, "y_max": float},
    ...
]}
```
Coordinates should be normalized to `[0, 1]` relative to image dimensions.

**`point(image, instruction)`** must return:
```python
{"points": [
    {"x": float, "y": float},
    ...
]}
```
Coordinates should be normalized to `[0, 1]` relative to image dimensions.

---

## Existing VLM Backends

For reference, these backends are already implemented:

| Class | `VLM_MODEL` value | Description |
|-------|-------------------|-------------|
| `MoondreamREST` | `moondream` | Self-hosted Moondream HTTP server |
| `MoondreamServer` | `moondream` | Moondream hosted API (requires `moondream` pip package + API key) |
| `MoondreamLocal` | `moondream` + `VLM_MODEL_URL=local` | Local HuggingFace Transformers loading |
| `QwenVLServer` | `qwen` | OpenAI-compatible Qwen2.5-VL server |
| `QwenVLLocal` | `qwen` + `VLM_MODEL_URL=local` | Local HuggingFace Transformers loading |

The factory auto-selects the appropriate class based on the `VLM_MODEL` value
and whether a `VLM_MODEL_URL` is provided.

---

## Thinking Modes

Some models — particularly Qwen3 — were released with a *hybrid* thinking mode, where the model can reason step-by-step (producing a `<think>...</think>` block) or respond directly, controlled by special system tokens injected at the start of the conversation.

SCOPE handles this automatically via `ThinkingMode`:

| Mode | Behaviour | When to use |
|------|-----------|-------------|
| `NEVER` | No tokens injected (default for unregistered models) | Any model without hybrid thinking |
| `TOGGLE` | Injects `/no_think` (or `/think`) system message | Qwen3 models — disables reasoning by default |
| `ALWAYS` | Always injects `/think` | Force reasoning on for every call |
| `LEVELS` | Sets `reasoning_effort` parameter | Models like gpt-oss-120b |

**Why `/no_think` by default?**
During development, Qwen3's thinking mode produced verbose reasoning traces that caused tool-call parsing failures. Disabling it (via `/no_think`) improved reliability significantly — especially for multi-step tool-call sequences. The SCOPE benchmark results were obtained with thinking disabled.

**For new users:** if you are adding a Qwen3 model, register it with `ThinkingMode.TOGGLE` and thinking will be automatically disabled unless you explicitly set `enable_thinking=True` in `agent.ask()`.

**For non-Qwen models:** leave the model unregistered. `ThinkingMode.NEVER` is the default and injects nothing — it works safely with any standard chat model.

To register a new model:

```python
# scope/agent/thinking.py
MODEL_CATALOG = {
    ...
    "your-org/your-model": {"thinking": ThinkingMode.TOGGLE},
}
```

---

## Creating a Config Preset

Once your SLM+VLM combination is working, save it as a reusable preset:

```bash
cp configs/presets/custom_template.yaml configs/presets/my_config.yaml
# Edit my_config.yaml with your model details
```

Presets can be activated by copying them to the main config:

```bash
cp configs/presets/my_config.yaml configs/agent_config.yaml
```
