# SCOPE Architecture

This document describes the internal architecture of the SCOPE framework:
how the agent loop operates, how tools are dispatched, and how the SLM and
VLM backends interact with the Blender simulation.

---

## High-Level Overview

```
 User prompt
      |
      v
 +--------------------+      +-------------------+
 |   AgentClient      | ---> | OpenAI-compatible |
 |   (scope/agent/    |      | SLM server        |
 |    client.py)      | <--- | (Ollama/vLLM)     |
 +----+----------+----+      +-------------------+
      |          |
      |   tool calls (JSON function-calling format)
      |          |
      v          v
 +----------+  +------------------+
 | PTZ      |  | Perception       |
 | Tools    |  | Tools            |
 | (Blender)|  | (VLM inference)  |
 +----------+  +------------------+
      |               |
      v               v
 +----------------------------+
 |    Blender 3D Scene        |
 |  (bpy camera, presets,     |
 |   rendered frames)         |
 +----------------------------+
```

---

## Components

### 1. AgentClient (`scope/agent/client.py`)

The `AgentClient` class is the central orchestrator. It manages:

- **Conversation history** -- A list of message dicts in OpenAI chat format
  (`system`, `user`, `assistant`, `tool` roles).
- **Tool schema loading** -- The tool definitions are loaded from
  `scope/tools/schema.json` at import time and passed to every SLM
  completion request.
- **Tool dispatch** -- A mapping from tool name to Python function is built
  automatically by introspecting `scope/tools/blender_tools.py`.
- **VLM binding** -- On initialization, the client creates a VLM instance
  from environment variables and binds it to the tool module via
  `blender_tools.set_vlm(vlm)`.
- **Thinking mode** -- The client consults `scope/agent/thinking.py` to
  inject the appropriate thinking-mode directive (`/think`, `/no_think`,
  or a reasoning level) into the system prompt based on the SLM model ID.

**Key methods:**

| Method | Description |
|--------|-------------|
| `ask(prompt, ...)` | Blocking call. Runs the full tool loop and returns `(text, messages, timings, call_tree)`. |
| `ask_iter(prompt, ...)` | Generator variant. Yields each assistant and tool message as it is produced, for streaming UIs or batch evaluation. |
| `set_presets(presets)` | Override the preset list injected into the system prompt. |
| `warmup_once()` | Send a throwaway request to warm up the SLM backend. |

### 2. Tool Schema (`scope/tools/schema.json`)

A JSON array of OpenAI function-calling tool definitions. Each entry has the
structure:

```json
{
  "type": "function",
  "function": {
    "name": "tool_name",
    "description": "...",
    "parameters": {
      "type": "object",
      "properties": { ... },
      "required": [ ... ]
    }
  }
}
```

This schema is passed verbatim to the SLM in every chat completion request.
The same schema is used in both simulation (Blender) and real-world (PTZ
camera API) deployments, enabling sim-to-real transfer without schema
changes.

### 3. Tool Implementations (`scope/tools/blender_tools.py`)

Each function in this module corresponds to one tool in the schema. Tools
fall into two categories:

**PTZ tools** (camera manipulation only, no VLM):
- `ptz_adjust` -- Pan, tilt, zoom by numeric amounts.
- `go_to_preset` -- Move to a named Blender camera preset.
- `home_action` -- Return to the home position.
- `get_presets` -- List available preset names.
- `take_image` -- Capture and save the current frame.
- `track_object` -- Track a described object (stub in simulation).

**Perception tools** (require a bound VLM):
- `count_pointing` -- Render frame, run VLM point detection, return count.
- `query_answer` -- Render frame, run VLM VQA, return answer text.
- `zoom_bounding` -- Render frame, run VLM detection/pointing, zoom camera
  to the detected bounding box.

Every tool returns a dict with at minimum a `"result"` key (string for the
SLM) and a `"timings"` key (dict with `vlm`, `script`, and optionally
`camera` durations in seconds).

Some tools (`count_pointing`, `query_answer`) are **generator functions**.
When `view_type="full"` is specified, they yield intermediate panorama
capture steps. The `AgentClient` consumes these yields transparently and
extracts the final return value via `StopIteration`.

### 4. VLM Clients (`scope/tools/vlm_clients.py`)

The VLM layer defines an abstract `VLMClient` base class with four
capabilities:

| Capability | Method | Description |
|------------|--------|-------------|
| `caption` | `caption(image)` | Generate a text caption for an image |
| `vqa` | `query(image, question)` | Answer a question about an image |
| `detect` | `detect(image, instruction)` | Return bounding boxes for described objects |
| `point` | `point(image, instruction)` | Return point coordinates for described objects |

Concrete implementations:

| Class | Backend | Notes |
|-------|---------|-------|
| `MoondreamREST` | Self-hosted Moondream HTTP server | Sends images as multipart form uploads |
| `MoondreamServer` | Moondream hosted API (SDK) | Uses `moondream` Python package with API key |
| `MoondreamLocal` | HuggingFace Transformers | Loads model weights locally on GPU |
| `QwenVLServer` | OpenAI-compatible VL server | Sends base64 images via chat completions API |
| `QwenVLLocal` | HuggingFace Transformers | Loads Qwen2.5-VL locally |

The factory function `create_vlm_from_env()` reads `VLM_MODEL`,
`VLM_MODEL_URL`, `VLM_MODEL_ID`, and `VLM_API_KEY` from the environment
to instantiate the correct client.

### 5. Blender Helpers (`scope/blender/`)

Two modules provide the low-level Blender integration:

**`helper_funcs.py`**:
- `screenshot_camera_view(path)` -- Render the camera view and crop to the
  camera frustum.
- `blender_zoom(cam, x1, y1, x2, y2)` -- Perspective-aware area zoom with
  cosine correction. Adjusts focal length and rotation to center and fill
  the specified normalized bounding box.
- `start_panorama_capture(path, overlap_ratio)` / `capture_panorama_step()`
  -- Panorama sweep stubs (extend for full 360-degree stitching).

**`preset_helpers.py`**:
- `list_presets()` -- Scan Blender's system and user preset directories for
  camera presets.
- `apply_preset(name)` -- Execute a preset script to set camera transform
  and lens.
- `create_preset(name)` -- Write the current camera state as a reusable
  preset `.py` file.

### 6. Configuration (`scope/utils/config.py`)

The configuration loader reads YAML files and recursively resolves
`${ENV_VAR}` patterns in string values. The resolution order for finding
the config file is:

1. Explicit path argument
2. `SCOPE_CONFIG` environment variable
3. `configs/agent_config.yaml` relative to the project root

### 7. Thinking Modes (`scope/agent/thinking.py`)

Different SLM architectures handle chain-of-thought reasoning differently.
The `ThinkingMode` class defines four modes:

| Mode | Behavior |
|------|----------|
| `NEVER` | No thinking directive is injected |
| `TOGGLE` | Inject `/think` or `/no_think` based on the `enable_thinking` flag |
| `ALWAYS` | Always inject `/think` |
| `LEVELS` | Inject a reasoning level (`off`, `low`, `medium`, `high`) |

The `MODEL_CATALOG` dict maps known model IDs to their thinking mode. New
models can be added to this catalog; unrecognized models default to `NEVER`.

---

## Agent Loop in Detail

The following describes the execution flow of `AgentClient.ask(prompt)`:

```
1. Seed conversation (if first call):
   a. Determine thinking mode from model ID
   b. Inject thinking directive as system message
   c. Inject system prompt (with preset list) as system message

2. Append user message

3. LOOP:
   a. Call SLM via OpenAI chat completions API
      (model, messages, tools, temperature, tool_choice="auto")
   b. Append assistant message to conversation
   c. If no tool_calls in response -> strip reasoning tags, return final text
   d. For each tool_call:
      i.   Parse function name and JSON arguments
      ii.  Look up function in TOOL_FUNCTIONS mapping
      iii. Call function(**args)
      iv.  If function is a generator, consume yields until StopIteration
      v.   Extract timings from result dict
      vi.  Append tool result message to conversation
   e. Continue loop (back to step 3a)
```

The loop terminates when the SLM produces a response with no tool calls.
The return value includes:

- `text` -- The final assistant response with reasoning tags stripped
- `messages` -- The full conversation history
- `timings` -- Aggregated timing breakdown (`total`, `llm`, `vlm`, `script`, `camera`)
- `call_tree` -- A structured record of tool calls and their durations

---

## Timing Breakdown

Every tool returns a `timings` dict. The `AgentClient` aggregates these
across all tool calls in a session:

| Key | What it measures |
|-----|-----------------|
| `llm` | Time spent in SLM inference (chat completion API calls) |
| `vlm` | Time spent in VLM inference (caption, VQA, detect, point) |
| `script` | Time spent in Blender scripting (camera moves, renders) |
| `camera` | Time spent in physical camera operations (zero in simulation) |
| `total` | Sum of all components |

This breakdown enables profiling bottlenecks across the SLM planning,
VLM perception, and simulation execution stages.
