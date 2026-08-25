# SCOPE

**Simulation and Camera Operations for Perception and Evaluation**

SCOPE is a modular multimodal agent for natural-language PTZ camera work: a language model plans, vision tools inspect frames, and camera tools act on the scene. It gives builders a Blender-based path to prototype, extend, and evaluate that loop before attaching it to a camera integration of their own.

[![Paper](https://img.shields.io/badge/Paper-HRI%20'26-blue)](https://doi.org/10.1145/3757279.3785641)
[![arXiv](https://img.shields.io/badge/arXiv-2606.02951-b31b1b?logo=arxiv)](https://arxiv.org/abs/2606.02951)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/datasets/HindsboNikolaj/scope-benchmark)
[![Space](https://img.shields.io/badge/Space-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/spaces/HindsboNikolaj/scope)
[![Collection](https://img.shields.io/badge/Collection-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/collections/HindsboNikolaj/scope-hri-26-6a1626e0b8e9b9205c09fffc)

![SCOPE architecture: a language-model planner uses PTZ and perception tools to control a camera or simulation and query a vision model](docs/images/scope-architecture.svg)

> The planner receives tool results as text rather than raw image tokens. The same tool boundary keeps camera control, perception, and the planner independently replaceable.

## Choose your path

| If you want to… | Start here | What you get |
| --- | --- | --- |
| Run a working camera agent in simulation | [Quick start](#quick-start) | A configured Blender smoke run through runner, judge, and metrics |
| Add a camera or perception capability | [Add a tool](#add-a-tool) | A runnable Blender tutorial, then the persistent extension points |
| Compare planner/VLM pairs | [Run the benchmark](#run-the-benchmark) | Repeatable traces and an LLM-as-judge report |
| Change model backends | [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) | Ollama, vLLM, hosted planner, and supported VLM setup |
| Hand setup to Claude Code or Codex | [`AGENT_INSTRUCTIONS.md`](AGENT_INSTRUCTIONS.md) | An install brief that stops at the first failure |

## Quick start

Run these from the repository root. You need Python 3.10+, Blender 4.0+, an OpenAI-compatible planner endpoint, a supported VLM, and a judge endpoint for the full evaluation pipeline.

```bash
# Install SCOPE and create local configuration.
python -m pip install -e .
test -f .env || cp .env.example .env

# Edit .env, then load it into this shell. At minimum configure:
# AGENT_API_BASE, AGENT_MODEL_ID, AGENT_API_KEY,
# a VLM option (for example MOONDREAM_API_KEY or VLM_MODEL_URL),
# and OPENAI_API_KEY (or JUDGE_API_BASE/JUDGE_API_KEY) for evaluation.
set -a; source .env; set +a

# Download scenes and install their Blender camera presets.
# The model pull is only for the Ollama route; skip it for vLLM or hosted APIs.
bash scripts/02_pull_models.sh qwen3:30b-a3b
bash scripts/03_download_scenes.sh
"${BLENDER_BIN:-blender}" --background --python scripts/04_install_presets.py

# Exercise the complete runner → judge → metrics path on five tasks.
BENCH_LIMIT=5 SCOPE_RESUME=0 bash scripts/run_eval_pipeline.sh
```

If Blender is not on `PATH`, set `BLENDER_BIN` in `.env`; on macOS this is commonly `/Applications/Blender.app/Contents/MacOS/Blender`. The evaluator intentionally launches Blender **with a UI**, not `--background`, because viewport capture needs 3D viewport context. `--background` is safe for the preset installer above.

For an agent-assisted install, paste [`AGENT_INSTRUCTIONS.md`](AGENT_INSTRUCTIONS.md) into a fresh Claude Code or Codex session.

## Why SCOPE

| Builder concern | SCOPE’s boundary | What you can verify |
| --- | --- | --- |
| A planner chooses the wrong action | A small language model receives a fixed tool list and text results | Tool name, JSON arguments, trace, and final answer |
| A visual model is the bottleneck | Perception lives behind tools such as `count_pointing` and `query_answer` | Swap a supported VLM without rewriting planner logic |
| A camera routine needs more than one primitive | Tools may package short workflows, such as detect-then-zoom | The planner calls one stable capability rather than reimplementing the sequence |
| A change is hard to reproduce on hardware | Blender scenes, presets, questions, and evaluation metadata are versioned | Run the same scene/question path and inspect the resulting trace |

The repository ships the agent loop, nine Blender-facing tool definitions, VLM adapters, a 541-task benchmark CSV, and an LLM-as-judge evaluation harness. It is a simulation and evaluation codebase; it does **not** include a public production AXIS-camera backend.

## Architecture and action space

[`AgentClient`](src/scope/agent/client.py) sends the nine definitions in [`src/scope/tools/schema.json`](src/scope/tools/schema.json) to a planner through an OpenAI-compatible chat/tool-calling API. When the planner calls a tool, SCOPE dispatches to a Python function, appends its text result to the conversation, and repeats until the planner returns an answer. In the shipped implementation, those functions operate on Blender’s active camera and call a configured VLM where perception is needed.

| Tool family | Shipped examples | Implementation boundary |
| --- | --- | --- |
| Camera control | `ptz_adjust`, `go_to_preset`, `home_action`, `take_image` | [`blender_tools.py`](src/scope/tools/blender_tools.py) manipulates Blender camera state and captures frames |
| Perception | `count_pointing`, `query_answer`, `zoom_bounding` | The same module captures a frame and delegates to a configured VLM client |
| Scene context | `get_presets`, `track_object` | Presets are read from Blender; `track_object` is a simulation stub, not a deployed tracker |

The diagram above is exported from the project research page and stored as a self-contained asset; it makes no network request when GitHub renders this README.

## Add a tool

Start with the runnable [custom-tool tutorial](examples/add_new_tool.py):

```bash
blender my_scene.blend --python examples/add_new_tool.py
```

It registers a `measure_distance` tool for that Blender process, verifies the schema/function registration, and calls the function directly against two scene objects. Set `SCOPE_RUN_AGENT_DEMO=1` only after configuring a planner endpoint if you also want to ask a live agent to use the tool.

For a persistent SCOPE tool, make both parts of the contract explicit:

1. Add its function schema to [`src/scope/tools/schema.json`](src/scope/tools/schema.json).
2. Implement a Python function with the same keyword arguments in [`src/scope/tools/blender_tools.py`](src/scope/tools/blender_tools.py); return at least `result`, and include timings when they matter to evaluation.
3. Add benchmark rows or an integration test before treating the new capability as measured.

### Extend into other agent runtimes or cameras

SCOPE’s shipped schema is **OpenAI-compatible function-calling JSON**, and its shipped execution layer is Python in Blender. MCP is a separate client/server protocol with different transport and result conventions; an OpenAI-compatible tool schema is not an MCP server definition.

To use SCOPE capabilities from MCP or another runtime, write a thin adapter around the boundary you need: map that runtime’s tool declaration and request payload to the SCOPE function arguments, call the implementation, and translate `result`/errors/timings back into the runtime’s result format. Keep the mapping and validation explicit instead of assuming schemas or wire formats are interchangeable.

To attach physical hardware, implement a camera adapter for the device/vendor API you operate, preserve the documented tool argument and result contracts where that is useful, and validate it separately from the Blender benchmark. The paper and demos include real AXIS-camera work, but the public repository does not contain an AXIS backend to reuse or imply deployment readiness.

## Results from the paper

**Scope of these numbers:** the HRI paper reports results on a supported **536-task published subset**, judged by GPT-4o. The current [`benchmark/scope_536.csv`](benchmark/scope_536.csv) filename is historical and the repository ships **541 tasks**; five shipped rows fall outside the published score subset. The public Git history begins with the 541-row CSV, so it does not establish how those rows relate to the paper subset. Report a fresh 541-task run separately rather than comparing it directly with the paper values below.

| Rank | Planner | Planner type | Vision model | Paper accuracy (536 tasks) |
| :--: | --- | --- | --- | :--: |
| 1 | Qwen3-30B-A3B | MoE | Moondream3 | **73.8%** |
| 2 | Qwen3-30B-A3B | MoE | Qwen2.5-VL-7B | 72.4% |
| 3 | Qwen3-32B | Dense | Moondream3 | 71.6% |
| 4 | Qwen3-32B | Dense | Qwen2.5-VL-7B | 70.9% |
| 5 | Qwen3-30B-A3B | MoE | Moondream2 | 69.5% |

`MoE` and `Dense` identify the documented planner architectures; the accuracy values are the paper matrix reproduced in [`docs/RESULTS_FULL.md`](docs/RESULTS_FULL.md). The full page links the 20 published planner/VLM pairings and the paper contains the per-category analysis.

## Run the benchmark

```bash
./scripts/run_eval_pipeline.sh configs/agent_config.yaml results/
```

| Variable | Default | Use it when… |
| --- | --- | --- |
| `BENCH_LIMIT` | `0` | You want a smoke test; set `5` before committing to a full run |
| `SCOPE_RESUME` | `1` | You want to continue a partial run; set `0` for a clean run |
| `REPEATS` | `1` | You need repeated executions per question |
| `QUESTIONS_CSV` | `benchmark/scope_536.csv` | You want a different benchmark input; the default file currently has 541 rows |
| `OUT_CSV` | `results/run_<ts>/raw_results.csv` | You want to resume into or retain a particular raw-results file |

The pipeline runs the Blender runner, judges the trace/final response, and prints aggregate metrics. See [`benchmark/README.md`](benchmark/README.md) for the CSV schema and task categories, [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) for backend setup, and [`docs/architecture.md`](docs/architecture.md) for the full loop and extension points.

## Repository guide

| Location | Contents |
| --- | --- |
| [`src/scope/`](src/scope/) | Agent loop, Blender helpers, tools, VLM clients, runner, and judge |
| [`benchmark/`](benchmark/) | The 541-row CSV and camera presets; scene downloads are kept out of Git for size |
| [`configs/`](configs/) | Planner/VLM configurations and paper pairing presets |
| [`prompts/`](prompts/) | The live planner system prompt and judge prompts |
| [`examples/`](examples/) | Quick start, custom model, and custom tool tutorials |
| [`docs/`](docs/) | Configuration, architecture, tool reference, results, and scene-authoring detail |

## Citation

```bibtex
@inproceedings{Armada2026SCOPE,
  title         = {SCOPE: A Real-Time Natural Language Camera Agent at the Edge:
                   A Sim-to-Real Benchmark and Analysis of Open-Source Vision
                   and Language Agents for PTZ Camera Tasks},
  author        = {Hindsbo, Nikolaj and Ehsani, Sina and Mishra, Pragyana},
  booktitle     = {Proceedings of the ACM/IEEE International Conference on
                   Human-Robot Interaction (HRI '26)},
  year          = {2026},
  publisher     = {ACM},
  doi           = {10.1145/3757279.3785641},
  eprint        = {2606.02951},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
}
```

## License

MIT. See [LICENSE](LICENSE).
