#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
custom_model.py -- Configure SCOPE with custom SLM and VLM backends.

Run inside Blender:

    blender my_scene.blend --python examples/custom_model.py

This example shows how to:
  1. Use a vLLM-served SLM instead of Ollama
  2. Use Qwen2.5-VL as the VLM instead of Moondream
  3. Wire them together into an AgentClient
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import bpy

from scope.agent.client import AgentClient
from scope.tools import blender_tools
from scope.tools.vlm_clients import (
    VLMClient,
    QwenVLServer,
    QwenVLLocal,
    MoondreamREST,
    MoondreamServer,
    create_vlm,
)

# ===================================================================
# EXAMPLE 1: Custom SLM via vLLM server
# ===================================================================
#
# If you are running a model with vLLM instead of Ollama, point the
# AgentClient at your vLLM server.  vLLM exposes an OpenAI-compatible
# API on /v1 by default.
#
#   vllm serve Qwen/Qwen3-30B-A3B --enable-auto-tool-choice \
#        --tool-call-parser hermes --port 8000
#
# Then create the agent:

agent_vllm = AgentClient(
    model_id="Qwen/Qwen3-30B-A3B",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",                      # vLLM does not require a real key
    temperature=0.7,
)
print(f"[vLLM agent] model={agent_vllm.model_id}")


# ===================================================================
# EXAMPLE 2: Custom VLM -- Qwen2.5-VL via a vLLM/OpenAI server
# ===================================================================
#
# SCOPE's visual tools (zoom_bounding, count_pointing, query_answer)
# need a VLM.  By default they use Moondream, but you can swap in
# any VLM that supports captioning, VQA, detection, or pointing.
#
# Option A: Qwen2.5-VL served by vLLM
#   vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001

vlm_qwen_server = QwenVLServer(
    base_url="http://localhost:8001/v1",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    api_key="EMPTY",
)
print(f"[VLM] Qwen server: {vlm_qwen_server.name}")
print(f"[VLM] Capabilities: caption={vlm_qwen_server.caps.caption}, "
      f"vqa={vlm_qwen_server.caps.vqa}, detect={vlm_qwen_server.caps.detect}, "
      f"point={vlm_qwen_server.caps.point}")


# Option B: Qwen2.5-VL loaded locally via Transformers (needs GPU)
# Uncomment to use:
#
# vlm_qwen_local = QwenVLLocal(model_id="Qwen/Qwen2.5-VL-7B-Instruct")


# Option C: Moondream self-hosted REST server
# If you run the Moondream server locally:
#   python -m moondream.server --port 3475
#
# vlm_moondream_rest = MoondreamREST(base_url="http://localhost:3475")


# Option D: Moondream cloud API (requires API key)
# vlm_moondream_cloud = MoondreamServer(api_key="your-moondream-api-key")


# Option E: Use the factory function with a kind string
# vlm_auto = create_vlm(kind="qwen2.5-vl", base_url="http://localhost:8001")


# ===================================================================
# EXAMPLE 3: Bind the custom VLM and use the agent
# ===================================================================
#
# After creating a VLM client, bind it to the Blender tools so that
# zoom_bounding, count_pointing, and query_answer use your VLM.

blender_tools.set_vlm(vlm_qwen_server)
print("[VLM] Bound Qwen VL server to simulation tools.")

# Now ask a question -- the agent will use the vLLM SLM for reasoning
# and the Qwen VLM for visual perception.

answer, messages, timings, tree = agent_vllm.ask(
    "How many people can you see?",
    reset_history=True,
)

print(f"\nAnswer: {answer}")
print(f"Timings: {timings}")


# ===================================================================
# EXAMPLE 4: Using environment variables (alternative approach)
# ===================================================================
#
# Instead of passing arguments directly, you can set environment
# variables before launching Blender:
#
#   export AGENT_API_BASE="http://localhost:8000/v1"
#   export AGENT_MODEL_ID="Qwen/Qwen3-30B-A3B"
#   export AGENT_API_KEY="EMPTY"
#   export VLM_MODEL="qwen2.5-vl"
#   export VLM_MODEL_URL="http://localhost:8001/v1"
#   export VLM_MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
#   blender scene.blend --python examples/quick_start.py
#
# The AgentClient() constructor and create_vlm_from_env() will read
# these automatically.
