#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quick_start.py -- Minimal SCOPE agent example.

Run inside Blender with a scene already open:

    blender my_scene.blend --python examples/quick_start.py

Prerequisites:
  - Ollama running with a model pulled (e.g. `ollama pull qwen3:8b`)
  - A VLM configured (Moondream API key, or local server)
  - A .blend scene with an active camera

This example creates an AgentClient, asks one question, and prints
the response along with timing breakdown.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so scope.* imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# This runs inside Blender, so bpy is available
import bpy

from scope.agent.client import AgentClient

# ---------------------------------------------------------------------------
# Create the agent with default settings.
#
# By default, AgentClient reads from these environment variables:
#   AGENT_API_BASE  -> Ollama at http://localhost:11434/v1
#   AGENT_MODEL_ID  -> auto-detected from /models endpoint
#   AGENT_API_KEY   -> "ollama"
#
# You can also pass them explicitly:
#   agent = AgentClient(
#       model_id="qwen3:8b",
#       base_url="http://localhost:11434/v1",
#   )
# ---------------------------------------------------------------------------
agent = AgentClient()

print(f"Model:    {agent.model_id}")
print(f"Base URL: {agent.base_url}")
print()

# ---------------------------------------------------------------------------
# Ask a question about the current scene.
#
# agent.ask() returns a 4-tuple:
#   answer   -- the final text response from the agent
#   messages -- the full conversation history (list of dicts)
#   timings  -- timing breakdown: {total, llm, vlm, script, camera}
#   tree     -- call tree for debugging tool invocations
# ---------------------------------------------------------------------------
question = "Describe what you see in the current camera view."

print(f"Question: {question}")
print("Waiting for agent response...")
print()

answer, messages, timings, tree = agent.ask(question)

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
print("=" * 60)
print("ANSWER:")
print(answer)
print()
print("TIMINGS:")
print(f"  Total:   {timings['total']:.2f}s")
print(f"  LLM:     {timings['llm']:.2f}s")
print(f"  VLM:     {timings['vlm']:.2f}s")
print(f"  Script:  {timings['script']:.2f}s")
print(f"  Camera:  {timings['camera']:.2f}s")
print()
print(f"Conversation turns: {len(messages)}")
print("=" * 60)
