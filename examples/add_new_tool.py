#!/usr/bin/env python3
"""
Register a process-local SCOPE tool in Blender.

Run with a Blender scene open:

    blender my_scene.blend --python examples/add_new_tool.py

The default run needs no planner endpoint: it registers the tool and calls it
directly against two objects in the scene. To also ask a configured planner to
use the tool, run:

    SCOPE_RUN_AGENT_DEMO=1 blender my_scene.blend --python examples/add_new_tool.py

This demonstrates temporary registration only. To ship a tool with SCOPE, add
its schema to src/scope/tools/schema.json and its implementation to
src/scope/tools/blender_tools.py.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import bpy

from scope.agent import client as agent_client
from scope.agent.client import AgentClient


MEASURE_DISTANCE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "measure_distance",
        "description": (
            "Measure the straight-line distance in Blender units between two "
            "named objects in the current scene. Use it when asked for the "
            "distance or separation between two specific objects."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_a": {
                    "type": "string",
                    "description": "Name of the first Blender object.",
                },
                "object_b": {
                    "type": "string",
                    "description": "Name of the second Blender object.",
                },
            },
            "required": ["object_a", "object_b"],
        },
    },
}


def measure_distance(object_a: str, object_b: str) -> dict[str, Any]:
    """Return the world-space Euclidean distance between two scene objects."""
    started = time.time()
    first = bpy.data.objects.get(object_a)
    second = bpy.data.objects.get(object_b)

    if first is None or second is None:
        missing = object_a if first is None else object_b
        return {
            "result": f"Object '{missing}' was not found in the scene.",
            "timings": {"script": round(time.time() - started, 3)},
        }

    delta = first.matrix_world.translation - second.matrix_world.translation
    distance = math.sqrt(delta.x**2 + delta.y**2 + delta.z**2)
    elapsed = round(time.time() - started, 3)
    return {
        "result": (
            f"Distance between '{object_a}' and '{object_b}': "
            f"{distance:.3f} Blender units."
        ),
        "distance": round(distance, 3),
        "timings": {"script": elapsed},
    }


def register_tool(schema: dict[str, Any], function: Callable[..., dict[str, Any]]) -> None:
    """Register a schema and callable for this Python process.

    AgentClient initializes the built-in bindings lazily. Initialize those
    bindings before adding this custom schema; otherwise its automatic lookup
    would try to find this example function in blender_tools.py.
    """
    tool = schema.get("function", {})
    name = tool.get("name")
    if schema.get("type") != "function" or not isinstance(name, str) or not name:
        raise ValueError("Expected an OpenAI-compatible function schema with a name.")

    existing_names = {
        item.get("function", {}).get("name")
        for item in agent_client.TOOL_DEFS
    }
    if name in existing_names:
        raise ValueError(f"Tool '{name}' is already registered.")

    if agent_client.TOOL_FUNCTIONS is None:
        agent_client.TOOL_FUNCTIONS = agent_client._load_tool_functions()

    agent_client.TOOL_DEFS.append(schema)
    agent_client.TOOL_FUNCTIONS[name] = function


def object_names(limit: int = 10) -> list[str]:
    """Return a short, stable list of object names from the active file."""
    return [obj.name for obj in list(bpy.data.objects)[:limit]]


def run_direct_demo(names: list[str]) -> None:
    """Exercise the function without requiring an LLM endpoint."""
    if len(names) < 2:
        print("[demo] Scene has fewer than two objects; registration succeeded.")
        return

    payload = measure_distance(names[0], names[1])
    assert "result" in payload and "timings" in payload
    print("[demo] Direct tool result:")
    print(json.dumps(payload, indent=2))


def run_agent_demo(names: list[str]) -> None:
    """Optionally ask a configured planner to use the newly registered tool."""
    if len(names) < 2:
        print("[demo] Skipping agent demo: choose a scene with at least two objects.")
        return

    agent = AgentClient()
    question = (
        f"Use the measure_distance tool to find the distance between "
        f"'{names[0]}' and '{names[1]}'."
    )
    answer, messages, timings, _ = agent.ask(question, reset_history=True)
    called_tools = [
        call["function"]["name"]
        for message in messages
        for call in (message.get("tool_calls") or [])
    ]

    print(f"[agent] Question: {question}")
    print(f"[agent] Answer: {answer}")
    print(f"[agent] Tool calls: {called_tools}")
    print(f"[agent] Timings: {timings}")
    if "measure_distance" not in called_tools:
        print("[agent] The planner did not select the requested tool; inspect its model, "
              "prompt, and tool-call support before relying on this behavior.")


register_tool(MEASURE_DISTANCE_SCHEMA, measure_distance)
names = object_names()

print("[tool] Registered: measure_distance")
print(f"[tool] Available tools: {[item['function']['name'] for item in agent_client.TOOL_DEFS]}")
print(f"[scene] Objects (first {len(names)}): {names}")
run_direct_demo(names)

if os.getenv("SCOPE_RUN_AGENT_DEMO") == "1":
    run_agent_demo(names)
else:
    print(
        "[demo] Skipped live-agent call. Set SCOPE_RUN_AGENT_DEMO=1 after "
        "configuring AGENT_API_BASE, AGENT_MODEL_ID, and AGENT_API_KEY."
    )
