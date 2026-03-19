#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_new_tool.py -- Register a custom tool with the SCOPE agent.

Run inside Blender:

    blender my_scene.blend --python examples/add_new_tool.py

This example demonstrates how to:
  1. Define a new tool using the OpenAI function-calling JSON schema
  2. Implement the tool function
  3. Register it with the SCOPE agent so the SLM can invoke it

The example adds a "measure_distance" tool that measures the Blender
world-space distance between two named objects in the scene.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import bpy

from scope.agent.client import AgentClient, TOOL_DEFS, TOOL_FUNCTIONS

# ===================================================================
# STEP 1: Define the tool schema
# ===================================================================
#
# Tools follow the OpenAI function-calling schema format.  Each tool
# is a dict with "type": "function" and a "function" key containing
# the name, description, and parameter spec.
#
# The description is critical -- the SLM uses it to decide when and
# how to call the tool.  Be specific about what it does and when to
# use it.

MEASURE_DISTANCE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "measure_distance",
        "description": (
            "Measure the straight-line distance (in Blender units) between "
            "two named objects in the scene. Use this when the user asks "
            "about the distance or separation between two specific objects. "
            "Returns the distance as a float."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_a": {
                    "type": "string",
                    "description": "Name of the first Blender object (e.g. 'Cube', 'Camera')"
                },
                "object_b": {
                    "type": "string",
                    "description": "Name of the second Blender object"
                },
            },
            "required": ["object_a", "object_b"],
        },
    },
}


# ===================================================================
# STEP 2: Implement the tool function
# ===================================================================
#
# The function signature must match the "required" and "properties"
# keys from the schema above.  The function should return a dict with:
#   - "result": a human-readable string the SLM will see
#   - "timings": optional timing breakdown (vlm, script, camera)
#   - any additional data you want to log

def measure_distance(object_a: str, object_b: str) -> dict:
    """
    Measure the Euclidean distance between two Blender objects.

    Parameters:
        object_a: Name of the first object in bpy.data.objects
        object_b: Name of the second object in bpy.data.objects

    Returns:
        Dict with "result" (str) and "distance" (float).
    """
    t0 = time.time()

    # Look up objects in the current Blender scene
    obj_a = bpy.data.objects.get(object_a)
    obj_b = bpy.data.objects.get(object_b)

    if obj_a is None:
        return {
            "result": f"Object '{object_a}' not found in scene.",
            "timings": {"vlm": 0.0, "script": round(time.time() - t0, 3)},
        }
    if obj_b is None:
        return {
            "result": f"Object '{object_b}' not found in scene.",
            "timings": {"vlm": 0.0, "script": round(time.time() - t0, 3)},
        }

    # Compute Euclidean distance between world-space locations
    loc_a = obj_a.matrix_world.translation
    loc_b = obj_b.matrix_world.translation
    dist = math.sqrt(
        (loc_a.x - loc_b.x) ** 2 +
        (loc_a.y - loc_b.y) ** 2 +
        (loc_a.z - loc_b.z) ** 2
    )

    elapsed = round(time.time() - t0, 3)
    return {
        "result": f"Distance between '{object_a}' and '{object_b}': {dist:.3f} Blender units.",
        "distance": round(dist, 3),
        "timings": {"vlm": 0.0, "script": elapsed},
    }


# ===================================================================
# STEP 3: Register the tool with the SCOPE agent
# ===================================================================
#
# The agent's tool registry has two parts:
#   TOOL_DEFS      -- list of JSON schema dicts (sent to the SLM)
#   TOOL_FUNCTIONS -- dict mapping function names to callables
#
# We append our schema and register our function.

def register_tool(schema: dict, func) -> None:
    """Register a new tool with the SCOPE agent runtime."""
    name = schema["function"]["name"]

    # Add to the schema list (so the SLM knows about it)
    TOOL_DEFS.append(schema)

    # Add to the function lookup (so the agent can call it)
    TOOL_FUNCTIONS[name] = func

    print(f"[TOOL] Registered new tool: {name}")


register_tool(MEASURE_DISTANCE_SCHEMA, measure_distance)

# Verify registration
print(f"[TOOL] Total tools available: {len(TOOL_DEFS)}")
print(f"[TOOL] Tool names: {[td['function']['name'] for td in TOOL_DEFS]}")


# ===================================================================
# STEP 4: Test the tool via the agent
# ===================================================================
#
# Create an agent and ask a question that should trigger the new tool.

agent = AgentClient()

# List objects in the scene for reference
object_names = [obj.name for obj in bpy.data.objects[:10]]
print(f"\nScene objects (first 10): {object_names}")

if len(object_names) >= 2:
    question = (
        f"What is the distance between '{object_names[0]}' "
        f"and '{object_names[1]}'?"
    )
    print(f"\nQuestion: {question}")

    answer, messages, timings, tree = agent.ask(question, reset_history=True)

    print(f"Answer: {answer}")
    print(f"Timings: {timings}")
else:
    print("\nScene has fewer than 2 objects -- skipping agent test.")
    print("You can test manually:")
    print("  result = measure_distance('Camera', 'Light')")
    print("  print(result)")


# ===================================================================
# NOTES ON ADDING TOOLS
# ===================================================================
#
# - Tool names must be unique. Do not reuse names from schema.json.
# - The function must accept keyword arguments matching the schema.
# - Return a dict with at least a "result" key (string the SLM sees).
# - Include "timings" if you want timing data in benchmark results.
# - Generator-based tools (like count_pointing) are also supported:
#   yield intermediate status dicts, then return the final result.
# - For persistent tools, add the schema to scope/tools/schema.json
#   and the function to scope/tools/blender_tools.py.
