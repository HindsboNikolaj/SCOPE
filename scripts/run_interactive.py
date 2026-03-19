#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_interactive.py -- Set up SCOPE for interactive use inside Blender.

Usage:
    blender scene.blend --python scripts/run_interactive.py

This script:
  1. Adds the SCOPE project root to sys.path
  2. Loads the agent configuration
  3. Initializes and binds the VLM for perception tools
  4. Creates a ready-to-use AgentClient stored in bpy.scope_agent
  5. Optionally registers the agent_banner Blender addon (if present)

After running, you can interact with the agent from Blender's Python console:

    >>> agent = bpy.scope_agent
    >>> answer, msgs, timings, tree = agent.ask("What do you see?")
    >>> print(answer)

Or use the agent_banner addon UI panel if it is installed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: add project root to sys.path
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import bpy

from scope.utils.config import load_agent_config
from scope.agent.client import AgentClient
from scope.tools import blender_tools
from scope.tools.vlm_clients import create_vlm_from_env, create_vlm, VLMClient

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
config_path = os.getenv("SCOPE_CONFIG", str(PROJECT_ROOT / "configs" / "agent_config.yaml"))
print(f"[SCOPE] Loading config from: {config_path}")
cfg = load_agent_config(config_path)

slm_cfg = cfg.get("agent", {}).get("slm", {})
vlm_cfg = cfg.get("agent", {}).get("vlm", {})

# ---------------------------------------------------------------------------
# Initialize VLM
# ---------------------------------------------------------------------------
vlm: VLMClient | None = None

vlm_backend = vlm_cfg.get("backend", "").lower()
vlm_base_url = vlm_cfg.get("base_url", "")
vlm_model_id = vlm_cfg.get("model_id", "")
vlm_api_key = vlm_cfg.get("api_key", "")

try:
    if vlm_backend:
        vlm = create_vlm(
            kind=vlm_backend,
            base_url=vlm_base_url or None,
            model_id=vlm_model_id or None,
            api_key=vlm_api_key or None,
        )
    else:
        # Fall back to environment variables
        vlm = create_vlm_from_env()
    blender_tools.set_vlm(vlm)
    print(f"[SCOPE] VLM initialized: {vlm.name} (caps: caption={vlm.caps.caption}, "
          f"vqa={vlm.caps.vqa}, detect={vlm.caps.detect}, point={vlm.caps.point})")
except Exception as e:
    print(f"[SCOPE] WARNING: VLM initialization failed: {e}")
    print("[SCOPE] Visual tools (zoom_bounding, query_answer, count_pointing) will not work.")
    print("[SCOPE] Check your VLM configuration in configs/agent_config.yaml")

# ---------------------------------------------------------------------------
# Create AgentClient
# ---------------------------------------------------------------------------
agent = AgentClient(
    model_id=slm_cfg.get("model_id"),
    base_url=slm_cfg.get("base_url"),
    api_key=slm_cfg.get("api_key", "ollama"),
    temperature=float(slm_cfg.get("temperature", 0.7)),
)

# Store the agent on bpy so it is accessible from Blender's Python console
# and from addon panels.
bpy.scope_agent = agent  # type: ignore[attr-defined]

print(f"[SCOPE] AgentClient ready: model={agent.model_id}, base_url={agent.base_url}")
print(f"[SCOPE] Access the agent via: bpy.scope_agent")

# ---------------------------------------------------------------------------
# Register agent_banner addon (if present)
# ---------------------------------------------------------------------------
# The agent_banner addon provides a Blender UI panel for chat-based
# interaction.  It is optional -- SCOPE works without it.

addon_candidates = [
    PROJECT_ROOT / "scope" / "blender" / "agent_banner.py",
    PROJECT_ROOT / "addons" / "agent_banner.py",
]

addon_loaded = False
for addon_path in addon_candidates:
    if addon_path.is_file():
        try:
            # Load the addon script so its register() is called
            exec(compile(open(addon_path).read(), str(addon_path), "exec"))
            print(f"[SCOPE] Loaded agent_banner addon from {addon_path}")
            addon_loaded = True
            break
        except Exception as e:
            print(f"[SCOPE] WARNING: Failed to load agent_banner from {addon_path}: {e}")

if not addon_loaded:
    print("[SCOPE] agent_banner addon not found (optional).")
    print("[SCOPE] You can interact with the agent via bpy.scope_agent.ask('...')")

# ---------------------------------------------------------------------------
# Warmup (optional: pre-load the model into memory)
# ---------------------------------------------------------------------------
try:
    agent.warmup_once()
    print("[SCOPE] Model warmup complete.")
except Exception:
    pass

print("\n[SCOPE] Interactive session ready.")
print("[SCOPE] Example:")
print("  >>> answer, msgs, timings, tree = bpy.scope_agent.ask('Describe the scene.')")
print("  >>> print(answer)")
