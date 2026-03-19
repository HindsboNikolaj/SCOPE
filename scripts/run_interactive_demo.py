#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_interactive_demo.py — Standalone SCOPE agent demo (outside Blender).

Useful for smoke-testing your SLM + VLM setup without launching Blender.
Sends a single question to the agent and prints the response and timings.

Usage:
    python scripts/run_interactive_demo.py \\
        --question "How many people are visible right now?"

    python scripts/run_interactive_demo.py \\
        --question "Go to the store-front preset" \\
        --model qwen3:4b \\
        --base-url http://localhost:11434/v1

Note:
    This script runs outside Blender, so Blender-specific tools
    (zoom_bounding, count_pointing, take_image, etc.) are not available.
    The agent will respond using only text reasoning. To use the full
    tool suite, run inside Blender via:
        blender scene.blend --python scripts/run_interactive.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# -- Bootstrap: add project root to sys.path --------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -- Load .env if present ---------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass  # python-dotenv is optional here


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SCOPE interactive demo — test your agent setup outside Blender.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--question", "-q",
        default="What tools do you have available?",
        help="Question to send to the agent (default: ask about tools)",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model ID override (e.g. qwen3:4b). Falls back to AGENT_MODEL_ID env var.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="API base URL override (e.g. http://localhost:11434/v1). Falls back to AGENT_API_BASE.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key override. Falls back to AGENT_API_KEY.",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=False,
        help="Enable thinking/reasoning mode (for Qwen3 hybrid models).",
    )
    args = parser.parse_args()

    # -- Set env vars from CLI overrides ------------------------------------
    if args.model:
        os.environ["AGENT_MODEL_ID"] = args.model
    if args.base_url:
        os.environ["AGENT_API_BASE"] = args.base_url
    if args.api_key:
        os.environ["AGENT_API_KEY"] = args.api_key

    # -- Import AgentClient -------------------------------------------------
    try:
        from scope.agent.thinking import ThinkingMode, thinking_mode_for_model
    except ImportError as e:
        print(f"[ERROR] Could not import SCOPE: {e}")
        print("        Make sure you ran:  pip install -r requirements.txt")
        sys.exit(1)

    # -- Show config --------------------------------------------------------
    base_url = os.getenv("AGENT_API_BASE", "http://localhost:11434/v1")
    model_id = os.getenv("AGENT_MODEL_ID", "(auto-detect)")
    print()
    print("─" * 60)
    print("  SCOPE Interactive Demo")
    print("─" * 60)
    print(f"  SLM endpoint : {base_url}")
    print(f"  Model        : {model_id}")
    print(f"  Question     : {args.question}")
    print("─" * 60)
    print()
    print("  Note: Running outside Blender — Blender tools are not active.")
    print("        The agent will respond with text reasoning only.")
    print()

    # -- Build a minimal tool-free agent for outside-Blender use -----------
    # We import openai directly to avoid the blender_tools lazy-load
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=base_url,
            api_key=os.getenv("AGENT_API_KEY", "ollama"),
        )

        # Auto-pick model if not set
        resolved_model = os.getenv("AGENT_MODEL_ID")
        if not resolved_model:
            try:
                models = client.models.list()
                candidates = [m.id for m in models.data]
                if candidates:
                    resolved_model = candidates[0]
                    print(f"  Auto-picked model: {resolved_model}")
                else:
                    print("[ERROR] No models available at the endpoint.")
                    sys.exit(1)
            except Exception as e:
                print(f"[ERROR] Could not list models: {e}")
                print("        Is your SLM endpoint running?")
                print(f"        Expected at: {base_url}")
                sys.exit(1)

        # Check thinking mode
        mode = thinking_mode_for_model(resolved_model)
        system_msgs = []
        if mode.name == "TOGGLE":
            token = "/think" if args.thinking else "/no_think"
            system_msgs = [{"role": "system", "content": token}]

        messages = system_msgs + [{"role": "user", "content": args.question}]

        print("Sending request...\n")
        import time
        t0 = time.time()
        resp = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
        )
        elapsed = round(time.time() - t0, 2)

        content = resp.choices[0].message.content or ""
        # Strip reasoning tags for display
        import re
        content_clean = re.sub(r"<think>.*?</think>", "", content, flags=re.S | re.I).strip()

        print("─" * 60)
        print(f"  Response ({elapsed}s):")
        print("─" * 60)
        print(content_clean)
        print()

    except ImportError:
        print("[ERROR] openai package not found. Install with:  pip install openai")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
