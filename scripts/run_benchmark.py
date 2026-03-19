#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_benchmark.py -- Run the SCOPE benchmark suite inside Blender.

Usage:
    blender --background --python scripts/run_benchmark.py -- [OPTIONS]

    Options:
      --config PATH       Path to agent_config.yaml (default: configs/agent_config.yaml
                          or SCOPE_CONFIG env var)
      --output PATH       Path for the results CSV (default: results/benchmark_results.csv)
      --scenes-dir PATH   Directory containing .blend scene files
                          (default: benchmark/scenes/)
      --limit N           Only run the first N questions (useful for testing)
      --dry-run           Parse the benchmark CSV and print questions without running

This script is designed to run INSIDE Blender's embedded Python interpreter.
It imports bpy and drives each scene programmatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Blender bootstrap: make sure the project root is on sys.path so that
# `scope.*` imports work when launched via `blender --python`.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# bpy must be available -- this script is meant to run inside Blender.
try:
    import bpy
except ImportError:
    print("ERROR: This script must be run inside Blender.")
    print("  blender --background --python scripts/run_benchmark.py -- [OPTIONS]")
    sys.exit(1)

from scope.utils.config import load_agent_config
from scope.agent.client import AgentClient
from scope.tools import blender_tools
from scope.tools.vlm_clients import create_vlm_from_env
from scope.blender.preset_helpers import apply_preset
from scope.eval.judge import main as judge_main
from scope.eval.metrics import print_report

# ---------------------------------------------------------------------------
# Argument parsing -- Blender swallows args before "--", so we parse after it.
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments that come after the Blender '--' separator."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Run the SCOPE benchmark inside Blender."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to agent_config.yaml (default: auto-detect via SCOPE_CONFIG or configs/agent_config.yaml)"
    )
    parser.add_argument(
        "--output", type=str, default=str(PROJECT_ROOT / "results" / "benchmark_results.csv"),
        help="Output CSV path for results"
    )
    parser.add_argument(
        "--scenes-dir", type=str, default=None,
        help="Override directory containing .blend scene files"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run the first N questions (useful for testing)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and print questions without executing them"
    )
    parser.add_argument(
        "--judge", action="store_true",
        help="After the benchmark completes, invoke the LLM-as-Judge on the output CSV"
    )
    parser.add_argument(
        "--judge-model", type=str, default=None,
        help="Judge model id (overrides JUDGE_MODEL_ID env var, default: gpt-4o)"
    )
    parser.add_argument(
        "--judge-base-url", type=str, default=None,
        help="Judge API base URL (overrides JUDGE_API_BASE env var)"
    )
    parser.add_argument(
        "--judge-api-key", type=str, default=None,
        help="Judge API key (overrides JUDGE_API_KEY env var)"
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Benchmark CSV loader
# ---------------------------------------------------------------------------

def load_benchmark(csv_path: str) -> list[dict]:
    """Read the benchmark CSV and return a list of row dicts."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Scene management
# ---------------------------------------------------------------------------

def open_scene(blend_path: str) -> bool:
    """Open a .blend file. Returns True on success."""
    if not os.path.isfile(blend_path):
        print(f"  [SKIP] Scene file not found: {blend_path}")
        return False
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    return True


def setup_preset(preset_name: str | None) -> None:
    """Move the camera to the starting preset, if specified."""
    if not preset_name or preset_name.lower() in ("none", ""):
        return
    try:
        apply_preset(preset_name)
        print(f"  [PRESET] Applied starting preset: {preset_name}")
    except Exception as e:
        print(f"  [WARN] Could not apply preset '{preset_name}': {e}")


# ---------------------------------------------------------------------------
# Result writer
# ---------------------------------------------------------------------------

RESULT_FIELDS = [
    "question_id", "file_location", "question", "expected_answer",
    "agent_answer", "question_type", "difficulty",
    "time_total", "time_llm", "time_vlm", "time_script", "time_camera",
    "model_id", "status",
]


def write_result_row(writer: csv.DictWriter, row: dict, answer: str,
                     timings: dict, model_id: str, status: str) -> None:
    writer.writerow({
        "question_id":    row.get("question_id", ""),
        "file_location":  row.get("file_location", ""),
        "question":       row.get("question", ""),
        "expected_answer": row.get("expected_answer", ""),
        "agent_answer":   answer,
        "question_type":  row.get("question_type", ""),
        "difficulty":     row.get("difficulty", ""),
        "time_total":     timings.get("total", 0.0),
        "time_llm":       timings.get("llm", 0.0),
        "time_vlm":       timings.get("vlm", 0.0),
        "time_script":    timings.get("script", 0.0),
        "time_camera":    timings.get("camera", 0.0),
        "model_id":       model_id,
        "status":         status,
    })


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(args: argparse.Namespace) -> None:
    # -- Load configuration --------------------------------------------------
    cfg = load_agent_config(args.config)
    slm_cfg = cfg.get("agent", {}).get("slm", {})
    blender_cfg = cfg.get("blender", {})

    scenes_dir = args.scenes_dir or blender_cfg.get("scenes_dir", "benchmark/scenes/")
    if not os.path.isabs(scenes_dir):
        scenes_dir = str(PROJECT_ROOT / scenes_dir)

    benchmark_csv = cfg.get("evaluation", {}).get("benchmark", "benchmark/scope_536.csv")
    if not os.path.isabs(benchmark_csv):
        benchmark_csv = str(PROJECT_ROOT / benchmark_csv)

    # -- Load benchmark rows -------------------------------------------------
    print(f"[BENCHMARK] Loading benchmark from {benchmark_csv}")
    rows = load_benchmark(benchmark_csv)
    if args.limit:
        rows = rows[:args.limit]
    print(f"[BENCHMARK] {len(rows)} questions to process")

    if args.dry_run:
        for r in rows:
            print(f"  {r.get('question_id', '?')}: {r.get('question', '(no question)')}")
        print("[BENCHMARK] Dry run complete.")
        return

    # -- Create agent --------------------------------------------------------
    agent = AgentClient(
        model_id=slm_cfg.get("model_id"),
        base_url=slm_cfg.get("base_url"),
        api_key=slm_cfg.get("api_key", "ollama"),
        temperature=float(slm_cfg.get("temperature", 0.7)),
    )
    print(f"[BENCHMARK] Agent model: {agent.model_id}")
    print(f"[BENCHMARK] Agent base_url: {agent.base_url}")

    # -- Prepare output directory --------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -- Run each question ---------------------------------------------------
    current_scene = None
    completed = 0
    errors = 0
    t_start = time.time()

    with open(output_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=RESULT_FIELDS)
        writer.writeheader()

        for i, row in enumerate(rows):
            qid = row.get("question_id", f"Q_{i}")
            question = row.get("question", "")
            file_loc = row.get("file_location", "")
            preset = row.get("preset_start", "")

            # Resolve scene path relative to scenes_dir
            if file_loc:
                scene_path = os.path.join(scenes_dir, file_loc)
            else:
                scene_path = ""

            print(f"\n--- [{i+1}/{len(rows)}] {qid} ---")
            print(f"  Scene:    {file_loc}")
            print(f"  Preset:   {preset}")
            print(f"  Question: {question}")

            # Open scene if it changed
            if scene_path and scene_path != current_scene:
                if not open_scene(scene_path):
                    write_result_row(writer, row, "", {}, agent.model_id, "scene_not_found")
                    errors += 1
                    continue
                current_scene = scene_path

            # Apply starting preset
            setup_preset(preset)

            # Ask the agent
            try:
                answer, messages, timings, call_tree = agent.ask(
                    question, reset_history=True
                )
                status = "ok"
                completed += 1
                print(f"  Answer:   {answer[:200]}")
                print(f"  Timings:  {timings}")
            except Exception as e:
                answer = ""
                timings = {}
                status = f"error: {e}"
                errors += 1
                print(f"  ERROR:    {e}")

            write_result_row(writer, row, answer, timings, agent.model_id, status)
            out_f.flush()

    elapsed = time.time() - t_start
    print(f"\n[BENCHMARK] Complete: {completed} ok, {errors} errors, {elapsed:.1f}s total")
    print(f"[BENCHMARK] Results written to {output_path}")

    # -- Optionally run the judge on the results --------------------------------
    if args.judge:
        judged_path = output_path.with_suffix(".judged.csv")
        print(f"\n[JUDGE] Running LLM-as-Judge on {output_path} ...")

        judge_base = (
            args.judge_base_url
            or os.getenv("JUDGE_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        judge_model = args.judge_model or os.getenv("JUDGE_MODEL_ID") or "gpt-4o"
        judge_key = (
            args.judge_api_key
            or os.getenv("JUDGE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "EMPTY"
        )

        # Build a fake sys.argv for judge_main (it uses argparse internally)
        _saved_argv = sys.argv
        sys.argv = [
            "judge",
            "-i", str(output_path),
            "-o", str(judged_path),
            "--base-url", judge_base,
            "--model", judge_model,
            "--api-key", judge_key,
        ]
        try:
            judge_main()
        finally:
            sys.argv = _saved_argv

        print(f"[JUDGE] Judged results written to {judged_path}")

        # Print accuracy report
        print()
        print_report(str(judged_path))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(args)
