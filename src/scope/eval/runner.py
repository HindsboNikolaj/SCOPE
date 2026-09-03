#!/usr/bin/env python3
"""
runner.py  --  SCOPE Batch Evaluation Runner (sanitized)

Run inside Blender (WITH UI) once, iterate over a CSV of questions,
drive the PTZ agent iteratively per question, and append results to CSV.

Env expected (set by launcher/config):
  QUESTIONS_CSV : required
  OUT_CSV       : required
  REPEATS       : optional (default 1)

Agent env (all optional):
  AGENT_API_BASE   : OpenAI-compatible base (default http://localhost:11434/v1)
  AGENT_MODEL_ID   : explicit model id; if omitted the agent will auto-pick from /models
  AGENT_API_KEY    : auth to pass through (default "ollama")
  AGENT_AUTO_PICK  : if set (e.g. "1"), and multiple models are served, pick the first

Notes:
- This script does NOT mutate your .blend file.
- It reuses the same agent/tool stack across questions.
"""

import bpy, os, sys, csv, json, re, traceback, datetime, time as _time, math
from pathlib import Path
from typing import Any, Dict, List
from collections import Counter

# --- Paths & imports ---------------------------------------------------------
# After the src/ layout move this file lives at src/scope/eval/runner.py, so:
#   parents[0] = src/scope/eval
#   parents[1] = src/scope
#   parents[2] = src                <- add to sys.path so `import scope.*` works
#   parents[3] = <project root>     <- for benchmark/, prompts/, etc.
_HERE = Path(__file__).resolve()
SRC_DIR = _HERE.parents[2]
PROJECT_ROOT = _HERE.parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Tool schema (now lives under src/scope/tools/)
tools_path = str(SRC_DIR / "scope" / "tools" / "schema.json")
with open(tools_path, "r", encoding="utf-8") as f:
    TOOL_DEFS = json.load(f)

# Tool implementations
from scope.tools import blender_tools

# Agent client
from scope.agent.client import AgentClient

# Blender helpers
from scope.blender.helper_funcs import (
    screenshot_camera_view,
)
from scope.blender.preset_helpers import (
    list_presets,
    create_preset,
    apply_preset,
)


def get_vlm_label() -> str:
    """Return a human-readable label for the active VLM, or empty string."""
    try:
        vlm = getattr(blender_tools, "_VLM", None)
        if vlm is not None:
            return getattr(vlm, "name", "") or str(type(vlm).__name__)
    except Exception:
        pass
    return ""


# How the viewport is shaded when a frame is captured.
#
# This used to be left to whatever the .blend happened to open with, and that is not stable
# across machines. Each benchmark scene carries nine or ten workspaces, each with its own
# shading: book-nook has Shading set to Material Preview, Layout set to Solid with random
# per-object colours, and Texture Paint set to Solid with textures. Which one is active
# decides what the model is shown. Blender restores the file's own workspace only while the
# user's "Load UI" preference is on, which is the default but is a preference, so a user who
# turns it off silently captures in their own workspace's mode instead and nothing reports it.
#
# The difference is not cosmetic. Asked what is inside the bookshop window, a vision model
# given a Material Preview capture answers "bookshelves filled with books"; given a Solid
# capture of the same camera it answers "a dark interior", because Solid draws glass opaque.
# Signage and object counts survive both. Anything behind glass does not.
#
# So the mode is set here rather than inherited. Material Preview is the default because it
# is what the scenes are saved as, what the published results used, and about a second a
# frame on a machine with a GPU. Override it when software rendering makes that impractical;
# docs/SETUP.md has the measurements.
SCOPE_VIEW_SHADING = os.environ.get("SCOPE_SHADING", "MATERIAL").strip().upper()
SCOPE_VIEW_COLOR = os.environ.get("SCOPE_SHADING_COLOR", "TEXTURE").strip().upper()


def prepare_view_for_capture():
    """Put the 3D viewport into a known state before any frame is captured.

    Sets the camera view, the shading mode, and hides the overlays and gizmos that would
    otherwise appear in the image. Everything here is deliberate: anything left unset is
    inherited from the file or from the user, and then two machines disagree.
    """
    try:
        for area in bpy.context.screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type != "VIEW_3D":
                    continue
                space.region_3d.view_perspective = "CAMERA"
                space.camera = bpy.context.scene.camera

                sh = space.shading
                if SCOPE_VIEW_SHADING in ("SOLID", "MATERIAL", "RENDERED"):
                    sh.type = SCOPE_VIEW_SHADING
                if sh.type == "SOLID" and SCOPE_VIEW_COLOR:
                    try:
                        sh.color_type = SCOPE_VIEW_COLOR
                    except (TypeError, AttributeError):
                        pass
                # Studio lighting, and the scene world left out of it. A scene whose world
                # texture is missing would otherwise tint every frame magenta, which is the
                # state whitechapel ships in.
                for attr, val in (("light", "STUDIO"), ("use_scene_lights", False),
                                  ("use_scene_world", False)):
                    try:
                        setattr(sh, attr, val)
                    except (TypeError, AttributeError):
                        pass

                # Editor furniture must not reach the model. screenshot_camera_view hides
                # these too, but setting them here means the state is right for any capture
                # backend, including the offscreen one used headless.
                ov = space.overlay
                for attr in ("show_overlays", "show_floor", "show_axis_x", "show_axis_y",
                             "show_axis_z", "show_cursor", "show_text", "show_stats",
                             "show_extras", "show_relationship_lines", "show_outline_selected"):
                    try:
                        setattr(ov, attr, False)
                    except (TypeError, AttributeError):
                        pass
                for attr in ("show_gizmo", "show_region_ui", "show_region_toolbar",
                             "show_region_header"):
                    try:
                        setattr(space, attr, False)
                    except (TypeError, AttributeError):
                        pass
            break
    except Exception as _e:
        print(f"[runner] WARN: could not standardise the viewport: {_e}")


# --- Config via env vars -----------------------------------------------------
# SCOPE_CONFIG (if set) provides defaults; explicit env vars always win.
# This preserves the documented "env-var-driven knobs" contract while making
# the YAML config a real source of truth instead of a no-op.
_cfg: Dict[str, Any] = {}
_cfg_slm: Dict[str, Any] = {}
_cfg_eval: Dict[str, Any] = {}
if os.environ.get("SCOPE_CONFIG"):
    try:
        from scope.utils.config import load_agent_config
        _cfg = load_agent_config(os.environ["SCOPE_CONFIG"]) or {}
        _cfg_slm = ((_cfg.get("agent") or {}).get("slm") or {})
        _cfg_eval = (_cfg.get("evaluation") or {})
        print(f"[runner] Loaded SCOPE_CONFIG: {os.environ['SCOPE_CONFIG']}")
    except Exception as _e:
        print(f"[runner] WARN: failed to load SCOPE_CONFIG: {_e}")

QUESTIONS_CSV = os.environ.get("QUESTIONS_CSV") or _cfg_eval.get("benchmark") or ""
OUT_CSV       = os.environ.get("OUT_CSV") or ""
REPEATS       = int(os.environ.get("REPEATS") or "1")
# Resume: skip question_ids that already have a non-empty final_answer in OUT_CSV.
# Default ON (Blender can crash mid-run); set SCOPE_RESUME=0 to force a fresh run.
RESUME        = (os.environ.get("SCOPE_RESUME", "1").strip().lower()
                 not in ("0", "false", "no", "off", ""))
# Optional smoke-test knob: cap the number of input questions actually processed.
try:
    BENCH_LIMIT = int(os.environ.get("BENCH_LIMIT") or "0")
except ValueError:
    BENCH_LIMIT = 0

# Optional agent wiring (env wins, then config, then hard-coded default).
API_BASE      = (os.environ.get("AGENT_API_BASE")
                 or _cfg_slm.get("base_url")
                 or "http://localhost:11434/v1")
MODEL_ID      = (os.environ.get("AGENT_MODEL_ID")
                 or os.environ.get("MODEL_ID")
                 or _cfg_slm.get("model_id")
                 or "").strip()
API_KEY       = (os.environ.get("AGENT_API_KEY")
                 or _cfg_slm.get("api_key")
                 or "ollama")

if not QUESTIONS_CSV or not OUT_CSV:
    print("[FATAL] QUESTIONS_CSV and OUT_CSV must be set in the environment.")
    bpy.ops.wm.quit_blender()

# --- CSV IO helpers ----------------------------------------------------------

def read_questions(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def append_result(path, row, header):
    exists = Path(path).exists()
    with open(path, "a" if exists else "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)


def prepare_resume(path: str, header: list) -> set:
    """Read an existing OUT_CSV and return the set of (question_id, repeat_idx)
    pairs that are already complete (non-empty ``final_answer``). Errored rows
    (empty ``final_answer``) are dropped from the file so they get retried, and
    the file is rewritten with the canonical header for append-compat.

    Keying on (qid, repeat) -- not just qid -- means a REPEATS>1 run that
    crashed partway through a question's repeats correctly resumes the
    remaining repeats instead of skipping them.

    If the existing header doesn't match ``header``, abort: a mismatched schema
    almost certainly means an older runner wrote it and interleaving columns
    would corrupt the file. The user can pass SCOPE_RESUME=0 to overwrite.
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return set()
    # Header compat check.
    with open(p, "r", encoding="utf-8", newline="") as rf:
        first_line = rf.readline()
    existing_header = next(csv.reader([first_line])) if first_line else []
    if existing_header != header:
        print(f"[runner] ERROR: existing {p} has incompatible schema; "
              f"set SCOPE_RESUME=0 or delete it to start fresh.")
        bpy.ops.wm.quit_blender()
        return set()

    # Drop a trailing partial line (interrupted write) before parsing.
    with open(p, "r", encoding="utf-8", newline="") as rf:
        raw = rf.read()
    lines = raw.splitlines(keepends=True)
    if lines and not lines[-1].endswith(("\n", "\r")):
        print(f"[runner] Dropping partial trailing row from {p}")
        with open(p, "w", encoding="utf-8", newline="") as wf:
            wf.write("".join(lines[:-1]))

    done: set = set()
    kept: list = []
    errored = 0
    with open(p, "r", encoding="utf-8", newline="") as rf:
        reader = csv.DictReader(rf)
        for r in reader:
            qid = (r.get("question_id") or "").strip()
            if not qid:
                continue
            # repeat_idx is 1-based in the output. Default to 1 if missing
            # (e.g. legacy rows written before this resume model existed).
            try:
                rep = int((r.get("repeat_idx") or "1").strip() or "1")
            except ValueError:
                rep = 1
            if (r.get("final_answer") or "").strip():
                done.add((qid, rep))
                kept.append(r)
            else:
                errored += 1  # drop -> retry
    if errored:
        with open(p, "w", encoding="utf-8", newline="") as wf:
            w = csv.DictWriter(wf, fieldnames=header)
            w.writeheader()
            for r in kept:
                w.writerow({k: r.get(k, "") for k in header})
    print(f"[runner] Resume: {len(done)} (qid, repeat) pairs already complete; "
          f"{errored} errored rows dropped for retry.")
    return done


def now_iso():
    return datetime.datetime.utcnow().isoformat()


def parse_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "on"):
        return True
    if s in ("false", "0", "no", "n", "off"):
        return False
    return default


def _as_json_str(obj: Any, default: str):
    """Ensure CSV cell contains a JSON string for list/dict; pass through if already JSON-looking."""
    if obj is None or obj == "":
        return default
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, ensure_ascii=False)
    s = str(obj).strip()
    if s.startswith("{") or s.startswith("["):
        return s
    try:
        if default.strip().startswith("["):
            return json.dumps([s], ensure_ascii=False)
        return json.dumps(s, ensure_ascii=False)
    except Exception:
        return default


def _get(row: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for k in keys:
        if k in row and str(row[k]).strip() != "":
            return str(row[k]).strip()
    return default


def _get_json(row: Dict[str, Any], keys: List[str], default: str) -> str:
    val = _get(row, keys, "")
    return _as_json_str(val if val != "" else None, default)


def _get_bool(row: Dict[str, Any], keys: List[str], default: bool = False) -> bool:
    val = _get(row, keys, "")
    return parse_bool(val, default)


def parse_presets_field(val) -> list:
    """
    Accepts:
      [mailbox, hotel-m]
      ["mailbox","hotel-m"]
      mailbox, hotel-m
    Returns a clean list of strings.
    """
    if val is None:
        return []
    s = str(val).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
    return [p for p in parts if p]


# --- Tool-call extraction helpers --------------------------------------------

def extract_tool_order(messages):
    order = []
    for m in messages or []:
        if (m.get("role") == "assistant") and m.get("tool_calls"):
            for tc in (m["tool_calls"] or []):
                fn = (tc.get("function") or {}).get("name")
                if fn:
                    order.append(fn)
    return json.dumps(order, ensure_ascii=False)


def extract_actual_tool_calls(messages):
    """Return ordered executed calls as JSON string: [{"name":..., "arguments":{...}}, ...]"""
    calls = []
    for m in messages or []:
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                fn = (tc.get("function") or {}).get("name")
                args_raw = (tc.get("function") or {}).get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                except Exception:
                    args = {"_raw": str(args_raw)}
                if fn:
                    calls.append({"name": fn, "arguments": args})
    return json.dumps(calls, ensure_ascii=False)


# --- Final answer scraper ----------------------------------------------------

_DELIMS = ("Final Answer:", "Assistant:", "Tool", "===", "---")


def _is_delim(line: str) -> bool:
    s = (line or "").strip()
    return any(s.startswith(d) for d in _DELIMS)


def _clean_line(s: str) -> str:
    return (s or "").rstrip()


def scrape_final_answer(log_lines):
    """
    Extract the final answer block:
      1) Prefer the LAST 'Final Answer:' block, capturing subsequent lines
         until the next delimiter.
      2) Fallback to the LAST real 'Assistant:' block (skips '(tool call)').
    """
    lines = []
    for raw in (log_lines or []):
        lines.extend(str(raw).splitlines())

    # 1) LAST Final Answer block
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if "Final Answer:" in line:
            head = line.split("Final Answer:", 1)[-1].strip()
            block = [head] if head else []
            j = i + 1
            while j < len(lines) and not _is_delim(lines[j]):
                if lines[j].strip():
                    block.append(_clean_line(lines[j]))
                j += 1
            out = "\n".join(block).strip()
            if out:
                return out

    # 2) Fallback: LAST non-tool-call Assistant block
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if "Assistant:" in line:
            content = line.split(":", 1)[-1].strip()
            if not content or content.lower() == "(tool call)":
                continue
            block = [content]
            j = i + 1
            while j < len(lines) and not _is_delim(lines[j]):
                if lines[j].strip():
                    block.append(_clean_line(lines[j]))
                j += 1
            out = "\n".join(block).strip()
            if out:
                return out

    return ""


def scope_full_satisfied(answer_view: str, actual_calls_json: str) -> bool:
    if (answer_view or "").strip().lower() != "full":
        return True
    try:
        calls = json.loads(actual_calls_json) or []
    except Exception:
        return False
    full_ok_names = {"count_pointing", "query_answer", "count_objects", "query_scene"}
    for c in calls:
        name = (c.get("name") or "").strip()
        args = c.get("arguments") or {}
        vt = (args.get("view_type") or "").strip().lower()
        if name in full_ok_names and vt == "full":
            return True
    return False


def coverage_for_unordered(required_tools_policy: str, expected_tool_order_json: str,
                           expected_tools_multiset: str, actual_calls_json: str) -> bool:
    """
    Basic coverage check:
    - For unordered mode, if policy == 'all': every tool name in expected set must appear at least once.
      if 'any': at least one appears.
    - If expected_tools_multiset present: enforce minimum counts by name.
    """
    try:
        exp = json.loads(expected_tool_order_json) if expected_tool_order_json else []
    except Exception:
        exp = []
    try:
        multiset = json.loads(expected_tools_multiset) if expected_tools_multiset else {}
    except Exception:
        multiset = {}
    try:
        calls = json.loads(actual_calls_json) if actual_calls_json else []
    except Exception:
        calls = []
    actual_names = [(c.get("name") or "").strip() for c in calls]
    # Name-set policy
    exp_names = []
    for item in exp:
        if isinstance(item, dict):
            nm = (item.get("name") or "").strip()
            if nm:
                exp_names.append(nm)
        elif isinstance(item, str):
            exp_names.append(item.strip())
    if required_tools_policy == "all" and exp_names:
        for nm in set(exp_names):
            if nm not in actual_names:
                return False
    elif required_tools_policy == "any" and exp_names:
        if not any(nm in actual_names for nm in set(exp_names)):
            return False
    # Multiplicity policy
    if isinstance(multiset, dict) and multiset:
        c = Counter(actual_names)
        for nm, req in multiset.items():
            try:
                req_i = int(req)
            except Exception:
                req_i = 1
            if c.get(nm, 0) < req_i:
                return False
    return True


# --- Result header (Groups 1-4) ---------------------------------------------

AUTHOR_COLS = [
    "question_id", "file_location", "preset_start", "presets_available", "question", "expected_answer",
    "question_type", "eval_category", "multi_step_mode", "required_tools_policy", "expected_tool_order_json",
    "args_check", "expected_tool_args_json", "answer_view", "evaluation_notes", "difficulty",
    "scene_objects_mentioned", "tool_schema_version", "metadata_json",
    "nl_required", "text_normalization", "expected_tools_multiset", "expected_tool_order_signature",
    "scope_requirement", "arg_match_policy", "gt_evidence_json",
]

RUNTIME_COLS = [
    "repeat_idx", "start_ts", "end_ts", "llm_model", "vlm_model",
    "wall_total", "llm_total", "vlm_total", "camera_total", "script_total",
    "final_answer", "tool_call_order", "actual_tool_calls_json", "llm_raw", "llm_readable",
]

JUDGE_COLS = ["judge_error_mode", "judge_reason"]

METRIC_COLS = ["judge_is_correct", "coverage_satisfied", "scope_satisfied", "ruler_reward", "ruler_explanation"]

RESULT_HEADER = AUTHOR_COLS + RUNTIME_COLS + JUDGE_COLS + METRIC_COLS


# --- State -------------------------------------------------------------------
_STATE = 0
_AGENT = None
_ROWS: List[Dict[str, Any]] = []
_ROW_IDX = 0
_REPEAT_IDX = 0
_EVENTS = []
_LOG_LINES = []
_CSV_PRESETS: List[str] = []
_CUR_PROMPT = ""
_RUN_START_TS = ""
_DONE_QIDS: set = set()


# --- UI helpers --------------------------------------------------------------

def _assistant_preview(msg: dict) -> str:
    c = (msg.get("content") or "").strip()
    return f"Assistant: {c}" if c else "Assistant: (tool call)"


def format_event(ev):
    lines = []
    if isinstance(ev, dict) and ev.get("role") == "assistant":
        lines.append(_assistant_preview(ev))
        for call in (ev.get("tool_calls") or []):
            fn = (call.get("function") or {}).get("name", "unknown")
            args = (call.get("function") or {}).get("arguments", "{}")
            lines.append(f"  ToolCall: {fn}({args})")
    elif isinstance(ev, dict) and ev.get("role") == "tool":
        lines.append(f"  ToolResult(id={ev.get('tool_call_id', '?')}): {ev.get('content', '')}")
    elif isinstance(ev, tuple) and len(ev) == 2 and ev[1] == "final":
        lines.append(f"  Final Answer: {ev[0]}")
    else:
        lines.append(str(ev))
    return "\n".join(lines)


# --- Field mappers for authoring cols ----------------------------------------

def field_map(row: Dict[str, Any]) -> Dict[str, Any]:
    """Pull Group 1 authoring fields from the input CSV row (new schema, with legacy fallbacks)."""
    qid   = _get(row, ["question_id", "Question ID", "ID"], default=f"Q_{_ROW_IDX+1:03d}")
    floc  = _get(row, ["file_location", "File Location"], default="")
    pstart = _get(row, ["preset_start", "Preset Start", "Preset"], default="")
    pavl  = _get(row, ["presets_available", "Presets Available", "Presets"], default="")
    ques  = _get(row, ["question", "Question", "Question Text", "Prompt", "User Question"], default="")
    expt  = _get(row, ["expected_answer", "Expected Answer", "Expected", "Answer", "Ground Truth", "GT", "Label"], default="")
    qtype = _get(row, ["question_type", "Question Type"], default="")
    ecat  = _get(row, ["eval_category", "Eval Category", "Category"], default="")
    msmode = _get(row, ["multi_step_mode", "Multi-step Mode"], default="none")
    rpol  = _get(row, ["required_tools_policy", "Required Tools Policy"], default="")
    etoj  = _get_json(row, ["expected_tool_order_json", "Expected Tool Order JSON"], default="[]")
    acheck = _get_bool(row, ["args_check", "Args Check"], default=False)
    etaj  = _get_json(row, ["expected_tool_args_json", "Expected Tool Args JSON"], default="{}")
    aview = _get(row, ["answer_view", "Answer View"], default="current")
    notes = _get(row, ["evaluation_notes", "Evaluation Notes"], default="")
    diff  = _get(row, ["difficulty", "Difficulty"], default="")
    sobjs = _get(row, ["scene_objects_mentioned", "Scene Objects Mentioned"], default="")
    tver  = _get(row, ["tool_schema_version", "Tool Schema Version"], default="v1.0")
    meta  = _get_json(row, ["metadata_json", "Metadata JSON"], default="{}")
    nlreq = _get_bool(row, ["nl_required", "NL Required"], default=True)
    tnorm = _get_json(row, ["text_normalization", "Text Normalization"], default="{}")
    etmul = _get_json(row, ["expected_tools_multiset", "Expected Tools Multiset"], default="{}")
    etsig = _get_json(row, ["expected_tool_order_signature", "Expected Tool Order Signature"], default="[]")
    scope = _get(row, ["scope_requirement", "Scope Requirement"], default="any")
    ampol = _get(row, ["arg_match_policy", "Arg Match Policy"], default="exact")
    gtev  = _get_json(row, ["gt_evidence_json", "GT Evidence JSON"], default="{}")

    pavl_list = parse_presets_field(pavl) if pavl else []
    pavl_json = json.dumps(pavl_list, ensure_ascii=False)

    return {
        "question_id": qid,
        "file_location": floc,
        "preset_start": pstart,
        "presets_available": pavl_json,
        "question": ques,
        "expected_answer": expt,
        "question_type": qtype,
        "eval_category": ecat,
        "multi_step_mode": msmode,
        "required_tools_policy": rpol,
        "expected_tool_order_json": etoj,
        "args_check": acheck,
        "expected_tool_args_json": etaj,
        "answer_view": aview,
        "evaluation_notes": notes,
        "difficulty": diff,
        "scene_objects_mentioned": sobjs,
        "tool_schema_version": tver,
        "metadata_json": meta,
        "nl_required": nlreq,
        "text_normalization": tnorm,
        "expected_tools_multiset": etmul,
        "expected_tool_order_signature": etsig,
        "scope_requirement": scope,
        "arg_match_policy": ampol,
        "gt_evidence_json": gtev,
    }


# --- Main timer loop ---------------------------------------------------------
_STATE = 0


def _batch_step():
    global _STATE, _AGENT, _ROWS, _ROW_IDX, _REPEAT_IDX
    global _EVENTS, _CUR_PROMPT, _LOG_LINES, _CSV_PRESETS, _RUN_START_TS
    global _DONE_QIDS

    try:
        # 0) Load questions & init agent once
        if _STATE == 0:
            print("[runner] Loading questions from:", QUESTIONS_CSV)
            _ROWS = read_questions(QUESTIONS_CSV)
            if BENCH_LIMIT > 0:
                _ROWS = _ROWS[:BENCH_LIMIT]
                print(f"[runner] BENCH_LIMIT={BENCH_LIMIT} (truncated)")
            print(f"[runner] Loaded {len(_ROWS)} rows")

            if RESUME:
                _DONE_QIDS = prepare_resume(OUT_CSV, RESULT_HEADER)
            else:
                # Fresh run: nuke any existing OUT_CSV so we don't append onto stale data.
                try:
                    if Path(OUT_CSV).exists():
                        Path(OUT_CSV).unlink()
                        print(f"[runner] SCOPE_RESUME=0; removed {OUT_CSV}")
                except Exception as _e:
                    print(f"[runner] Could not remove {OUT_CSV}: {_e}")

            print("[runner] Initializing agent:")
            print(f"  base: {API_BASE}")
            print(f"  model: {MODEL_ID or '(auto-pick)'}")
            _AGENT = AgentClient(
                model_id=MODEL_ID or None,
                base_url=API_BASE,
                api_key=API_KEY,
            )
            try:
                _AGENT.warmup_once()
            except Exception:
                pass

            _STATE = 1
            return 0.2

        # 1) If all questions done, quit
        if _ROW_IDX >= len(_ROWS):
            print("[runner] All questions complete -- quitting Blender.")
            bpy.ops.wm.quit_blender()
            return None

        # 2) Begin a run for the current row & repeat
        if _STATE == 1:
            row = _ROWS[_ROW_IDX]
            author_fields = field_map(row)

            # Resume: skip (qid, repeat) pairs that already completed.
            # _DONE_QIDS holds 1-based (qid, repeat_idx) tuples, matching the
            # repeat_idx value we write to OUT_CSV further below.
            if (author_fields["question_id"], _REPEAT_IDX + 1) in _DONE_QIDS:
                print(f"[runner] Skipping completed qid={author_fields['question_id']} "
                      f"repeat={_REPEAT_IDX + 1}")
                _REPEAT_IDX += 1
                if _REPEAT_IDX >= REPEATS:
                    _REPEAT_IDX = 0
                    _ROW_IDX += 1
                return 0.01

            # Per-row scene switching: the 541-row benchmark spans 4 distinct
            # .blend scenes. file_location is prefixed with "scenes/..." but
            # the scenes actually live under PROJECT_ROOT/benchmark/scenes/,
            # so the resolution root is PROJECT_ROOT/benchmark (matches the
            # convention established in 8609789's scene-path fix).
            floc = (author_fields.get("file_location") or "").strip()
            if floc:
                abs_scene = (PROJECT_ROOT / "benchmark" / floc).resolve()
                cur_path = ""
                try:
                    cur_path = bpy.data.filepath or ""
                except Exception:
                    cur_path = ""
                if str(abs_scene) != cur_path:
                    if not abs_scene.exists():
                        print(f"[runner] scene_not_found: {abs_scene}")
                        # Persist a placeholder row so resume doesn't loop.
                        err_row = {k: "" for k in RESULT_HEADER}
                        err_row.update(author_fields)
                        err_row.update({
                            "repeat_idx": _REPEAT_IDX + 1,
                            "start_ts": now_iso(),
                            "end_ts": now_iso(),
                            "final_answer": "",
                            "judge_error_mode": "scene_not_found",
                            "judge_reason": f"missing .blend at {abs_scene}",
                        })
                        append_result(OUT_CSV, err_row, RESULT_HEADER)
                        _REPEAT_IDX += 1
                        if _REPEAT_IDX >= REPEATS:
                            _REPEAT_IDX = 0
                            _ROW_IDX += 1
                        return 0.01
                    print(f"[runner] Opening scene: {abs_scene}")
                    bpy.ops.wm.open_mainfile(filepath=str(abs_scene))

            _CUR_PROMPT = author_fields["question"]
            _CSV_PRESETS = json.loads(author_fields["presets_available"]) if author_fields["presets_available"] else []
            if _CSV_PRESETS:
                print(f"[runner] CSV presets for this question: {_CSV_PRESETS}")
                _AGENT.set_presets(_CSV_PRESETS)
            else:
                print("[runner] No CSV presets for this question -- using tool-fetched presets/default.")
                _AGENT.set_presets(None)

            preset = author_fields["preset_start"]
            if preset:
                print(f"[runner] Applying preset: {preset}")
                apply_preset(preset)
            else:
                print("[runner] No preset in row; continuing from current view.")

            prepare_view_for_capture()

            print(f"[runner] Ask (row={_ROW_IDX+1}/{len(_ROWS)}, repeat={_REPEAT_IDX+1}/{REPEATS}): {_CUR_PROMPT}")
            _AGENT._iter = _AGENT.ask_iter(_CUR_PROMPT, reset_history=True)

            _EVENTS = []
            _LOG_LINES = []
            _RUN_START_TS = now_iso()
            _STATE = 2
            return 0.1

        # 3) Drive iteration until final
        if _STATE == 2:
            ev = next(_AGENT._iter, None)
            if ev is None:
                _STATE = 3
            else:
                pretty = format_event(ev)
                for ln in pretty.splitlines():
                    print(ln)
                    _LOG_LINES.append(ln)
                _EVENTS.append(ev)
                if isinstance(ev, tuple) and ev[1] == "final":
                    _STATE = 3
            return 0.05

        # 4) Persist results for this (row, repeat), then advance indices
        if _STATE == 3:
            row = _ROWS[_ROW_IDX]
            author_fields = field_map(row)

            timings = getattr(_AGENT, "last_timings", {}) or {}
            msgs = getattr(_AGENT, "last_messages", None) or _AGENT.messages
            llm_raw_json = json.dumps(msgs, ensure_ascii=False)

            llm_model = getattr(_AGENT, "last_model_id", None) or getattr(_AGENT, "model_id", "")
            vlm_model = get_vlm_label()

            final_answer = scrape_final_answer(_LOG_LINES)
            tool_order_json = extract_tool_order(msgs)
            actual_calls_json = extract_actual_tool_calls(msgs)

            scope_ok = scope_full_satisfied(author_fields["answer_view"], actual_calls_json)
            cov_ok = True
            if author_fields["multi_step_mode"] == "unordered":
                cov_ok = coverage_for_unordered(
                    author_fields["required_tools_policy"],
                    author_fields["expected_tool_order_json"],
                    author_fields["expected_tools_multiset"],
                    actual_calls_json,
                )

            out_row = {k: "" for k in RESULT_HEADER}

            # Group 1 -- Authoring fields (verbatim)
            out_row.update({
                "question_id": author_fields["question_id"],
                "file_location": author_fields["file_location"],
                "preset_start": author_fields["preset_start"],
                "presets_available": author_fields["presets_available"],
                "question": author_fields["question"],
                "expected_answer": author_fields["expected_answer"],
                "question_type": author_fields["question_type"],
                "eval_category": author_fields["eval_category"],
                "multi_step_mode": author_fields["multi_step_mode"],
                "required_tools_policy": author_fields["required_tools_policy"],
                "expected_tool_order_json": author_fields["expected_tool_order_json"],
                "args_check": author_fields["args_check"],
                "expected_tool_args_json": author_fields["expected_tool_args_json"],
                "answer_view": author_fields["answer_view"],
                "evaluation_notes": author_fields["evaluation_notes"],
                "difficulty": author_fields["difficulty"],
                "scene_objects_mentioned": author_fields["scene_objects_mentioned"],
                "tool_schema_version": author_fields["tool_schema_version"],
                "metadata_json": author_fields["metadata_json"],
                "nl_required": author_fields["nl_required"],
                "text_normalization": author_fields["text_normalization"],
                "expected_tools_multiset": author_fields["expected_tools_multiset"],
                "expected_tool_order_signature": author_fields["expected_tool_order_signature"],
                "scope_requirement": author_fields["scope_requirement"],
                "arg_match_policy": author_fields["arg_match_policy"],
                "gt_evidence_json": author_fields["gt_evidence_json"],
            })

            # Group 2 -- Runtime outputs
            out_row.update({
                "repeat_idx": _REPEAT_IDX + 1,
                "start_ts": _RUN_START_TS or now_iso(),
                "end_ts": now_iso(),
                "llm_model": llm_model,
                "vlm_model": vlm_model,
                "wall_total": timings.get("total", 0.0),
                "llm_total": timings.get("llm", 0.0),
                "vlm_total": timings.get("vlm", 0.0),
                "camera_total": timings.get("camera", 0.0),
                "script_total": timings.get("script", 0.0),
                "final_answer": final_answer,
                "tool_call_order": tool_order_json,
                "actual_tool_calls_json": actual_calls_json,
                "llm_raw": llm_raw_json,
                "llm_readable": "\n".join(_LOG_LINES[-500:]),
            })

            # Group 3 -- Judge (empty placeholders for now)
            out_row.update({
                "judge_error_mode": "",
                "judge_reason": "",
            })

            # Group 4 -- Metrics
            out_row.update({
                "judge_is_correct": "",
                "coverage_satisfied": cov_ok,
                "scope_satisfied": scope_ok,
                "ruler_reward": "",
                "ruler_explanation": "",
            })

            append_result(OUT_CSV, out_row, RESULT_HEADER)

            # advance repeat / row
            _REPEAT_IDX += 1
            if _REPEAT_IDX >= REPEATS:
                _REPEAT_IDX = 0
                _ROW_IDX += 1
            _STATE = 1
            return 0.1

    except Exception as e:
        print("[runner] FATAL error in batch runner:", e)
        traceback.print_exc()
        bpy.ops.wm.quit_blender()
        return None


# Entrypoint: register timer
print("=== Starting SCOPE Batch Evaluation Runner ===")
bpy.app.timers.register(_batch_step, first_interval=0.2)
