# -*- coding: utf-8 -*-
"""
client.py — SCOPE AgentClient: OpenAI-compatible chat+tools client

Synchronous `.ask(...)` that executes tool calls in-loop and returns
(final_text, messages, timings, call_tree).

Environment overrides:
  AGENT_API_BASE    (default: "http://localhost:11434/v1" for Ollama)
  AGENT_MODEL_ID    (optional; if omitted we auto-pick from /models)
  AGENT_API_KEY     (default: "ollama")
  AGENT_AUTO_PICK   (optional; if truthy and multiple models served, pick first)
  AGENT_TOOL_CHOICE (default: "auto"; set "none" to omit tool_choice)
"""

from __future__ import annotations
import os, re, json, time, logging, copy
from typing import Dict, Any, Iterable, Tuple, Optional
from openai import OpenAI

from .thinking import ThinkingMode, thinking_mode_for_model

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("scope.agent")

# ─── Tool schema & bindings ───────────────────────────────────────────────────

TOOLS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools", "schema.json")
with open(TOOLS_PATH, "r", encoding="utf-8") as f:
    TOOL_DEFS = json.load(f)

def _get_blender_tools():
    """Lazy import of blender_tools (requires PIL + bpy — only available inside Blender)."""
    from ..tools import blender_tools as _bt
    return _bt

def _load_tool_functions():
    """Lazy-load blender_tools so 'import scope' works without PIL/bpy."""
    bt = _get_blender_tools()
    return {
        td["function"]["name"]: getattr(bt, td["function"]["name"])
        for td in TOOL_DEFS
    }

TOOL_FUNCTIONS: dict | None = None  # populated on first use inside Blender

try:
    from ..tools.vlm_clients import create_vlm_from_env
except Exception:
    create_vlm_from_env = None

# ─── Utilities ────────────────────────────────────────────────────────────────

_REASONING_TAGS = (
    (r"<think>.*?</think>", re.S|re.I),
    (r"<analysis>.*?</analysis>", re.S|re.I),
    (r"<reasoning>.*?</reasoning>", re.S|re.I),
)

def _strip_reasoning(text: str | None) -> str:
    if not text:
        return ""
    out = text
    for pattern, flags in _REASONING_TAGS:
        out = re.sub(pattern, "", out, flags=flags)
    return out.strip()

def _is_gen_like(x):
    return hasattr(x, "__iter__") and not isinstance(x, (dict, str, bytes))

# ─── AgentClient ──────────────────────────────────────────────────────────────

class AgentClient:
    """SCOPE agent with synchronous tool execution for Blender simulation."""

    def __init__(
        self,
        model_id: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        presets: Optional[list[str]] = None,
    ):
        self.last_timings: dict | None = None
        self.last_messages: list[dict] | None = None
        self.last_model_id: str | None = None

        self.base_url = base_url or os.getenv("AGENT_API_BASE", "http://localhost:11434/v1")
        self.api_key  = api_key  or os.getenv("AGENT_API_KEY", "ollama")
        self.temperature = temperature
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self._tool_choice_pref = (os.getenv("AGENT_TOOL_CHOICE", "auto") or "auto").strip().lower()
        if self._tool_choice_pref not in ("auto", "none"):
            self._tool_choice_pref = "auto"

        # Bind VLM for simulation tools (lazy — only runs inside Blender)
        if create_vlm_from_env is not None and os.getenv("VLM_AUTO_INIT", "1") in ("1", "true", "yes"):
            try:
                vlm = create_vlm_from_env()
                _get_blender_tools().set_vlm(vlm)
                log.info(f"[vlm] bound simulation tools to {vlm.name}")
            except Exception as e:
                log.warning(f"[vlm] auto-bind failed: {e}")

        self._presets_override = self._clean_presets(presets)
        self.model_id = self._resolve_model_id(model_id or os.getenv("AGENT_MODEL_ID"))

        self.messages: list[dict] = []
        self._seeded = False
        self.system_prompt = system_prompt or self._build_system_prompt()

    def _clean_presets(self, presets: Optional[list[str]]) -> Optional[list[str]]:
        if presets is None:
            return None
        cleaned = [p.strip() for p in presets if isinstance(p, str) and p.strip()]
        seen, out = set(), []
        for p in cleaned:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def set_presets(self, presets: Optional[list[str]]):
        self._presets_override = self._clean_presets(presets) if presets is not None else None
        self.system_prompt = self._build_system_prompt()
        self._seeded = False

    def _build_system_prompt(self) -> str:
        global TOOL_FUNCTIONS
        if TOOL_FUNCTIONS is None:
            TOOL_FUNCTIONS = _load_tool_functions()
        if self._presets_override is not None:
            presets = self._presets_override
        else:
            try:
                out = TOOL_FUNCTIONS["get_presets"]()
                presets = (out.get("presets") or []) if isinstance(out, dict) else []
                if not isinstance(presets, list):
                    presets = []
                else:
                    presets = self._clean_presets(presets) or []
            except Exception:
                presets = []

        sp = (
            "You are a SCOPE PTZ camera agent (simulation mode). "
            "You are a multi-turn agent; users may ask you to repeat actions.\n"
        )
        if presets:
            sp += "Available presets: " + ", ".join(presets) + ".\n"
        else:
            sp += "There are currently no presets defined.\n"

        sp += (
            "\nFollow multi-step and conditional instructions in order.\n"
            "Only narrate completed actions backed by tool results.\n"
            "Do not announce future actions; report completed ones.\n"
            "If unsure what to do, ask a clarifying question.\n"
        )
        return sp

    def _resolve_model_id(self, provided: Optional[str]) -> str:
        if provided and str(provided).strip():
            return str(provided).strip()
        try:
            ids = [m.id for m in self.client.models.list().data]
            if not ids:
                raise RuntimeError("No models served by this base URL")
            if len(ids) == 1:
                log.info(f"[models] auto-selecting: {ids[0]}")
                return ids[0]
            if str(os.getenv("AGENT_AUTO_PICK", "")).strip():
                log.info(f"[models] AGENT_AUTO_PICK set, using: {ids[0]}")
                return ids[0]
            raise RuntimeError(f"Multiple models available, set AGENT_MODEL_ID: {ids}")
        except Exception as e:
            raise RuntimeError(f"Failed to resolve model id: {e}") from e

    def _seed(self, enable_thinking: bool = False, reasoning_level: Optional[str] = None):
        mode = thinking_mode_for_model(self.model_id)

        if mode == ThinkingMode.TOGGLE:
            self.messages.append({"role": "system", "content": ("/think" if enable_thinking else "/no_think")})
        elif mode == ThinkingMode.ALWAYS:
            self.messages.append({"role": "system", "content": "/think"})
        elif mode == ThinkingMode.NEVER:
            self.messages.append({"role": "system", "content": "/no_think"})
        elif mode == ThinkingMode.LEVELS:
            lvl = (reasoning_level or "").strip().lower()
            if lvl in {"off","low","medium","high"}:
                self.messages.append({"role": "system", "content": f"Reasoning: {lvl}"})
            elif enable_thinking:
                self.messages.append({"role": "system", "content": "Reasoning: medium"})

        self.messages.append({"role": "system", "content": self.system_prompt})
        self._seeded = True

    def warmup_once(self):
        try:
            self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1, temperature=0.0,
            )
        except Exception:
            pass

    def _chat_create(self, messages: list[dict], *, temperature: float):
        kwargs = dict(model=self.model_id, messages=messages, tools=TOOL_DEFS, temperature=temperature)
        if self._tool_choice_pref == "auto":
            kwargs["tool_choice"] = "auto"
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as e:
            emsg = str(getattr(e, "message", "")) or str(e)
            if '"auto" tool choice requires' in emsg or "--enable-auto-tool-choice" in emsg:
                kwargs.pop("tool_choice", None)
                return self.client.chat.completions.create(**kwargs)
            raise

    # ── One-shot (blocking) ───────────────────────────────────────────────────

    def ask(
        self,
        prompt: str,
        *,
        reset_history: bool = False,
        enable_thinking: bool = False,
        reasoning_level: Optional[str] = None,
    ) -> Tuple[str, list[dict], Dict[str, float], Dict[str, Any]]:
        global TOOL_FUNCTIONS
        if TOOL_FUNCTIONS is None:
            TOOL_FUNCTIONS = _load_tool_functions()
        if reset_history:
            self.messages.clear(); self._seeded = False
        if not self._seeded:
            self._seed(enable_thinking, reasoning_level)

        self.messages.append({"role": "user", "content": prompt})
        llm_time = vlm_time = script_time = camera_time = 0.0
        children = []

        while True:
            t0 = time.time()
            resp = self._chat_create(self.messages, temperature=self.temperature)
            llm_time += (time.time() - t0)
            msg = resp.choices[0].message.model_dump()
            self.messages.append(msg)

            calls = msg.get("tool_calls") or []
            if not calls:
                text = _strip_reasoning(msg.get("content", "") or "")
                total = round(llm_time + vlm_time + script_time + camera_time, 3)
                return (
                    text,
                    self.messages,
                    {"total": total, "llm": round(llm_time, 3), "vlm": round(vlm_time, 3),
                     "script": round(script_time, 3), "camera": round(camera_time, 3)},
                    {"name": "session", "duration": total, "children": children},
                )

            for call in calls:
                name = call["function"]["name"]
                args = json.loads(call["function"].get("arguments") or "{}")
                tool_fn = TOOL_FUNCTIONS.get(name)
                if not tool_fn:
                    self.messages.append({
                        "role": "tool",
                        "content": json.dumps({"error": f"unknown tool {name}"}),
                        "tool_call_id": call.get("id"),
                    })
                    continue

                out = tool_fn(**args)

                if _is_gen_like(out):
                    gen = out
                    final_payload = None
                    while True:
                        try:
                            _ = next(gen)
                        except StopIteration as stop:
                            final_payload = stop.value
                            break
                    out = final_payload if isinstance(final_payload, dict) else {"result": final_payload}

                t = (out or {}).get("timings", {}) if isinstance(out, dict) else {}
                vlm_time    += float(t.get("vlm", 0.0))
                script_time += float(t.get("script", 0.0))
                camera_time += float(t.get("camera", 0.0))

                self.messages.append({
                    "role": "tool",
                    "content": json.dumps((out or {}).get("result", "") if isinstance(out, dict) else out),
                    "tool_call_id": call.get("id"),
                })

    # ── Iterator for batch runner ─────────────────────────────────────────────

    def ask_iter(
        self,
        prompt: str,
        *,
        reset_history: bool = False,
        enable_thinking: bool = False,
        reasoning_level: Optional[str] = None,
    ) -> Iterable[Any]:
        global TOOL_FUNCTIONS
        if TOOL_FUNCTIONS is None:
            TOOL_FUNCTIONS = _load_tool_functions()
        if reset_history:
            self.messages.clear(); self._seeded = False
        if not self._seeded:
            self._seed(enable_thinking, reasoning_level)
        self.messages.append({"role": "user", "content": prompt})

        llm_time = vlm_time = script_time = camera_time = 0.0

        while True:
            t0 = time.time()
            resp = self._chat_create(self.messages, temperature=self.temperature)
            llm_time += (time.time() - t0)
            msg = resp.choices[0].message.model_dump()
            self.messages.append(msg)
            yield msg

            calls = msg.get("tool_calls") or []
            if not calls:
                text = _strip_reasoning(msg.get("content", "") or "")
                total = round(llm_time + vlm_time + script_time + camera_time, 3)
                self.last_timings = {
                    "total": total, "llm": round(llm_time, 3),
                    "vlm": round(vlm_time, 3), "script": round(script_time, 3),
                    "camera": round(camera_time, 3),
                }
                self.last_messages = copy.deepcopy(self.messages)
                self.last_model_id = self.model_id
                yield (text, 'final')
                return

            for call in calls:
                name = call["function"]["name"]
                args = json.loads(call["function"].get("arguments") or "{}")
                tool_fn = TOOL_FUNCTIONS.get(name)
                if not tool_fn:
                    tool_msg = {
                        "role": "tool",
                        "content": json.dumps({"error": f"unknown tool {name}"}),
                        "tool_call_id": call.get("id"),
                    }
                    self.messages.append(tool_msg)
                    yield tool_msg
                    continue

                out = tool_fn(**args)

                if _is_gen_like(out):
                    gen = out
                    while True:
                        try:
                            step = next(gen)
                            tool_msg = {"role": "tool", "content": json.dumps(step), "tool_call_id": call.get("id")}
                            self.messages.append(tool_msg)
                            yield tool_msg
                        except StopIteration as stop:
                            out = stop.value if isinstance(stop.value, dict) else {"result": stop.value}
                            break

                if isinstance(out, dict):
                    t = out.get("timings", {}) or {}
                    vlm_time    += float(t.get("vlm", 0.0))
                    script_time += float(t.get("script", 0.0))
                    camera_time += float(t.get("camera", 0.0))

                tool_msg = {
                    "role": "tool",
                    "content": json.dumps((out or {}).get("result", "") if isinstance(out, dict) else out),
                    "tool_call_id": call.get("id"),
                }
                self.messages.append(tool_msg)
                yield tool_msg


def create_default_agent() -> AgentClient:
    """Create an AgentClient with default settings from environment."""
    return AgentClient()
