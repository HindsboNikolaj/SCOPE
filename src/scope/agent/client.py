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
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, Optional
from openai import OpenAI

from .thinking import ThinkingMode, thinking_mode_for_model

# Prompts live at repo-root/prompts so they're visible without digging into the package.
_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"
_AGENT_SYSTEM_PROMPT_TMPL = (_PROMPTS_DIR / "agent_system_prompt.md").read_text(encoding="utf-8")

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
    (r"<think>.*?</think>", re.S | re.I),
    (r"<analysis>.*?</analysis>", re.S | re.I),
    (r"<reasoning>.*?</reasoning>", re.S | re.I),
)

# Harmony channel markers, which several families leak into the recorded answer.
#
# gpt-oss writes <|channel|>analysis<|message|>...<|channel|>final<|message|>answer, and
# gemma-4 writes the pipes reversed, <|channel>thought ... <channel|>answer. tA09's tool calls
# were perfect while every answer began "<|channel>thought\n<channel|>", so the run looked
# healthy and would have been judged on the marker.
#
# Order matters and a flat list of substitutions is not enough: when a `final` channel exists
# the answer is what FOLLOWS it, so deleting the markers in place leaves the analysis text
# concatenated onto the answer.
_FINAL_CHANNEL = re.compile(r"<\|channel\|>\s*final\s*<\|message\|>", re.I)
_ANALYSIS_BLOCK = (
    re.compile(r"<\|channel\|>\s*(?:analysis|thought|commentary)\b.*?<\|message\|>", re.S | re.I),
    re.compile(r"<\|channel>\s*(?:analysis|thought|commentary)\b.*?<channel\|>", re.S | re.I),
)
_BARE_MARKER = re.compile(r"</?\|?(?:channel|message|start|end|return)\|?>", re.I)
# The recipient marker only ever appears immediately before a channel marker. Requiring that
# lookahead is what keeps it from eating ordinary prose: "The tomato is to=the left of the
# bowl" is a legitimate answer, and a looser pattern rewrote it to "is  left of the bowl".
_RECIPIENT = re.compile(r"\bto=[A-Za-z_][\w.]*\s*(?=<\|)", re.I)


def _strip_channels(text: str) -> str:
    m = None
    for m in _FINAL_CHANNEL.finditer(text):
        pass                      # keep the last final-channel marker
    if m:
        text = text[m.end():]
    for pat in _ANALYSIS_BLOCK:
        text = pat.sub("", text)
    text = _RECIPIENT.sub("", text)
    return _BARE_MARKER.sub("", text)


def _strip_reasoning(text: str | None) -> str:
    if not text:
        return ""
    out = text
    for pattern, flags in _REASONING_TAGS:
        out = re.sub(pattern, "", out, flags=flags)
    return _strip_channels(out).strip()


def _is_gen_like(x):
    return hasattr(x, "__iter__") and not isinstance(x, (dict, str, bytes))

# ─── AgentClient ──────────────────────────────────────────────────────────────

# Hard cap on planner turns for one question.
#
# There was none. Both agent loops were `while True`, so the only thing that could stop a
# planner that kept calling tools was running out of context -- about 150 turns and half an
# hour on a 24k model. tA04's city-street shard sat on one question for 143 turns while the
# other three waited. A capped episode is also the honest measurement: "did not converge in N
# turns" is a result, an unbounded loop is a hang.
MAX_TURNS = int(os.environ.get("SCOPE_MAX_TURNS", "40"))


class TurnLimitExceeded(RuntimeError):
    """The planner kept calling tools past MAX_TURNS without producing an answer.

    Raised rather than broken out of: both loops assemble their timings and final text inside
    the "no more tool calls" branch, so falling out of the loop would end the generator with no
    answer and no timings. The runner already records a per-row failure and advances, so this
    reuses that path and the row lands as a real, gradeable non-answer."""



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
        # Set by _seed once the thinking mode for this model is known.
        self._thinking_kwargs: dict | None = None
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

        if presets:
            presets_line = "Available presets: " + ", ".join(presets) + "."
        else:
            presets_line = "There are currently no presets defined."
        return _AGENT_SYSTEM_PROMPT_TMPL.format(presets_line=presets_line)

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
        """Seed exactly one system message.

        The thinking directive and the system prompt used to be appended as two separate system
        messages. Most chat templates tolerate that. The Qwen3.5 family does not: it rejects the
        whole conversation with "System message must be at the beginning", which names the
        position and means the *count*. Served by vLLM that is a 400 on every request, so the
        shard produced zero rows for forty-five minutes and looked like a hang.

        Merging them changes no content and no ordering -- the directive still leads.
        """
        mode = thinking_mode_for_model(self.model_id)

        directive = None
        if mode == ThinkingMode.TOGGLE:
            directive = "/think" if enable_thinking else "/no_think"
        elif mode == ThinkingMode.ALWAYS:
            directive = "/think"
        elif mode == ThinkingMode.NEVER:
            directive = "/no_think"
        elif mode == ThinkingMode.LEVELS:
            lvl = (reasoning_level or "").strip().lower()
            if lvl in {"off", "low", "medium", "high"}:
                directive = f"Reasoning: {lvl}"
            elif enable_thinking:
                directive = "Reasoning: medium"

        content = f"{directive}\n\n{self.system_prompt}" if directive else self.system_prompt
        self.messages.append({"role": "system", "content": content})

        # The per-request half of the same switch. `enable_thinking` is what the Qwen3 and
        # Qwen3.5 templates read; a model without it ignores the field.
        want = enable_thinking if mode in (ThinkingMode.TOGGLE, ThinkingMode.ALWAYS) else False
        if mode == ThinkingMode.ALWAYS:
            want = True
        self._thinking_kwargs = {"enable_thinking": bool(want)}

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

    @staticmethod
    def _reasoned(resp) -> bool:
        """Did this reply carry real reasoning?

        A `/no_think` Qwen3 still emits an empty `<think></think>`, so the tags alone prove
        nothing -- the content between them is the measurement. Some servers put it in a
        separate `reasoning_content` field instead.
        """
        try:
            msg = resp.choices[0].message
        except Exception:
            return False
        rc = getattr(msg, "reasoning_content", None)
        if rc and rc.strip():
            return True
        m = re.search(r"<think>(.*?)</think>", getattr(msg, "content", "") or "", re.S)
        return bool(m and len(m.group(1).strip()) > 8)

    def _chat_create(self, messages: list[dict], *, temperature: float):
        kwargs = dict(model=self.model_id, messages=messages, tools=TOOL_DEFS, temperature=temperature)
        if self._tool_choice_pref == "auto":
            kwargs["tool_choice"] = "auto"

        # Assert the thinking setting on *every* request, not once at the start.
        #
        # `/no_think` in the system message works -- measured on Qwen3-4B, the think block comes
        # back empty on turn one whether the directive sits in its own system message, merged
        # into the system prompt, or appended to the user turn. But it is applied per turn from
        # the recent conversation, and after a few tool results the model starts reasoning again:
        # 99 of 1134 turns in tA01 carried real reasoning, the worst 15,108 characters, always on
        # a later assistant turn. The directive does not decay because the model ignores it; it
        # decays because nothing re-states it.
        #
        # `chat_template_kwargs` is applied by the template at render time on every request, so
        # it cannot lapse. On the same probe it removed the block entirely rather than emptying
        # it. Models whose template has no such flag ignore the field.
        if self._thinking_kwargs:
            kwargs["extra_body"] = {"chat_template_kwargs": dict(self._thinking_kwargs)}
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
        # Per-turn detail. The scalars above are the sums; these keep the shape of the
        # conversation, so a report can say "the planner took 1.2 s on turn 1 and 6.4 s on
        # turn 3" instead of only "8.5 s for the answer".
        llm_turns: list[float] = []
        thinking_turns: list[int] = []   # 1 per turn whose reply carried a non-empty <think>
        vlm_calls: list[float] = []
        script_calls: list[float] = []
        tool_names: list[str] = []
        children = []

        while True:
            if len(llm_turns) >= MAX_TURNS:
                raise TurnLimitExceeded(
                    f"planner made {len(llm_turns)} turns without answering "
                    f"(cap {MAX_TURNS}, set SCOPE_MAX_TURNS to change)")
            t0 = time.time()
            resp = self._chat_create(self.messages, temperature=self.temperature)
            _dt = time.time() - t0
            llm_time += _dt
            llm_turns.append(round(_dt, 3))
            # Whether the directive actually held on *this* turn. It is asserted per request
            # now, but "asked for" and "obeyed" are different columns and only one is measured
            # by reading the reply.
            thinking_turns.append(1 if self._reasoned(resp) else 0)
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
                     "script": round(script_time, 3), "camera": round(camera_time, 3),
                     "llm_turns": llm_turns, "vlm_calls": vlm_calls,
                     "script_calls": script_calls, "tools": tool_names,
                     "thinking_turns": thinking_turns},
                    {"name": "session", "duration": total, "children": children},
                )

            for call in calls:
                # vLLM's harmony parser sometimes leaves a channel marker glued to the tool
                # name -- `query_answer<|channel|>commentary`. The model asked for the right
                # tool; the parser leaked a control token. Trimming it is faithful to what was
                # requested, where treating it as an unknown tool is not.
                name = call["function"]["name"].split("<|", 1)[0].strip()
                args = json.loads(call["function"].get("arguments") or "{}")
                tool_fn = TOOL_FUNCTIONS.get(name)
                if not tool_fn:
                    self.messages.append({
                        "role": "tool",
                        "content": json.dumps({"error": f"unknown tool {name}"}),
                        "tool_call_id": call.get("id"),
                    })
                    continue

                # A planner can call a tool with arguments the tool cannot take:
                # `pan_deg="in"`, `pan_deg="-50%"`, or a nested `arguments={}`. Those raise out
                # of the tool and, before this, out of the whole batch -- one malformed call
                # from a weak planner killed the shard and every remaining question in it went
                # unrun. The model is told what went wrong instead, which is both what a real
                # agent would do and the only way a weak planner can be measured at all.
                try:
                    out = tool_fn(**args)
                except Exception as exc:
                    out = {"result": f"Error: {type(exc).__name__}: {exc}",
                           "timings": {"vlm": 0.0, "script": 0.0}}

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
                vlm_calls.append(round(float(t.get("vlm", 0.0)), 3))
                script_calls.append(round(float(t.get("script", 0.0)), 3))
                tool_names.append(name)

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
        # Per-turn detail. The scalars above are the sums; these keep the shape of the
        # conversation, so a report can say "the planner took 1.2 s on turn 1 and 6.4 s on
        # turn 3" instead of only "8.5 s for the answer".
        llm_turns: list[float] = []
        thinking_turns: list[int] = []   # 1 per turn whose reply carried a non-empty <think>
        vlm_calls: list[float] = []
        script_calls: list[float] = []
        tool_names: list[str] = []

        while True:
            if len(llm_turns) >= MAX_TURNS:
                raise TurnLimitExceeded(
                    f"planner made {len(llm_turns)} turns without answering "
                    f"(cap {MAX_TURNS}, set SCOPE_MAX_TURNS to change)")
            t0 = time.time()
            resp = self._chat_create(self.messages, temperature=self.temperature)
            _dt = time.time() - t0
            llm_time += _dt
            llm_turns.append(round(_dt, 3))
            # Whether the directive actually held on *this* turn. It is asserted per request
            # now, but "asked for" and "obeyed" are different columns and only one is measured
            # by reading the reply.
            thinking_turns.append(1 if self._reasoned(resp) else 0)
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
                    "llm_turns": llm_turns, "vlm_calls": vlm_calls,
                    "script_calls": script_calls, "tools": tool_names,
                    "thinking_turns": thinking_turns,
                }
                self.last_messages = copy.deepcopy(self.messages)
                self.last_model_id = self.model_id
                yield (text, 'final')
                return

            for call in calls:
                # vLLM's harmony parser sometimes leaves a channel marker glued to the tool
                # name -- `query_answer<|channel|>commentary`. The model asked for the right
                # tool; the parser leaked a control token. Trimming it is faithful to what was
                # requested, where treating it as an unknown tool is not.
                name = call["function"]["name"].split("<|", 1)[0].strip()
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

                # A planner can call a tool with arguments the tool cannot take:
                # `pan_deg="in"`, `pan_deg="-50%"`, or a nested `arguments={}`. Those raise out
                # of the tool and, before this, out of the whole batch -- one malformed call
                # from a weak planner killed the shard and every remaining question in it went
                # unrun. The model is told what went wrong instead, which is both what a real
                # agent would do and the only way a weak planner can be measured at all.
                try:
                    out = tool_fn(**args)
                except Exception as exc:
                    out = {"result": f"Error: {type(exc).__name__}: {exc}",
                           "timings": {"vlm": 0.0, "script": 0.0}}

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
                    vlm_calls.append(round(float(t.get("vlm", 0.0)), 3))
                    script_calls.append(round(float(t.get("script", 0.0)), 3))
                    tool_names.append(name)

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
