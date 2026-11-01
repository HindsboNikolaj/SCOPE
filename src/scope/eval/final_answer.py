"""What the judge grades: the agent's final answer and its trajectory, nothing else.

A saved row carries the whole conversation (``llm_raw``): system prompt, every assistant turn
with its reasoning, every tool call and every tool result. The benchmark asks whether the
agent's *answer* is right and whether it *did* what the question asked. So the judge sees two
things only:

* the final answer: the text of the last assistant message, with reasoning removed. Some
  serving stacks put the model's planning notes into the content field before the real answer
  (Muse-Glimmer writes "We should narrate completed actions ... <|eom|>assistant <answer>").
  Only the text after the last end-of-message marker counts. Reasoning fields, marked thinking
  blocks and tool outputs never become the final answer.
* the trajectory: each tool call in order with its arguments and what the tool returned. That is
  the evidence for grounding (did a tool see what the answer claims?) and for the action (did
  the camera actually move the way the question asked?). The assistant's prose between calls
  is left out, so leaked planning cannot earn or lose credit.

A terminal assistant tool call with no text counts as a response for presence: a pure camera
command can be answered by doing it. A reply with neither final text nor a final tool call has
no answer and fails mechanically (``answer_present``).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

_REASONING_TAGS = ("think", "thinking", "analysis", "reasoning")
_HARMONY_FINAL = re.compile(r"<\|channel\|>\s*final\s*<\|message\|>", re.I)
_HARMONY_OTHER = re.compile(r"<\|channel\|>\s*(?:analysis|thought|commentary)\b.*$", re.S | re.I)
_GEMMA_THOUGHT = re.compile(r"<\|channel>\s*(?:analysis|thought|commentary)\b.*?(?:<channel\|>|$)",
                            re.S | re.I)
# Muse-Glimmer's chat template sometimes leaves its own turn separator in the content:
# "<planning notes><|eom|><|start|>assistant to=user<|message|><answer>". The answer is what
# follows the last one.
_END_OF_MESSAGE = re.compile(r"<\|eo[mt]\|>\s*(?:<\|start\|>)?\s*(?:assistant\b)?", re.I)
_RECIPIENT = re.compile(r"^\s*(?:assistant\s+)?to=[A-Za-z_][\w.]*\s*(?=<\|)")
_SPECIAL = re.compile(r"</?\|?(?:channel|message|start|end|return|eom|eot)\|?>", re.I)


def final_text(content: Optional[str]) -> str:
    """Strip reasoning and serving artefacts from one assistant message's content."""
    if content is None:
        return ""
    if not isinstance(content, str):
        raise TypeError(f"unknown content representation: {type(content).__name__}")
    text = content
    for name in _REASONING_TAGS:
        text = re.sub(rf"<{name}>.*?(?:</{name}>|$)", "", text, flags=re.S | re.I)
    # A dangling close tag means the reasoning started before the content field did.
    for name in _REASONING_TAGS:
        parts = re.split(rf"</{name}>", text, flags=re.I)
        text = parts[-1]
    finals = list(_HARMONY_FINAL.finditer(text))
    if finals:
        text = text[finals[-1].end():]
    else:
        text = _HARMONY_OTHER.sub("", text)
    text = _GEMMA_THOUGHT.sub("", text)
    ends = list(_END_OF_MESSAGE.finditer(text))
    if ends:
        text = text[ends[-1].end():]
    text = _RECIPIENT.sub("", text)
    return _SPECIAL.sub("", text).strip()


def conversation(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = (row.get("llm_raw") or "").strip()
    if not raw:
        return []
    msgs = json.loads(raw)
    if not isinstance(msgs, list):
        raise ValueError("llm_raw is not a message list")
    return msgs


def tool_name(name: Optional[str]) -> str:
    """gpt-oss can leak a channel marker into the name ("ptz_adjust<|channel|>commentary").
    The agent still runs the tool, so the name is what precedes the marker."""
    return re.split(r"<\|", name or "")[0].strip()


def _calls(msg: Dict[str, Any]) -> List[Tuple[str, Any, Optional[str]]]:
    out = []
    for c in msg.get("tool_calls") or []:
        fn = c.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                pass
        out.append((tool_name(fn.get("name")), args, c.get("id")))
    legacy = msg.get("function_call")
    if isinstance(legacy, dict) and legacy.get("name"):
        args = legacy.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                pass
        out.append((tool_name(legacy["name"]), args, None))
    return out


def answer(row: Dict[str, Any]) -> Tuple[bool, str, str]:
    """Return (present, final answer text, how it was found)."""
    msgs = conversation(row)
    if not msgs:
        return False, "", "no_transcript"
    last = msgs[-1]
    if last.get("role") != "assistant":
        return False, "", "no_terminal_assistant"
    text = final_text(last.get("content"))
    if text:
        return True, text, "terminal_content"
    if _calls(last):
        return True, "", "terminal_tool_call"
    return False, "", "empty_terminal_final"


def answer_present(row: Dict[str, Any]) -> bool:
    return answer(row)[0]


def tool_steps(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every tool call in order, with its result when the transcript has one."""
    msgs = conversation(row)
    results = {}
    for m in msgs:
        if m.get("role") == "tool" and m.get("tool_call_id"):
            results[m["tool_call_id"]] = m.get("content")
    steps = []
    pending: List[Dict[str, Any]] = []
    for m in msgs:
        if m.get("role") == "assistant":
            for name, args, cid in _calls(m):
                step = {"name": name, "arguments": args, "result": results.get(cid) if cid else None}
                steps.append(step)
                if cid is None:
                    pending.append(step)
        elif m.get("role") == "tool" and not m.get("tool_call_id") and pending:
            pending.pop(0)["result"] = m.get("content")
    return steps


def _short(value: Any, limit: int) -> str:
    if value is None:
        return "(no result recorded)"
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    try:
        decoded = json.loads(text)
        if isinstance(decoded, str):
            text = decoded
    except (ValueError, TypeError):
        pass
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit] + f" ... [{len(text) - limit} more characters]"


def trajectory_text(row: Dict[str, Any], result_limit: int = 1200) -> str:
    """The trajectory as the judge reads it: numbered calls, arguments and results."""
    steps = tool_steps(row)
    if not steps:
        return "(the agent made no tool calls)"
    lines = []
    for i, s in enumerate(steps, 1):
        args = json.dumps(s["arguments"], ensure_ascii=False, sort_keys=True)
        lines.append(f"[{i}] {s['name']}({args})\n    -> {_short(s['result'], result_limit)}")
    return "\n".join(lines)
