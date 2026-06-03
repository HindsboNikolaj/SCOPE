# Prompts

This directory holds the verbatim system prompts SCOPE uses at runtime. They
live at the repo root (not buried in the Python package) so reviewers and
downstream researchers can audit and modify them without reading source code.

The Python modules load these files at import time via
`Path(__file__).resolve().parents[3] / "prompts" / "<file>.md"` and substitute
runtime values through `str.format()`. Edit the `.md` files to change the
prompt; do not edit string literals in the Python source.

## Files

| File | Loaded by | Purpose |
| --- | --- | --- |
| `agent_system_prompt.md` | `src/scope/agent/client.py` | Main system prompt for the SLM planner. Placeholder: `{presets_line}`. |
| `thinking_modes.md` | (docs only) | Explains the `/think`, `/no_think`, and `Reasoning: low\|medium\|high` directives that `AgentClient._seed()` prepends per model. |
| `judge_general.md` | `src/scope/eval/judge.py` | System prompt for the LLM-as-Judge. Sets the evaluation ladder, scope discipline, and error-mode precedence. |
| `judge_category_*.md` (10 files) | `src/scope/eval/judge.py` | Per-category judge user-message templates. Selected by `build_category_key()`. |
| `vlm_detect.md`, `vlm_point.md` | `src/scope/tools/vlm_clients.py` (`QwenVLServer`) | JSON-schema instructions for detection/pointing on a Qwen2.5-VL OpenAI-compatible server (pixel coords). |
| `vlm_detect_qwen.md`, `vlm_point_qwen.md` | `src/scope/tools/vlm_clients.py` (`QwenVLLocal`) | Same, but for the local-Transformers Qwen2.5-VL backend (0..1 normalized coords). |

## Conventions

- Prompts are intentionally preserved byte-for-byte from the original sources,
  including any apparent quirks (e.g., literal `\n` two-character sequences in
  the VLM prompts, missing item 3 in `judge_category_ocr_identification.md`).
- `{double_braces}` are literal in the markdown because Python's `.format()`
  unescapes them. Single-brace fields are runtime substitutions.
- Do not add license headers or extra trailing newlines; the Python code
  compares prompts byte-for-byte in tests.
