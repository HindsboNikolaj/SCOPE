# Thinking-token injection

SCOPE prepends a short control directive to the system messages so the agent
model knows whether to emit chain-of-thought before tool calls. The exact
literal injected depends on the model's `ThinkingMode` (see
`src/scope/agent/thinking.py`):

| Mode | Injected literal (system message) | When |
| --- | --- | --- |
| `TOGGLE` (Qwen3 family) | `/think` | `enable_thinking=True` |
| `TOGGLE` | `/no_think` | `enable_thinking=False` |
| `ALWAYS` | `/think` | every call |
| `NEVER` | `/no_think` | every call |
| `LEVELS` (gpt-oss family) | `Reasoning: off` \| `Reasoning: low` \| `Reasoning: medium` \| `Reasoning: high` | per `reasoning_level` arg |
| `LEVELS`, no level given but `enable_thinking=True` | `Reasoning: medium` | fallback |

The directive is appended as a separate `{"role": "system"}` message immediately
before the main system prompt (`agent_system_prompt.md`). See
`AgentClient._seed()` in `src/scope/agent/client.py`.
