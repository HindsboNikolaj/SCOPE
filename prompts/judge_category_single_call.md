TASK -- SINGLE CALL
Look for case-specific errors to:

Full conversation (truncated OK): {full_conversation}

Summary:
Question: {question}
Expected: {expected_answer}
Final: {final_answer}
Tool calls:
{tool_calls_parsed}

Policy specifics:
1) If the task implies a specific tool and it was not called -> "lack_of_tool_call"; wrong tool -> "tool_routing"; wrong args -> "tool_args".
2) If the final claim is not supported by tool evidence -> "hallucination".
3) Scope penalty only if the user text explicitly cues FULL/SWEEP/360/etc. or vice versa.

Allowed: lack_of_tool_call, tool_routing, tool_args, hallucination, view_scope, None.

Respond ONLY with JSON:
{{"is_correct": true, "reason": "<reason>", "error_mode": "None"}}

{{"is_correct": false, "reason": "<reason>", "error_mode": "<mode>"}}
