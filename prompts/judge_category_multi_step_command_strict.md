TASK -- MULTI-STEP COMMAND (STRICT)
Detected region cue in question: {det_region_cue}
Lenient open-ended (from CSV notes): {lenient_open_ended}

Full conversation (truncated OK): {full_conversation}

Summary:
Question: {question}
Expected: {expected_answer}
Final: {final_answer}
Tool calls:
{tool_calls_parsed}

Policy specifics:
- Always normalize case/punctuation/whitespace when comparing text.
- Treat "about N" as +/-1 for numeric comparisons.
- Plural tolerance for nouns (e.g., barricade(s), bike(s)).
- Penalize scope/order/coverage only when the user text cues it explicitly.

Expected exact order: {expected_tool_order_json}
Tool calls:
{tool_calls_parsed}

Policy specifics:
1) Final must be grounded. If not -> "hallucination".
2) Enforce exact order only if the user text cues a sequence.
3) Content mismatches -> "vision_counting"/"VLM_query"/"Reasoning".

Allowed: order, lack_of_tool_call, tool_routing, tool_args, vision_counting, VLM_query, Reasoning, hallucination, None.

Respond ONLY with JSON:
{{"is_correct": true, "reason": "<reason>", "error_mode": "None"}}

{{"is_correct": false, "reason": "<reason>", "error_mode": "<mode>"}}
