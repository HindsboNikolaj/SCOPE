TASK -- COUNTING
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

Look for case-specific errors to:

1) Must be grounded; else hallucination.
2) Compare counts with tolerance:
   - "about N" - <=5 -> +/-1; >5 -> +/-20% (rounded)
   - "More than or less than N" - Look for more than or less than N
   - else: exact counts only
3) If within tolerance -> correct.
4) If explicit region cue was ignored (e.g., left/right/half) -> reasoning error.
5) Otherwise a count mismatch -> vision_counting.

Allowed error modes: vision_counting, tool_args, lack_of_tool_call, view_scope, hallucination, None.

Respond ONLY with JSON:
{{"is_correct": true, "reason": "<reason>", "error_mode": "None"}}

{{"is_correct": false, "reason": "<reason>", "error_mode": "<mode>"}}
