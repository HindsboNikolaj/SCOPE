TASK -- COMPARATIVE / RELATIONAL
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
2) Validate underlying reads first:
   - Counts differ beyond tolerance -> vision_counting
   - Attribute/text reads differ -> VLM_query
3) Required comparisons:
   - If the question asks compare A vs B and only one side was observed -> lack_of_tool_call.
   - If both observed but comparison logic wrong -> Reasoning.
4) Region/targeting and scope:
   - Explicit region cue ignored -> tool_args.
   - Scope misuse (FULL vs current) -> view_scope.

Allowed: vision_counting, VLM_query, lack_of_tool_call, Reasoning, tool_args, view_scope, hallucination, None.

Respond ONLY with JSON:
{{"is_correct": true, "reason": "<reason>", "error_mode": "None"}}

{{"is_correct": false, "reason": "<reason>", "error_mode": "<mode>"}}
