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

Grade only the claim the question asks for (criterion C1):
- The verdict is decided by the comparison itself: which side, view, preset or object has more,
  or how A relates to B. If the final answer states the same comparison as the expected answer,
  it is correct.
- Wrong details the answer volunteers beyond that claim (for example counts given alongside a
  correct "the left side has more") do NOT make it incorrect. Mention them in the reason only.
- Grounding still applies: if the tool outputs contradict the stated comparison, it is
  hallucination.

Look for case-specific errors to:

1) Must be grounded; else hallucination.
2) Validate underlying reads first, but only reads the question asks for:
   - The question asks for counts and they differ beyond tolerance -> vision_counting
   - The question asks for an attribute/text read and it differs -> VLM_query
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
