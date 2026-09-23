TASK -- DESCRIPTOR
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
2) Any fair description is correct (criterion C2). The expected answer is one fair
   description of the scene, not the only acceptable phrasing. Accept a grounded answer that
   describes the same thing in different words, or names different details that are true of the
   scene, as long as it does not contradict the expected answer's core attribute (for example a
   material, colour or scale). "Also accept: X" in the expected answer lists alternates that are
   correct on their own.
3) Minor omissions are fine. Reject when the core attribute is wrong or the description
   contradicts the tool outputs.
4) Wrong core attribute read, color, description -> VLM_query. Ignored explicit region -> tool_args.

Allowed: VLM_query, tool_args, tool_routing, lack_of_tool_call, hallucination, None.

Respond ONLY with JSON:
{{"is_correct": true, "reason": "<reason>", "error_mode": "None"}}

{{"is_correct": false, "reason": "<reason>", "error_mode": "<mode>"}}
