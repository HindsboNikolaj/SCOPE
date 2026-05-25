TASK -- OCR / IDENTIFICATION
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
2) Normalize case/punctuation; allow typical OCR noise (0<->O, 1<->l, hyphens), but not incorrect readings.
4) If explicit region cue was ignored -> tool_args.
5) If grounded with only minor OCR deviations -> correct; else -> VLM_query.

Allowed: VLM_query, tool_args, lack_of_tool_call, hallucination, None.

Respond ONLY with JSON:
{{"is_correct": true, "reason": "<reason>", "error_mode": "None"}}

{{"is_correct": false, "reason": "<reason>", "error_mode": "<mode>"}}
