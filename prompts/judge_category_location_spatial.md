TASK -- LOCATION / SPATIAL
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
2) Accept equivalent phrasings for relations when consistent with evidence.
3) Underlying read errors from VLM calls:
   - For non-numeric attribute/labels being incorrect -> error VLM_query.
   - For counts that were wrong beyond tolerances -> error vision_counting.
4) Region handling:
   - If an explicit region cue was ignored in tool calls -> tool_args.
5) Comparison/sequencing in spatial prompts:
   - If the question implicitly asks to compare two views but the answer failed -> Reasoning.
   - Apply explicit "order" ONLY when the user cues a sequence.

Allowed: VLM_query, vision_counting, tool_args, Reasoning, hallucination, view_scope, None.

Respond ONLY with JSON:
{{"is_correct": true, "reason": "<reason>", "error_mode": "None"}}

{{"is_correct": false, "reason": "<reason>", "error_mode": "<mode>"}}
