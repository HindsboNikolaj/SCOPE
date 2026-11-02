TASK -- COMPARATIVE / RELATIONAL
Detected region cue in question: {det_region_cue}
Lenient open-ended (from CSV notes): {lenient_open_ended}

Trajectory (the agent's tool calls in order, each with what the tool returned; the agent's own reasoning is not shown): {full_conversation}

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
- "Also acceptable" under the expected answer lists alternates (criterion C3). Each is a correct answer on
  its own: an answer that matches the expected answer or any one alternate is correct. Alternates never
  widen a count or a text reading beyond what they state, and grounding still applies.

Grade only the claim the question asks for (criterion C1):
- The verdict is decided by the comparison itself: which side, view, preset or object has more,
  or how A relates to B. If the final answer states the same comparison as the expected answer,
  it is correct.
- Wrong details the answer volunteers beyond that claim (for example counts given alongside a
  correct "the left side has more") do NOT make it incorrect. Mention them in the reason only.
- Grounding still applies: if the tool outputs contradict the stated comparison, it is
  hallucination.
- A tie, a hedge between options, or a comparison of different things (for example the full
  panorama against the current view) is not the same comparison as the expected answer.
- A before/after comparison needs both views observed in the trajectory: a count or description
  of only one of them is lack_of_tool_call, even if the stated direction is right.

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

Allowed: vision_request, vision_counting, VLM_query, lack_of_tool_call, Reasoning, tool_args, view_scope, hallucination, claimed_unperformed_action, admitted_failure, None.
claimed_unperformed_action = the final answer says the agent did, checked or looked at something, but the trajectory has no matching tool call or result (example: "I zoomed in and read the sign" with no zoom or reading call).
admitted_failure = the final answer plainly says the agent could not do, find or answer the task and claims nothing the tool outputs do not support (example: "I could not find a red car in this view."). Incorrect; if it also invents something it is hallucination.
vision_request = the planner asked the vision model the wrong thing: the wrong object, a vague query or the wrong view; a better request would likely have worked. vision_counting and VLM_query are the vision model's own errors on a reasonable request.
Respond ONLY with JSON:
{{"is_correct": true, "reason": "<one line: why the answer is right>", "error_mode": "None", "all_modes": [], "first_break": "none"}}

{{"is_correct": false, "reason": "<one line: why the answer is wrong>", "error_mode": "<the mode that best explains it>", "all_modes": ["<every mode that applies>"], "first_break": "<vision_output, planner or harness>"}}
