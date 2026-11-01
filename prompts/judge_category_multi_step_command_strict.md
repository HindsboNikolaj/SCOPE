TASK -- MULTI-STEP COMMAND (STRICT)
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

Expected exact order: {expected_tool_order_json}
Tool calls:
{tool_calls_parsed}

Policy specifics:
1) Final must be grounded. If not -> "hallucination".
2) Enforce exact order only if the user text cues a sequence.
3) Content mismatches -> "vision_counting"/"VLM_query"/"Reasoning".

Allowed: order, lack_of_tool_call, tool_routing, tool_args, vision_request, vision_counting, VLM_query, Reasoning, hallucination, claimed_unperformed_action, admitted_failure, None.
claimed_unperformed_action = the final answer says the agent did, checked or looked at something, but the trajectory has no matching tool call or result (example: "I zoomed in and read the sign" with no zoom or reading call).
admitted_failure = the final answer plainly says the agent could not do, find or answer the task and claims nothing the tool outputs do not support (example: "I could not find a red car in this view."). Incorrect; if it also invents something it is hallucination.
vision_request = the planner asked the vision model the wrong thing: the wrong object, a vague query or the wrong view; a better request would likely have worked. vision_counting and VLM_query are the vision model's own errors on a reasonable request.
Respond ONLY with JSON:
{{"is_correct": true, "reason": "<one line: why the answer is right>", "error_mode": "None", "all_modes": [], "first_break": "none"}}

{{"is_correct": false, "reason": "<one line: why the answer is wrong>", "error_mode": "<the mode that best explains it>", "all_modes": ["<every mode that applies>"], "first_break": "<vision_output, planner or harness>"}}
