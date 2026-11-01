TASK -- DESCRIPTOR
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

Look for case-specific errors to:

1) Must be grounded; else hallucination.
2) Any fair description is correct (criterion C2). The expected answer is one fair
   description of the scene, not the only acceptable phrasing. Accept a grounded answer that
   describes the same thing in different words, or names different details that are true of the
   scene, as long as it does not contradict the expected answer's core attribute (for example a
   material, colour or scale). The "Also acceptable" list under the expected answer gives
   alternates that are correct on their own.
   C2 does not widen numbers: a count stated in the answer must still match the expected answer
   exactly (or within an explicit "about N" tolerance).
   C2 does not widen what is in the scene: objects or features named in neither the tool outputs
   nor the expected answer are hallucination, even when plausible.
3) Minor omissions are fine. Reject when the core attribute is wrong or the description
   contradicts the tool outputs.
4) Wrong core attribute read, color, description -> VLM_query. Ignored explicit region -> tool_args.

Allowed: vision_request, VLM_query, tool_args, tool_routing, lack_of_tool_call, hallucination, claimed_unperformed_action, admitted_failure, None.
claimed_unperformed_action = the final answer says the agent did, checked or looked at something, but the trajectory has no matching tool call or result (example: "I zoomed in and read the sign" with no zoom or reading call).
admitted_failure = the final answer plainly says the agent could not do, find or answer the task and claims nothing the tool outputs do not support (example: "I could not find a red car in this view."). Incorrect; if it also invents something it is hallucination.
vision_request = the planner asked the vision model the wrong thing: the wrong object, a vague query or the wrong view; a better request would likely have worked. vision_counting and VLM_query are the vision model's own errors on a reasonable request.
Respond ONLY with JSON:
{{"is_correct": true, "reason": "<one line: why the answer is right>", "error_mode": "None", "all_modes": [], "first_break": "none"}}

{{"is_correct": false, "reason": "<one line: why the answer is wrong>", "error_mode": "<the mode that best explains it>", "all_modes": ["<every mode that applies>"], "first_break": "<vision_output, planner or harness>"}}
