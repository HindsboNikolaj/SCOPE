TASK -- SINGLE CALL
Look for case-specific errors to:

Trajectory (the agent's tool calls in order, each with what the tool returned; the agent's own reasoning is not shown): {full_conversation}

Summary:
Question: {question}
Expected: {expected_answer}
Final: {final_answer}
Tool calls:
{tool_calls_parsed}

Policy specifics:
1) If the task implies a specific tool and it was not called -> "lack_of_tool_call"; wrong tool -> "tool_routing"; wrong args -> "tool_args".
2) If the final claim is not supported by tool evidence -> "hallucination".
3) Scope penalty only if the user text explicitly cues FULL/SWEEP/360/etc. or vice versa.
4) "Also acceptable" under the expected answer lists alternates (criterion C3); an answer matching the
   expected answer or any one alternate is correct, and grounding still applies.

Allowed: lack_of_tool_call, tool_routing, tool_args, hallucination, claimed_unperformed_action, admitted_failure, view_scope, None.
claimed_unperformed_action = the final answer says the agent did, checked or looked at something, but the trajectory has no matching tool call or result (example: "I zoomed in and read the sign" with no zoom or reading call).
admitted_failure = the final answer plainly says the agent could not do, find or answer the task and claims nothing the tool outputs do not support (example: "I could not find a red car in this view."). Incorrect; if it also invents something it is hallucination.
Respond ONLY with JSON:
{{"is_correct": true, "reason": "<one line: why the answer is right>", "error_mode": "None", "all_modes": [], "first_break": "none"}}

{{"is_correct": false, "reason": "<one line: why the answer is wrong>", "error_mode": "<the mode that best explains it>", "all_modes": ["<every mode that applies>"], "first_break": "<vision_output, planner or harness>"}}
