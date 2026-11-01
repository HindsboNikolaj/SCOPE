# Judge-panel fixture

Inputs for `tests/test_judge_panel.py`: 41 records of runs tA01 and tB03 from the v2 judging,
one file per judge seat (`<run>__<seat>.csv`), and `expected.csv`, what the October scoring code
made of them.

**Real:** the question, its category, and each seat's verdict and error mode. The votes and
`expected.csv` are the October results.

**Synthetic:** every `llm_raw` conversation. The agents' real conversations are not published.
Each record carries the smallest conversation that leads the vote to the same result as the
real one:

- the question as the user turn;
- the camera calls the agent made (`ptz_adjust`, `go_to_preset`, `home_action`, `zoom_bounding`)
  with their recorded arguments, so the camera-action check passes or fails as it did;
- where the real run hit an unreadable vision reply, one `count_pointing` call returning that
  error;
- a fixed final answer, "Synthetic final answer.", or no conversation at all for the three
  records that had none (`no_final_answer`).

Other tool calls, tool results, system prompts and the agents' text are left out. The vote reads
`llm_raw` from the first seat (`gpt-oss-120b`) only, so the other seats' files leave it empty.
