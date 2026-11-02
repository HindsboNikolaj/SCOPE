"""The v2 judge panel (scripts/13_judge_panel.py), on CPU with no model.

``tests/fixtures/judge_panel/`` holds real stored votes from the v2 judging: 41 records of runs
tA01 and tB03, each seat's file cut to the columns the vote reads (verdict, error mode,
question, category and the saved answer). The vote reads the saved answer (``llm_raw``) from the
first seat only, so the other two seats' files leave it empty. The saved answers are synthetic
stand-ins, not the agents' conversations (see the fixture README). ``expected.csv`` is what the October scoring code
(the code that produced benchmark/results/v2-2026-10) made of those records. The full check,
all 29 runs x 536 records x 3 seats, ran on the maintainers' machine against the stored votes,
which are not published: ``13_judge_panel.py vote <judged> --check-against
benchmark/results/v2-2026-10`` matched every run and the overall summary.
"""
import csv
import importlib.util
import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "judge_panel"
spec = importlib.util.spec_from_file_location("judge_panel", ROOT / "scripts" / "13_judge_panel.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_vote_reproduces_october_scoring_on_stored_votes():
    expected = {(r["run"], r["question_id"]): r for r in rows(FIXTURE / "expected.csv")}
    runs = P.find_runs(FIXTURE)
    got = {}
    for run, seats in runs.items():
        for rec in P.vote_run([rows(seats[s]) for s in P.SEAT_NAMES]):
            got[(run, rec["question_id"])] = rec
    assert set(got) == set(expected) and len(got) == 41
    for k, e in expected.items():
        g = got[k]
        assert (g["votes"], g["code_check"], str(g["unparseable_vision"]), str(g["pass"])) == \
               (e["votes"], e["code_check"], e["unparseable_vision"], e["pass"]), k
        assert g["code_check_matches_seats"] == 1, k


def test_fixture_covers_every_rule():
    exp = rows(FIXTURE / "expected.csv")
    kinds = {(e["votes"].count("1"), e["code_check"], e["unparseable_vision"], e["pass"]) for e in exp}
    assert (2, "", "0", "1") in kinds            # 2 of 3 passes
    assert (1, "", "0", "0") in kinds            # 1 of 3 fails
    assert (0, "no_final_answer", "0", "0") in kinds
    assert (0, "action_contradiction", "0", "0") in kinds
    assert (3, "", "1", "0") in kinds            # an unreadable vision reply fails even 3 of 3


def test_vote_cli_writes_tables(tmp_path):
    assert P.main(["vote", str(FIXTURE), "-o", str(tmp_path)]) == 0
    per_run = {r["run"]: r for r in rows(tmp_path / "panel_runs.csv")}
    exp = rows(FIXTURE / "expected.csv")
    for run in ("tA01", "tB03"):
        assert int(per_run[run]["passes"]) == sum(int(e["pass"]) for e in exp if e["run"] == run)
    summary = json.loads((tmp_path / "panel_summary.json").read_text())
    assert summary["records"] == 41 and summary["unparseable_vision"] == 7


def test_missing_verdict_is_never_a_fail(tmp_path):
    for f in FIXTURE.glob("tA01__*.csv"):
        shutil.copy(f, tmp_path / f.name)
    path = tmp_path / "tA01__muse-glimmer-judge.csv"
    data = rows(path)
    data[0]["judge_is_correct"] = ""
    P.write_rows(path, data, list(data[0]))
    with pytest.raises(SystemExit, match="verdicts missing"):
        P.main(["vote", str(tmp_path)])


def test_check_against_reports_differences(tmp_path):
    pub = tmp_path / "pub"
    pub.mkdir()
    cols = ["tag", "tasks_scored", "new_passes", "new_pct", "new_no_final_answer",
            "new_action_contradiction", "new_unparseable_vision"]
    P.write_rows(pub / "models.csv", [dict(zip(cols, ["tA01", 15, 99, 1.0, 0, 3, 0])),
                                      dict(zip(cols, ["tB03", 26, 99, 1.0, 3, 2, 7]))], cols)
    (pub / "summary.json").write_text(json.dumps({"groups": {"all": {"runs": ["tZ99"]}}}))
    assert P.main(["vote", str(FIXTURE), "--check-against", str(pub)]) == 1


def test_seat_settings_are_octobers():
    assert P.SEAT_NAMES == ["gpt-oss-120b", "muse-glimmer-judge", "qwen3-nemotron-genrm"]
    common = {"JUDGE_INPUT": "final+trajectory", "JUDGE_STRUCTURED": "1", "JUDGE_MAX_TOKENS": "8192",
              "JUDGE_REASON_MAXLEN": "0", "JUDGE_TRANSCRIPT_CAP": "24000"}
    assert P.seat_env("gpt-oss-120b") == common
    assert P.seat_env("muse-glimmer-judge") == common
    assert P.seat_env("qwen3-nemotron-genrm") == {**common, "JUDGE_DISABLE_THINKING": "1", "JUDGE_MAX_TOKENS": "3072"}
    last = P.seat_env("qwen3-nemotron-genrm", last_try=True)
    assert last["JUDGE_MAX_TOKENS"] == "8192" and last["JUDGE_REASON_MAXLEN"] == "400" and last["JUDGE_MODES_BOUND"] == "1"


def test_endpoints_from_env_or_config(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "ROOT", tmp_path)        # no repo .env
    for _, _, prefix, _ in P.SEATS:
        for suffix in ("_URL", "_MODEL", "_API_KEY", "_CONCURRENCY"):
            monkeypatch.delenv(prefix + suffix, raising=False)
    (tmp_path / ".env").write_text("PANEL_MUSE_URL=http://muse.example/v1  # comment\nOTHER=1\n")
    monkeypatch.setenv("PANEL_GPT_OSS_URL", "http://gpt.example/v1")
    ep = P.seat_endpoints()
    assert ep["gpt-oss-120b"]["base_url"] == "http://gpt.example/v1"
    assert ep["muse-glimmer-judge"]["base_url"] == "http://muse.example/v1"
    assert ep["qwen3-nemotron-genrm"]["base_url"] == ""
    assert ep["gpt-oss-120b"]["model"] == "gpt-oss-120b"
    cfg = tmp_path / "panel.json"
    cfg.write_text(json.dumps({"qwen3-nemotron-genrm": {"base_url": "http://nem.example/v1", "model": "nem",
                                                        "concurrency": 3}}))
    ep = P.seat_endpoints(cfg)
    assert ep["qwen3-nemotron-genrm"] == {"base_url": "http://nem.example/v1", "model": "nem",
                                          "api_key": "EMPTY", "concurrency": 3}
    with pytest.raises(SystemExit, match="no endpoint for: qwen3-nemotron-genrm"):
        P.main(["judge", "-i", "x.csv", "-o", str(tmp_path)])


# --------------------------------------------------------------------------- judge, end to end

class StubJudge(BaseHTTPRequestHandler):
    """An OpenAI-compatible endpoint that passes every answer, and garbles its first reply to
    any question containing GARBLE so the panel must ask again."""
    bodies, garbled = [], set()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        StubJudge.bodies.append(body)
        user = body["messages"][-1]["content"]
        content = json.dumps({"is_correct": True, "reason": "matches", "error_mode": "None",
                              "all_modes": [], "first_break": "none"})
        if "GARBLE" in user and body["model"] not in StubJudge.garbled:
            StubJudge.garbled.add(body["model"])
            content = "I think it is right"
        reply = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                            "choices": [{"index": 0, "finish_reason": "stop",
                                         "message": {"role": "assistant", "content": content}}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):
        pass


def answer(text, calls=()):
    msgs = [{"role": "user", "content": "task"}]
    for i, (name, args) in enumerate(calls):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "done"})
    msgs.append({"role": "assistant", "content": text})
    return json.dumps(msgs)


def test_judge_asks_each_seat_with_its_settings_and_resumes(tmp_path, monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), StubJudge)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    StubJudge.bodies.clear()
    StubJudge.garbled.clear()
    inp = tmp_path / "tX01.csv"
    base = {"repeat_idx": "1", "eval_category": "counting", "expected_answer": "2", "also_accept": "[]"}
    P.write_rows(inp, [
        {**base, "question_id": "Q_1", "question": "How many cars?", "llm_raw": answer("There are 2 cars.")},
        {**base, "question_id": "Q_2", "question": "How many GARBLE bikes?", "llm_raw": answer("Two bikes.")},
        {**base, "question_id": "Q_3", "question": "How many signs?", "llm_raw": json.dumps([{"role": "user", "content": "t"}])},
        {**base, "question_id": "Q_4", "question": "Zoom out 2x, then count the cars.",
         "llm_raw": answer("2 cars.", [("ptz_adjust", {"zoom_factor": 2})])},
    ])
    cfg = tmp_path / "panel.json"
    cfg.write_text(json.dumps({s: {"base_url": url, "model": f"m-{s}", "concurrency": 2} for s in P.SEAT_NAMES}))
    out = tmp_path / "judged"
    try:
        assert P.main(["judge", "-i", str(inp), "-o", str(out), "--config", str(cfg)]) == 0
        first = len(StubJudge.bodies)
        # Resume: everything has a verdict, so a second run asks nothing.
        assert P.main(["judge", "-i", str(inp), "-o", str(out), "--config", str(cfg)]) == 0
        assert len(StubJudge.bodies) == first
    finally:
        server.shutdown()
    # Q_3 (no final answer) and Q_4 (zoomed in when told to zoom out) never reach a judge;
    # Q_1 once per seat, Q_2 twice per seat (its garbled reply is asked again).
    assert first == 3 * (1 + 2)
    by_model = {}
    for b in StubJudge.bodies:
        by_model.setdefault(b["model"], []).append(b)
    nem, gpt = by_model["m-qwen3-nemotron-genrm"][0], by_model["m-gpt-oss-120b"][0]
    assert nem["chat_template_kwargs"] == {"enable_thinking": False} and nem["max_tokens"] == 3072
    assert "chat_template_kwargs" not in gpt and gpt["max_tokens"] == 8192
    assert all(b["response_format"]["type"] == "json_schema" and b["temperature"] == 0 for b in StubJudge.bodies)
    for seat in P.SEAT_NAMES:
        judged = {r["question_id"]: r for r in rows(out / f"tX01__{seat}.csv")}
        assert {q: judged[q]["judge_is_correct"] for q in judged} == \
               {"Q_1": "True", "Q_2": "True", "Q_3": "False", "Q_4": "False"}
        assert judged["Q_3"]["judge_error_mode"] == "no_final_answer"
        assert judged["Q_4"]["judge_error_mode"] == "action_contradiction"
    recs = {r["question_id"]: r for r in P.vote_run([rows(out / f"tX01__{s}.csv") for s in P.SEAT_NAMES])}
    assert {q: r["pass"] for q, r in recs.items()} == {"Q_1": 1, "Q_2": 1, "Q_3": 0, "Q_4": 0}
    assert all(r["code_check_matches_seats"] for r in recs.values())


def test_changed_answer_is_judged_again(tmp_path):
    dest = tmp_path / "tX01__gpt-oss-120b.csv"
    old = {"question_id": "Q_1", "repeat_idx": "1", "question": "q", "llm_raw": "a", "judge_is_correct": "True"}
    P.write_rows(dest, [old, {**old, "question_id": "Q_2"}])
    assert P.drop_stale(dest, [{**old, "llm_raw": "b"}, {**old, "question_id": "Q_2"}]) == 1
    assert [r["question_id"] for r in rows(dest)] == ["Q_2"]
