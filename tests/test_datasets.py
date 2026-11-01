"""The shipped dataset files are the files the docs describe, and v2 is the default everywhere."""
import ast
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scope.eval import judge as J

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_fast", ROOT / "scripts/run_fast.py")
fast = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fast)

FILES = {  # path: (rows, sha256), as recorded in benchmark/archive/README.md
    "benchmark/scope_v2.csv": (536, "b45c601ba1d4c6fee61636163084dcde37633b4afb23c6e6a710851295cc72b6"),
    "benchmark/archive/scope_536_v1.csv": (536, "2003c5f40fdac248a58a4b7aca61dc3594a099a6c116befe877c01d0afe2affd"),
    "benchmark/archive/lineage_541_to_536.csv": (41, "dcad0915ff58e8cd5ff5ea3394d1cef0e3158c017b1b24d06fb770cc96505691"),
}


def rows(rel):
    with open(ROOT / rel, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("rel", sorted(FILES))
def test_shipped_files_match_their_recorded_rows_and_hashes(rel):
    n, sha = FILES[rel]
    assert len(rows(rel)) == n
    assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == sha


def test_v2_is_v1_plus_corrections_and_nothing_else():
    out = subprocess.run([sys.executable, str(ROOT / "scripts/apply_corrections.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    v1, v2 = rows("benchmark/archive/scope_536_v1.csv"), rows("benchmark/scope_v2.csv")
    assert [r["question_id"] for r in v1] == [r["question_id"] for r in v2]
    changed = [a["question_id"] for a, b in zip(v1, v2)
               if any(a[k] != b[k] for k in a) or b["also_accept"] != "[]"]
    assert len(changed) == 187
    for r in v2:
        assert isinstance(json.loads(r["also_accept"]), list)


def test_lineage_accounts_for_the_541_row_file_row_by_row():
    v1 = {r["question_id"] for r in rows("benchmark/archive/scope_536_v1.csv")}
    lin = rows("benchmark/archive/lineage_541_to_536.csv")
    removed = [r for r in lin if r["change"] == "removed"]
    inserted = [r for r in lin if r["change"] == "inserted"]
    assert 541 - len(removed) + len(inserted) == len(v1) == 536
    for r in removed:  # each removed row carries the 541-row file's whole row
        old = json.loads(r["row_in_541_file_json"])
        assert len(old) == 25 and old["question_id"] == r["id_in_541_file"]
        assert (old["question"], old["expected_answer"]) == (r["question"], r["expected_answer"])
        assert set(r["v1_ids_same_question"].split()) <= v1
    assert not any(r["row_in_541_file_json"] for r in inserted)
    orphans = [r["id_in_541_file"] for r in removed if not r["v1_ids_same_question"]]
    assert orphans == ["Q_225", "Q_229", "Q_539", "Q_540", "Q_541"]


@pytest.mark.parametrize("flag,expected", [([], 536), (["--dataset", "v1"], 536)])
def test_launcher_defaults_to_v2_and_selects_older_versions_by_flag(tmp_path, capsys, flag, expected):
    assert fast.main(flag + ["--out", str(tmp_path / "absent"), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["expected_rows"] == expected
    assert fast.DATASETS["v2"].name == "scope_v2.csv"
    assert sorted(fast.DATASETS) == ["v1", "v2"]


def test_other_entry_points_default_to_v2():
    assert 'benchmark: "benchmark/scope_v2.csv"' in (ROOT / "configs/agent_config.yaml").read_text()
    sh = (ROOT / "scripts/run_eval_pipeline.sh").read_text()
    assert 'case "${SCOPE_DATASET:-v2}" in' in sh and "scope_v2.csv" in sh
    assert not (ROOT / "benchmark/scope_536.csv").exists()
    assert "v1-541" not in sh and not (ROOT / "benchmark/archive/scope_541_as_released.csv").exists()


def test_runner_records_alternates_next_to_the_expected_answer():
    tree = ast.parse((ROOT / "src/scope/eval/runner.py").read_text())
    cols = next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "AUTHOR_COLS")
    assert cols[cols.index("expected_answer") + 1] == "also_accept"


def test_judge_shows_alternates_only_when_present():
    row = {"question": "Which view has more lamps?", "expected_answer": "The after view.",
           "eval_category": "comparative_relational", "final_answer": "after"}
    # Every October prompt explains the list (criterion C3); the list itself appears only with alternates.
    head = "Also acceptable (any of these also passes):"
    plain = J.build_messages(dict(row, also_accept="[]"))[-1]["content"]
    assert head not in plain
    alt = J.build_messages(dict(row, also_accept=json.dumps(["More lamps after panning"])))[-1]["content"]
    assert head in alt and "- More lamps after panning" in alt


def test_scene_download_never_replaces_the_repository_csv():
    sh = (ROOT / "scripts/03_download_scenes.sh").read_text()
    downloads = [b for b in sh.split("download \"$HF_REPO\"")[1:]]
    assert len(downloads) == 2
    assert all('--include "scenes/*"' in b.split("\n\n")[0] for b in downloads)
    # The Python fallback is limited to scenes/ too.
    assert 'allow_patterns=["scenes/*"]' in sh


def test_smoke_set_follows_its_rule_and_spans_every_world_and_category():
    import collections
    out = subprocess.run([sys.executable, str(ROOT / "scripts/make_smoke_set.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    smoke, v2 = rows("benchmark/scope_v2_smoke.csv"), rows("benchmark/scope_v2.csv")
    assert len(smoke) == 24
    assert all(r in v2 for r in smoke)  # whole v2 rows, unchanged
    assert set(collections.Counter(r["file_location"] for r in smoke).values()) == {6}
    assert set(collections.Counter(r["eval_category"] for r in smoke).values()) == {3}
    assert len({r["eval_category"] for r in smoke}) == len({r["eval_category"] for r in v2}) == 8


def test_per_task_verdicts_add_up_to_the_published_runs():
    import collections
    import re
    per_task = rows("benchmark/results/v2-2026-10/per-task.csv")
    models = rows("benchmark/results/v2-2026-10/models.csv")
    v2 = [r["question_id"] for r in rows("benchmark/scope_v2.csv")]
    assert list(per_task[0]) == ["pairing", "question_id", "panel_verdict"]
    assert {r["panel_verdict"] for r in per_task} == {"pass", "fail"}
    by_run = collections.defaultdict(list)
    for r in per_task:
        by_run[r["pairing"]].append(r)
    assert sorted(by_run) == sorted(m["tag"] for m in models)
    for m in models:
        got = by_run[m["tag"]]
        assert [r["question_id"] for r in got] == v2
        assert sum(r["panel_verdict"] == "pass" for r in got) == int(m["new_passes"])
    # The README's smoke-set comparison for the default setup and its repeat.
    smoke = {r["question_id"] for r in rows("benchmark/scope_v2_smoke.csv")}
    score = {tag: sum(r["panel_verdict"] == "pass" for r in by_run[tag] if r["question_id"] in smoke)
             for tag in ("tB08", "tRPL2")}
    assert next(m for m in models if m["tag"] == "tRPL2")["repeat_of"] == "tB08"
    readme = " ".join((ROOT / "README.md").read_text().split())
    assert re.search(rf"passed {score['tB08']} of 24 in v2 \(`tB08`.*repeat run {score['tRPL2']} of 24", readme)
