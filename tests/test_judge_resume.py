"""Behavioral regression guards without starting Blender or model APIs."""
import ast
import csv
import datetime
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from scope.agent import client as C
from scope.eval import judge as J
from scope.tools import frame_probe as P
ROOT = Path(C.__file__).resolve().parents[3]

def functions(relative, names, scope):
    # runner starts Blender at import; isolate its ordinary I/O helpers without running it.
    tree = ast.parse((ROOT / relative).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), relative, 'exec'), scope)
    return scope

def write_rows(path, rows):
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def run_judge(monkeypatch, inp, out):
    monkeypatch.setattr('sys.argv', ['judge', '-i', str(inp), '-o', str(out), '--concurrency', '1', '--model', 'test', '--base-url', 'http://invalid.test/v1'])
    monkeypatch.setattr(J, 'build_messages', lambda _: [])
    monkeypatch.setattr(J, 'precheck', lambda _: None)  # these rows carry no transcript to check
    monkeypatch.setattr(J, 'call_judge', lambda *_: {'is_correct': True, 'reason': 'fake verdict'})
    J.main()

def test_judge_row_identity_includes_repeat():
    assert J._row_key({'question_id': 'Q1', 'repeat_idx': '1'}) != J._row_key({'question_id': 'Q1', 'repeat_idx': '2'})

def test_judge_resume_keeps_latest_usable_duplicate(monkeypatch, tmp_path):
    inp, out = tmp_path / 'input.csv', tmp_path / 'output.csv'
    row = {'question_id': 'Q1', 'question': 'one', 'final_answer': 'yes'}
    write_rows(inp, [row])
    write_rows(out, [dict(row, judge_is_correct='True'), dict(row, judge_is_correct='')])
    run_judge(monkeypatch, inp, out)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    assert rows[0]['judge_is_correct'] == 'True'

def test_judge_checkpoint_preserves_previously_graded_later_rows(monkeypatch, tmp_path):
    inp, out = tmp_path / 'input.csv', tmp_path / 'output.csv'
    rows = [{'question_id': f'Q{i}', 'question': f'question {i}', 'final_answer': 'yes'} for i in range(31)]
    write_rows(inp, rows)
    write_rows(out, [dict(r, judge_is_correct='True') for r in rows[25:]])
    real_replace = os.replace
    def interrupt_after_checkpoint(src, dst):
        real_replace(src, dst)
        if str(src).endswith('.checkpoint'):
            raise SystemExit('simulated interruption after checkpoint publication')
    monkeypatch.setattr(J.os, 'replace', interrupt_after_checkpoint)
    with pytest.raises(SystemExit, match='simulated interruption'):
        run_judge(monkeypatch, inp, out)
    saved = list(csv.DictReader(out.open()))
    assert len(saved) == 31
    assert {r['question_id'] for r in saved} == {r['question_id'] for r in rows}
    assert all(r['judge_is_correct'] == 'True' for r in saved)
