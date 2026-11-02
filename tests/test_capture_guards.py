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

def capture_scope():
    return functions('src/scope/tools/blender_tools.py', {'_check_frame'}, {
        '_COLOURS': [], '_BLANK_STATS': {'blank': 0, 'underdrawn': 0, 'wait_ms': 0, 'settled': 0, 'still_flat': 0},
        'DETAIL_WAIT_MS': 400, 'DETAIL_WAIT_STEP_MS': 120,
        '_time': SimpleNamespace(sleep=lambda _: None), '_note_blank_frame': lambda _: None})

def test_retry_uses_fresh_cached_probe_and_records_final_sent_colour(monkeypatch):
    initial, final = (20, 80, 0.5, 0.0), (30, 200, 10.0, 20.0)
    monkeypatch.setattr(P, '_CACHE', {'frame': initial})
    def cached_only(path):
        assert path in P._CACHE, 'draw cached a fresh probe; it must not be discarded and decoded again'
        return P._CACHE[path]
    monkeypatch.setattr(P, 'probe_path', cached_only)
    def draw(path):
        assert path not in P._CACHE, 'stale cache must be removed before drawing'
        P._CACHE[path] = final
    scope = capture_scope()
    scope['_check_frame']('frame', draw)
    assert scope['_BLANK_STATS']['settled'] == 1
    assert scope['_COLOURS'] == [20.0]

def test_capture_names_distinguish_frames_in_same_second():
    times = iter([datetime.datetime(2026, 1, 1, microsecond=10), datetime.datetime(2026, 1, 1, microsecond=20)])
    scope = functions('src/scope/tools/blender_tools.py', {'_now_ts'}, {
        'datetime': SimpleNamespace(now=lambda: next(times))})
    assert scope['_now_ts']() != scope['_now_ts']()
