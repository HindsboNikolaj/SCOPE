"""Behavioral regression guards without starting Blender or model APIs."""
import ast
import csv
import datetime
import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace
import pytest
from scope.agent import client as C
ROOT = Path(C.__file__).resolve().parents[3]

def functions(relative, names, scope):
    # runner starts Blender at import; isolate its ordinary I/O helpers without running it.
    tree = ast.parse((ROOT / relative).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), relative, 'exec'), scope)
    return scope

def test_runner_resumes_when_runtime_columns_are_added_before_judge_columns(tmp_path):
    def unexpected_quit():
        pytest.fail('compatible header was rejected')
    scope = functions('src/scope/eval/runner.py', {'_upgrade_header', 'prepare_resume'}, {
        'csv': csv, 'os': os, 'Path': Path,
        'bpy': SimpleNamespace(ops=SimpleNamespace(wm=SimpleNamespace(quit_blender=unexpected_quit)))})
    fp = tmp_path / 'old.csv'
    fp.write_text('question_id,repeat_idx,final_answer,judge_is_correct\nQ1,1,yes,True\n')
    header = ['question_id', 'repeat_idx', 'final_answer', 'turn_timings_json', 'judge_is_correct']
    assert scope['prepare_resume'](str(fp), header) == {('Q1', 1)}
    with fp.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == header
    assert rows == [dict(zip(header, ['Q1', '1', 'yes', '', 'True']))]

def test_runner_preserves_preset_application_across_question_states():
    """Exercise actual state 1 -> 2 -> 3 transitions without a Blender event loop."""
    tree = ast.parse((ROOT / 'src/scope/eval/runner.py').read_text())
    author = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'AUTHOR_COLS' for t in n.targets))
    def fields(row):
        result = dict.fromkeys(author, '')
        result.update(question_id=row['question_id'], preset_start=row['preset'],
                      question='test question', multi_step_mode='none', answer_view='current')
        return result
    def unexpected_quit():
        pytest.fail('runner attempted to quit before writing the rows')
    written = []
    applied = []
    homes = []
    agent = SimpleNamespace(set_presets=lambda _: None, ask_iter=lambda *_, **__: iter([('answer', 'final')]),
                            messages=[], model_id='fake-model')
    scope = functions('src/scope/eval/runner.py', {'_batch_step'}, {
        '_STATE': 1, '_AGENT': agent, '_ROWS': [
            {'question_id': 'Q1', 'preset': 'valid'}, {'question_id': 'Q2', 'preset': 'missing'},
            {'question_id': 'Q3', 'preset': ''}],
        '_ROW_IDX': 0, '_REPEAT_IDX': 0, '_DONE_QIDS': set(), '_PRESET_APPLIED': None,
        'field_map': fields, 'json': json, 'traceback': traceback, 'REPEATS': 1,
        'apply_preset': lambda preset: applied.append(preset) or preset == 'valid',
        'prepare_view_for_capture': lambda: None, 'now_iso': lambda: '2026-01-01T00:00:00',
        'format_event': str, 'get_vlm_label': lambda: 'fake-vlm',
        'scrape_final_answer': lambda _: 'answer', 'extract_tool_order': lambda _: '[]',
        'extract_actual_tool_calls': lambda _: '[]', 'scope_full_satisfied': lambda *_: True,
        '_take_panorama_provenance': lambda: '', 'RESULT_HEADER': author + ['preset_applied'],
        'OUT_CSV': 'unused.csv', 'append_result': lambda _, row, __: written.append(dict(row)),
        'HOME_PER_TASK': True,
        'blender_tools': SimpleNamespace(reset_home=lambda: homes.append(len(applied))),
        'bpy': SimpleNamespace(ops=SimpleNamespace(wm=SimpleNamespace(quit_blender=unexpected_quit)))})
    for _ in range(3):
        assert scope['_STATE'] == 1
        scope['_batch_step']()
        assert scope['_STATE'] == 2
        scope['_batch_step']()
        assert scope['_STATE'] == 3
        scope['_batch_step']()
    assert applied == ['valid', 'missing']
    assert [r['preset_applied'] for r in written] == [True, False, None]
    assert [r['question_id'] for r in written] == ['Q1', 'Q2', 'Q3']
    # Home is reset once per question, after that question's preset (v2 runs did the same).
    assert homes == [1, 2, 2]


def _direct_loop(steps, rows, row_idx, failed=False):
    """Run the runner's direct-loop driver with a scripted _batch_step; return the exit code."""
    exits = []
    def fake_exit(code):
        exits.append(code)
    delays = list(steps)
    scope = functions('src/scope/eval/runner.py', {'_drive_by_loop', '_direct_loop_exit_code'}, {
        '_batch_step': lambda: delays.pop(0), '_ROWS': rows, '_ROW_IDX': row_idx,
        '_RUN_FAILED': [failed], '_time': SimpleNamespace(sleep=lambda _: None),
        'os': SimpleNamespace(_exit=fake_exit), 'sys': __import__('sys')})
    scope['_drive_by_loop']()
    assert not delays, 'driver stopped before the batch said it was finished'
    return exits


def test_direct_loop_exits_blender_after_the_last_row():
    # No window manager: quit_blender() never takes effect, so the driver must exit itself,
    # or the serial pipeline waits forever before the judge (fresh-install report, fix 4).
    assert _direct_loop([0.2, 0.1, None], rows=[{}, {}], row_idx=2) == [0]


def test_direct_loop_exits_nonzero_after_a_fatal_error():
    assert _direct_loop([0.2, None], rows=[{}, {}], row_idx=0, failed=True) == [1]
    assert _direct_loop([None], rows=[{}, {}], row_idx=1) == [1]

class Message:
    def __init__(self, content='', tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
    def model_dump(self):
        return {'role': 'assistant', 'content': self.content, 'tool_calls': self.tool_calls}

@pytest.mark.parametrize('iterator', [False, True])
@pytest.mark.parametrize('failure', ['malformed_json', 'non_object_json', 'generator'])
def test_one_bad_tool_call_is_reported_and_agent_can_finish(monkeypatch, iterator, failure):
    def broken_generator(**_):
        yield {'progress': 'started'}
        raise RuntimeError('tool failed while yielding')
    tool = broken_generator if failure == 'generator' else lambda **_: {'result': 'ok'}
    monkeypatch.setattr(C, 'TOOL_FUNCTIONS', {'test_tool': tool})
    args = '{' if failure == 'malformed_json' else '[]' if failure == 'non_object_json' else '{}'
    replies = [Message(tool_calls=[{'id': 'a', 'function': {'name': 'test_tool', 'arguments': args}}]),
               Message('finished')]
    agent = C.AgentClient.__new__(C.AgentClient)
    agent.messages, agent._seeded, agent.temperature = [], True, 0.0
    agent.model_id = 'fake-model'
    agent._chat_create = lambda *_, **__: SimpleNamespace(choices=[SimpleNamespace(message=replies.pop(0))])
    if iterator:
        assert list(agent.ask_iter('test'))[-1] == ('finished', 'final')
    else:
        assert agent.ask('test')[0] == 'finished'
    errors = [m['content'] for m in agent.messages if m['role'] == 'tool' and 'Error:' in m['content']]
    assert len(errors) == 1
