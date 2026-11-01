"""Exercise partitioning, interrupted outputs and resume through a fake Docker boundary."""
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("run_fast", Path(__file__).resolve().parents[1] / "scripts/run_fast.py")
fast = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fast)


def dataset():
    return [{"question_id": str(i), "file_location": f"scenes/{scene}/{scene}.blend", "question": "Count objects"}
            for i, scene in enumerate(["whitechapel"] * 3 + ["book-nook"] * 5 +
                                       ["city-street"] * 5 + ["postwar-city"] * 3)]


def test_shards_cover_every_question_once_and_keep_each_world_loaded():
    rows = dataset()
    shards = fast.plan(rows)
    assert len(shards) == 6
    assert sorted(r["question_id"] for s in shards for r in s["rows"]) == sorted(r["question_id"] for r in rows)
    assert all(len({r["file_location"] for r in s["rows"]}) == 1 for s in shards)
    with pytest.raises(ValueError, match="unique"):
        fast.plan(rows + rows[:1])
    with pytest.raises(ValueError, match="relative"):
        fast.plan([{"question_id": "bad", "file_location": "../scene.blend"}])


def test_merge_checks_unique_repeat_keys_and_preserves_answer_after_failed_retry(tmp_path):
    rows = dataset()[:1]
    shard = fast.plan(rows)[0]
    fields = ["question_id", "repeat_idx", "final_answer"]
    fast.write_csv(tmp_path / f"{shard['id']}.csv", fields, [
        {"question_id": "0", "repeat_idx": "1", "final_answer": "one\nobject"},
        {"question_id": "0", "repeat_idx": "1", "final_answer": ""},
        {"question_id": "0", "repeat_idx": "2", "final_answer": "two"}])
    report = fast.merge([shard], tmp_path, rows, 2)
    assert report == {"expected": 2, "rows": 2, "missing": 0, "failed_answers": 0, "duplicate_attempts": 1}
    assert fast.read_csv(tmp_path / "raw_results.csv")[1][0]["final_answer"] == "one\nobject"


def test_launch_resume_and_input_change_guard(tmp_path, monkeypatch):
    rows = dataset()
    questions, scenes, presets = tmp_path / "questions.csv", tmp_path / "scenes", tmp_path / "presets"
    fast.write_csv(questions, list(rows[0]), rows)
    presets.mkdir()
    for row in rows:
        path = scenes / Path(row["file_location"]).relative_to("scenes")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setenv("AGENT_MODEL_ID", "test-planner")
    monkeypatch.setenv("VLM_MODEL", "test-vision")
    monkeypatch.setenv("VLM_MODEL_URL", "http://localhost:2020")
    monkeypatch.setenv("AGENT_API_KEY", "test-only-placeholder")
    calls = []
    def run(cmd, **kw):
        assert cmd[:2] == ["docker", "run"]
        settings = dict(cmd[i+1].split("=", 1) for i, arg in enumerate(cmd[:-1])
                        if arg == "-e" and "=" in cmd[i+1])
        assert "AGENT_API_KEY" in cmd  # pass name, never interpolate the secret
        assert "test-only-placeholder" not in " ".join(cmd)
        mounts = [cmd[i+1] for i, arg in enumerate(cmd[:-1]) if arg == "-v"]
        output_dir = Path(next(m[:-5] for m in mounts if m.endswith(":/run")))
        shard_rows = fast.read_csv(output_dir / Path(settings["QUESTIONS_CSV"]).name)[1]
        result = [{**r, "repeat_idx": "1", "final_answer": "two"} for r in shard_rows]
        fast.write_csv(output_dir / Path(settings["OUT_CSV"]).name, list(result[0]), result)
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    args = ["--questions", str(questions), "--out", str(tmp_path / "run"),
            "--scenes", str(scenes), "--presets", str(presets)]
    monkeypatch.setattr(fast, "ROOT", Path(__file__).resolve().parents[1])
    def output(cmd, **kw):
        if cmd[0] == "docker":
            return "sha256:test-image\n"
        return "test-source-revision\n"
    monkeypatch.setattr(subprocess, "check_output", output)
    assert fast.main(args) == 0
    assert len(calls) == 6
    assert len(fast.read_csv(tmp_path / "run/raw_results.csv")[1]) == 16
    assert fast.main(args) == 0
    assert len(calls) == 6  # completed shards are reused
    scene = scenes / "whitechapel/whitechapel.blend"
    scene.write_bytes(b"changed scene")
    with pytest.raises(SystemExit):
        fast.main(args)
    scene.write_bytes(b"")
    assert len(calls) == 6
    monkeypatch.setenv("AGENT_MODEL_ID", "different-model")
    with pytest.raises(SystemExit):
        fast.main(args)
    assert len(calls) == 6


def test_dry_run_does_not_create_output(tmp_path):
    rows = dataset()
    questions = tmp_path / "questions.csv"
    fast.write_csv(questions, list(rows[0]), rows)
    assert fast.main(["--questions", str(questions), "--out", str(tmp_path / "absent"), "--dry-run"]) == 0
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("url", ["", "photon", "local", "HTTP://localhost:2020"])
def test_launcher_requires_shared_http_vision(tmp_path, monkeypatch, url):
    rows = dataset()
    questions = tmp_path / "questions.csv"
    fast.write_csv(questions, list(rows[0]), rows)
    monkeypatch.setenv("AGENT_MODEL_ID", "test-planner")
    monkeypatch.setenv("VLM_MODEL", "Moondream2")
    monkeypatch.setenv("VLM_MODEL_URL", url)
    with pytest.raises(SystemExit) as result:
        fast.main(["--questions", str(questions), "--out", str(tmp_path / "absent")])
    assert result.value.code == 2
    assert not (tmp_path / "absent").exists()


def test_full_pipeline_invokes_existing_judge_then_metrics_and_propagates_failure(tmp_path, monkeypatch):
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        assert kwargs["check"] is True
        assert str(fast.ROOT / "src") in kwargs["env"]["PYTHONPATH"]
    monkeypatch.setattr(subprocess, "run", run)
    assert fast.judge_and_report(tmp_path) >= 0
    assert calls[0][2] == "scope.eval.judge"
    assert calls[1][2:4] == ["scope.eval.metrics", "report"]
    assert str(tmp_path / "raw_results.csv") in calls[0]
    def fail(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)
    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        fast.judge_and_report(tmp_path)


def fake_launch(tmp_path, monkeypatch, rows, pass_presets=True):
    """Run main() against a fake Docker boundary; return every docker run command."""
    questions, scenes, presets = tmp_path / "questions.csv", tmp_path / "scenes", tmp_path / "presets"
    fast.write_csv(questions, list(rows[0]), rows)
    presets.mkdir(exist_ok=True)
    for row in rows:
        path = scenes / Path(row["file_location"]).relative_to("scenes")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setenv("AGENT_MODEL_ID", "test-planner")
    monkeypatch.setenv("VLM_MODEL", "test-vision")
    monkeypatch.setenv("VLM_MODEL_URL", "http://localhost:2020")
    calls = []
    def run(cmd, **kw):
        settings = dict(cmd[i+1].split("=", 1) for i, arg in enumerate(cmd[:-1])
                        if arg == "-e" and "=" in cmd[i+1])
        out = Path(next(m[:-5] for m in (cmd[i+1] for i, a in enumerate(cmd[:-1]) if a == "-v")
                        if m.endswith(":/run")))
        shard_rows = fast.read_csv(out / Path(settings["QUESTIONS_CSV"]).name)[1]
        result = [{**r, "repeat_idx": "1", "final_answer": "two"} for r in shard_rows]
        fast.write_csv(out / Path(settings["OUT_CSV"]).name, list(result[0]), result)
        calls.append(cmd)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "check_output",
                        lambda cmd, **kw: "sha256:test-image\n" if cmd[0] == "docker" else "rev\n")
    code = fast.main(["--questions", str(questions), "--out", str(tmp_path / "run"),
                      "--scenes", str(scenes)] + (["--presets", str(presets)] if pass_presets else []))
    return code, calls


def option(cmd, flag):
    return cmd[cmd.index(flag) + 1]


@pytest.mark.parametrize("gpu, prefix", [(None, None), ("5", "alice-scope"),
                                         ("GPU-00000000-1111-2222-3333-444444444444", None)])
def test_capture_card_and_container_names_come_from_env(tmp_path, monkeypatch, gpu, prefix):
    for name, value in (("SCOPE_GPU_DEVICE", gpu), ("SCOPE_CONTAINER_PREFIX", prefix)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    code, calls = fake_launch(tmp_path, monkeypatch, dataset())
    assert code == 0 and len(calls) == 6
    names = [option(c, "--name") for c in calls]
    assert len(set(names)) == 6
    assert all(n.startswith((prefix or "scope") + "-") for n in names)
    assert {option(c, "--gpus") for c in calls} == {f"device={gpu or '0'}"}


def test_container_prefix_must_be_a_docker_name(tmp_path, monkeypatch):
    monkeypatch.setenv("SCOPE_CONTAINER_PREFIX", "bad name")
    with pytest.raises(SystemExit) as result:
        fake_launch(tmp_path, monkeypatch, dataset())
    assert result.value.code == 2


def test_default_presets_prefer_the_in_image_install(tmp_path, monkeypatch):
    host = tmp_path / "host-blender/presets/camera"
    host.mkdir(parents=True)
    monkeypatch.setattr(fast, "HOST_PRESETS", host)
    monkeypatch.setattr(fast, "ROOT", tmp_path / "repo")
    assert fast.default_presets() == host
    in_image = tmp_path / "repo/.scope-presets/presets/camera"
    in_image.mkdir(parents=True)
    assert fast.default_presets() == in_image


def test_missing_preset_folder_stops_before_any_worker(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fast, "HOST_PRESETS", tmp_path / "absent")
    monkeypatch.setattr(fast, "ROOT", tmp_path / "repo")
    with pytest.raises(SystemExit) as result:
        fake_launch(tmp_path, monkeypatch, dataset(), pass_presets=False)
    assert result.value.code == 2
    assert f"camera preset folder not found: {tmp_path / 'absent'}" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_presets_the_questions_name_must_be_installed(tmp_path, monkeypatch, capsys):
    rows = [{**r, "preset_start": "store-front", "presets_available": '["store-front", "road1"]'}
            for r in dataset()]
    with pytest.raises(SystemExit) as result:
        fake_launch(tmp_path, monkeypatch, rows)
    assert result.value.code == 2
    assert f"camera presets missing from {tmp_path / 'presets'}: road1, store-front" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()
    second = tmp_path / "second"
    second.mkdir()
    (second / "presets").mkdir()
    for name in ("store-front", "road1"):
        (second / f"presets/{name}.py").touch()
    code, calls = fake_launch(second, monkeypatch, rows)
    assert code == 0 and len(calls) == 6
    assert f"camera presets from {second / 'presets'}" in capsys.readouterr().err


STUB_VGLRUN = """#!/bin/sh
case "$*" in
  *glxspheres64*)
    echo "OpenGL Renderer: $FAKE_RENDERER"
    # The real glxspheres64 keeps drawing after printing the renderer unless -f bounds it.
    case " $* " in *" -f "*) exit 0 ;; esac
    exec sleep 600 ;;
  *) echo "$DISPLAY" > "$BLENDER_MARKER" ;;
esac
"""
STUB_XVFB = """#!/bin/sh
echo "$FAKE_DISPLAY" >&3
exec sleep 600
"""


def run_worker(tmp, renderer, real_xvfb=False, tag="1"):
    """Run the worker script outside Docker with stub GPU tools; return (exit code, display)."""
    import os
    import signal
    import stat
    import shutil
    stubs = tmp / "bin"
    stubs.mkdir(exist_ok=True)
    tools = {"vglrun": STUB_VGLRUN, "nvidia-smi": "#!/bin/sh\necho stub GPU\n"}
    if not real_xvfb:
        tools["Xvfb"] = STUB_XVFB
    for name, body in tools.items():
        (stubs / name).write_text(body)
        (stubs / name).chmod(stat.S_IRWXU)
    work = tmp / f"worker-{tag}"
    (work / "run").mkdir(parents=True)
    script = (fast.INNER.replace("/run/", f"{work}/run/").replace("/tmp/xdisplay", f"{work}/xdisplay")
              .replace("cd /repo", f"cd {work}"))
    marker = work / "blender-display"
    env = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}", "FAKE_RENDERER": renderer,
           "FAKE_DISPLAY": "42", "BLENDER_MARKER": str(marker)}
    proc = subprocess.Popen(["bash", "-c", script, "scope-worker", "scene.blend", f"shard-{tag}"],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        proc.communicate(timeout=90)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)  # the background X server
        except ProcessLookupError:
            pass
    return proc.returncode, marker.read_text().strip() if marker.exists() else None


def test_worker_takes_the_display_xvfb_reports_and_starts_blender_on_nvidia(tmp_path):
    assert ":99" not in fast.INNER and "X99" not in fast.INNER
    assert run_worker(tmp_path, "NVIDIA A100-SXM4-80GB/PCIe/SSE2") == (0, ":42")


def test_software_renderer_stops_the_worker_before_blender(tmp_path):
    code, display = run_worker(tmp_path, "llvmpipe (LLVM 15.0.7, 256 bits)")
    assert code != 0 and display is None


@pytest.mark.skipif(not __import__("shutil").which("Xvfb"), reason="needs Xvfb")
def test_two_workers_sharing_a_network_namespace_get_different_displays(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda t: run_worker(tmp_path, "NVIDIA A100", real_xvfb=True, tag=t),
                                ["1", "2"]))
    assert [code for code, _ in results] == [0, 0]
    assert len({display for _, display in results}) == 2
