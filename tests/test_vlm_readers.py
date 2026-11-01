"""The served-vision readers ported from the October 2026 runtime that answered the v2 runs.

Each reader is keyed to the served model family; the Moondream clients and their image path
are unchanged. No network: QwenVLServer._chat is replaced by a scripted reply.
"""
import importlib

import pytest
from PIL import Image

from scope.tools import vlm_clients as V


def served(model_id, reply, sent=None):
    vlm = V.QwenVLServer(base_url="http://127.0.0.1:9/v1", model_id=model_id, api_key="x")
    def chat(messages, max_tokens=256):
        if sent is not None:
            sent.append(messages)
        return reply
    vlm._chat = chat
    return vlm


IMG = Image.new("RGB", (1920, 1080))


def test_gemma_boxes_are_read_y_first_on_a_1000_grid():
    vlm = served("gemma-4-26b-a4b", '```json\n[{"box_2d": [100, 200, 300, 600], "label": "tree"}]\n```')
    assert vlm.detect(IMG, "tree") == {"objects": [
        {"x_min": 0.2, "y_min": 0.1, "x_max": 0.6, "y_max": 0.3}]}


def test_gemma_points_are_read_y_first():
    vlm = served("google/gemma-4-26B-A4B-it", '[{"point_2d": [250, 750], "label": "car"}]')
    assert vlm.point(IMG, "car") == {"points": [{"x": 0.75, "y": 0.25}]}


def test_qwen3_points_use_the_native_prompt_and_x_first():
    sent = []
    vlm = served("Qwen/Qwen3.6-35B-A3B", '[{"point_2d": [250, 750], "label": "car"}]', sent)
    assert vlm.point(IMG, "car") == {"points": [{"x": 0.25, "y": 0.75}]}
    # One user message in Qwen's own words, no JSON-points system prompt.
    assert [m["role"] for m in sent[0]] == ["user"]
    assert "Point to every car" in sent[0][0]["content"][1]["text"]


def test_qwen3_boxes_are_read_from_bbox_2d():
    vlm = served("qwen36-35b-v", '[{"bbox_2d": [100, 200, 300, 600], "label": "bike"}]')
    assert vlm.detect(IMG, "bike") == {"objects": [
        {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.6}]}


def test_an_unreadable_native_reply_is_flagged_not_counted_zero():
    vlm = served("Qwen/Qwen3.8-27B", "I can see several cars near the corner.")
    out = vlm.point(IMG, "car")
    assert out["points"] == [] and out["unparseable"] is True


def test_an_explicit_none_is_zero_points():
    vlm = served("Qwen/Qwen3.8-27B", "There are none.")
    assert vlm.point(IMG, "car") == {"points": []}


def test_molmo2_points_markup_and_no_boxes():
    vlm = served("allenai/Molmo2-8B", '<points coords="1 1 500 250 2 100 900">cars</points>')
    assert vlm.point(IMG, "car") == {"points": [{"x": 0.5, "y": 0.25}, {"x": 0.1, "y": 0.9}]}
    assert vlm.detect(IMG, "car") == {"objects": []}


def test_thinking_is_turned_off_only_for_qwen3_vision(monkeypatch):
    monkeypatch.delenv("SCOPE_VLM_THINKING", raising=False)
    assert V.vision_thinking_off("Qwen/Qwen3.6-35B-A3B")
    assert V.vision_thinking_off("cyankiwi/Qwen3.8-27B-AWQ-INT4")
    assert not V.vision_thinking_off("google/gemma-4-26B-A4B-it")
    assert not V.vision_thinking_off("Qwen/Qwen2.5-VL-7B-Instruct")


def test_generic_json_points_still_read_for_other_models():
    vlm = served("some-vision-model", '{"points": [{"x": 0.5, "y": 0.5}]}')
    assert vlm.point(IMG, "car") == {"points": [{"x": 0.5, "y": 0.5}]}


def test_pixel_budget_applies_to_served_vision_only(monkeypatch):
    monkeypatch.delenv("SCOPE_VLM_MAX_PIXELS", raising=False)
    mod = importlib.reload(V)
    try:
        big = Image.new("RGB", (3240, 3240))
        # Moondream clients call _to_pil with no budget: frames go through untouched, as before.
        assert mod._to_pil(big).size == (3240, 3240)
        # The served-vision path keeps the 1080p-area budget the v2 runs used.
        w, h = mod._to_pil(big, mod.SERVER_VLM_MAX_PIXELS).size
        assert w * h <= 1920 * 1080 and w == h
        assert mod.VLM_COORD_SPACE == "auto"
    finally:
        importlib.reload(V)


def test_moondream_clients_do_not_use_the_served_vision_readers():
    """The Moondream path is the one the paper and most v2 runs used; the port leaves it alone."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(V))
    ported = {"qwen_native_grounding", "gemma_native_grounding", "is_gemma", "point_adapter",
              "molmo2_points", "clean_vqa", "vision_thinking_off", "strip_reasoning",
              "SERVER_VLM_MAX_PIXELS"}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.startswith("Moondream"):
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            assert not names & ported, (node.name, names & ported)
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "_to_pil":
                    assert len(call.args) == 1 and not call.keywords, node.name
    assert type(V.create_vlm("Moondream2", base_url="http://127.0.0.1:2020")).__name__ == "MoondreamREST"


@pytest.mark.parametrize("kind, model_id", [("gemma", "gemma4-26b-a4b"), ("Qwen", "qwen36-35b-v"),
                                            ("molmo", "molmo2-8b")])
def test_served_vision_client_is_named_after_the_served_model(kind, model_id):
    # The name is the vlm_model column of every result row; a Gemma run once said "Qwen".
    vlm = V.create_vlm(kind, base_url="http://127.0.0.1:9/v1", model_id=model_id, api_key="x")
    assert model_id in vlm.name and vlm.label == vlm.name
    assert kind == "Qwen" or "Qwen" not in vlm.name
