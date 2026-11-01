# vlm_clients.py — unified VLM layer for SCOPE
from __future__ import annotations
import base64, io, json, os, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from PIL import Image

# Prompts live at repo-root/prompts so they're visible without digging into the package.
_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"
_VLM_DET_PROMPT      = (_PROMPTS_DIR / "vlm_detect.md").read_text(encoding="utf-8")
_VLM_PT_PROMPT       = (_PROMPTS_DIR / "vlm_point.md").read_text(encoding="utf-8")
_VLM_DET_QWEN_PROMPT = (_PROMPTS_DIR / "vlm_detect_qwen.md").read_text(encoding="utf-8")
_VLM_PT_QWEN_PROMPT  = (_PROMPTS_DIR / "vlm_point_qwen.md").read_text(encoding="utf-8")

@dataclass
class VLMCaps:
    caption: bool = False
    vqa: bool = False
    detect: bool = False
    point: bool = False

class VLMClient:
    name: str = "base"
    caps: VLMCaps = VLMCaps()
    label: str = "VLM"

    def caption(self, image: Image.Image) -> Dict[str, Any]: raise NotImplementedError
    def query(self, image: Image.Image, question: str) -> Dict[str, Any]: raise NotImplementedError
    def detect(self, image: Image.Image, instruction: str) -> Dict[str, Any]: raise NotImplementedError
    def point(self, image: Image.Image, instruction: str) -> Dict[str, Any]: raise NotImplementedError

# ─── utils ────────────────────────────────────────────────────────────────────

# How many pixels a vision model is given, as a budget rather than an edge length.
#
# The scenes render at very different shapes: 1920x1080 for whitechapel and postwar-city,
# 2309x1867 for city-street, and 3240x3240 for book-nook. A cap on the longest edge treats
# those unfairly, because a square frame at edge 1920 is 3.7 megapixels where a 16:9 frame at
# the same edge is 2.1. Budgeting area instead means "no more detail than a 1080p frame,
# whatever shape it is".
#
# Set SCOPE_VLM_MAX_PIXELS=2073600 for a 1080p-area budget. In practice that leaves the two 1080p
# scenes untouched, takes city-street from about 1440p to 1080p, and takes book-nook from a
# square 4K frame to 1440x1440.
#
# Sending more than this made the vision model the dominant cost on the large scenes and not on
# the small ones, which reads as one scene being hard when it is only being photographed bigger.
# The models resize internally, so the extra pixels only cost encode, transfer and decode.
# SCOPE_VLM_MAX_PIXELS=0 sends frames untouched.
VLM_MAX_PIXELS = int(os.environ.get("SCOPE_VLM_MAX_PIXELS", "0") or 0)

# The OpenAI-compatible vision server path (QwenVLServer: Qwen, Gemma, Molmo and any other
# served vision model) keeps the 1080p-area budget it was measured with in the v2 runs (October
# 2026 runtime default) unless SCOPE_VLM_MAX_PIXELS is set. The Moondream clients use
# VLM_MAX_PIXELS above, unchanged.
SERVER_VLM_MAX_PIXELS = int(os.environ.get("SCOPE_VLM_MAX_PIXELS", str(1920 * 1080)) or 0)

# What coordinate space a served VLM answers in, for detect and point.
#
#   auto     (default) 0..1 unless a value exceeds 1, then treat the answer as pixels
#   unit     always 0..1, clamp anything outside
#   pixel    always pixels of the frame we sent, divide by width/height
#   percent  always 0..100
#
# There is no universal answer: the prompt asks for 0..1 and models comply at very different
# rates, so this is per model rather than a constant. `auto` is what the campaign has used.
VLM_COORD_SPACE = (os.environ.get("SCOPE_VLM_COORD_SPACE") or "auto").strip().lower()


def _to_unit(v: float, extent: int) -> float:
    """One coordinate into 0..1, given the frame extent along that axis."""
    if VLM_COORD_SPACE == "unit":
        return _clamp01(v)
    if VLM_COORD_SPACE == "pixel":
        return _clamp01(v / extent) if extent else _clamp01(v)
    if VLM_COORD_SPACE == "percent":
        return _clamp01(v / 100.0)
    return _clamp01(v / extent) if (v > 1.0 and extent) else _clamp01(v)


def _fit_for_vlm(im: Image.Image, budget: Optional[int] = None) -> Image.Image:
    """Scale a frame down to the pixel budget. Never crops: the model sees the whole frame.

    BOX rather than LANCZOS. This runs on every capture, and on a 10.5 megapixel frame LANCZOS
    costs 152 ms against 43 ms for BOX, which is real time taken out of the saving this is
    supposed to produce. Measured against a LANCZOS result the two agree to 0.99977 by
    correlation, and the model resizes again internally regardless, so the extra sharpness is
    spent on pixels nothing looks at. BOX is also the correct filter for pure downscaling: it
    averages the pixels it is discarding rather than sampling through them.
    """
    budget = VLM_MAX_PIXELS if budget is None else budget
    px = im.width * im.height
    if not budget or px <= budget:
        return im
    scale = (budget / px) ** 0.5
    return im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                     Image.BOX)


def _to_pil(img, budget: Optional[int] = None) -> Image.Image:
    if isinstance(img, Image.Image):
        return _fit_for_vlm(img, budget)
    try:
        import numpy as np
        if isinstance(img, np.ndarray):
            return _fit_for_vlm(Image.fromarray(img), budget)
    except Exception:
        pass
    raise TypeError("Unsupported image type for VLM")

def _png_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()

def _b64_data_url(img: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(_png_bytes(img)).decode("utf-8")

def _extract_json(text: str) -> Optional[dict]:
    s = (text or "").strip()
    if s.startswith("{") and s.endswith("}"):
        try: return json.loads(s)
        except Exception: pass
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None

def _clamp01(x: float) -> float:
    try: v = float(x)
    except Exception: return 0.0
    return 0.0 if v < 0 else 1.0 if v > 1 else v

# ─── Moondream REST (self-hosted server) ──────────────────────────────────────

class MoondreamREST(VLMClient):
    def __init__(self, base_url: str):
        self.name = "Moondream2 REST"
        self.label = "Moondream2(REST)"
        self.caps = VLMCaps(caption=True, vqa=True, detect=True, point=True)
        self.url = base_url.rstrip("/")

    def _post_file(self, route: str, pil: Image.Image, data: Dict[str, Any] | None = None) -> Dict[str, Any]:
        files = {"image": ("image.png", _png_bytes(pil), "image/png")}
        r = requests.post(f"{self.url}{route}", files=files, data=data or {}, timeout=120)
        r.raise_for_status()
        return r.json()

    def caption(self, image):
        j = self._post_file("/caption", _to_pil(image))
        return {"caption": j.get("caption", "")}

    def query(self, image, question: str):
        j = self._post_file("/query", _to_pil(image), data={"question": question})
        return {"answer": j.get("answer", "")}

    def detect(self, image, instruction: str):
        j = self._post_file("/detect", _to_pil(image), data={"instruction": instruction})
        objs = []
        for o in (j.get("objects") or []):
            try:
                objs.append({
                    "x_min": float(o["x_min"]), "y_min": float(o["y_min"]),
                    "x_max": float(o["x_max"]), "y_max": float(o["y_max"]),
                })
            except Exception:
                continue
        return {"objects": objs}

    def point(self, image, instruction: str):
        j = self._post_file("/point", _to_pil(image), data={"instruction": instruction})
        pts = j.get("points") or []
        if isinstance(pts, dict): pts = [pts]
        out = []
        for p in pts:
            try: out.append({"x": float(p["x"]), "y": float(p["y"])})
            except Exception: continue
        return {"points": out}

# ─── Moondream hosted (SDK via API key) ───────────────────────────────────────

class MoondreamServer(VLMClient):
    def __init__(self, api_key: Optional[str] = None):
        import moondream
        key = api_key or os.getenv("VLM_API_KEY") or os.getenv("MOONDREAM_API_KEY") or ""
        self.model = moondream.vl(api_key=key)
        self.name = "Moondream hosted"
        self.label = "Moondream(hosted)"
        self.caps = VLMCaps(caption=True, vqa=True, detect=True, point=True)

    def caption(self, image):
        out = self.model.caption(_to_pil(image))
        if isinstance(out, dict): return {"caption": out.get("caption") or out.get("text") or ""}
        return {"caption": str(out)}

    def query(self, image, question: str):
        out = self.model.query(_to_pil(image), question)
        if isinstance(out, dict): return {"answer": out.get("answer") or out.get("text") or ""}
        return {"answer": str(out)}

    def detect(self, image, instruction: str):
        out = self.model.detect(_to_pil(image), instruction)
        objs = []
        if isinstance(out, dict):
            for o in (out.get("objects") or []):
                try:
                    objs.append({
                        "x_min": _clamp01(o["x_min"]), "y_min": _clamp01(o["y_min"]),
                        "x_max": _clamp01(o["x_max"]), "y_max": _clamp01(o["y_max"]),
                    })
                except Exception: continue
        return {"objects": objs}

    def point(self, image, instruction: str):
        out = self.model.point(_to_pil(image), instruction)
        pts = out.get("points") if isinstance(out, dict) else None
        if isinstance(pts, dict): pts = [pts]
        res = []
        for p in (pts or []):
            try: res.append({"x": _clamp01(p["x"]), "y": _clamp01(p["y"])})
            except Exception: continue
        return {"points": res}

# ─── Moondream local (Transformers) ───────────────────────────────────────────

class MoondreamLocal(VLMClient):
    def __init__(self, model_id: str = "vikhyatk/moondream2", revision: Optional[str] = None):
        from transformers import AutoModelForCausalLM
        import torch
        device = os.getenv("VLM_DEVICE", "").strip().lower()
        if not device:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        kwargs: Dict[str, Any] = {
            "revision": revision,
            "trust_remote_code": True,
        }
        if device == "cuda":
            kwargs["device_map"] = {"": "cuda"}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, **kwargs
        )
        if device != "cuda":
            self.model.to(device)
        self.device = device
        self.name = "Moondream local"
        self.label = "Moondream(local)"
        self.caps = VLMCaps(caption=True, vqa=True, detect=True, point=True)

    def caption(self, image):
        out = self.model.caption(_to_pil(image))
        if isinstance(out, dict): return {"caption": out.get("caption") or out.get("text") or ""}
        return {"caption": str(out)}

    def query(self, image, question: str):
        out = self.model.query(_to_pil(image), question)
        if isinstance(out, dict): return {"answer": out.get("answer") or out.get("text") or ""}
        return {"answer": str(out)}

    def detect(self, image, instruction: str):
        out = self.model.detect(_to_pil(image), instruction)
        objs = []
        if isinstance(out, dict):
            for o in (out.get("objects") or []):
                try:
                    objs.append({
                        "x_min": _clamp01(o["x_min"]), "y_min": _clamp01(o["y_min"]),
                        "x_max": _clamp01(o["x_max"]), "y_max": _clamp01(o["y_max"]),
                    })
                except Exception: continue
        return {"objects": objs}

    def point(self, image, instruction: str):
        out = self.model.point(_to_pil(image), instruction)
        pts = out.get("points") if isinstance(out, dict) else None
        if isinstance(pts, dict): pts = [pts]
        res = []
        for p in (pts or []):
            try: res.append({"x": _clamp01(p["x"]), "y": _clamp01(p["y"])})
            except Exception: continue
        return {"points": res}

# ─── Moondream via Photon (local GPU, kestrel runtime) ────────────────────────

class MoondreamPhoton(VLMClient):
    """Moondream running on Photon, its own inference runtime.

    Why this is separate from MoondreamLocal. That class loads the model through
    transformers, which works anywhere and is slow. Photon is Moondream's purpose-built
    runtime, and on an A100 the difference is not incremental:

        transformers path   seconds per call
        Photon              34 ms for a query, 93 ms for a point call

    That changes what the benchmark is bound by. Across a 541-row run the vision model
    stops being the dominant cost and the graphics becomes it, which is worth knowing
    before anyone optimises the wrong half.

    Requirements: `pip install moondream`, a CUDA or Apple Silicon device, and a torch build
    matching the installed driver. A torch compiled for a newer CUDA than the driver makes
    `cuda.is_available()` false, and Photon then refuses with a message about finding no
    accelerator, which reads like missing hardware rather than a version mismatch.

    Configure with:
        VLM_MODEL=moondream  VLM_MODEL_URL=photon  VLM_MODEL_ID=moondream3.1-9B-A2B
    """

    def __init__(self, model_id: Optional[str] = None, max_batch_size: int = 1,
                 device: Optional[str] = None):
        import moondream as md
        model = (model_id or os.getenv("VLM_MODEL_ID") or "moondream3.1-9B-A2B").strip()
        kwargs: Dict[str, Any] = {"local": True, "model": model,
                                  "max_batch_size": max_batch_size}
        dev = device or os.getenv("VLM_DEVICE", "").strip()
        if dev:
            kwargs["device"] = dev
        # The first call loads weights and builds kernels, about two minutes on an A100.
        # Every call after it is in the tens of milliseconds.
        self.model = md.vl(**kwargs)
        self.model_id = model
        self.name = f"Moondream Photon ({model})"
        self.label = "Moondream(photon)"
        self.caps = VLMCaps(caption=True, vqa=True, detect=True, point=True)

    def caption(self, image):
        out = self.model.caption(_to_pil(image))
        if isinstance(out, dict):
            return {"caption": out.get("caption") or out.get("text") or ""}
        return {"caption": str(out)}

    def query(self, image, question: str):
        out = self.model.query(_to_pil(image), question)
        if isinstance(out, dict):
            return {"answer": out.get("answer") or out.get("text") or ""}
        return {"answer": str(out)}

    def detect(self, image, instruction: str):
        out = self.model.detect(_to_pil(image), instruction)
        objs = []
        if isinstance(out, dict):
            for o in (out.get("objects") or []):
                try:
                    objs.append({
                        "x_min": _clamp01(o["x_min"]), "y_min": _clamp01(o["y_min"]),
                        "x_max": _clamp01(o["x_max"]), "y_max": _clamp01(o["y_max"]),
                    })
                except Exception:
                    continue
        return {"objects": objs}

    def point(self, image, instruction: str):
        out = self.model.point(_to_pil(image), instruction)
        pts = out.get("points") if isinstance(out, dict) else None
        if isinstance(pts, dict):
            pts = [pts]
        res = []
        for p in (pts or []):
            try:
                res.append({"x": _clamp01(p["x"]), "y": _clamp01(p["y"])})
            except Exception:
                continue
        return {"points": res}


# ─── Qwen VL server (OpenAI-compatible /v1) ───────────────────────────────────

def qwen_native_grounding(raw: str, key: str, size: int):
    """Read Qwen-VL's own grounding format, which it answers in whatever schema is asked for.

    Qwen3-family vision models reply to a pointing or detection prompt with
    ``[{"point_2d": [x, y], "label": ...}]`` or ``[{"bbox_2d": [x1, y1, x2, y2], ...}]`` on a
    0..1000 grid, not the requested ``{"points": [{"x", "y"}]}``. The generic parser found no
    points, so every ``count_pointing`` call on a Qwen vision server returned 0 (80% of them in
    the September Qwen-vision runs). Each item is read on its own, so a reply cut off at the
    token cap still yields the items it completed.
    """
    num = r"(-?\d+(?:\.\d+)?)"
    pattern = rf'"{key}"\s*:\s*\[\s*' + r"\s*,\s*".join([num] * size) + r"\s*\]"
    out = []
    for m in re.finditer(pattern, raw or ""):
        out.append([_clamp01(float(v) / 1000.0) for v in m.groups()])
    return out


def is_gemma(model_id: Optional[str]) -> bool:
    return "gemma" in (model_id or "").lower()


def gemma_native_grounding(raw: str, key: str, size: int):
    """Read Gemma's own grounding format: ``"box_2d": [y1, x1, y2, x2]`` and ``"point_2d": [y, x]``
    on a 0..1000 grid, y before x. Gemma 4 answers the generic detection prompt this way (tB07 and
    tRPL2, September 2026: "I did not find any bounding box detections for tree" above a reply
    that held a tree box), and the readers built for Qwen either missed the key (box_2d) or would
    swap the axes (point_2d). Returned as [x, y] / [x1, y1, x2, y2] in 0..1, like the Qwen reader.
    """
    out = []
    for item in qwen_native_grounding(raw, key, size):
        if size == 2:
            y, x = item
            out.append([x, y])
        else:
            y1, x1, y2, x2 = item
            out.append([x1, y1, x2, y2])
    return out


def vision_thinking_off(model_id: str) -> bool:
    """Ask the served model not to think on vision calls.

    On by default for Qwen3-family checkpoints, whose chat templates read ``enable_thinking``.
    SCOPE_VLM_THINKING=on keeps the model's default; =off forces the flag for any model.
    """
    mode = os.getenv("SCOPE_VLM_THINKING", "auto").strip().lower()
    if mode == "on":
        return False
    if mode == "off":
        return True
    return bool(re.search(r"qwen3", (model_id or "").lower().replace(".", "").replace("-", "")))


def strip_reasoning(text: Optional[str]) -> str:
    """Drop a reasoning block if a server still returns one inside the content."""
    text = text or ""
    text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.S | re.I)
    if re.search(r"</think>", text, flags=re.I):
        text = re.split(r"</think>", text, flags=re.I)[-1]
    return text.strip()


# Per-model prompt adapters for pointing. Qwen3-family
# vision models point in their own format: asked in their own words, they answer
# [{"point_2d": [x, y], "label": ...}] on a 0..1000 grid. The generic JSON-points system prompt
# made Qwen3.8-27B AWQ INT4 reply "{}", so pointing never worked for it. The native wording is
# sent on the first call; a reply is never re-asked.
QWEN_POINT_ADAPTER = {
    "name": "qwen-native-point",
    "user_text": "Point to every {instruction} in the image. Output their coordinates in JSON format.",
    "system_prompt": None,
    "reply_format": "point_2d on a 0..1000 grid",
}


# Molmo2 (a tool-server error is the harness's to fix and re-run, not the model's failure). Its chat
# template accepts no system message ("Conversation roles must alternate"), so the generic
# JSON-points system prompt got HTTP 400 on 546 count_pointing calls in tB02 and tB03. It points in
# its own markup, <points coords="t i x y i x y ...">label</points> on a 0..1000 grid ("There are
# none." when absent), which nothing read, so 339 replies reached the planner raw.
MOLMO2_POINT_ADAPTER = {
    "name": "molmo2-native-point",
    "user_text": "Point to all {instruction} in the image.",
    "system_prompt": None,
    "reply_format": "<points coords=\"t i x y ...\">label</points> on a 0..1000 grid; 'There are none.' when absent",
}


def point_adapter(model_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """The pointing prompt adapter for a served vision model, or None for the generic prompt."""
    name = (model_id or "").lower().replace(".", "").replace("-", "")
    if re.search(r"qwen3", name):
        return QWEN_POINT_ADAPTER
    if re.search(r"molmo", name):
        return MOLMO2_POINT_ADAPTER
    return None


_MOLMO_MARKUP = re.compile(r"<points?\b([^>]*)>(.*?)</points?>", re.S)
_NONE_REPLY = re.compile(r"^\W*(there (are|is) (none|no\b.*)|none|no\b[^.]*\bfound|\[\s*\])\W*$", re.I)


def molmo2_points(raw: Optional[str]) -> Optional[List[List[float]]]:
    """Points from a Molmo2 reply as [x, y] in 0..1; [] for an explicit none; None if unreadable.

    Reads the three shapes seen in the October 2026 traces: one point ``x y``; a frame id followed
    by ``i x y`` triples; and VideoPoint's several frames separated by ``;``. Points are 0..1000 and
    keyed by their index, so a point listed in two frames counts once. The older Molmo form
    ``<point x="12.5" y="40">`` is on a 0..100 grid.
    """
    raw = (raw or "").strip()
    if _NONE_REPLY.match(raw):
        return []
    found = {}
    order = []
    for m in _MOLMO_MARKUP.finditer(raw):
        attrs = m.group(1)
        c = re.search(r'coords="([^"]*)"', attrs)
        if c:
            for seg in c.group(1).split(";"):
                nums = re.findall(r"\d+(?:\.\d+)?", seg)
                if len(nums) == 2:
                    key = ("single", len(order))
                    found[key] = [_clamp01(float(nums[0]) / 1000.0), _clamp01(float(nums[1]) / 1000.0)]
                    order.append(key)
                    continue
                body = nums[1:] if len(nums) % 3 == 1 else nums
                for i in range(0, len(body) - 2, 3):
                    key = ("i", body[i])
                    if key not in found:
                        order.append(key)
                    found[key] = [_clamp01(float(body[i + 1]) / 1000.0), _clamp01(float(body[i + 2]) / 1000.0)]
            continue
        x, y = re.search(r'\bx\d*="([\d.]+)"', attrs), re.search(r'\by\d*="([\d.]+)"', attrs)
        if x and y:
            key = ("xy", len(order))
            found[key] = [_clamp01(float(x.group(1)) / 100.0), _clamp01(float(y.group(1)) / 100.0)]
            order.append(key)
    if not found:
        return None
    return [found[k] for k in order]


def clean_vqa(text: Optional[str]) -> str:
    """A question's answer without point markup: <points ...>label</points> becomes the label."""
    return _MOLMO_MARKUP.sub(lambda m: m.group(2).strip(), text or "").strip()


class QwenVLServer(VLMClient):
    # The 0..1 variants, matching QwenVLLocal. The generic prompts name no coordinate space,
    # and a served VLM answers in pixels of its own resized input by default -- which the
    # caller then reads as a fraction of the frame, so every zoom lands in the top-left
    # corner. Saying "0..1" in the prompt is the cheap half of the fix; normalising anything
    # that still comes back in pixels is the other half, below.
    SYS_JSON_DET = _VLM_DET_QWEN_PROMPT
    SYS_JSON_POINT = _VLM_PT_QWEN_PROMPT
    def __init__(self, base_url: str, model_id: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"): self.base_url += "/v1"
        self.model_id = model_id or os.getenv("VLM_MODEL_ID") or "Qwen/Qwen2.5-VL-7B-Instruct"
        self.api_key = api_key or os.getenv("VLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
        # Any OpenAI-compatible server lands here (Gemma, Molmo and others too), so name the
        # served model: this is the vlm_model column of every result row.
        self.name = f"OpenAI-compatible vision server ({self.model_id})"
        self.label = self.name
        self.caps = VLMCaps(caption=True, vqa=True, detect=True, point=True)

    def _chat(self, messages: List[Dict[str, Any]], max_tokens: int = 256) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body = {"model": self.model_id, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0}
        # Qwen3.x thinks by default, and the vision budgets here (96-512 tokens) are spent on
        # reasoning: the September 2026 Qwen-vision runs got "The user wants... Let me look
        # carefully at" cut off mid-sentence as the tool result in about 80% of records, so the
        # planner answered with no real vision reply. A vision tool call wants the answer.
        if vision_thinking_off(self.model_id):
            body["chat_template_kwargs"] = {"enable_thinking": False}
        r = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=body, timeout=120)
        r.raise_for_status()
        data = r.json()
        try: content = data["choices"][0]["message"]["content"]
        except Exception: return json.dumps(data)
        return strip_reasoning(content)

    def _img(self, img: Image.Image) -> Dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": _b64_data_url(img)}}

    def caption(self, image):
        pil = _to_pil(image, SERVER_VLM_MAX_PIXELS)
        messages = [{"role":"user","content":[{"type":"text","text":"Describe this image in one concise sentence."}, self._img(pil)]}]
        return {"caption": self._chat(messages, max_tokens=96).strip()}

    def query(self, image, question: str):
        pil = _to_pil(image, SERVER_VLM_MAX_PIXELS)
        messages = [{"role":"user","content":[{"type":"text","text":question}, self._img(pil)]}]
        # Molmo2 answers some questions with point markup inline; the planner gets the words.
        return {"answer": clean_vqa(self._chat(messages, max_tokens=128))}

    def detect(self, image, instruction: str):
        pil = _to_pil(image, SERVER_VLM_MAX_PIXELS)
        if point_adapter(self.model_id) is MOLMO2_POINT_ADAPTER:
            # Molmo2 points; it does not draw boxes, and its chat template rejects the system
            # message this request needs (HTTP 400 "Conversation roles must alternate"), so every
            # zoom_bounding call in tB02 and tB03 came back "Could not locate" without trying the
            # pointing fallback (inbox 090). No boxes here: zoom_bounding then frames its points.
            return {"objects": []}
        system = {"role":"system","content":[{"type":"text","text": self.SYS_JSON_DET}]}
        user   = {"role":"user","content":[self._img(pil), {"type":"text","text": f"Instruction: {instruction}"}]}
        raw = self._chat([system, user], max_tokens=512).strip()
        j = _extract_json(raw) or {"objects": []}
        W, H = pil.size
        objs = []
        if not j.get("objects"):
            boxes = (gemma_native_grounding(raw, "box_2d", 4) if is_gemma(self.model_id)
                     else qwen_native_grounding(raw, "bbox_2d", 4))
            if boxes:
                return {"objects": [{"x_min": b[0], "y_min": b[1], "x_max": b[2], "y_max": b[3]} for b in boxes]}
        for o in (j.get("objects") or []):
            try:
                b = [float(o["x_min"]), float(o["y_min"]), float(o["x_max"]), float(o["y_max"])]
            except Exception:
                continue
            objs.append({"x_min": _to_unit(b[0], W), "y_min": _to_unit(b[1], H),
                         "x_max": _to_unit(b[2], W), "y_max": _to_unit(b[3], H)})
        return {"objects": objs}

    def point(self, image, instruction: str):
        pil = _to_pil(image, SERVER_VLM_MAX_PIXELS)
        adapter = point_adapter(self.model_id)
        if adapter:
            text = adapter["user_text"].format(instruction=instruction)
            raw = self._chat([{"role": "user", "content": [self._img(pil), {"type": "text", "text": text}]}],
                             max_tokens=256).strip()
            if adapter is MOLMO2_POINT_ADAPTER:
                pts = molmo2_points(raw)
                if pts is None:   # a reply in the generic schema is read too
                    j = _extract_json(raw) or {}
                    gp = j.get("points") if isinstance(j, dict) else None
                    if isinstance(gp, dict):
                        gp = [gp]
                    if gp:
                        W, H = pil.size
                        try:
                            pts = [[_to_unit(float(p["x"]), W), _to_unit(float(p["y"]), H)] for p in gp]
                        except (KeyError, TypeError, ValueError):
                            pts = None
            else:
                pts = qwen_native_grounding(raw, "point_2d", 2)
                if not pts and not _NONE_REPLY.match(raw) and "point_2d" not in raw:
                    pts = None
            if pts is None:   # an unparseable reply is a failure, tallied apart
                return {"points": [], "unparseable": True, "raw": raw[:300]}
            return {"points": [{"x": x, "y": y} for x, y in pts]}
        system = {"role":"system","content":[{"type":"text","text": self.SYS_JSON_POINT}]}
        user   = {"role":"user","content":[self._img(pil), {"type":"text","text": f"Instruction: {instruction}"}]}
        raw = self._chat([system, user], max_tokens=256).strip()
        j = _extract_json(raw) or {"points": []}
        pts = j.get("points") or []
        if isinstance(pts, dict): pts = [pts]
        if not pts:
            native = (gemma_native_grounding(raw, "point_2d", 2) if is_gemma(self.model_id)
                      else qwen_native_grounding(raw, "point_2d", 2))
            if native:
                return {"points": [{"x": x, "y": y} for x, y in native]}
        W, H = pil.size
        out = []
        for p in pts:
            try: x, y = float(p["x"]), float(p["y"])
            except Exception: continue
            out.append({"x": _to_unit(x, W), "y": _to_unit(y, H)})
        return {"points": out}

# ─── Qwen VL local (Transformers) ─────────────────────────────────────────────

class QwenVLLocal(VLMClient):
    SYS_JSON_DET = _VLM_DET_QWEN_PROMPT
    SYS_JSON_POINT = _VLM_PT_QWEN_PROMPT
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        from transformers import AutoModelForCausalLM, AutoProcessor
        import torch
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto", device_map="auto")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = "Qwen VL local"
        self.label = f"Qwen(local:{model_id})"
        self.caps = VLMCaps(caption=True, vqa=True, detect=True, point=True)

    def _gen(self, messages: List[dict], max_new_tokens: int = 256) -> str:
        import torch
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
        imgs = []
        for m in messages:
            for part in (m.get("content") or []):
                if isinstance(part, dict) and part.get("type") == "image":
                    imgs.append(part["image"])
        if imgs:
            imt = self.processor(images=imgs, return_tensors="pt")
            for k, v in imt.items():
                if k not in inputs: inputs[k] = v
                else: inputs[k] = torch.cat([inputs[k], v], dim=0)
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        return text.strip()

    def _img(self, img: Image.Image) -> Dict[str, Any]:
        return {"type": "image", "image": img}

    def caption(self, image):
        pil = _to_pil(image)
        msgs = [{"role":"user","content":[self._img(pil), {"type":"text","text":"Describe the image in one concise sentence."}]}]
        return {"caption": self._gen(msgs, max_new_tokens=128)}

    def query(self, image, question: str):
        pil = _to_pil(image)
        msgs = [{"role":"user","content":[self._img(pil), {"type":"text","text": f"Answer concisely: {question}"}]}]
        return {"answer": self._gen(msgs, max_new_tokens=256)}

    def detect(self, image, instruction: str):
        pil = _to_pil(image)
        msgs = [
            {"role":"system","content":[{"type":"text","text": self.SYS_JSON_DET}]},
            {"role":"user","content":[self._img(pil), {"type":"text","text": f"Instruction: {instruction}"}]}
        ]
        data = _extract_json(self._gen(msgs, max_new_tokens=256)) or {"objects": []}
        return {"objects": data.get("objects", [])}

    def point(self, image, instruction: str):
        pil = _to_pil(image)
        msgs = [
            {"role":"system","content":[{"type":"text","text": self.SYS_JSON_POINT}]},
            {"role":"user","content":[self._img(pil), {"type":"text","text": f"Instruction: {instruction}"}]}
        ]
        data = _extract_json(self._gen(msgs, max_new_tokens=256)) or {"points": []}
        pts = data.get("points") or []
        if isinstance(pts, dict): pts = [pts]
        out = []
        for p in pts:
            try: out.append({"x": _clamp01(p["x"]), "y": _clamp01(p["y"])})
            except Exception: continue
        return {"points": out}

# ─── Factory ──────────────────────────────────────────────────────────────────

def create_vlm(kind: Optional[str], base_url: Optional[str] = None,
               model_id: Optional[str] = None, mode_hint: Optional[str] = None,
               api_key: Optional[str] = None) -> VLMClient:
    k = (kind or "").strip().lower()
    url = (base_url or "").strip()
    hint = (mode_hint or "").strip().lower() if mode_hint else ""
    if "moondream" in k or k in ("md2","md3"):
        if url.startswith("http"): return MoondreamREST(url)
        # "photon" selects Moondream's own runtime, roughly two orders of magnitude
        # faster than the transformers path. See MoondreamPhoton.
        if hint == "photon" or url.lower() == "photon":
            return MoondreamPhoton(model_id or None)
        if hint == "local" or url.lower() == "local": return MoondreamLocal(model_id or "vikhyatk/moondream2")
        return MoondreamServer(api_key=api_key)
    if "qwen" in k:
        if url.startswith("http"): return QwenVLServer(base_url=url, model_id=model_id, api_key=api_key)
        return QwenVLLocal(model_id or "Qwen/Qwen2.5-VL-7B-Instruct")
    if url.startswith("http"):
        # Anything else with a URL: assume an OpenAI-compatible vision endpoint.
        return QwenVLServer(base_url=url, model_id=model_id or kind, api_key=api_key)
    raise ValueError(f"Unsupported VLM kind: {kind!r} (no VLM_MODEL_URL to fall back on)")

def create_vlm_from_env() -> VLMClient:
    kind = os.getenv("VLM_MODEL") or os.getenv("VLM_KIND") or "Moondream2"
    url  = os.getenv("VLM_MODEL_URL") or os.getenv("VLM_BASE_URL") or ""
    mid  = os.getenv("VLM_MODEL_ID") or ""
    # Moondream-first precedence: the Moondream branch must not leak an
    # OpenAI key as its Moondream Cloud token. Order: explicit VLM_API_KEY,
    # then the backend-specific key (MOONDREAM_API_KEY), then OPENAI_API_KEY
    # as a last-resort fallback (only meaningful for the Qwen-via-vLLM path).
    key  = os.getenv("VLM_API_KEY") or os.getenv("MOONDREAM_API_KEY") or os.getenv("OPENAI_API_KEY") or None
    hint = None
    if url and url.lower() in ("local","photon","api_key","api-key","apikey","api key"):
        low = url.lower()
        hint = "photon" if low == "photon" else ("local" if "local" in low else "api_key")
        url = ""
    return create_vlm(kind, base_url=url, model_id=mid, mode_hint=hint, api_key=key)

def create_vlm_from_config(cfg: Dict[str, Any]) -> VLMClient:
    kind = cfg.get("vlm_model") or os.getenv("VLM_MODEL") or "Moondream2"
    url  = cfg.get("vlm_model_url") or os.getenv("VLM_MODEL_URL") or ""
    mid  = cfg.get("vlm_model_id") or os.getenv("VLM_MODEL_ID") or ""
    # Same Moondream-first precedence as create_vlm_from_env.
    key  = cfg.get("vlm_api_key") or os.getenv("VLM_API_KEY") or os.getenv("MOONDREAM_API_KEY") or os.getenv("OPENAI_API_KEY") or None
    hint = None
    if url and url.lower() in ("local","photon","api_key","api-key","apikey","api key"):
        low = url.lower()
        hint = "photon" if low == "photon" else ("local" if "local" in low else "api_key")
        url = ""
    return create_vlm(kind, base_url=url, model_id=mid, mode_hint=hint, api_key=key)
