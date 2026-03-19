# vlm_clients.py — unified VLM layer for SCOPE
from __future__ import annotations
import base64, io, json, os, re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import requests
from PIL import Image

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

def _to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    try:
        import numpy as np
        if isinstance(img, np.ndarray):
            return Image.fromarray(img)
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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, trust_remote_code=True, device_map={"": "cuda"}
        )
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

# ─── Qwen VL server (OpenAI-compatible /v1) ───────────────────────────────────

class QwenVLServer(VLMClient):
    SYS_JSON_DET = (
        "You are a vision model. Given a user instruction about what to find, "
        "respond with ONLY JSON in this schema:\\n"
        "{\\n"
        '  "objects": [\\n'
        '    {"x_min": <float>, "y_min": <float>, "x_max": <float>, "y_max": <float>}\\n'
        "  ]\\n"
        "}\\n"
        "No extra text."
    )
    SYS_JSON_POINT = (
        "You are a vision model. Return ONLY JSON with points (pixel coords):\\n"
        "{\\n"
        '  "points": [ {"x": <float>, "y": <float>} ]\\n'
        "}\\n"
        "No extra text."
    )
    def __init__(self, base_url: str, model_id: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"): self.base_url += "/v1"
        self.model_id = model_id or os.getenv("VLM_MODEL_ID") or "Qwen/Qwen2.5-VL-7B-Instruct"
        self.api_key = api_key or os.getenv("VLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
        self.name = "Qwen VL server"
        self.label = f"Qwen(server:{self.model_id})"
        self.caps = VLMCaps(caption=True, vqa=True, detect=True, point=True)

    def _chat(self, messages: List[Dict[str, Any]], max_tokens: int = 256) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body = {"model": self.model_id, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0}
        r = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=body, timeout=120)
        r.raise_for_status()
        data = r.json()
        try: return data["choices"][0]["message"]["content"]
        except Exception: return json.dumps(data)

    def _img(self, img: Image.Image) -> Dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": _b64_data_url(img)}}

    def caption(self, image):
        pil = _to_pil(image)
        messages = [{"role":"user","content":[{"type":"text","text":"Describe this image in one concise sentence."}, self._img(pil)]}]
        return {"caption": self._chat(messages, max_tokens=96).strip()}

    def query(self, image, question: str):
        pil = _to_pil(image)
        messages = [{"role":"user","content":[{"type":"text","text":question}, self._img(pil)]}]
        return {"answer": self._chat(messages, max_tokens=128).strip()}

    def detect(self, image, instruction: str):
        pil = _to_pil(image)
        system = {"role":"system","content":[{"type":"text","text": self.SYS_JSON_DET}]}
        user   = {"role":"user","content":[self._img(pil), {"type":"text","text": f"Instruction: {instruction}"}]}
        raw = self._chat([system, user], max_tokens=512).strip()
        j = _extract_json(raw) or {"objects": []}
        objs = []
        for o in (j.get("objects") or []):
            try:
                objs.append({
                    "x_min": float(o["x_min"]), "y_min": float(o["y_min"]),
                    "x_max": float(o["x_max"]), "y_max": float(o["y_max"]),
                })
            except Exception: continue
        return {"objects": objs}

    def point(self, image, instruction: str):
        pil = _to_pil(image)
        system = {"role":"system","content":[{"type":"text","text": self.SYS_JSON_POINT}]}
        user   = {"role":"user","content":[self._img(pil), {"type":"text","text": f"Instruction: {instruction}"}]}
        raw = self._chat([system, user], max_tokens=256).strip()
        j = _extract_json(raw) or {"points": []}
        pts = j.get("points") or []
        if isinstance(pts, dict): pts = [pts]
        out = []
        for p in pts:
            try: out.append({"x": float(p["x"]), "y": float(p["y"])})
            except Exception: continue
        return {"points": out}

# ─── Qwen VL local (Transformers) ─────────────────────────────────────────────

class QwenVLLocal(VLMClient):
    SYS_JSON_DET = (
        "You are a vision model. Given a user instruction about what to find, "
        "respond with ONLY JSON in this schema:\\n"
        "{\\n"
        '  "objects": [\\n'
        '    {"x_min": <float 0..1>, "y_min": <float 0..1>, "x_max": <float 0..1>, "y_max": <float 0..1>}\\n'
        "  ]\\n"
        "}\\n"
        "No extra text."
    )
    SYS_JSON_POINT = (
        "You are a vision model. Return ONLY JSON with points (0..1 normalized):\\n"
        "{\\n"
        '  "points": [ {"x": <float>, "y": <float>} ]\\n'
        "}\\n"
        "No extra text."
    )
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
        if hint == "local" or url.lower() == "local": return MoondreamLocal(model_id or "vikhyatk/moondream2")
        return MoondreamServer(api_key=api_key)
    if "qwen" in k:
        if url.startswith("http"): return QwenVLServer(base_url=url, model_id=model_id, api_key=api_key)
        return QwenVLLocal(model_id or "Qwen/Qwen2.5-VL-7B-Instruct")
    raise ValueError(f"Unsupported VLM kind: {kind!r}")

def create_vlm_from_env() -> VLMClient:
    kind = os.getenv("VLM_MODEL") or os.getenv("VLM_KIND") or "Moondream2"
    url  = os.getenv("VLM_MODEL_URL") or os.getenv("VLM_BASE_URL") or ""
    mid  = os.getenv("VLM_MODEL_ID") or ""
    key  = os.getenv("VLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("MOONDREAM_API_KEY") or None
    hint = None
    if url and url.lower() in ("local","api_key","api-key","apikey","api key"):
        hint = "local" if "local" in url.lower() else "api_key"
        url = ""
    return create_vlm(kind, base_url=url, model_id=mid, mode_hint=hint, api_key=key)

def create_vlm_from_config(cfg: Dict[str, Any]) -> VLMClient:
    kind = cfg.get("vlm_model") or os.getenv("VLM_MODEL") or "Moondream2"
    url  = cfg.get("vlm_model_url") or os.getenv("VLM_MODEL_URL") or ""
    mid  = cfg.get("vlm_model_id") or os.getenv("VLM_MODEL_ID") or ""
    key  = cfg.get("vlm_api_key") or os.getenv("VLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("MOONDREAM_API_KEY") or None
    hint = None
    if url and url.lower() in ("local","api_key","api-key","apikey","api key"):
        hint = "local" if "local" in url.lower() else "api_key"
        url = ""
    return create_vlm(kind, base_url=url, model_id=mid, mode_hint=hint, api_key=key)
