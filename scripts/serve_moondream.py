"""Serve one Moondream on one GPU, for many benchmark workers.

Why this exists. The benchmark shards well by scene: four Blender processes, four scenes, four
times the throughput. But each of those processes builds its own VLM, and a local Moondream is
about 20 GB of weights, so four shards want 80 GB of vision model before Blender or the planner
have asked for anything. On a single 80 GB card that does not fit, and on a bigger one it is
still four copies of the same weights answering one question at a time.

Loading it once and answering over HTTP costs a few milliseconds of transfer per call and gives
the shards back to the scheduler. Measured against an in-process Photon, a point call went from
0.85s to 0.9s; the model is doing the same work either way.

This speaks the contract src/scope/tools/vlm_clients.py:MoondreamREST already expects, so
pointing a worker at it needs no code:

    VLM_MODEL=Moondream2  VLM_MODEL_URL=http://<host>:2020

Routes, all multipart with an `image` file part:

    POST /caption                          -> {"caption": str}
    POST /query    question=<str>          -> {"answer": str}
    POST /detect   instruction=<str>       -> {"objects": [{x_min,y_min,x_max,y_max}, ...]}
    POST /point    instruction=<str>       -> {"points": [{x,y}, ...]}
    GET  /health                           -> {"ok": true, "model": str, "served": int}

Usage:

    python3 scripts/serve_moondream.py --port 2020
    VLM_MODEL_ID=moondream3.1-9B-A2B python3 scripts/serve_moondream.py

  --backend photon   Moondream's own runtime, needs CUDA (default when available)
  --backend cloud    the hosted API, needs MOONDREAM_API_KEY; useful for a smoke test
"""
import argparse
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

_MODEL = None
_LOCK = threading.Lock()
_STATS = {"served": 0, "errors": 0, "seconds": 0.0}


def build_model(backend, model_id):
    import moondream as md
    if backend == "cloud":
        key = os.getenv("MOONDREAM_API_KEY") or os.getenv("VLM_API_KEY") or ""
        if not key:
            raise SystemExit("--backend cloud needs MOONDREAM_API_KEY")
        return md.vl(api_key=key), "cloud"
    return md.vl(local=True, model=model_id), f"photon:{model_id}"


def _parts(body, ctype):
    """Pull the image and the form fields out of a multipart body.

    Written by hand rather than pulled from a framework because this script has to run under
    whatever interpreter is nearby, including Blender's, and a dependency that is missing there
    turns a one-file server into an install problem.
    """
    boundary = ctype.split("boundary=")[-1].strip().strip('"').encode()
    image, fields = None, {}
    for chunk in body.split(b"--" + boundary):
        if b"\r\n\r\n" not in chunk:
            continue
        head, _, payload = chunk.partition(b"\r\n\r\n")
        payload = payload.rstrip(b"\r\n-")
        head_s = head.decode("latin1")
        name = ""
        for token in head_s.split(";"):
            token = token.strip()
            if token.startswith("name="):
                name = token[5:].strip('"').split("\r\n")[0]
        if not name:
            continue
        if name == "image":
            image = Image.open(io.BytesIO(payload)).convert("RGB")
        else:
            fields[name] = payload.decode("utf-8", "replace")
    return image, fields


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._json({"ok": True, "model": MODEL_NAME, **_STATS})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        t0 = time.time()
        try:
            n = int(self.headers.get("Content-Length") or 0)
            image, fields = _parts(self.rfile.read(n), self.headers.get("Content-Type", ""))
            if image is None:
                return self._json({"error": "no image part"}, 400)
            route = self.path.rstrip("/")

            # One model, many callers. Photon is not documented as thread safe and the GPU is
            # the bottleneck anyway, so calls are serialised rather than raced.
            with _LOCK:
                if route == "/caption":
                    out = {"caption": _norm(_MODEL.caption(image), "caption")}
                elif route == "/query":
                    out = {"answer": _norm(_MODEL.query(image, fields.get("question", "")), "answer")}
                elif route == "/detect":
                    r = _MODEL.detect(image, fields.get("instruction", ""))
                    out = {"objects": (r or {}).get("objects", [])}
                elif route == "/point":
                    r = _MODEL.point(image, fields.get("instruction", ""))
                    pts = (r or {}).get("points", [])
                    out = {"points": [pts] if isinstance(pts, dict) else pts}
                else:
                    return self._json({"error": f"unknown route {route}"}, 404)
            _STATS["served"] += 1
            _STATS["seconds"] = round(_STATS["seconds"] + time.time() - t0, 2)
            self._json(out)
        except Exception as e:
            _STATS["errors"] += 1
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def _norm(out, key):
    if isinstance(out, dict):
        return out.get(key) or out.get("text") or ""
    return str(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.getenv("MOONDREAM_PORT", "2020")))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--backend", choices=["photon", "cloud"], default="photon")
    ap.add_argument("--model", default=os.getenv("VLM_MODEL_ID", "moondream3.1-9B-A2B"))
    a = ap.parse_args()

    print(f"loading {a.backend} {a.model} ...", flush=True)
    _MODEL, MODEL_NAME = build_model(a.backend, a.model)
    print(f"ready on {a.host}:{a.port}  ({MODEL_NAME})", flush=True)
    print(f"point workers at it with:  VLM_MODEL=Moondream2 "
          f"VLM_MODEL_URL=http://<host>:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
