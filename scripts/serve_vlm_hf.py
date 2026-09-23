"""Serve any transformers vision model behind a minimal OpenAI /v1/chat/completions.

Why this exists. vLLM is the right way to serve a VLM and it is not always available: the
Qwen3.5 / 3.6 / 3.8 architectures need vLLM 0.17 or newer, every such wheel links against
libcudart.so.13, and a CUDA 12.8 driver cannot load it. Transformers can load those models
today. This is the bridge -- slower per call, but it is the difference between measuring a
model and not measuring it.

It speaks just enough of the OpenAI API for scope.tools.vlm_clients.QwenVLServer:

    POST /v1/chat/completions   {"model":..., "messages":[{role, content:[{type:"text"|"image_url"}]}]}
    GET  /v1/models

so a worker needs no code change:

    VLM_MODEL=qwen  VLM_MODEL_URL=http://<host>:2021  VLM_MODEL_ID=<served-name>

One model, one lock. Requests are serialised because the GPU is the bottleneck and batching
across HTTP would only add a scheduler this does not need.

    python3 scripts/serve_vlm_hf.py --model Qwen/Qwen3.8-27B --served-name qwen38-27b --port 2021
"""
import argparse, base64, io, json, os, re, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

_MODEL = _PROC = None
# How many generate() calls may be in flight. One is the safe default and it is also the reason
# a four-shard benchmark spends most of its time queueing: the GPU is nowhere near busy with a
# small model, and the callers are four separate Blender processes.
THINKING = None      # None = leave the template alone, False = ask it not to reason
FLATTEN_TOOLS = [False]   # flipped on the first template that rejects structured tool turns
REASON_HEADROOM = 512   # extra generation budget spent on reasoning and then discarded
_LOCK = threading.BoundedSemaphore(1)
_STATS = {"served": 0, "errors": 0, "seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
SERVED = "hf-vlm"


def load(model_id, dtype, device_map, trust):
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
    from transformers import AutoConfig
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust)
    td = {"bfloat16": torch.bfloat16, "float16": torch.float16, "auto": "auto"}[dtype]

    # Order matters. AutoModelForCausalLM on a multimodal checkpoint loads the *text* backbone
    # and the processor still emits pixel_values, so generate() dies with "model_kwargs are not
    # used by the model: ['pixel_values', 'image_grid_thw']" -- which reads like a processor bug
    # and is really the wrong class. Load the image-text class whenever the config has vision.
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust)
    multimodal = hasattr(cfg, "vision_config") or hasattr(getattr(cfg, "text_config", None), "vision_config")
    classes = []
    if multimodal:
        for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
            try:
                classes.append(getattr(__import__("transformers", fromlist=[name]), name))
            except (ImportError, AttributeError):
                pass
    classes.append(AutoModelForCausalLM)

    last = None
    for cls in classes:
        try:
            m = cls.from_pretrained(model_id, dtype=td, device_map=device_map,
                                    trust_remote_code=trust)
            print(f"  loaded with {cls.__name__}", flush=True)
            break
        except Exception as e:
            last = e
    else:
        raise last
    m.eval()
    return m, proc


def _as_obj(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except json.JSONDecodeError:
            return a
    return a if a is not None else {}


def _decode_image(url):
    if url.startswith("data:"):
        return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
    raise ValueError("only data: image URLs are accepted; this server does not fetch URLs")


def to_chat(messages, flatten_tools=False):
    """OpenAI messages -> the (chat_messages, images) shape a transformers processor wants.

    `flatten_tools` rewrites the tool exchange into plain assistant/user turns carrying the
    model's own `<tool_call>` / `<tool_response>` markup. Qwen3.5's template rejects the
    structured shape outright -- an assistant message with `tool_calls`, or any `role: "tool"`
    message, comes back as *"TemplateError: System message must be at the beginning"*, which
    names the wrong problem entirely. Since the model emits that markup itself, feeding it back
    the same way is the shape it was trained on.
    """
    out, images = [], []
    seen_system = False
    for m in messages:
        role = m.get("role")
        c = m.get("content")
        # Only the first system message may stay a system message. SCOPE reuses one agent
        # across a shard and re-seeds, so a second system turn can appear mid-conversation, and
        # Qwen3.5's template rejects the whole list with "System message must be at the
        # beginning" -- naming the symptom, not the turn. A later system turn is still
        # instruction, so it is carried as a user turn rather than dropped.
        if role == "system":
            if seen_system:
                role = "user"
            seen_system = True
        # An assistant turn that called tools carries the calls, not prose. Chat templates
        # expect them back in the same place they emitted them.
        if role == "assistant" and m.get("tool_calls"):
            if flatten_tools:
                blocks = "".join(
                    "<tool_call>" + json.dumps({
                        "name": (tc.get("function") or {}).get("name", ""),
                        "arguments": _as_obj((tc.get("function") or {}).get("arguments"))
                    }) + "</tool_call>"
                    for tc in m["tool_calls"])
                out.append({"role": "assistant",
                            "content": [{"type": "text", "text": (c or "") + blocks}]})
            else:
                out.append({"role": role, "content": c or "", "tool_calls": m["tool_calls"]})
            continue
        if role == "tool":
            txt = c if isinstance(c, str) else json.dumps(c)
            if flatten_tools:
                out.append({"role": "user", "content": [
                    {"type": "text", "text": f"<tool_response>{txt}</tool_response>"}]})
            else:
                out.append({"role": "tool", "content": txt})
            continue
        if isinstance(c, str):
            out.append({"role": role, "content": [{"type": "text", "text": c}]})
            continue
        parts = []
        for p in c or []:
            if p.get("type") == "text":
                parts.append({"type": "text", "text": p.get("text", "")})
            elif p.get("type") == "image_url":
                images.append(_decode_image((p.get("image_url") or {}).get("url", "")))
                parts.append({"type": "image"})
        out.append({"role": role, "content": parts})
    return out, images


_TOOL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_XML_FN = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.S)
_XML_PARAM = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.S)


def _coerce(v):
    """"10" -> 10, "true" -> True, "{...}" -> dict. The XML form carries no types."""
    v = v.strip()
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def split_tool_calls(text):
    """Pull tool calls out of a completion, in either shape models actually emit.

    Two formats, both wrapped in <tool_call>:

        {"name": "ptz_adjust", "arguments": {"pan_deg": 10}}          hermes JSON (Qwen3, most)
        <function=ptz_adjust><parameter=pan_deg>10</parameter></function>   XML (Qwen3.5)

    vLLM has parsers for these behind --tool-call-parser; this shim has to do it itself,
    because without tool calls a planner served here can plan but never act. Qwen3.5-0.8B was
    what surfaced the second form: it answered correctly and the call was dropped on the floor
    because only the JSON shape was recognised.
    """
    calls = []
    for block in _TOOL_BLOCK.finditer(text):
        body = block.group(1).strip()
        try:
            j = json.loads(body)
        except json.JSONDecodeError:
            j = None
        if isinstance(j, dict):
            args = j.get("arguments", j.get("parameters", {}))
            calls.append({"name": j.get("name", ""),
                          "arguments": args if isinstance(args, str) else json.dumps(args)})
            continue
        for fn in _XML_FN.finditer(body):
            args = {k: _coerce(v) for k, v in _XML_PARAM.findall(fn.group(2))}
            # Some models wrap the real arguments in a literal `arguments` parameter. Passed
            # through as-is that becomes `take_image(arguments={})`, a TypeError from the tool.
            if set(args) == {"arguments"} and isinstance(args["arguments"], dict):
                args = args["arguments"]
            calls.append({"name": fn.group(1), "arguments": json.dumps(args)})

    out = [{"id": f"call_{i}", "type": "function", "function": c} for i, c in enumerate(calls)]
    return _TOOL_BLOCK.sub("", text).strip(), out


_THINK_CLOSE = re.compile(r"^.*?</think\s*>", re.S)


def strip_reasoning(txt):
    """Drop a reasoning block, including the common case with no opening tag.

    A thinking chat template puts the opening `<think>` in the *generation prompt*, so the
    model only ever emits the closing one. Matching `<think>.*?</think>` therefore matches
    nothing and the whole reasoning pass is handed back as the answer -- which reads as a model
    that cannot follow "answer in one short sentence", and is really a parsing miss.
    """
    return _THINK_CLOSE.sub("", txt, count=1).strip()


def generate(messages, max_tokens, temperature, tools=None):
    import torch
    chat, images = to_chat(messages, flatten_tools=FLATTEN_TOOLS[0])
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    # Qwen3.5-family templates default to a reasoning pass, and this model writes it as plain
    # prose rather than inside <think> tags -- so it lands in the answer ("The user wants a
    # single concise sentence... I need to look carefully at the elements: ..."), which a judge
    # reads as a wrong answer, and it costs most of the generation budget. The flag is ignored
    # by templates that do not have it.
    if THINKING is False:
        kwargs["enable_thinking"] = False
    if tools:
        kwargs["tools"] = tools
    def render(ch):
        return _PROC.apply_chat_template(ch, **kwargs)

    try:
        text = render(chat)
    except Exception as e:
        msg = str(e)
        if not FLATTEN_TOOLS[0] and ("System message must be at the beginning" in msg
                                     or "Can only get item pairs from a mapping" in msg):
            # Qwen3.5's template rejects an assistant turn carrying `tool_calls`, and any
            # `role: "tool"` message, and reports it as a system-message ordering problem --
            # which names the wrong thing entirely. Re-render with the exchange flattened into
            # the model's own <tool_call>/<tool_response> markup, and stay flattened.
            print("  template rejects structured tool messages; flattening from here on",
                  flush=True)
            FLATTEN_TOOLS[0] = True
            chat, images = to_chat(messages, flatten_tools=True)
            text = render(chat)
        elif isinstance(e, TypeError):
            kwargs.pop("enable_thinking", None)
            try:
                text = render(chat)
            except TypeError:
                # A template that does not take `tools` still has to see them, or the model is
                # asked to call functions it was never shown.
                kwargs.pop("tools", None)
                if tools:
                    chat = [{"role": "system", "content": [{"type": "text", "text":
                             "You may call these tools. To call one, emit "
                             "<tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call>.\n"
                             + json.dumps(tools)}]}] + chat
                text = render(chat)
        else:
            raise
    kw = {"text": [text], "return_tensors": "pt", "padding": True}
    if images:
        kw["images"] = images
    inputs = _PROC(**kw).to(_MODEL.device)  # noqa: E501
    n_in = int(inputs["input_ids"].shape[-1])
    # Headroom for the reasoning pass. The caller's max_tokens is an *answer* budget: the SCOPE
    # clients ask for 96 to 512. A thinking model spends that budget reasoning, the budget runs
    # out before it answers, and the caller gets the reasoning instead -- 128 tokens of "The
    # user wants a count of shapes... let me look carefully" and no count. The extra is
    # discarded with the reasoning, so it costs time, not correctness.
    budget = max_tokens + (REASON_HEADROOM if THINKING is not False else 0)
    with torch.no_grad():
        ids = _MODEL.generate(**inputs, max_new_tokens=budget, do_sample=temperature > 0,
                              temperature=temperature or None)
    trimmed = ids[:, n_in:]
    txt = _PROC.batch_decode(trimmed, skip_special_tokens=True)[0]
    return strip_reasoning(txt), n_in, int(trimmed.shape[-1])


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
        if self.path.startswith("/v1/models"):
            self._json({"object": "list",
                        "data": [{"id": SERVED, "object": "model", "owned_by": "local"}]})
        elif self.path.startswith("/health"):
            self._json({"ok": True, "model": SERVED, **_STATS})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._json({"error": f"unknown route {self.path}"}, 404)
        t0 = time.time()
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            with _LOCK:
                txt, n_in, n_out = generate(body.get("messages") or [],
                                            int(body.get("max_tokens") or 256),
                                            float(body.get("temperature") or 0.0),
                                            tools=body.get("tools"))
            txt, calls = split_tool_calls(txt)
            _STATS["served"] += 1
            _STATS["seconds"] = round(_STATS["seconds"] + time.time() - t0, 2)
            _STATS["prompt_tokens"] += n_in
            _STATS["completion_tokens"] += n_out
            self._json({
                "id": f"chatcmpl-{_STATS['served']}", "object": "chat.completion",
                "created": int(t0), "model": SERVED,
                "choices": [{"index": 0,
                             "finish_reason": "tool_calls" if calls else "stop",
                             "message": {"role": "assistant", "content": txt,
                                         **({"tool_calls": calls} if calls else {})}}],
                "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                          "total_tokens": n_in + n_out},
            })
        except Exception as e:
            _STATS["errors"] += 1
            import traceback; traceback.print_exc()
            self._json({"error": {"message": f"{type(e).__name__}: {e}"}}, 500)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--served-name", default=None)
    ap.add_argument("--port", type=int, default=2021)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "auto"])
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="generate() calls allowed in flight. transformers has no scheduler, "
                         "so this is plain concurrency on one model: fine for a small model "
                         "where the GPU is idle between calls, wasteful of memory for a large "
                         "one. Verify outputs match the serial ones before raising it.")
    ap.add_argument("--reason-headroom", type=int, default=512,
                    help="Extra generation budget on top of the caller's max_tokens, for a "
                         "reasoning pass that is stripped before the answer is returned.")
    ap.add_argument("--thinking", choices=["on", "off", "default"], default="default",
                    help="off passes enable_thinking=False to the chat template. The "
                         "default leaves the template alone, because on Qwen3.8 turning "
                         "reasoning off made detect and point return nothing at all: the "
                         "spatial answers are what the reasoning pass is for.")
    a = ap.parse_args()
    SERVED = a.served_name or a.model.split("/")[-1]
    THINKING = {"on": True, "off": False, "default": None}[a.thinking]
    REASON_HEADROOM = a.reason_headroom
    if a.concurrency > 1:
        _LOCK = threading.BoundedSemaphore(a.concurrency)

    print(f"loading {a.model} ({a.dtype}, device_map={a.device_map}) ...", flush=True)
    t = time.time()
    _MODEL, _PROC = load(a.model, a.dtype, a.device_map, a.trust_remote_code)
    print(f"ready on {a.host}:{a.port} as '{SERVED}' in {time.time()-t:.0f}s", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
