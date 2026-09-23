#!/usr/bin/env python3
"""
judge.py -- SCOPE LLM-as-Judge (sanitized)

Reads a results CSV, sends each row to an LLM judge for correctness
evaluation, and writes the judged output CSV with additional columns.

No runtime pip installs. Falls back to a tiny OpenAI-compatible HTTP
client if the ``openai`` package is not installed.

Env vars:
  JUDGE_API_BASE  : OpenAI-compatible base URL (default: https://api.openai.com/v1)
  JUDGE_API_KEY   : API key (falls back to OPENAI_API_KEY)
  JUDGE_MODEL_ID  : Model to use for judging (default: gpt-4o)

LOCAL JUDGE SUPPORT:
  The judge uses an OpenAI-compatible chat API via ``_HTTPChatClient``,
  so any local model served via vLLM, Ollama, or similar works by
  setting JUDGE_API_BASE and JUDGE_MODEL_ID appropriately. Example:
    JUDGE_API_BASE=http://localhost:11434/v1 JUDGE_MODEL_ID=llama3 python -m scope.eval.judge -i results.csv -o judged.csv
"""

import os
import shutil, sys, json, csv, re, argparse, difflib, hashlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Prompts live at repo-root/prompts so they're visible without digging into the package.
_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


# --------------------------------------------------------------------------- #
# OpenAI-compatible HTTP client (no pip dependency required)                    #
# --------------------------------------------------------------------------- #

class _HTTPChatClient:
    """Minimal OpenAI-compatible chat completions client using urllib only."""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key or "EMPTY"
        self.base_url = base_url.rstrip("/")

    class _ChatCompletions:
        def __init__(self, outer):
            self.outer = outer

        def create(self, model: str, messages: List[Dict[str, str]], temperature: float = 0):
            import urllib.request, urllib.error
            url = f"{self.outer.base_url}/chat/completions"
            data = json.dumps({
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }).encode("utf-8")
            req = urllib.request.Request(url, data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("Authorization", f"Bearer {self.outer.api_key}")
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    raw = resp.read().decode("utf-8", "ignore")
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "ignore")
                raise RuntimeError(f"HTTP {e.code} from {url}: {body}")
            except Exception as e:
                raise RuntimeError(f"HTTP error calling {url}: {e}")
            obj = json.loads(raw)
            try:
                content = obj["choices"][0]["message"]["content"]
            except Exception:
                content = obj.get("choices", [{}])[0].get("text", "")
            return type("Resp", (), {
                "choices": [type("C", (), {
                    "message": type("M", (), {"content": content})()
                })()]
            })

    @property
    def chat(self):
        return type("Chat", (), {"completions": self._ChatCompletions(self)})()


def _load_openai_client():
    """Return a factory that creates an OpenAI-compatible client."""
    try:
        from openai import OpenAI  # type: ignore
        return lambda api_key, base_url: OpenAI(api_key=api_key or "EMPTY", base_url=base_url)
    except Exception:
        return lambda api_key, base_url: _HTTPChatClient(api_key=api_key, base_url=base_url)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _norm_str(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _row_key(row: Dict[str, Any]) -> str:
    """Stable identity for a row across retries."""
    qid = _norm_str(row.get("question_id"))
    if qid:
        return f"qid::{qid}"
    fields = [
        "file_location", "preset_start", "question", "expected_answer",
        "answer_view", "repeat_idx", "start_ts", "end_ts", "final_answer",
    ]
    parts = [f"{k}={_norm_str(row.get(k))}" for k in fields if k in row]
    sig = "|".join(parts)
    return "sig::" + hashlib.sha1(sig.encode("utf-8", "ignore")).hexdigest()[:20]


def jget(row: Dict[str, Any], key: str, default: str = "") -> str:
    v = row.get(key)
    return default if v is None else str(v)


def jget_bool(row: Dict[str, Any], key: str) -> bool:
    s = str(row.get(key) or "").lower()
    return "lenient_open_ended=true" in s or s in ("true", "1", "yes", "y", "on")


def parse_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "on"):
        return True
    if s in ("false", "0", "no", "n", "off"):
        return False
    return default


def extract_tool_calls_pretty(actual_calls_json: str) -> str:
    try:
        calls = json.loads(actual_calls_json) if actual_calls_json else []
    except Exception:
        calls = []
    parts = []
    for idx, c in enumerate(calls, start=1):
        nm = c.get("name") or ""
        args = c.get("arguments", {})
        parts.append(f"[{idx}] {nm}({json.dumps(args, ensure_ascii=False)})")
    return "\n".join(parts) if parts else "(no tool calls)"


def norm_int_like(text: str) -> Tuple[int, int]:
    """Return (number, tolerance). 'about N' -> (N, 1). If none found -> (None, 0)."""
    s = (text or "").lower()
    m = re.search(r"\babout\s+(\d+)\b", s)
    if m:
        return int(m.group(1)), 1
    m = re.search(r"\b(\d+)\b", s)
    return (int(m.group(1)), 0) if m else (None, 0)


def text_norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def close_text(a: str, b: str, ratio: float = 0.88) -> bool:
    A, B = text_norm(a), text_norm(b)
    if A == B:
        return True
    return difflib.SequenceMatcher(None, A, B).ratio() >= ratio


def singularize_tokens(s: str) -> str:
    toks = text_norm(s).split()
    out = []
    for t in toks:
        if len(t) > 2 and t.endswith("s"):
            out.append(t[:-1])
        else:
            out.append(t)
    return " ".join(out)


def names_match(a: str, b: str) -> bool:
    return singularize_tokens(a) == singularize_tokens(b)


def detect_region_cue(text: str) -> bool:
    REGION = re.compile(
        r"\b(left|right|center|middle|upper|lower|top|bottom|half|third|quadrant|front|back)\b",
        re.I,
    )
    return bool(REGION.search(text or ""))


# --------------------------------------------------------------------------- #
# Judge prompts                                                                #
# --------------------------------------------------------------------------- #

GENERAL_PROMPT = (_PROMPTS_DIR / "judge_general.md").read_text(encoding="utf-8")

_CATEGORY_KEYS = [
    "counting", "ocr_identification", "descriptor", "location_spatial",
    "comparative_relational", "single_call",
    "multi_step_command_unordered", "multi_step_command_strict",
    "multi_step_reasoning_unordered", "multi_step_reasoning_strict",
]

TEMPLATES = {
    k: (_PROMPTS_DIR / f"judge_category_{k}.md").read_text(encoding="utf-8")
    for k in _CATEGORY_KEYS
}


# --------------------------------------------------------------------------- #
# Category key builder                                                         #
# --------------------------------------------------------------------------- #

def build_category_key(eval_category: str, multi_step_mode: str) -> str:
    cat = (eval_category or "").strip().lower()
    if cat in ("multi_step_command", "multi-step_command", "multi step command"):
        mode = (multi_step_mode or "none").strip().lower()
        return f"multi_step_command_{mode}"
    if cat in ("multi_step_reasoning", "multi-step_reasoning", "multi step reasoning"):
        mode = (multi_step_mode or "none").strip().lower()
        return f"multi_step_reasoning_{mode}"
    return cat


# --------------------------------------------------------------------------- #
# Message builder                                                              #
# --------------------------------------------------------------------------- #

# The raw conversation is pasted into the judge prompt whole. On a verbose planner that is
# tens of thousands of tokens and the judge returns a 400 before grading anything -- tA02
# failed this way on every seat, every attempt, from Sep 6, and produced only header-sized
# partials. Keep the head and the tail: the setup and the final answer carry the verdict, the
# middle is repeated tool chatter.
TRANSCRIPT_CAP = int(os.environ.get("JUDGE_TRANSCRIPT_CAP", "24000"))


def _cap(text: str) -> str:
    if not text or len(text) <= TRANSCRIPT_CAP:
        return text
    head = TRANSCRIPT_CAP * 2 // 3
    tail = TRANSCRIPT_CAP - head
    return (f"{text[:head]}\n\n... [{len(text) - TRANSCRIPT_CAP} characters of tool chatter "
            f"elided by the harness] ...\n\n{text[-tail:]}")


def build_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    eval_category = jget(row, "eval_category")
    multi_step_mode = jget(row, "multi_step_mode")
    key = build_category_key(eval_category, multi_step_mode)
    tmpl = TEMPLATES.get(key) or TEMPLATES["descriptor"]

    full_conversation = _cap(jget(row, "llm_raw") or jget(row, "llm_readable"))
    question = jget(row, "question")
    expected_answer = jget(row, "expected_answer")
    final_answer = jget(row, "final_answer")
    tool_calls_parsed = extract_tool_calls_pretty(jget(row, "actual_tool_calls_json"))
    required_tools_policy = jget(row, "required_tools_policy")
    expected_tool_order_json = jget(row, "expected_tool_order_json")

    det_region_cue = detect_region_cue(question)
    lenient_open_ended = jget_bool(row, "evaluation_notes")

    user_block = tmpl.format(
        full_conversation=full_conversation,
        question=question,
        expected_answer=expected_answer,
        final_answer=final_answer,
        tool_calls_parsed=tool_calls_parsed,
        required_tools_policy=required_tools_policy,
        expected_tool_order_json=expected_tool_order_json,
        evaluation_notes=jget(row, "evaluation_notes"),
        det_region_cue=str(det_region_cue),
        lenient_open_ended=str(lenient_open_ended),
    )

    return [
        {"role": "system", "content": GENERAL_PROMPT},
        {"role": "user", "content": user_block},
    ]


# --------------------------------------------------------------------------- #
# Judge caller                                                                 #
# --------------------------------------------------------------------------- #

def call_judge(messages: List[Dict[str, str]], base: str, model: str, api_key: str) -> Dict[str, Any]:
    """Send messages to the judge LLM and parse the JSON response."""
    OpenAIFactory = _load_openai_client()
    client = OpenAIFactory(api_key=api_key or "EMPTY", base_url=base)
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
    text = getattr(resp.choices[0].message, "content", "")
    try:
        return json.loads(text)
    except Exception:
        pass

    # Take the first complete JSON object and ignore whatever follows.
    #
    # This used to be re.search(r"\{.*\}", text, re.S), which is greedy: it spans from the first
    # brace to the *last* one, so a judge that emits its verdict and then explains itself
    # produces "Extra data: line 6 column 1" and the row is scored wrong for a formatting
    # reason. Measured on ERNIE-4.5 that was 36% of rows -- a judge that was answering
    # perfectly well, marked down for punctuation.
    dec = json.JSONDecoder()
    last_err = None
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text[i:])
        except ValueError as e:
            last_err = e
            continue
        if isinstance(obj, dict) and ("is_correct" in obj or "reason" in obj):
            return obj
    if last_err is None:
        return {"is_correct": False, "reason": f"Judge returned non-JSON: {text[:200]!r}",
                "error_mode": "format"}
    return {"is_correct": False, "reason": f"Judge JSON decode error: {last_err}",
            "error_mode": "format"}


# --------------------------------------------------------------------------- #
# Programmatic single-row judge                                                #
# --------------------------------------------------------------------------- #

def judge_row(row: Dict[str, Any], base_url: str = None, model: str = None, api_key: str = None) -> Dict[str, Any]:
    """Judge a single row programmatically.

    Parameters
    ----------
    row : dict
        A single result row (dict with CSV column names as keys).
    base_url : str, optional
        Judge API base URL.  Falls back to JUDGE_API_BASE env, then
        ``https://api.openai.com/v1``.
    model : str, optional
        Judge model id.  Falls back to JUDGE_MODEL_ID env, then ``gpt-4o``.
    api_key : str, optional
        API key.  Falls back to JUDGE_API_KEY, then OPENAI_API_KEY env.

    Returns
    -------
    dict
        ``{"is_correct": bool, "reason": str, "error_mode": str}``
    """
    base_url = base_url or os.getenv("JUDGE_API_BASE") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    model = model or os.getenv("JUDGE_MODEL_ID") or "gpt-4o"
    api_key = api_key or os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
    messages = build_messages(row)
    return call_judge(messages, base_url, model, api_key)


# --------------------------------------------------------------------------- #
# CLI main                                                                     #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="SCOPE LLM-as-Judge: evaluate agent results via an LLM judge.",
    )
    ap.add_argument("-i", "--in", dest="in_path", required=True,
                    help="Input CSV path (results from the batch runner).")
    ap.add_argument("-o", "--out", dest="out_path", required=True,
                    help="Output CSV path for judged results.")
    ap.add_argument("--base-url", dest="base_url", default=None,
                    help="Judge API base URL (overrides JUDGE_API_BASE env var).")
    ap.add_argument("--model", dest="model", default=None,
                    help="Judge model id (overrides JUDGE_MODEL_ID env var).")
    ap.add_argument("--api-key", dest="api_key", default=None,
                    help="Judge API key (overrides JUDGE_API_KEY env var).")
    ap.add_argument("--no-resume", action="store_true",
                    help="Do not resume; overwrite output instead of appending remaining rows.")
    ap.add_argument("--concurrency", type=int,
                    default=int(os.getenv("JUDGE_CONCURRENCY", "8")),
                    help="Judge calls in flight at once. One at a time is fine for a single "
                         "run and hopeless for a sweep: 25 pairings x 536 rows x 3 judges is "
                         "40k calls, and a slow judge at 5 rows a minute turns that into days. "
                         "The server batches them; 1 restores the old serial behaviour.")
    args = ap.parse_args()

    in_path = args.in_path
    out_path = args.out_path

    base = args.base_url or os.getenv("JUDGE_API_BASE") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    model = args.model or os.getenv("JUDGE_MODEL_ID") or "gpt-4o"
    key = args.api_key or os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"

    # Load input rows
    with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        in_rows = list(reader)
        in_header = list(reader.fieldnames or [])

    # Desired columns (ensure present in output)
    wanted_cols = [
        "judge_is_correct", "coverage_satisfied", "scope_satisfied",
        "judge_reason", "judge_error_mode", "model_id_judge", "base_url_judge",
    ]

    # Determine output header = input header + wanted_cols (preserve order)
    out_header = list(in_header)
    lower = [h.lower() for h in out_header]
    for col in wanted_cols:
        if col.lower() not in lower:
            out_header.append(col)
            lower.append(col.lower())

    # RESUME: collect the row keys we should skip.
    #
    # A row is skippable when it carries a USABLE VERDICT, not merely when it is present. A row
    # the judge could not grade is still written, with judge_error in its error mode and an empty
    # judge_is_correct, so keying resume on presence made a failed row permanently unjudgeable:
    # tA06's muse seat held 536 rows of which 385 were errors, and every rerun reported "done
    # (151 graded)" without attempting one of them. Any tool that asks "is this row graded?" (a
    # resume gate, a completeness check, a report) must use this same predicate: in the
    # campaign two gates that each wrote their own disagreed silently, and both reported success.
    GRADED = {"true", "1", "yes", "false", "0", "no"}

    def _is_graded(row):
        return (row.get("judge_is_correct") or "").strip().lower() in GRADED

    existing_keys = set()
    out_exists = os.path.exists(out_path)
    if out_exists and not args.no_resume:
        try:
            with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
                r = csv.DictReader(f)
                total = 0
                for row in r:
                    total += 1
                    if _is_graded(row):
                        existing_keys.add(_row_key(row))
            skipped = total - len(existing_keys)
            print(f"[resume] {len(existing_keys)} of {total} rows already judged in {out_path}; "
                  f"will skip them"
                  + (f" and re-judge {skipped} that carry no verdict." if skipped else "."))
        except Exception as e:
            print(f"[resume] Failed to read existing output (will rewrite): {e}")
            existing_keys.clear()
            out_exists = False

    # Resume by rewriting, not by appending.
    #
    # Appending assumes every row already in the file was recognised as already judged. When
    # that fails, for any reason, the run adds a second copy of rows that are already there and
    # nothing complains: the file has one header, the counts in the log add up, and the only
    # symptom is more rows out than went in. It happened here across three restarts and
    # produced 792 rows from a 541 row input, 251 of them duplicated, with the accuracy
    # computed over the duplicates.
    #
    # So the previously judged rows are carried forward explicitly and the file is written
    # whole. The output then cannot contain a key twice however many times a run is restarted,
    # and a crash mid-write costs the temporary file rather than the results.
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    carried = {}
    if out_exists and not args.no_resume:
        try:
            with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
                for row in csv.DictReader(f):
                    carried[_row_key(row)] = row
        except Exception as e:
            print(f"[resume] could not read {out_path}, judging everything again: {e}")
            carried.clear()
            existing_keys.clear()
    concurrency = max(1, int(args.concurrency))
    block_size = max(1, concurrency * 4)
    pending = deque(i for i, r in enumerate(in_rows)
                    if not (existing_keys and _row_key(r) in existing_keys))
    results: Dict[int, Dict[str, Any]] = {}

    def judge_one(i: int) -> Dict[str, Any]:
        """A row the judge refuses must cost that row, not the whole tag.

        Before this, one over-long row raised out of the worker and the run died with nothing
        written, so a tag could fail for two days while the batch script reported it done.
        """
        try:
            return call_judge(build_messages(in_rows[i]), base, model, key)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"[judge] row {i} ungradeable — {msg[:200]}", flush=True)
            return {"is_correct": None, "reason": f"judge failed: {msg[:500]}",
                    "error_mode": "judge_error"}

    if concurrency > 1:
        print(f"[judge] {len(pending)} rows to judge, {concurrency} calls in flight")

    mode = "w"
    tmp_path = out_path + f".partial.{os.getpid()}"
    judged_count = 0
    skipped_count = 0

    with open(tmp_path, mode, encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_header, extrasaction="ignore")

        if mode == "w":
            w.writeheader()

        for idx, row in enumerate(in_rows):
            row_key = _row_key(row)
            if existing_keys and row_key in existing_keys:
                skipped_count += 1
                prev = carried.get(row_key)
                if prev:
                    w.writerow(prev)
                continue

            # Carry judge endpoint info (for provenance)
            row.setdefault("model_id_judge", model)
            row.setdefault("base_url_judge", base)

            # Optional CSV-provided booleans; leave empty if unknown
            if "coverage_satisfied" not in row or str(row["coverage_satisfied"]).strip() == "":
                row["coverage_satisfied"] = ""
            if "scope_satisfied" not in row or str(row["scope_satisfied"]).strip() == "":
                row["scope_satisfied"] = ""

            # Judge in blocks, write in order.
            #
            # The calls go out concurrently because the serving side batches them happily and
            # the round trip, not the GPU, is what a serial loop spends its time on. The write
            # loop still walks in_rows in order, so the output file and its checkpoints are
            # exactly what they were before.
            if idx not in results:
                block = [idx]
                while pending and pending[0] <= idx:
                    pending.popleft()
                while pending and len(block) < block_size:
                    block.append(pending.popleft())
                if concurrency > 1 and len(block) > 1:
                    with ThreadPoolExecutor(max_workers=concurrency) as ex:
                        for k, obj in zip(block, ex.map(judge_one, block)):
                            results[k] = obj
                else:
                    for k in block:
                        results[k] = judge_one(k)
            judge_obj = results.pop(idx)

            row["judge_is_correct"] = judge_obj.get("is_correct", False)
            row["judge_reason"] = judge_obj.get("reason", "")
            row["judge_error_mode"] = judge_obj.get("error_mode", "None")

            for col in wanted_cols:
                row.setdefault(col, "")

            w.writerow(row)
            judged_count += 1
            # Publish progress without disturbing the open handle. The file is built at a
            # temporary path and swapped into place at the end, so a crash cannot leave a
            # half-written results file. On a 541 row run that also meant nothing was visible
            # until the very end, so every 25 rows the partial file is copied over the real
            # one: progress can be watched, and an interrupted run keeps nearly all its work.
            if judged_count % 25 == 0:
                try:
                    f.flush()
                    shutil.copyfile(tmp_path, out_path)
                except Exception:
                    pass

            if judged_count % 10 == 0:
                print(f"[judge] Processed {judged_count} rows...")

    # Swap the finished file into place in one step, so an interrupted run never leaves a
    # half-written results file where a complete one used to be.
    os.replace(tmp_path, out_path)
    print(f"[judge] Done. Judged {judged_count} rows, skipped {skipped_count} (already judged).")
    print(f"[judge] Output: {out_path}")


if __name__ == "__main__":
    main()
