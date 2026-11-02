#!/usr/bin/env python3
"""Grade runs the v2 way: three judges, 2 of 3 to pass, code checks first.

The v2 results (docs/RESULTS_V2.md) were graded by this panel in October 2026:

* three seats, each a different model: gpt-oss-120b, Muse-Glimmer-30B and
  Qwen3-Nemotron-32B-GenRM, each with the client settings it was judged with;
* the October rubric (criteria C1 to C3, fixes J1 to J6, accepted alternates), in
  ``prompts/judge_*.md`` and ``scope.eval.judge``;
* code checks before any judge call: a reply with no final answer, or a camera action the
  question commands and the recorded tool calls do not carry out, fails
  (``scope.eval.final_answer``, ``scope.eval.action_check``);
* a record passes when at least two seats pass it, no code check fails it and its vision
  reply could be read. A missing verdict is asked again, never counted as a fail.

Subcommands:

  judge    judge result CSVs with each seat; resumes, re-asks rows without a verdict
  vote     combine the three seats' verdicts per record and score each run
  seats    show the seat endpoints this run would use

Seat endpoints come from the environment (or a repo-root ``.env``), or from ``--config``:

  PANEL_GPT_OSS_URL, PANEL_MUSE_URL, PANEL_NEMOTRON_URL     OpenAI-compatible base URLs (.../v1)
  PANEL_<SEAT>_MODEL        served model name (default: the seat name, e.g. gpt-oss-120b)
  PANEL_<SEAT>_API_KEY      API key (default: none)
  PANEL_<SEAT>_CONCURRENCY  calls in flight (default 8; changes speed only, not verdicts)

``--config panel.json`` takes {"<seat>": {"base_url", "model", "api_key", "concurrency"}}.

Examples:

  python scripts/13_judge_panel.py judge -i results/my-pairing/raw_results.csv --run my-pairing -o judged/
  python scripts/13_judge_panel.py vote judged/ -o judged/
  python scripts/13_judge_panel.py vote judged/ --check-against benchmark/results/v2-2026-10

``judge`` writes ``<out>/<run>__<seat>.csv`` (run = ``--run`` or the input file name without
.csv), the layout ``vote`` reads. See docs/JUDGING.md, section 3.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from collections import Counter
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
csv.field_size_limit(sys.maxsize)

# name, Hugging Face model, environment prefix, client settings beyond COMMON_ENV.
SEATS = [
    ("gpt-oss-120b", "openai/gpt-oss-120b", "PANEL_GPT_OSS", {}),
    ("muse-glimmer-judge", "meta-models/Muse-Glimmer-30B", "PANEL_MUSE", {}),
    # Thinking off: with it on, the final-answer prompt sends Nemotron into a thinking loop
    # until the context runs out. A verdict takes under 300 tokens; a reply that loops is cut
    # at 3,072 and asked again.
    ("qwen3-nemotron-genrm", "nvidia/Qwen3-Nemotron-32B-GenRM-Principle", "PANEL_NEMOTRON",
     {"JUDGE_DISABLE_THINKING": "1", "JUDGE_MAX_TOKENS": "3072"}),
]
SEAT_NAMES = [s[0] for s in SEATS]
# The weights judged with in October 2026 (Hugging Face revisions).
REVISIONS = {"gpt-oss-120b": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
             "muse-glimmer-judge": "a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
             "qwen3-nemotron-genrm": "9fd58dd68f50bfd03fc1cd95f1413a8457eda5ab"}
# Every seat grades the final answer plus trajectory, answers in schema-checked JSON with
# replies capped at 8,192 tokens, and reads at most 24,000 characters of trajectory (head and
# tail kept). JUDGE_REASON_MAXLEN=0 leaves the 400-character reason bound out of the grammar
# (it slows decoding about 15x); the reason is cut to 400 characters after parsing.
COMMON_ENV = {"JUDGE_INPUT": "final+trajectory", "JUDGE_STRUCTURED": "1", "JUDGE_MAX_TOKENS": "8192",
              "JUDGE_REASON_MAXLEN": "0", "JUDGE_TRANSCRIPT_CAP": "24000"}
# A row without a verdict is asked again at most this many more times. The last try lifts the
# reply cap and puts the reason and mode bounds back in the grammar: a judge that repeats
# itself in the reason would otherwise be cut off by any cap.
VERDICT_RETRIES = 2
LAST_TRY_ENV = {"JUDGE_MAX_TOKENS": "8192", "JUDGE_REASON_MAXLEN": "400", "JUDGE_MODES_BOUND": "1"}
# What a verdict depends on: a saved verdict whose task or answer changed is dropped and the
# row judged again. A verdict given before also_accept existed had no alternates: same as [].
VERDICT_INPUTS = ("question", "expected_answer", "also_accept", "eval_category", "multi_step_mode",
                  "expected_tool_order_json", "expected_tool_args_json", "args_check", "arg_match_policy",
                  "required_tools_policy", "evaluation_notes", "answer_view", "scope_requirement",
                  "preset_start", "final_answer", "llm_raw")
# Set by scope.tools.blender_tools when the vision model's points cannot be read: such a record
# fails (the maintainer's rule of 9 October 2026) and is tallied apart.
UNPARSEABLE = "could not be read as points (unparseable)"
CODE_CHECKS = ("no_final_answer", "action_contradiction")


# --------------------------------------------------------------------------- files

def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def key(row):
    return row["question_id"], (row.get("repeat_idx") or "1").strip() or "1"


def verdict(row):
    """1 or 0 for a seat's verdict, None when the row has none."""
    v = (row.get("judge_is_correct") or "").strip().lower()
    return 1 if v in {"true", "1", "yes"} else 0 if v in {"false", "0", "no"} else None


# --------------------------------------------------------------------------- seats

def load_dotenv():
    """PANEL_* settings from the repo-root .env, unless already set in the environment."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split(" #", 1)[0].strip()
        if line.startswith("export "):
            line = line[7:].strip()
        if not line.startswith("PANEL_") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def seat_endpoints(config=None):
    """{seat: {base_url, model, api_key, concurrency}} from --config or PANEL_* variables."""
    load_dotenv()
    given = json.loads(Path(config).read_text()) if config else {}
    out = {}
    for name, _, prefix, _ in SEATS:
        c = given.get(name, {})
        out[name] = {"base_url": c.get("base_url") or os.environ.get(f"{prefix}_URL", ""),
                     "model": c.get("model") or os.environ.get(f"{prefix}_MODEL") or name,
                     "api_key": c.get("api_key") or os.environ.get(f"{prefix}_API_KEY") or "EMPTY",
                     "concurrency": int(c.get("concurrency") or os.environ.get(f"{prefix}_CONCURRENCY") or 8)}
    return out


def seat_env(name, last_try=False):
    extra = next(s[3] for s in SEATS if s[0] == name)
    return {**COMMON_ENV, **extra, **(LAST_TRY_ENV if last_try else {})}


# --------------------------------------------------------------------------- judge

def fingerprint(row):
    return tuple((row.get(k) or "[]") if k == "also_accept" else (row.get(k) or "") for k in VERDICT_INPUTS)


def drop_stale(dest, rows):
    """Remove saved verdicts whose task or answer differs from the input (judged again)."""
    if not dest.exists():
        return 0
    current = {key(r): fingerprint(r) for r in rows}
    saved = read_rows(dest)
    keep = [r for r in saved if current.get(key(r)) == fingerprint(r)]
    if len(keep) != len(saved):
        write_rows(dest, keep, list(saved[0]))
    return len(saved) - len(keep)


def judge_file(src, out_dir, seats, endpoints, run=None):
    rows = read_rows(src)
    run = run or src.stem
    missing_any = {}
    for name in seats:
        ep = endpoints[name]
        dest = out_dir / f"{run}__{name}.csv"
        dropped = drop_stale(dest, rows)
        if dropped:
            print(f"[panel] {run} {name}: {dropped} saved verdicts no longer match their answer; judging them again")
        for attempt in range(1 + VERDICT_RETRIES):
            cmd = [sys.executable, str(Path(__file__).resolve()), "_seat", "-i", str(src), "-o", str(dest),
                   "--base-url", ep["base_url"], "--model", ep["model"], "--api-key", ep["api_key"],
                   "--concurrency", str(ep["concurrency"])]
            env = {**os.environ, **seat_env(name, last_try=attempt == VERDICT_RETRIES),
                   "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")]))}
            subprocess.run(cmd, env=env, check=True)
            judged = read_rows(dest)
            missing = [r["question_id"] for r in judged if verdict(r) is None]
            assert len(judged) == len(rows), (run, name, len(judged), len(rows))
            if not missing:
                break
            print(f"[panel] {run} {name}: {len(missing)} rows without a verdict after try {attempt + 1}; asking again")
        if missing:
            missing_any[f"{run} {name}"] = missing
        else:
            print(f"[panel] {run} {name}: {len(judged)} rows, every row has a verdict")
    return missing_any


def seat_main(argv):
    """One seat over one file, with replies that are not a full verdict counted as missing.

    scope.eval.judge turns a reply its parser cannot read into a failing vote; the panel needs
    three real votes, so such a reply is recorded without a verdict and asked again.
    """
    from scope.eval import judge

    original = judge.call_judge

    def checked(*args, **kwargs):
        result = original(*args, **kwargs)
        bad = not isinstance(result, dict) or type(result.get("is_correct")) is not bool
        if isinstance(result, dict):
            # An October verdict also carries all_modes and first_break.
            bad |= not isinstance(result.get("all_modes"), list) or not result.get("first_break")
            bad |= str(result.get("reason", "")).startswith(("Judge returned non-JSON:", "Judge JSON decode error:"))
        if bad:
            return {"is_correct": None, "reason": f"Invalid judge reply: {result!r}"[:1000], "error_mode": "judge_error"}
        return result

    judge.call_judge = checked
    dest = Path(argv[argv.index("-o") + 1])
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Two processes on one seat file would rewrite it at once: serialize them.
    with dest.with_suffix(dest.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sys.argv = ["scope.eval.judge", *argv]
        judge.main()


# --------------------------------------------------------------------------- vote

def code_check(row):
    """'no_final_answer', 'action_contradiction' or None, decided from the answer alone."""
    from scope.eval import action_check, final_answer as fa
    if not fa.answer(row)[0]:
        return "no_final_answer"
    if not action_check.check(row.get("question", ""), fa.tool_steps(row))["ok"]:
        return "action_contradiction"
    return None


def vote_run(seat_rows):
    """Panel records for one run from its three seats' rows (lists of dicts, SEATS order)."""
    keyed = [{key(r): r for r in rows} for rows in seat_rows]
    keys = set(keyed[0])
    for name, k in zip(SEAT_NAMES, keyed):
        if set(k) != keys:
            raise SystemExit(f"seat {name} graded different records from {SEAT_NAMES[0]}")
    missing = [(n, k) for n, seat in zip(SEAT_NAMES, keyed) for k, r in seat.items() if verdict(r) is None]
    if missing:
        raise SystemExit(f"{len(missing)} seat verdicts missing (first: {missing[:3]}); run judge again, "
                         "a missing verdict is never counted as a fail")
    records = []
    for k in sorted(keys):
        rows = [seat[k] for seat in keyed]
        votes = [verdict(r) for r in rows]
        check = code_check(rows[0])
        saved = {r.get("judge_error_mode") for r in rows if r.get("judge_error_mode") in CODE_CHECKS}
        unparseable = UNPARSEABLE in (rows[0].get("llm_raw") or "")
        records.append({
            "question_id": k[0], "repeat_idx": k[1], "category": rows[0].get("eval_category", ""),
            "votes": "".join(map(str, votes)), "code_check": check or "",
            # The judge applies the same checks before calling a model; a seat file whose
            # checks disagree was judged with other settings.
            "code_check_matches_seats": int(saved == ({check} if check else set())),
            "unparseable_vision": int(unparseable),
            "pass": int(sum(votes) >= 2 and not check and not unparseable)})
    return records


def find_runs(folder):
    runs = {}
    for p in sorted(Path(folder).glob("*__*.csv")):
        run, seat = p.stem.split("__", 1)
        if seat in SEAT_NAMES:
            runs.setdefault(run, {})[seat] = p
    incomplete = {r: sorted(set(SEAT_NAMES) - set(s)) for r, s in runs.items() if len(s) < 3}
    if incomplete:
        raise SystemExit(f"runs without all three seats: {incomplete}")
    if not runs:
        raise SystemExit(f"no <run>__<seat>.csv files in {folder}")
    return runs


def run_summary(run, recs):
    n = len(recs)
    checks = Counter(r["code_check"] for r in recs if r["code_check"])
    return {"run": run, "records": n, "passes": sum(r["pass"] for r in recs),
            "pct": round(100 * sum(r["pass"] for r in recs) / n, 1),
            "no_final_answer": checks["no_final_answer"], "action_contradiction": checks["action_contradiction"],
            "unparseable_vision": sum(r["unparseable_vision"] for r in recs)}


def overall(records):
    n = len(records)
    cats = {}
    for r in records:
        c = cats.setdefault(r["category"], [0, 0])
        c[0] += 1
        c[1] += r["pass"]
    return {"records": n, "passes": sum(r["pass"] for r in records),
            "pct": round(100 * sum(r["pass"] for r in records) / n, 2),
            "seat_pass_pct": {s: round(100 * sum(int(r["votes"][i]) for r in records) / n, 1)
                              for i, s in enumerate(SEAT_NAMES)},
            "all_three_agree_pct": round(100 * sum(r["votes"] in {"000", "111"} for r in records) / n, 1),
            "code_checks": dict(Counter(r["code_check"] for r in records if r["code_check"])),
            "unparseable_vision": sum(r["unparseable_vision"] for r in records),
            "category_pct": {c: round(100 * v[1] / v[0], 1) for c, v in cats.items()}}


def check_against(published, per_run, by_run):
    """Compare with a published results folder (summary.json, models.csv); return differences."""
    published = Path(published)
    models = {r["tag"]: r for r in read_rows(published / "models.csv")}
    summary = json.loads((published / "summary.json").read_text())
    diffs = []
    for s in per_run:
        m = models.get(s["run"])
        if m is None:
            diffs.append(f"{s['run']}: not in models.csv")
            continue
        for mine, theirs in (("records", "tasks_scored"), ("passes", "new_passes"), ("pct", "new_pct"),
                             ("no_final_answer", "new_no_final_answer"),
                             ("action_contradiction", "new_action_contradiction"),
                             ("unparseable_vision", "new_unparseable_vision")):
            if float(s[mine]) != float(m[theirs]):
                diffs.append(f"{s['run']} {mine}: {s[mine]} here, {m[theirs]} published")
    report = {"runs_compared": len(per_run)}
    # The overall figures count each distinct configuration once (repeat runs left out).
    ranked = [r for r in by_run if r not in summary.get("repeats", {})]
    if set(ranked) >= {t for g in summary["groups"].values() for t in g["runs"]}:
        got = overall([rec for r in ranked for rec in by_run[r]])
        for k in ("records", "passes", "pct", "seat_pass_pct", "all_three_agree_pct", "code_checks",
                  "unparseable_vision", "category_pct"):
            if got[k] != summary[k]:
                diffs.append(f"summary {k}: {got[k]} here, {summary[k]} published")
        passes = {s["run"]: s["passes"] for s in per_run}
        for name, g in summary["groups"].items():
            mean = round(100 * sum(passes[t] for t in g["runs"]) / sum(len(by_run[t]) for t in g["runs"]), 2)
            if mean != g["mean_pct"]:
                diffs.append(f"group {name}: {mean} here, {g['mean_pct']} published")
        report["summary_compared"] = True
    else:
        report["summary_compared"] = False
    return diffs, report


def vote_main(args):
    runs = find_runs(args.folder)
    by_run, per_run = {}, []
    for run, seats in runs.items():
        recs = vote_run([read_rows(seats[name]) for name in SEAT_NAMES])
        by_run[run] = recs
        per_run.append(run_summary(run, recs))
    mismatched = sum(1 - r["code_check_matches_seats"] for recs in by_run.values() for r in recs)
    for s in per_run:
        print(f"{s['run']:>8}  {s['passes']:>4} of {s['records']} pass  {s['pct']:5.1f}%  "
              f"(no final answer {s['no_final_answer']}, action not carried out {s['action_contradiction']}, "
              f"vision reply unreadable {s['unparseable_vision']})")
    total = overall([r for recs in by_run.values() for r in recs])
    print(f"all runs: {total['passes']:,} of {total['records']:,} pass ({total['pct']}%); "
          f"code checks that disagree with the seat files: {mismatched}")
    if args.out:
        out = Path(args.out)
        write_rows(out / "panel_records.csv", [{"run": run, **r} for run, recs in by_run.items() for r in recs])
        write_rows(out / "panel_runs.csv", per_run)
        (out / "panel_summary.json").write_text(json.dumps(total, indent=1) + "\n")
        print(f"wrote {out}/panel_records.csv, panel_runs.csv, panel_summary.json")
    status = 0
    if mismatched:
        print(f"WARNING: {mismatched} records' code checks differ from the seat files' (other judge settings?)")
        status = 1
    if args.check_against:
        diffs, report = check_against(args.check_against, per_run, by_run)
        what = f"{report['runs_compared']} runs" + (" and the overall summary" if report["summary_compared"] else "")
        if diffs:
            print(f"DIFFERS from {args.check_against} ({what}):\n  " + "\n  ".join(diffs))
            status = 1
        else:
            print(f"MATCHES {args.check_against} exactly ({what})")
    return status


# --------------------------------------------------------------------------- main

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["_seat"]:
        return seat_main(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)
    j = sub.add_parser("judge", help="judge result CSVs with each seat (resumes)")
    j.add_argument("-i", "--in", dest="inputs", nargs="+", required=True, help="result CSVs, one per run")
    j.add_argument("-o", "--out", required=True, help="folder for <run>__<seat>.csv")
    j.add_argument("--run", help="run name for a single input (default: its file name without .csv)")
    j.add_argument("--seats", nargs="+", choices=SEAT_NAMES, default=SEAT_NAMES)
    j.add_argument("--config", help="JSON file with seat endpoints (instead of PANEL_* variables)")
    v = sub.add_parser("vote", help="2 of 3 vote and code checks over a judged folder")
    v.add_argument("folder", help="folder with <run>__<seat>.csv for all three seats")
    v.add_argument("-o", "--out", help="write panel_records.csv, panel_runs.csv, panel_summary.json here")
    v.add_argument("--check-against", help="published results folder to compare with "
                                           "(e.g. benchmark/results/v2-2026-10)")
    s = sub.add_parser("seats", help="show the seat endpoints and client settings")
    s.add_argument("--config")
    args = ap.parse_args(argv)

    if args.action == "vote":
        return vote_main(args)
    endpoints = seat_endpoints(args.config)
    if args.action == "seats":
        for name, hf, prefix, _ in SEATS:
            ep = endpoints[name]
            print(f"{name} ({hf} at {REVISIONS[name][:8]}): {ep['base_url'] or f'NOT SET ({prefix}_URL)'} model={ep['model']} "
                  f"concurrency={ep['concurrency']} env={seat_env(name)}")
        return 0
    unset = [f"{n} ({p}_URL)" for n, _, p, _ in SEATS if n in args.seats and not endpoints[n]["base_url"]]
    if unset:
        raise SystemExit("no endpoint for: " + ", ".join(unset) + "; set it in .env or pass --config")
    if args.run and len(args.inputs) > 1:
        raise SystemExit("--run names a single input")
    if args.run and ("__" in args.run or "/" in args.run):
        raise SystemExit("--run may not contain '__' or '/'")
    missing = {}
    for src in args.inputs:
        missing.update(judge_file(Path(src), Path(args.out), args.seats, endpoints, args.run))
    if missing:
        print("Rows still without a verdict (never counted as a fail; run judge again):", file=sys.stderr)
        for k, qs in missing.items():
            print(f"  {k}: {len(qs)} rows: {qs[:10]}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
