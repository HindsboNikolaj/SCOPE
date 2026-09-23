#!/usr/bin/env python3
"""Build benchmark/scope_536.csv from the paper's task set plus the audit corrections.

    python scripts/apply_corrections.py            # writes benchmark/scope_536.csv
    python scripts/apply_corrections.py --check    # exits 1 if the committed file is stale

The input is benchmark/scope_536_paper.csv, the 536-task set the HRI '26 numbers were
computed on. benchmark/corrections.json lists every adjudicated change by question_id. Each
entry records the text it expects to find, so a correction applied to the wrong row, or to a
file that has moved on, fails loudly instead of silently editing something else.
"""
import argparse
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "benchmark" / "scope_536_paper.csv"
CORR = ROOT / "benchmark" / "corrections.json"
OUT = ROOT / "benchmark" / "scope_536.csv"


def build() -> str:
    with open(BASE, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    by_id = {r["question_id"]: r for r in rows}
    if len(by_id) != len(rows):
        sys.exit(f"{BASE.name}: duplicate question_id")

    for c in json.loads(CORR.read_text(encoding="utf-8"))["corrections"]:
        qid, action = c["question_id"], c["action"]
        r = by_id.get(qid)
        if r is None:
            sys.exit(f"{qid}: not in {BASE.name}")
        for field, key in (("question", "question"), ("expected_answer", "expected_answer")):
            if r[field] != c[key]:
                sys.exit(f"{qid}: {field} does not match corrections.json; refusing to apply")
        if action == "keep":
            continue
        if action == "replace-question":
            old_q, new_q = r["question"], c["new_question"]
            r["question"] = new_q
            # The expected tool calls sometimes quote the question verbatim as the VLM
            # instruction; keep them in step with the new wording.
            for col in ("expected_tool_order_json", "expected_tool_args_json",
                        "expected_tool_order_signature"):
                if old_q in r[col]:
                    r[col] = r[col].replace(old_q, new_q)
        elif action != "replace-answer":
            sys.exit(f"{qid}: unknown action {action!r}")
        r["expected_answer"] = c["new_expected_answer"]
        for col, val in c.get("set_fields", {}).items():
            if col not in r:
                sys.exit(f"{qid}: unknown column {col!r}")
            r[col] = val

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=header, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify the committed file is current")
    a = ap.parse_args()
    text = build()
    if a.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            sys.exit(f"{OUT.relative_to(ROOT)} is stale; run scripts/apply_corrections.py")
        print(f"{OUT.relative_to(ROOT)} is current")
        return
    OUT.write_text(text, encoding="utf-8")
    n = sum(1 for _ in csv.DictReader(io.StringIO(text)))
    print(f"wrote {OUT.relative_to(ROOT)} ({n} rows)")


if __name__ == "__main__":
    main()
