#!/usr/bin/env python3
"""Which preliminary results survive the dataset corrections, derived row by row.

    python scripts/14_rerun_status.py            # markdown table to stdout
    python scripts/14_rerun_status.py --json     # the same, machine-readable

Every preliminary run in benchmark/results/preliminary-2026-09/ answered the paper's task set
(benchmark/scope_536_paper.csv) and was graded with the judge prompts as they were before
criteria C1 and C2. benchmark/corrections.json says what changed since. A row is:

  re-run    its question changed, so the recorded answer is to a different question
  re-grade  the answer exists and its input is unchanged, but grading changed: a new expected
            answer, or a category whose judging criterion changed (C1, C2)
  valid     nothing that affects it changed

A run's status is the worst of its rows. Accuracy on the valid rows alone is the one number
that can be read today without re-judging anything.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "benchmark" / "results" / "preliminary-2026-09"


def classify():
    corr = json.loads((ROOT / "benchmark" / "corrections.json").read_text(encoding="utf-8"))
    crit_cats = {c["applies_to"] for c in corr["judging_criteria"]}
    rerun = {c["question_id"] for c in corr["corrections"] if c["action"] == "replace-question"}
    new_answer = {c["question_id"] for c in corr["corrections"] if c["action"] == "replace-answer"}
    with open(ROOT / "benchmark" / "scope_536_paper.csv", encoding="utf-8") as f:
        cat = {r["question_id"]: r["eval_category"] for r in csv.DictReader(f)}
    status = {}
    for q, c in cat.items():
        if q in rerun:
            status[q] = "re-run"
        elif q in new_answer or c in crit_cats:
            status[q] = "re-grade"
        else:
            status[q] = "valid"
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    status = classify()
    pairings = json.loads((RES / "pairings.json").read_text(encoding="utf-8"))["pairings"]
    with open(RES / "verdicts.csv", encoding="utf-8") as f:
        verdicts = list(csv.DictReader(f))

    out = []
    for p in pairings:
        t = p["tag"]
        rows = [(v["question_id"], int(v[t])) for v in verdicts if v.get(t, "") != ""]
        n = {k: sum(1 for q, _ in rows if status[q] == k) for k in ("valid", "re-grade", "re-run")}
        valid = [ok for q, ok in rows if status[q] == "valid"]
        worst = "needs re-run" if n["re-run"] else "needs re-grade" if n["re-grade"] else "valid"
        out.append(dict(tag=t, pairing=f"{p['planner']} x {p['vision']}", rows=len(rows),
                        accuracy_as_graded=round(100 * sum(ok for _, ok in rows) / len(rows), 1),
                        accuracy_valid_rows=round(100 * sum(valid) / len(valid), 1),
                        **{f"rows_{k.replace('-', '_')}": v for k, v in n.items()},
                        status=worst))

    if a.json:
        print(json.dumps(out, indent=2))
        return
    print("| Tag | Pairing | Status | Valid rows | Re-grade rows | Re-run rows "
          "| Accuracy as graded (536) | Accuracy on valid rows |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for r in sorted(out, key=lambda r: -r["accuracy_as_graded"]):
        print(f"| {r['tag']} | {r['pairing']} | {r['status']} | {r['rows_valid']} "
              f"| {r['rows_re_grade']} | {r['rows_re_run']} | {r['accuracy_as_graded']}% "
              f"| {r['accuracy_valid_rows']}% |")


if __name__ == "__main__":
    main()
