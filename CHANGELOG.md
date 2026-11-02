# Changelog

## 0.2.0 (2026-11-01)

- **Dataset v2** (`benchmark/scope_v2.csv`), now the default: the paper's 536 tasks with 187 rows
  corrected and a new `also_accept` column, built from v1 plus `benchmark/corrections.json` by
  `scripts/apply_corrections.py`. Every change and its reason is in `benchmark/CHANGES_v2.md`.
  The paper's tasks stay available as v1 (`benchmark/archive/scope_536_v1.csv`, `--dataset v1`).
- The 541-row `benchmark/scope_536.csv` that 0.1.0 shipped is removed; `benchmark/archive/`
  maps its rows to v1.
- **v2 results**: 26 planner and vision pairings on all 536 tasks, graded by a three-judge panel
  (`scripts/13_judge_panel.py`, 2 of 3 to pass), with per-task verdicts for every run in
  `benchmark/results/v2-2026-10/`. See `docs/RESULTS.md`. The paper's numbers are unchanged in
  `docs/PAPER_RESULTS.md`.
- **Faster runs**: `scripts/run_fast.py` runs a pairing as six scene workers with GPU capture and
  resume; a full pairing took 30 min instead of 157 on A100s. Headless servers are supported.
- **Install**: Blender's own Python packages (`scripts/install_blender_deps.py`), a scene
  download that fetches only `scenes/` with fallbacks, camera presets and the Blender check inside
  the GPU image, and the judge on Python 3.10 and 3.11.
- Default models: gemma-4-26B-A4B-it as planner and vision on one server.
- Each task now starts from Home; `SCOPE_HOME_PER_TASK=0` restores the 0.1.0 behaviour.

## 0.1.0

The code and benchmark that accompany the HRI 2026 paper.
