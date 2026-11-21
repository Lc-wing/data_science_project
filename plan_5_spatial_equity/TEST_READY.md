# TEST_READY — Plan 5 Phase A E2E (A18)

Status: **ready for the 12/03 Go/No-Go checkpoint**, on the toolchain below.
This file records the M5 run against `origin/main` at `bdfad99` (PR #5 merged) plus the Tier 5 single-fault cases added on this branch.

No M1–M4 product code was changed. No scientific threshold was relaxed to obtain a pass.

## Command

From the Plan 5 folder (`plan_5_spatial_equity/`), with `uv` on `PATH`:

```bash
uv sync --all-groups
uv run pytest tests/e2e --project-root .
```

Omit `--project-root` only when the default (the Plan 5 folder that contains `tests/e2e/`) is the tree under test. Do not set `MACAO_E2E_IN_SELF_TEST`; that variable is the child-process recursion guard.

`tests/e2e/test_e2e_self_validation.py` is in scope (`TEST_INFRA.md` §§3 and 5). `tests/e2e/test_tier5_adversarial.py` is the Tier 5 hardening (`TEST_INFRA.md` §6).

## Toolchain this run used

| Tool | Role |
|---|---|
| `uv` 0.13.9, CPython 3.12.3, locked `.venv` | `uv run pytest` |
| Poppler `pdfinfo` / `pdftotext` 24.02.0 | PDF page size and main-text fraction |
| pdfTeX 3.141592653-2.6-1.40.25 (TeX Live 2023/Debian), `pdflatex` and `bibtex` | Tier 4 scenario 5 replays `scripts/run_phase_a.py --skip-download`, which compiles the report |

Without `pdflatex`, scenario 5 exits 1 inside `src/macao_equity/viz/report.py` (`FileNotFoundError: pdflatex`). That is an environment prerequisite of the existing CLI contract, not a test or product defect. This run installed TeX Live and re-ran the scenario; it passed with no code change.

## Counts

One command, 2026-11-20, after the Tier 5 cases and the diagnostic fixes below:

```text
uv run pytest tests/e2e --project-root . --tb=line -q
204 passed in 28.99s
exit code 0
```

Collection of that same tree (no skips):

| Suite | File | Tests | Result |
|---|---|---:|---|
| Tier 1 | `tests/e2e/test_tier1_feature_coverage.py` | 91 | passed |
| Tier 2 | `tests/e2e/test_tier2_boundary_corner.py` | 71 | passed |
| Tier 3 | `tests/e2e/test_tier3_cross_feature.py` | 12 | passed |
| Tier 4 | `tests/e2e/test_tier4_real_world_scenarios.py` | 9 | passed |
| Self-validation | `tests/e2e/test_e2e_self_validation.py` | 16 | passed |
| Tier 5 | `tests/e2e/test_tier5_adversarial.py` | 5 | passed |
| **Total** | `tests/e2e` | **204** | **0 failed, 0 errors, 0 skipped** |

The self-validation valid-synthetic test requires the child process to report exactly **183 passed, 0 failed, 0 errors, 0 skipped** on Tiers 1–4 (91+71+12+9). That assertion passed. Each of the 15 bundled mutation checks exited 1 with its named diagnostic and `errors == 0`.

Tier 5 adds the five single-fault mutations that §5 names separately and that self-validation only injected inside `graph_edge_and_island_drop` or `table1_and_pdf_report_violations`:

| Fault | Required diagnostics | Result |
|---|---|---|
| `graph_edge_zero_or_gt10` | `snap singularity`, `exceeds 10.0`, `Asymmetric walk_time_min` | detected, child exit 1 |
| `graph_dropped_islands` | `Taipa-Cotai-Coloane pedestrian edges missing` | detected, child exit 1 |
| `table1_numeric_drift` | `missing from Table 1` | detected, child exit 1 |
| `pdf_out_of_budget_and_us_letter` | `is not A4` and `outside Phase A budget` | detected, child exit 1 |
| `figure_low_dpi_150` | `DPI below 300` | detected, child exit 1 |

## Test edits (diagnostics only)

These keep the same pass condition on a valid tree. They stop a fail-fast assert from hiding a second clause of a named fault.

- `test_t2_graph_edges_walk_time_strictly_positive_le_10` reports every edge outside `(0, 10]`, so `t=0` does not hide `t>10`.
- `test_t2_graph_edges_undirected_symmetry_unique` names an unequal reverse edge `Asymmetric walk_time_min`.
- `test_t2_pdf_a4_dimensions_every_page` still requires at least 4 pages and A4 dimensions, and reports both when a 3-page US Letter file hits the page-count check first.
- `test_t2_graph_edges_no_self_loops_reciprocal_weight` rejects `t<=0` before dividing, so that fault is an assertion failure rather than `ZeroDivisionError`. Tier 5 does not select this test; `t=0` is already required on the walk-time check.

## Remaining gaps

- Live re-download (`scripts/run_phase_a.py` without `--skip-download`) is not part of this suite. Scenario 5 replays from the frozen and interim inputs.
- The 15 self-validation checks still include two bundled mutators. Tier 5 covers those faults one at a time. The bundles were left in place.
- Cover and contribution student-name slots may stay unresolved. The existing tests exempt those slots and still forbid `NaN` / `TODO` / `TBD_NUM` / `??` in analytical text. Nothing new was opened there.

No E2E test was left failing, and none was weakened to force a pass.

## Forensic spot check (same tree)

- SHA-256 of `data/frozen/FREEZE_MANIFEST.json`: `7c793b6f21bf20e87c4cad20aef3bebe2649a24588b5e81f9f3223daa8a72ac8`
- Every path in that manifest matches the on-disk SHA-256 (0 mismatches).
- All 145 rows in `report/go_nogo_audit.md` are `PASS`. The 134 CSV/JSON/config source values match the reported figures within the audit's stated absolute error. Population 681,293 / 82,751, cell counts 287 and 63, facility counts 240 / 11 / 15 / 587 / 85, 23 zones, 7 parishes, station `9488798` imputed `capacity=12.0`, and `WALK_SPEED_KMH = 4.5` match the files named in the audit.
- Poppler measure of the committed PDF: 6 pages, each 595.276 × 841.89 pt, main text **3.763** A4 pages (audit states 3.76).
