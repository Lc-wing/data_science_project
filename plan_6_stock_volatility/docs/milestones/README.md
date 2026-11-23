# Plan 6 Phase A — Milestone Delivery Ledger & AI Execution Log Convention (里程碑交付台账与 AI 执行记录规范)

> **Delivery Inspection Purpose (最终交付检查说明)**
>
> Per the user's delivery inspection mandate, every milestone (`M0` through `M6`) in Plan 6 Phase A (`plan_6_stock_volatility/`) is documented as a Git-tracked Markdown file in `plan_6_stock_volatility/docs/milestones/`. Each document records:
> 1. **Part A — Milestone Content & Deliverables (里程碑交付内容与技术实现):** Exact files created or modified, data contracts, schemas, cryptographic SHA-256 checksums, and quantitative metrics.
> 2. **Part B — What the AI Agents Did in This Milestone (本里程碑内 AI 完成的工作全记录):** Chronological multi-agent exploration and live upstream API probes, data/engineering traps discovered and resolved, Test-Driven Development (TDD) implementation, automated test suite results, and verdicts from the 5-way independent verification gate (Forensic Auditor + 2 Reviewers + 2 Challengers).

---

## 1. Milestone Delivery Ledger (M0–M6 里程碑交付状态总表)

| Milestone | Document | Title & Scope (里程碑范围) | Status | Baseline / Commit SHA | Verification Gate |
| --- | --- | --- | --- | --- | --- |
| **M0** | [`M0_repo_cleanup_and_scaffold.md`](M0_repo_cleanup_and_scaffold.md) | Repository Cleanup, TeX Live/Poppler Toolchain & `plan_6_stock_volatility/` Scaffold (仓库整理、工具链配置与 Plan 6 脚手架) | `COMPLETED & PUSHED` | `124fc67eb8d578a0495733092598b60aa2b763e0` | `5/5 PASS` (Auditor `CLEAN`, 2 Reviewers `APPROVE`, 2 Challengers `PASS`; `26/26` pytest) |
| **M1** | [`M1_data_ingest_and_day1_audit.md`](M1_data_ingest_and_day1_audit.md) | 60-Stock OHLCV, Macro (`^VIX`/FRED `VIXCLS`/`T10Y2Y`/`SPY`), Paginated SEC EDGAR (`8-K/10-Q/10-K` + `BLK`/`XOM` Predecessor CIKs), Day-1 & Cross-Period Coverage Audit, Milestone Docs (数据采集、Day-1 质量与跨期覆盖率审计及里程碑文档体系) | `COMPLETED & PUSHED` | `5370d1835215f5463b38f3855bffa96211ceba69` | `VERIFIED PASS` (`ruff`: 0 errors; `pytest`: `35/35` passed; offline `--skip-download` bit-identical; `9/9` raw + `5/5` derived SHA-256 verified; `0` sibling diffs; `0` secrets) |
| **M2** | [`M2_panel_splits_and_freeze.md`](M2_panel_splits_and_freeze.md) | Strict `16:00 America/New_York` & `13:00 ET` Early-Close Cutoff, Deduplication & `<= 5`/day Cap, `<= t` Features & `t+1..t+5` Targets, Purged+Embargoed Splits, `analysis_spec.json` & SHA-256 Freeze (`FREEZE_MANIFEST.json`) (面板构建、防泄漏切分与 Phase A→B 冻结清单) | `COMPLETED & PUSHED` | `3c9bba99b09c53a838d787a60b0b7ebb9c4fb025` | `VERIFIED PASS` (`ruff`: 0 errors; `pytest`: `43/43` passed; `6/6` frozen SHA-256 verified; offline `--refreeze` bit-identical; `0` sibling diffs; `0` secrets) |
| **M3** | [`M3_eda_and_h1_panel_har_rv.md`](M3_eda_and_h1_panel_har_rv.md) | Cross-Cycle EDA & H1 Stock-FE Extended Panel HAR-RV (Driscoll–Kraay SE + Holm CIs + 5 Sensitivities) (跨周期 EDA 与 H1 面板扩展 HAR-RV 回归) | `COMPLETED & PUSHED` | Parent: `3c9bba99b09c53a838d787a60b0b7ebb9c4fb025` (`M3` commit on `origin/main`) | `VERIFIED PASS` (`ruff`: 0 errors; `pytest`: `51/51` passed; DK SE cross-checked vs `statsmodels`; offline deterministic; `0` sibling diffs; `0` secrets) |
| **M4** | `M4_h2_forecasting_calibration_and_shap.md` *(planned)* | H2 Out-of-Sample Baselines (Persistence, GJR-GARCH, HAR-RV), Global LightGBM, 2024 Val Calibration, 2025 Holdout Test, `h=5` HLN-DM/Holm & Train-Background TreeSHAP | `PLANNED` | `TBD` | Pending M4 Gate |
| **M5** | `M5_figures_table1_and_cs482_report.md` *(planned)* | 300-DPI Figure 1 & Figure 2, Table 1, Audited CS482 Phase A Draft PDF (3.5–3.8 A4 pp), and `run_phase_a.py --skip-download` Replay CLI | `PLANNED` | `TBD` | Pending M5 Gate |
| **M6** | `M6_e2e_and_adversarial_hardening.md` *(planned)* | Opaque-Box E2E Tiers 1–4 Pass & Tier 5 White-Box Adversarial Hardening | `PLANNED` | `TBD` | Pending M6 Final Gate |

---

## 2. Mandatory Per-Milestone Documentation Policy (每个里程碑的强制记录规范)

Before any milestone (`M0` through `M6`) is committed and pushed to GitHub `origin/main`, the milestone worker and verification gate MUST enforce the following 6 rules:

1. **Dual-Content Requirement (双重记录要求):**
   Every milestone report (`docs/milestones/M<N>_*.md`) MUST document both:
   - **(a) Milestone Content & Deliverables (里程碑交付内容):** Exact files created/modified, schemas, data contracts, SHA-256 checksums, and key quantitative metrics.
   - **(b) What the AI Did (AI 工作全过程记录):** Exploration and live probes, data/tooling traps discovered and fixed, TDD RED-GREEN-REFACTOR implementation steps, automated test suite results, and the 5-way independent verification gate across Forensic Auditor + 2 Reviewers + 2 Challengers.
2. **Update Master Ledger (同步更新总表):**
   Update the status table in `docs/milestones/README.md` (and `plan_6_stock_volatility/README.md`) with the new milestone report link and the previous milestone's pushed Git commit SHA.
3. **Git Trackability Verification (防忽略检查):**
   Run `git check-ignore -v --no-index plan_6_stock_volatility/docs/milestones/*.md` from the repository root and confirm it exits with code `1` and empty output (verifying that no `.gitignore` or `.git/info/exclude` rule hides the milestone documentation).
4. **Reproducible Verification Commands (可复现验证命令):**
   Every milestone report must include copy-pasteable offline CLI, `ruff check`, and `pytest` verification commands with exact exit codes and observed outputs.
5. **Pre-Flight Git Guardrails (Git 提交前安全门禁):**
   Stage files via explicit `git add <paths>` (never `git add -A` or `git add .`), verify zero `.agents/` or local coordination doc (`/PROJECT.md`, `/TEST_INFRA.md`, `/TEST_READY.md`) leakage, verify every staged file is `< 25 MB`, and verify zero secrets/tokens.
6. **Automated Test Enforcement (自动化测试校验):**
   Each milestone's pytest suite must assert that its `docs/milestones/M<N>_*.md` report exists, is non-empty, contains all mandatory sections, is linked from `README.md`, and is not gitignored.

---

## 3. Mandatory Milestone Markdown Template for M2–M6 (后续里程碑 M2–M6 标准 Markdown 模板)

All future milestones (`M2` through `M6`) must copy and populate the following standard template:

```markdown
# Milestone M<N>: <English Title> (里程碑 M<N>：<中文标题>)

- **Milestone**: `M<N> — <English Title> (<中文标题>)`
- **Status**: `COMPLETED & VERIFIED`
- **Parent / Baseline Commit SHA**: `<40-char parent commit SHA>`
- **Date (UTC)**: `<YYYY-MM-DD>`
- **Verification Verdict**: `5/5 PASS (Forensic Auditor CLEAN, Reviewer 1 APPROVE, Reviewer 2 APPROVE, Challenger 1 PASS, Challenger 2 PASS)`

---

## 1. Milestone Objectives & Scope (里程碑目标与范围)

- Summarize the pre-registered technical objectives and strict write-ownership boundaries for Milestone `M<N>`.

## 2. Milestone Deliverables & File Inventory (交付物清单与文件契约)

| File Path (Relative to `plan_6_stock_volatility/`) | Category | Size / Rows | SHA-256 / Contract Summary |
| --- | --- | --- | --- |
| `<path>` | Code / Data / Output / Test / Doc | `<size / rows>` | `<schema & contract summary>` |

## 3. Technical Implementation & Key Metrics (核心技术实现与关键指标)

- Document mathematical formulas, anti-leakage boundaries, and exact quantitative outputs matching committed JSON/CSV/Parquet artifacts.

## 4. What the AI Agents Did in This Milestone (本里程碑内 AI 完成的工作全记录)

### 4.1 Exploration & Live Probes (前期调研与实测探测)
- Record read-only explorer and spec-miner investigations and empirical findings.

### 4.2 Traps Discovered & Engineering Resolutions (发现的陷阱与工程解决)
- Detail every data, numerical, schema, or tooling trap caught during exploration/implementation and its exact engineering fix.

### 4.3 Implementation & TDD Workflow (代码实现与测试驱动开发)
- Document the TDD RED (failing test first) -> GREEN (minimal production code) -> REFACTOR cycle and test coverage breakdown.

### 4.4 Independent 5-Verifier Gate Audit (五路独立验证门禁结果)
- Summarize verification checks performed by Forensic Auditor, Reviewer 1, Reviewer 2, Challenger 1, and Challenger 2.

## 5. Verification Commands & Observed Results (独立验证命令与实测输出)

```bash
cd plan_6_stock_volatility
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync ruff check .
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync pytest -v
```

## 6. Git Hygiene & Delivery Checklist (Git 交付检查清单)

- [ ] Dual-content record complete (Deliverables + AI execution log)
- [ ] `docs/milestones/README.md` and `plan_6_stock_volatility/README.md` updated
- [ ] `git check-ignore -v --no-index plan_6_stock_volatility/docs/milestones/*.md` exited `1`
- [ ] Zero sibling diffs in `plan_2_tourism_hotel/`, `plan_5_spatial_equity/`, and `plans/`
- [ ] All staged files `< 25 MB`, zero `.agents/` or secret leakage
```
