# Milestone M9: Report-Aligned Repository Layout Unification & Documentation Rewrite (里程碑 M9：与五节报告一致的仓库结构合并与文档重写)

- **Milestone**: `M9 — Report-Aligned Repository Layout Unification & Documentation Rewrite (仓库结构统一合并与文档重写)`
- **Status**: `COMPLETED & VERIFIED`
- **Parent / Baseline Commit SHA**: `23fd06570938a0b620fda215136300f4381a3b13`
- **Date (UTC)**: `2026-11-27`
- **Verification Verdict**: `VERIFIED PASS (ruff: 0 errors; pytest: 104/104 passed; report_audit.md: PASS 9/9 gates, 61/61 claims verified, 4.46 A4 main pp; 0 frozen/sibling diffs)`

---

## 1. Milestone Objectives & Scope (里程碑目标与范围)

- Eliminate the historical Phase A vs. Phase B split across source modules, CLI scripts, notebooks, outputs, and report directories so the entire `plan_6_stock_volatility/` workspace mirrors the unified 5-section CS482 submission report (`report/CS482_GroupX_StockVolatilityLLM.pdf`).
- Merge deep time-series (`PatchTST`, `Chronos`) and event-anchored financial LLM (`FinBERT`, `google/gemma-4-12B-it` / `Qwen/Qwen3.8-27B` 4-bit NF4 QLoRA) training and meta-fusion directly inside the Data Analysis package (`src/stock_volatility/analysis/dl/`).
- Consolidate all outputs cleanly by artifact type (`outputs/{figures,metrics,predictions,tables}/`), remove obsolete Phase A draft reports and standalone `PHASE_B.md`, and rewrite `README.md` and `TEST_READY.md`.
- Preserve `data/frozen/*` (`FREEZE_MANIFEST.json`) and `data/raw/*` 100% bit-for-bit untouched.

## 2. Milestone Deliverables & File Inventory (交付物清单与文件契约)

| File Path (Relative to `plan_6_stock_volatility/`) | Category | Summary |
| --- | --- | --- |
| `src/stock_volatility/analysis/{eda,panel_har_rv,baselines,lgbm,treeshap}.py` | Code | Unified Section 3 classical econometrics & tabular ML modules |
| `src/stock_volatility/analysis/dl/{datasets,patchtst,chronos,finbert,qlora_llm,fusion}.py` | Code | Unified Section 3.3 deep time-series & event-anchored QLoRA LLM modules |
| `src/stock_volatility/report.py` | Code | Unified 5-section CS482 LaTeX/BibTeX/PDF builder & 61-claim auditor |
| `scripts/{01..07}_*.py`, `scripts/run_pipeline.py` | CLI | Sequential Steps `01`–`07` + unified offline `run_pipeline.py` CLI |
| `notebooks/dl_gpu_training_colab.ipynb` | Notebook | Colab single-GPU L4/A100 runner pointing to `scripts/06_dl_and_llm_gpu.py` |
| `outputs/{figures,metrics,predictions,tables}/` | Outputs | Orthogonal artifact directories (all 13 `.parquet` files in `predictions/`, `h3_dl_summary.json` in `metrics/`, `table2_h3_ablation.*` in `tables/`) |
| `report/{CS482_GroupX_StockVolatilityLLM.pdf,report.tex,references.bib,report_audit.md}` | Report | Single authoritative CS482 submission package (`4.46` A4 main pp, `61/61` claims) |
| `README.md`, `TEST_READY.md` | Docs | Rewritten unified documentation aligned with the 5-section report |

## 3. Verification Commands & Observed Results (独立验证命令与实测输出)

```bash
cd plan_6_stock_volatility
uv run --no-sync ruff check .
uv run --no-sync python scripts/run_pipeline.py --report-only
uv run --no-sync pytest -v
```
