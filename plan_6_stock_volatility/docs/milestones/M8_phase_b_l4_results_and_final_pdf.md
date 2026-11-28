# Milestone M8: Phase B Colab L4 Execution Results & Final Audited CS482 Submission PDF (里程碑 M8：阶段 B Colab L4 实测结果与 CS482 最终提交版 PDF 整合验收)

- **Milestone**: `M8 — Phase B Colab Single-L4 Execution Results & Final CS482 Submission PDF Sign-Off (阶段 B 单卡 L4 实测结果与 CS482 最终报告整合验收)`
- **Status**: `COMPLETED & VERIFIED (FINAL PLAN 6 DELIVERY)`
- **Parent / Baseline Commit SHA**: `1b6827f3beab5611854d5ab880844e924140c0c4` (`M7` + L4/batched QLoRA upgrades on `origin/main`)
- **Date (UTC)**: `2026-11-23`
- **Verification Verdict**: `VERIFIED PASS` (`ruff check .`: 0 errors; `pytest tests/test_phase_b.py -v`: `7/7` passed; full `pytest -v`: `104/104` passed; Final CS482 PDF audit `PASS` `9/9` gates, `61/61` numeric claims verified, `4.46` A4 main pages [`5` pages occupied, `8` total pages], 5-section CS482 rubric layout with Phase B merged into `3 Data Analysis`, `18` verified references, `0` forbidden causal words, `0` frozen/sibling diffs, `0` secrets)

---

## 1. Milestone Objectives & Scope (里程碑目标与范围)

Milestone M8 integrates the **real, non-fabricated Colab NVIDIA L4 (24 GB VRAM) Phase B experimental outputs** (`outputs/phase_b/`) against the cryptographically untouched Phase A frozen dataset (`data/frozen/FREEZE_MANIFEST.json`, SHA-256 `8aa3b379b8beef2b077d5c3e6651733be089eea45e0ac56b738de07a30aa3b1b`), deterministically generates the final publication-grade Table 2 (`phase_b_comparison.{csv,md,tex}`), compiles and audits the unified 5-section CS482 submission report **`report/CS482_GroupX_StockVolatilityLLM.pdf`** (merging Phase B directly into `3 Data Analysis`, `4 Method Evaluation`, and `5 Results and Discussion`), and verifies the entire 104-test Plan 6 repository suite cleanly offline.

1. **Zero Phase A Mutation**: All 6 frozen artifacts in `data/frozen/` and all Phase A econometric/LightGBM baselines remain byte-identical.
2. **Real Colab Single-L4 GPU Execution (`outputs/phase_b/`)**:
   - **Hardware Disclosure**: Colab A100/G4 compute quota was unavailable, so Phase B ran cleanly on a single **NVIDIA L4 (23.66 GB VRAM)** using our VRAM-aware selector (`select_default_llm_for_vram()` -> **`google/gemma-4-12B-it`**, `11.96B` ungated `Gemma4UnifiedForConditionalGeneration`, HF Hub `main` commit `43c9d014b231` last modified `2026-10-30T15:47:58Z`; peak VRAM `15.336 GB / 23.66 GB`, `0` OOMs).
   - **Route A**: 5-seed multivariate `PatchTST` capacity scaling ablation (`1.09M` vs. `14.65M` parameters, CUDA `bf16` autocast, 60-day lookback, ticker/sector embeddings, non-crossing Pinball quantiles).
   - **Route B (Time-Series Foundation Models)**: Strictly causal 5-day volatility forecasting with `amazon/chronos-2` (120M) and `amazon/chronos-t5-large` (710M).
   - **Route B (H3 Event-Anchored Multimodal Text Ablation)**: Frozen `ProsusAI/finbert` (110M) vs. `google/gemma-4-12B-it` Zero-Shot vs. `google/gemma-4-12B-it` 4-bit NF4 QLoRA (`r=16, alpha=32, dropout=0.05`, `328` text-backbone linear modules, `65,568,768` trainable params, 1 epoch across all `4,929` modeling-complete Train pre-close event days, `epoch_loss = 2.0774`, `6,914.6s`), fused via leakage-free **2024-Val 4-quarter OOF meta-fusion** onto Phase A Global LightGBM and evaluated single-shot on **2025 Holdout (`14,700` full rows; `895` scorable pre-close event days)**.
3. **Final Unified 5-Section CS482 Submission PDF & Automated Audit**:
   - Built `report/CS482_GroupX_StockVolatilityLLM.pdf` (`8` total pages = `1` Cover + `5` Main Text pages [`4.46` A4 pp est., single-column, 11pt, 1.5 line spacing, strictly inside `[3.0, 5.0]`] + `1` References page [`18` verified citations in `report/final_references.bib`] + `1` Separate Contribution Statement & AI Usage Declaration page) and `report/phase_b_audit.md` (`VERDICT: PASS`, `9/9` gates, `61/61` numeric claims traced, `0` forbidden causal words, `0` unresolved placeholders).

---

## 2. Final 2025 Single-Shot Holdout Results (表 2：阶段 B 真实样本外消融结果)

All predictions evaluated on identical 2025 Holdout (`N = 14,700` stock-days; `895` scorable pre-close SEC EDGAR event days; `h = 5` Bartlett-PSD HLN-DM Holm-Bonferroni `p`-values vs. Phase A LightGBM; `—` indicates volatility-only models without tail-risk classifier heads):

| Model | 2025 Full QLIKE | Event-Day QLIKE | Tail PR-AUC (Event) | ECE | HLN-DM Holm `p` (Q / B vs LGBM) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase A LightGBM (Price-Only) | `0.2306` | `0.1980` | `0.1231 (0.1076)` | `0.0102` | `Ref.` |
| PatchTST (1M, 5-seed; 80% cov=63.1%) | `0.3142` | `0.2800` | `0.0964 (0.0887)` | `0.4407` | `<0.0001 / <0.0001` |
| PatchTST (15M, 5-seed; 80% cov=68.3%) | `0.2819` | `0.2473` | `0.1129 (0.1027)` | `0.4349` | `<0.001 / <0.0001` |
| chronos-2 (Zero-Shot; 80% cov=89.2%) | `0.3505` | `0.3181` | `—` | `—` | `0.006 / —` |
| chronos-t5-large (Zero-Shot; 80% cov=22.3%) | `0.5848` | `0.5448` | `—` | `—` | `0.004 / —` |
| **LightGBM + FinBERT (110M)** | **`0.2251`** | `0.2005` | **`0.1249 (0.1398)`** | **`0.0101`** | **`0.020`** `/ 0.928` |
| LightGBM + `gemma-4-12B-it` (Zero-Shot) | `0.2252` | `0.2009` | `0.1230 (0.1108)` | `0.0102` | **`0.020`** `/ 0.681` |
| LightGBM + `gemma-4-12B-it` (4-bit QLoRA) | `0.2252` | `0.2008` | `0.1224 (0.1076)` | `0.0102` | **`0.020`** `/ 0.759` |

### Key Scientific Takeaways (核心学术发现)
1. **Route A & Time-Series Foundation Baselines**: Scaling `PatchTST` from `1M -> 15M` improves QLIKE (`0.3142 -> 0.2819`), Tail PR-AUC (`0.0964 -> 0.1129`), and 80% empirical coverage (`63.1% -> 68.3%`), and `amazon/chronos-2` (`0.3505`, `89.2%` coverage) dramatically outperforms `chronos-t5-large` (`0.5848`, `22.3%` coverage); yet all four deep/foundation time-series architectures still trail Phase A tabular LightGBM (`0.2306`, Holm `p <= 0.006`).
2. **H3 Significant Full-Universe QLIKE Gain**: All three 2024-Val OOF meta-fused text arms significantly reduce 2025 full-universe QLIKE from `0.2306` to **`0.2251–0.2252`** (`h = 5` HLN-DM Holm **`p = 0.020`**) while maintaining tight calibration (`ECE = 0.0101–0.0102`).
3. **H3 Event-Day Tail PR-AUC — Honest Null QLoRA Result vs. Domain-Pretrained `FinBERT`**:
   - Domain-pretrained **`ProsusAI/finbert` (110M)** delivers the strongest 2025 event-day tail-risk ranking lift (**`0.1076 -> 0.1398`**, `+30.0%` relative gain).
   - **`google/gemma-4-12B-it` Zero-Shot** modestly improves event-day PR-AUC (`0.1108`; earnings-day PR-AUC `0.1042` vs. `0.0948`), while **1-epoch 4-bit QLoRA (`0.1076`)** yields no incremental event-day PR-AUC gain over Zero-Shot or domain-pretrained `FinBERT` — an honest, pre-registered null QLoRA result highlighting sample-efficiency limits when adapting `65.6M` LoRA parameters on `4,929` short SEC filing titles with sparse (~8%) tail labels.

---

## 3. Milestone Deliverables & File Inventory (交付物清单)

| File Path (Relative to `plan_6_stock_volatility/`) | Category | Purpose Summary |
| --- | --- | --- |
| `outputs/phase_b/phase_b_summary.json` | Output | Real Colab NVIDIA L4 run summary & verified H3 metrics (`1,128 B`) |
| `outputs/phase_b/llm_run_metadata.json` | Output | Exact `google/gemma-4-12B-it` QLoRA telemetry (`15.336 GB` peak VRAM, `65,568,768` trainable params, `6,914.6s`) |
| `outputs/phase_b/phase_b_comparison.{csv,md,tex}` | Output | Unified 8-arm 2025 Holdout comparison table (`$895$` scorable event days; honest `---` for volatility-only Chronos arms) |
| `outputs/phase_b/*_metrics.json`, `outputs/phase_b/*.parquet` | Output | All real Colab L4 prediction & score artifacts (`2.27 MB` total across 20 files, zero model weights) |
| `report/final_report.tex`, `report/final_references.bib` | Report | Unified 5-section CS482 Final Report LaTeX source (Phase B merged into `3 Data Analysis` + Table 2 + `18` verified citations) |
| `report/CS482_GroupX_StockVolatilityLLM.pdf` | Report | Final deterministic CS482 submission PDF (`8` pp, `4.46` A4 main pp, `61/61` claims, `PASS`) |
| `report/phase_b_audit.md` | Audit | Automated 9-gate + 61-claim traceability audit report (`VERDICT: PASS`) |
| `src/stock_volatility/phase_b/{fusion_and_eval,report_integration}.py` | Code | Dynamic `$895$` event caption, volatility-only Chronos `---` formatting, unified 5-section CS482 rubric layout, `18`-entry `final_references.bib` & AI Usage Declaration |
| `scripts/colab_phase_b.py` | CLI | Consistent PatchTST/Chronos `80% cov` table labels, Stage 1–4 `--resume`, and subset `--routes` state hydration |
| `tests/test_phase_b.py` | Test | Added `test_committed_m8_phase_b_outputs_final_pdf_and_milestone_doc` (`7` Phase B tests; `104` repo tests total) |
| `docs/milestones/M8_phase_b_l4_results_and_final_pdf.md` | Doc | This M8 final delivery & AI execution record |
| `docs/milestones/README.md`, `README.md`, `../README.md` | Doc | Updated milestone ledger with M7 SHA `1b6827f` and M8 final completion |

---

## 4. What the AI Agents Did in This Milestone (本里程碑内 AI 完成的工作全记录)

### 4.1 Real-Time Colab L4 Execution Support & Optimization (Colab L4 实测调优)
- Responded to Colab A100/G4 quota unavailability by evaluating `L4 (24 GB)` vs. `v6e-1 TPU (32 GB HBM)` (identifying `bitsandbytes` CUDA dependency & `bf16` HBM math) and live-probing `google/gemma-4-12B-it` (`safetensors` 88 KB header probe confirming `11.96B` params, `gemma4_unified`, tied embeddings, `~11–15 GB` QLoRA peak on L4).
- Diagnosed and fixed two live Colab bottlenecks during user execution:
  1. **Route A `PatchTST` 10–15× GPU speedup** (`87c3664`): replaced CPU `DataLoader` FP32 loop with GPU-resident window tensors, CUDA `bf16` autocast, `batch_size=512`, and per-seed live logs (`PatchTST 1M` finished in `95.0s`, `15M` in `296.8s`).
  2. **Route B Batched QLoRA + `--resume`** (`1b6827f`): switched from single-sample `bs=1` (`ETA ~108 min`) to `train_batch_size=8`, `eval_batch_size=16`, `padding_side="left"`, `logits_to_keep=1`, plus `--resume` so already-completed Stage 1–3 outputs on Colab disk were preserved without recomputation.

### 4.2 Post-Colab Forensic Audit & Visual PDF QA (产出审计与排版目视精修)
1. **Dynamic Event-Day Caption Fix**: Caught that `phase_b_comparison.tex` hardcoded `$899$` while only `895` 2025 pre-close event days have complete `t+1..t+5` forward targets; updated `evaluate_phase_b_h3_ablation()` to compute `int(ev_mask.sum())` dynamically and deterministically refreshed `phase_b_comparison.{csv,md,tex}` directly from the exact Colab `.parquet` predictions.
2. **Honest Volatility-Only Formatting for Chronos**: Fixed `fusion_and_eval.py` so univariate volatility-only `chronos-2` & `chronos-t5-large` report `—` / `---` instead of echoing Phase A LightGBM's tail classification metrics, and excluded them from Brier Holm family adjustment.
3. **Unified 5-Section CS482 Rubric Layout & Visual PDF Inspection (`pdftoppm` + `view_file`)**: Merged Phase B directly into `3 Data Analysis`, `4 Method Evaluation`, and `5 Results and Discussion` (`4.46` A4 main pages across `5` main-text pages, single-column 11pt 1.5× line spacing, `18` verified bibliography entries in `report/final_references.bib`, and explicit AI Usage Declaration & Separate Contribution Statement) and visually inspected every rendered page (`0` overfull boxes, `0` heading/table inversions).

---

## 5. Verification Commands & Observed Results (独立验证命令与实测输出)

```bash
cd plan_6_stock_volatility
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync python scripts/07_integrate_phase_b_report.py
# Verdict: PASS | 8 total pages | 5 main pages occupied | 4.46 A4 main pp | 61/61 claims verified

MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync ruff check .
# All checks passed! (exit 0)

MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync pytest -v
# 104 passed (97 Phase A M0-M6 + 7 Phase B M7-M8; exit 0)

cd ..
git check-ignore -v --no-index \
  plan_6_stock_volatility/docs/milestones/M8_phase_b_l4_results_and_final_pdf.md \
  plan_6_stock_volatility/report/CS482_GroupX_StockVolatilityLLM.pdf \
  plan_6_stock_volatility/outputs/phase_b/phase_b_summary.json
# Expected: empty stdout, exit code 1
```

---

## 6. Final Delivery Sign-Off Checklist (最终交付验收清单)

- [x] Real Colab L4 Phase B outputs (`outputs/phase_b/`, `2.27 MB`, zero model weights, zero fabrication) committed & verified
- [x] Final CS482 submission PDF (`report/CS482_GroupX_StockVolatilityLLM.pdf`) visually inspected & passed `9/9` automated audit gates (`4.46` A4 main pp, `61/61` claims, `0` forbidden words)
- [x] Dual-content M8 milestone record complete and linked from `docs/milestones/README.md`, `plan_6_stock_volatility/README.md`, and root `README.md`
- [x] `data/frozen/FREEZE_MANIFEST.json` (`6/6` SHA-256), `plan_2_tourism_hotel/`, `plan_5_spatial_equity/`, and `plans/` untouched (`0` diffs)
- [x] All `104/104` pytest tests and `ruff check .` passed fresh after final edits; `0` secrets; all blobs `< 25 MB`
