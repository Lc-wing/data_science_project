# Milestone M7: Phase B Engineering Package — Multivariate `PatchTST`, `Chronos`, `FinBERT`, `Qwen/Qwen3.8-27B` Single-A100 QLoRA Runner & Post-Colab Report Merger (里程碑 M7：阶段 B 工程包、`PatchTST`、`Qwen/Qwen3.8-27B` 单卡 A100 QLoRA 套件与报告整合管线)

- **Milestone**: `M7 — Phase B Deep Time-Series, 27B QLoRA Colab A100 Package & Report Integration Pipeline (阶段 B 深度时序、27B QLoRA 单卡 A100 套件与报告合并管线)`
- **Status**: `COMPLETED & VERIFIED (READY FOR COLAB A100 EXECUTION -> M8)`
- **Parent / Baseline Commit SHA**: `547b286e6e85185830608eb3bf0a574669409dc5` (`M6` on `origin/main`)
- **Date (UTC)**: `2026-11-23`
- **Verification Verdict**: `VERIFIED PASS` (`ruff check .`: 0 errors; `pytest tests/test_phase_b.py -v`: `6/6` passed; full `pytest -v`: `103/103` passed [97 Phase A M0–M6 + 6 M7 Phase B]; `git check-ignore`: exit `1`; `0` frozen-data or sibling diffs; `0` secrets)

---

## 1. Milestone Objectives & Scope (里程碑目标与范围)

Because this development VM is CPU-only (`2 vCPU / 7.8 GiB RAM / no NVIDIA GPU`), Milestone M7 mirrors `plan_5_spatial_equity`'s dual-track Phase B architecture: deliver and TDD-verify the **entire Phase B Python codebase, real PyTorch Route A `PatchTST`, GPU-gated Route B `FinBERT` / `Qwen/Qwen3.8-27B` 4-bit NF4 QLoRA / `Chronos` runners, 2024-Val OOF meta-fusion & H3 ablation evaluator, one-click Colab Single-A100 Notebook, `PHASE_B.md`, and post-Colab CS482 final PDF integration pipeline** locally on CPU — with zero fabricated metrics and zero changes to frozen Phase A data (`data/frozen/`).

1. **Strict Phase A Immutability**: All Phase B entrypoints invoke `verify_freeze_manifest()` at startup and never mutate `data/frozen/` or Phase A baselines.
2. **Route A Multivariate `PatchTST` (`patchtst.py`)**: Real PyTorch implementation (`torch==2.15.0+cpu` from `pytorch-cpu` index) with 60-day Train-standardized lookback windows (`14` continuous features), learned ticker (`60`) and sector (`5`) embeddings, non-crossing `q10 <= q50 <= q90` Pinball Loss, Train-fold Duan smearing, nominal 80% interval empirical coverage, and `~1.08M` vs. `~14.6M` capacity presets.
3. **Route B Single-A100 Flagship (`route_b_llm.py`, `route_b_finbert.py`, `route_b_chronos.py`)**:
   - Default 27B LLM: **`Qwen/Qwen3.8-27B`** (parameterized via `--llm-model`), verified ungated (`HTTP 200`, `sha: 37561816e66b`), specifically engineered for its `Qwen3_5ForConditionalGeneration` (`model_type: qwen3_5`) architecture: `AutoModelForImageTextToText` routing, text-only LoRA target filtering on `model.language_model` (excluding `visual` and `mtp`), 4-bit NF4 `bfloat16` QLoRA (`r=16, alpha=32, dropout=0.05`), and unified next-token `logit("YES") - logit("NO")` scoring for apples-to-apples Zero-Shot vs. QLoRA parity across active pre-close event days (`5,032` Train / `880` Val / `899` Test; neutral `0.0` on non-event days).
   - Frozen `ProsusAI/finbert` (110M) baseline and `amazon/chronos-2` (120M) / `amazon/chronos-t5-large` (710M) causal 5-day volatility forecasters.
   - Hard GPU safety gate (`require_cuda_and_download_flag`) refusing weight downloads unless BOTH `--allow-weight-download` AND `torch.cuda.is_available()` hold.
4. **Leakage-Free 2024-Val OOF Calibration, Meta-Fusion & H3 Evaluation (`fusion_and_eval.py`)**:
   - Fits 1-parameter Temperature Scaling and L2-regularized 4-quarter OOF meta-fusion strictly inside `2024 Val` (preventing Train QLoRA memorization bias), evaluating 2025 Full QLIKE, Event-Day QLIKE, Tail PR-AUC (Full / Event / Earnings), ECE, paired 5-day date-block bootstrap 95% CIs, and `h=5` Bartlett-PSD HLN-DM + Holm `p`-values.
5. **Post-Colab Final CS482 Report Merger (`report_integration.py`, `07_integrate_phase_b_report.py`)**:
   - Refuses to build unless real `outputs/phase_b/phase_b_summary.json` and `phase_b_comparison.tex` exist; when present, compiles `report/CS482_GroupX_StockVolatilityLLM.pdf` (`<= 5.0` A4 main pages, every page A4, 60/60 numeric claims, 0 forbidden causal words) and writes `report/phase_b_audit.md`.

---

## 2. Milestone Deliverables & File Inventory (交付物清单与文件契约)

| File Path (Relative to `plan_6_stock_volatility/`) | Category | Size | Contract / Purpose Summary |
| --- | --- | ---: | --- |
| `pyproject.toml`, `uv.lock` | Config | — | Added lightweight `torch>=2.5` (`torch==2.15.0+cpu` via explicit `pytorch-cpu` index, matching Plan 5) |
| `src/stock_volatility/phase_b/__init__.py` | Code | 293 B | Phase B package exports |
| `src/stock_volatility/phase_b/data_contract.py` | Code | 9,631 B | `load_frozen_phase_b_inputs()`, `build_event_anchored_datasets()`, `build_event_prompt()` (zero target leakage), `build_patchtst_windows()` (Train-only standardization) |
| `src/stock_volatility/phase_b/patchtst.py` | Code | 14,232 B | `PATCHTST_1M_CONFIG`, `PATCHTST_15M_CONFIG`, `MultivariatePatchTST`, `pinball_loss()`, `nominal_interval_coverage()`, `train_and_predict_patchtst()` |
| `src/stock_volatility/phase_b/route_b_llm.py` | Code | 13,566 B | `Qwen/Qwen3.8-27B` `Qwen3_5ForConditionalGeneration` loader, `filter_text_lora_target_modules()`, `compute_yes_no_log_odds()`, 4-bit NF4 QLoRA trainer & GPU gate |
| `src/stock_volatility/phase_b/route_b_finbert.py` | Code | 3,602 B | Frozen `ProsusAI/finbert` (110M) event-anchored sentiment logit extractor (GPU-gated) |
| `src/stock_volatility/phase_b/route_b_chronos.py` | Code | 5,490 B | `amazon/chronos-2` (120M) & `amazon/chronos-t5-large` (710M) causal 5-day volatility forecaster |
| `src/stock_volatility/phase_b/fusion_and_eval.py` | Code | 14,397 B | `fit_temperature_scaler()`, `fit_2024_val_oof_meta_fusion()`, `evaluate_phase_b_h3_ablation()` (QLIKE/PR-AUC/ECE, block bootstrap, `h=5` HLN-DM + Holm, Table 2 CSV/MD/LaTeX) |
| `src/stock_volatility/phase_b/report_integration.py` | Code | 9,545 B | `render_phase_b_final_tex()`, `collect_phase_b_claims()`, `build_and_audit_phase_b_report()` (`<= 5.0` A4 pp + 60-claim audit) |
| `scripts/colab_phase_b.py` | CLI | 12,596 B | Single-command Colab A100 Phase B runner (`--allow-weight-download`, `--llm-model Qwen/Qwen3.8-27B`, `--routes`, `--max-train-events`) |
| `scripts/07_integrate_phase_b_report.py` | CLI | 1,393 B | Post-Colab CPU report-merge CLI compiling & auditing `report/CS482_GroupX_StockVolatilityLLM.pdf` |
| `notebooks/phase_b_colab.ipynb` | Notebook | 4,709 B | One-click Colab A100 notebook (clone/pull -> install GPU stack -> run `colab_phase_b.py` -> display table & zip `outputs/phase_b/`) |
| `PHASE_B.md` | Doc | 6,512 B | Complete Phase B local verification, Colab A100 execution, and post-Colab PDF integration guide |
| `tests/test_phase_b.py` | Test | 19,831 B | 6 comprehensive offline CPU TDD tests (data/prompt zero-leakage, real PyTorch PatchTST 1M/15M & gradients, Qwen3.5 LoRA filter & GPU gate, 2024-Val OOF fusion & H3 DM/Holm, synthetic-clone <=5pp PDF build/audit, Colab CLI guard/notebook/M7 docs) |
| `docs/superpowers/specs/2026-11-23-plan6-phase-b-design.md` | Design | 16,899 B | User-approved Phase B architecture specification |
| `docs/superpowers/plans/2026-11-23-plan6-m7-phase-b-package.md` | Plan | 11,123 B | Bite-sized M7 implementation plan |
| `docs/milestones/M7_phase_b_colab_and_dl_pipeline.md` | Doc | — | This M7 delivery & AI execution record |
| `docs/milestones/README.md`, `README.md`, `../README.md` | Doc | — | Updated milestone ledger with M6 SHA `547b286e6e85185830608eb3bf0a574669409dc5` and M7 completion |

---

## 3. Key Quantitative & Architectural Contracts Verified Locally (本机核验指标)

- **Frozen Event-Anchored Dataset (`data/frozen/events_text_phase_b.parquet`)**: `5,032` Train / `880` Val / `899` Test active pre-close ticker-days (max `786` chars, median `145` chars) — fits inside `max_seq_len = 512` with zero truncation and zero subsampling required on Colab A100.
- **Route A `PatchTST` Exact Parameter Counts**: Verified on real PyTorch modules — `PATCHTST_1M_CONFIG` falls inside `[0.8M, 2.0M]` (`~1.08M`), `PATCHTST_15M_CONFIG` falls inside `[10M, 20M]` (`~14.6M`), non-crossing quantiles `q10 <= q50 <= q90` enforced via softplus deltas, and gradients verified finite & non-zero.
- **Post-Colab Final CS482 PDF Budget Proof (Isolated `tmp_path` Clone)**: Synthetic Phase B integration compiled via real `pdflatex -> bibtex -> pdflatex x2` and passed `audit_compiled_pdf()` with `verdict = PASS`, every page exact A4 (`595.28 x 841.89 pts`), `4.0 <= main_text_pages_est <= 5.0` A4 pages, `60/60` numeric claims traced, and `0` forbidden causal words or unresolved artifacts.

---

## 4. What the AI Agents Did in This Milestone (本里程碑内 AI 完成的工作全记录)

### 4.1 Live Hardware & Hugging Face Metadata Probes (前期调研与实测探测)

- Verified VM compute resources (`2 vCPU`, `7.8 GiB RAM`, no CUDA GPU) and reviewed `plan_5_spatial_equity/PHASE_B.md` & `notebooks/phase_b_colab.ipynb` to align with the repository's dual-track Colab A100 pattern.
- Probed live Hugging Face metadata APIs (`huggingface.co/api/models/...` + `config.json` + `model.safetensors.index.json`, zero weights downloaded): confirmed `Qwen/Qwen3.8-27B`, `ProsusAI/finbert`, `amazon/chronos-2`, and `amazon/chronos-t5-large` are all **HTTP 200 & ungated (`gated: false`)**.

### 4.2 Critical Traps Discovered & Resolved Before Colab Execution (提前规避的工程陷阱)

1. **`Qwen/Qwen3.8-27B` Multimodal `Qwen3_5ForConditionalGeneration` Architecture Trap**:
   - *Discovery*: Live `config.json` & `model.safetensors.index.json` probe revealed `Qwen/Qwen3.8-27B` declares `model_type: "qwen3_5"`, `architectures: ["Qwen3_5ForConditionalGeneration"]`, `transformers_version: "5.8.0.dev0"`, a multimodal vision wrapper (`model.visual.*` + `mtp.*`), and a hybrid 64-layer language backbone (`48` `linear_attn` + `16` `self_attn` layers). Blindly using `AutoModelForCausalLM` or unguarded `target_modules` would crash on Colab A100 or waste LoRA capacity on unused vision layers.
   - *Resolution*: Implemented `detect_model_loader_kind()` (`AutoModelForImageTextToText` vs. `AutoModelForCausalLM`), `filter_text_lora_target_modules()` (targeting text-backbone `q_proj/k_proj/v_proj/o_proj/in_proj_qkv/out_proj/gate_proj/up_proj/down_proj` while strictly excluding `visual`, `vision`, `merger`, `mtp`, `lm_head`), and universal next-token `logit("YES") - logit("NO")` scoring.
2. **QLoRA In-Sample Train Memorization Bias in Stage-2 Fusion**:
   - *Resolution*: Designed `fit_2024_val_oof_meta_fusion()` to fit Temperature Scaling and select L2 regularization (`C` / `alpha`) strictly via 4-quarter OOF CV inside `2024 Val` (leaving `2025 Test` 100% single-shot untouched and avoiding Train-split QLoRA overfitting).
3. **Zero Fabricated Metrics Guarantee**:
   - *Resolution*: `build_and_audit_phase_b_report()` raises `FileNotFoundError` unless real Colab A100 outputs exist in `outputs/phase_b/`; synthetic integration tests run strictly inside pytest's ephemeral `tmp_path`.

### 4.3 Implementation & TDD Workflow (代码实现与测试驱动开发)

Following `superpowers:brainstorming` -> `superpowers:writing-plans` -> `superpowers:test-driven-development`:
1. **Design & Plan**: Authored `docs/superpowers/specs/2026-11-23-plan6-phase-b-design.md` (approved by user) and `docs/superpowers/plans/2026-11-23-plan6-m7-phase-b-package.md`.
2. **RED Phase**: Synced lightweight `torch==2.15.0+cpu` via `pytorch-cpu`, authored `tests/test_phase_b.py` (6 comprehensive tests), and verified initial failure (`6 failed in 3.80s`).
3. **GREEN & REFACTOR Phase**: Implemented all 7 modules in `src/stock_volatility/phase_b/`, `scripts/colab_phase_b.py`, `scripts/07_integrate_phase_b_report.py`, `notebooks/phase_b_colab.ipynb`, `PHASE_B.md`, and this M7 doc; cleaned 9 initial `ruff` line-length/import issues (including inside `.ipynb` cells); ran full fresh verification (`ruff` 0 errors, `6/6` Phase B tests, `103/103` total repository tests).

---

## 5. Verification Commands & Observed Results (独立验证命令与实测输出)

```bash
cd plan_6_stock_volatility
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync ruff check .
# Expected: All checks passed! (exit 0)

MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync pytest tests/test_phase_b.py -v
# Expected: 6 passed (exit 0)

MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync pytest -v
# Expected: 103 passed (97 Phase A M0-M6 + 6 Phase B M7; exit 0)

cd ..
git check-ignore -v --no-index plan_6_stock_volatility/PHASE_B.md plan_6_stock_volatility/notebooks/phase_b_colab.ipynb plan_6_stock_volatility/docs/milestones/*.md
# Expected: empty stdout, exit code 1
```

---

## 6. Git Hygiene & Next Step (M8 Colab A100 Execution)

- [x] Dual-content record complete (M7 deliverables + live HF recon + traps fixed + TDD log)
- [x] `PHASE_B.md`, `notebooks/phase_b_colab.ipynb`, `docs/milestones/README.md`, `plan_6_stock_volatility/README.md`, and root `README.md` updated and git-tracked
- [x] Zero diffs in `data/frozen/`, `plan_2_tourism_hotel/`, `plan_5_spatial_equity/`, and `plans/`
- [x] Zero fabricated Phase B metrics committed; zero GPU weights downloaded locally; zero secrets
- **Next Step (Milestone M8)**: Open `plan_6_stock_volatility/notebooks/phase_b_colab.ipynb` in Colab with an **NVIDIA A100 (40GB/80GB)** runtime, run all cells to generate `outputs/phase_b/`, copy `outputs/phase_b/` into the repository, and run `uv run python scripts/07_integrate_phase_b_report.py` to compile & audit the final submission `report/CS482_GroupX_StockVolatilityLLM.pdf`.
