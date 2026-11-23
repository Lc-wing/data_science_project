# Plan 6 Milestone M7 — Phase B Deep Time-Series, 27B QLoRA & Report Integration Implementation Plan

> **For agentic workers:** Native execution in this session following `superpowers:test-driven-development` and `superpowers:verification-before-completion`. Every task runs RED -> GREEN -> VERIFY before advancing.

**Goal:** Deliver the complete, TDD-verified Plan 6 Phase B software package (`src/stock_volatility/phase_b/`), one-click Colab single-A100 runner (`scripts/colab_phase_b.py` + `notebooks/phase_b_colab.ipynb` + `PHASE_B.md`), post-Colab CS482 report merger (`scripts/07_integrate_phase_b_report.py`), and M7 milestone documentation, without altering any frozen Phase A data (`data/frozen/`) or Phase A results.

**Architecture:** Mirror `plan_5_spatial_equity`'s dual-track Phase B architecture: lightweight `torch` (CPU wheel from `pytorch-cpu`) lets `PatchTST` (1M & 15M capacity configs) run genuine PyTorch forward/backward/Pinball-Loss/coverage tests locally, while GPU-heavy Hugging Face models (`ProsusAI/finbert`, `Qwen/Qwen3.8-27B` `Qwen3_5ForConditionalGeneration` 4-bit NF4 QLoRA, `amazon/chronos-2`, `amazon/chronos-t5-large`) are lazy-imported behind an explicit `--allow-weight-download` + CUDA gate and executed on Colab A100.

**Tech Stack:** Python 3.11 (`uv`), PyTorch (`pytorch-cpu` locally / CUDA on Colab), `scikit-learn`, `lightgbm`, `scipy`, `pandas`, `pyarrow`, `pypdf`, `pdflatex`/`bibtex`/`pdfinfo`/`pdftotext`; Colab-only: `transformers` (5.x / `qwen3_5`), `peft`, `bitsandbytes`, `accelerate`, `chronos-forecasting`.

**Spec:** `plan_6_stock_volatility/docs/superpowers/specs/2026-11-23-plan6-phase-b-design.md`

## Global Constraints
- `plan_6_stock_volatility/data/frozen/` and sibling workspaces (`plan_2_tourism_hotel/`, `plan_5_spatial_equity/`, `plans/`) MUST remain 100% untouched (`0` diffs).
- Every Phase B entrypoint calls `verify_freeze_manifest()` before reading frozen artifacts.
- Default 27B LLM is `Qwen/Qwen3.8-27B` (`Qwen3_5ForConditionalGeneration`), configurable via `--llm-model`.
- Zero GPU weight downloads unless BOTH `--allow-weight-download` AND `torch.cuda.is_available()` hold.
- No fabricated Phase B metrics in `outputs/phase_b/` or `report/` on disk; report integration tests run inside `tmp_path` with clearly labeled synthetic fixtures.
- `ruff check .` (line-length `100`, rules `E,F,I,W,B,UP`) must exit `0`; all 97 existing Phase A tests + all new M7 tests must pass (`0` failures, `0` skips).
- Non-causal wording (`0` hits of `impact`, `driver`, `elasticity`, `caused`, `causes`) and explicit non-investment-advice boundary enforced everywhere.

## Review Focus (Top 5 Failure Modes Pinned by Tests)
1. **Multimodal `Qwen3_5ForConditionalGeneration` LoRA leakage into `visual`/`mtp`**: `filter_text_lora_target_modules()` must exclude any `visual`/`mtp` weights and target only text-backbone linear projections.
2. **Future-target leakage into Phase B prompts or 60-day windows**: `build_event_prompt()` must never embed `rv_5d_fwd`, `y_log_rv`, `cum_ret_5d_fwd`, or `y_tail`; 60-day sequences must end strictly at `date <= t`.
3. **QLoRA in-sample Train memorization leaking into Stage-2 fusion**: `fit_2024_val_oof_meta_fusion()` must fit Temperature Scaling and meta-fusion strictly on `2024 Val` (4-quarter OOF CV inside 2024 Val), never using in-sample Train QLoRA predictions to over-weight the LLM.
4. **Accidental weight download or CUDA crash on CPU VM**: `scripts/colab_phase_b.py` must refuse GPU routes cleanly with non-zero exit & actionable message when `--allow-weight-download` is omitted or CUDA is absent.
5. **Final CS482 PDF exceeding 5.0 A4 pages or leaking forbidden wording after Phase B Section 6 merge**: `build_and_audit_phase_b_report()` must verify `3.0 <= main_text_pages_est <= 5.0`, every page A4 (`595.28 x 841.89 pts`), `0` forbidden words/placeholders, and complete Phase B claim traceability.

---

### Task 1: Add Lightweight `torch` (`pytorch-cpu`) to `pyproject.toml` & Sync Environment
- **Files**: Modify `pyproject.toml`, `uv.lock`; Test existing `pytest -q` (97 tests).
- Add `torch>=2.5` sourced from `pytorch-cpu` (`https://download.pytorch.org/whl/cpu`) identical to `plan_5_spatial_equity/pyproject.toml`, run `uv lock && uv sync --all-groups`, confirm `torch` imports cleanly on CPU and existing Phase A tests pass untouched.

### Task 2: Phase B Data Contract & Zero-Leakage Prompt / Sequence Builders (`data_contract.py`)
- **Files**: Create `src/stock_volatility/phase_b/__init__.py`, `src/stock_volatility/phase_b/data_contract.py`; Test in `tests/test_phase_b.py`.
- **Produces**:
  - `PHASE_B_CONT_FEATURES: tuple[str, ...]` (14 strictly `<= t` continuous features)
  - `load_frozen_phase_b_inputs(project_root: Path | None = None) -> dict[str, Any]` (verifies `FREEZE_MANIFEST.json`, loads `panel_daily`, `events_text_phase_b`, `feature_manifest`, `split_metadata`)
  - `build_patchtst_windows(panel_df, feature_manifest, lookback=60) -> dict[str, dict[str, np.ndarray]]` (strictly Train-standardized `<= t` 60-day windows for `train`, `val`, `test`, plus ticker/sector integer IDs and targets)
  - `build_event_prompt(row: Mapping[str, Any]) -> str` (strictly `<= t` market & pre-close SEC EDGAR event text ending with `YES`/`NO` tail-risk prompt, `<= 1,800` chars / `<= 512` tokens, zero target fields)
  - `build_event_anchored_datasets(panel_df, events_phase_b_df) -> dict[str, pd.DataFrame]` (active `has_news == 1` & modeling-complete slices for `train` [`<= 5,032`], `val` [`880`], `test` [`899`])

### Task 3: Route A Multivariate `PatchTST` Model & Capacity Ablation (`patchtst.py`)
- **Files**: Create `src/stock_volatility/phase_b/patchtst.py`; Test in `tests/test_phase_b.py`.
- **Produces**:
  - `PatchTSTConfig` dataclass + presets `PATCHTST_1M_CONFIG` (`[0.8M, 2.0M]` params) and `PATCHTST_15M_CONFIG` (`[10M, 20M]` params)
  - `MultivariatePatchTST(nn.Module)` (channel-mixed patch tokenization `P=5, S=5` + learned `ticker` [60] & `sector` [5] embeddings + Pre-LN TransformerEncoder + dual head outputting `[y_log_rv_point, q10, q50, q90, tail_logit]`)
  - `pinball_loss(pred_quantiles, target, quantiles=(0.10, 0.50, 0.90)) -> torch.Tensor`
  - `nominal_interval_coverage(y_true, q_lo, q_hi) -> float`
  - `train_and_predict_patchtst(windows, cfg, seeds=(42,43,44,45,46), device=...) -> dict[str, Any]` (early-stopping on `2024 Val` QLIKE, Train-fold Duan smearing, 2024 Val Temperature Scaling on `tail_logit`, multi-seed ensemble + per-seed dispersion)

### Task 4: Route B GPU Modules — `FinBERT`, `Qwen/Qwen3.8-27B` QLoRA & `Chronos` (`route_b_finbert.py`, `route_b_llm.py`, `route_b_chronos.py`)
- **Files**: Create `src/stock_volatility/phase_b/route_b_finbert.py`, `route_b_llm.py`, `route_b_chronos.py`; Test in `tests/test_phase_b.py`.
- **Produces**:
  - `require_cuda_and_download_flag(allow_weight_download: bool, component: str) -> None` (hard runtime guard)
  - `route_b_finbert.py`: `DEFAULT_FINBERT_MODEL = "ProsusAI/finbert"`, `extract_finbert_event_scores(...)` (scores active event days, returns neutral `0.0` on non-event days, records HF revision)
  - `route_b_llm.py`:
    - `DEFAULT_LLM_MODEL = "Qwen/Qwen3.8-27B"`, `SUPPORTED_LLM_MODELS`
    - `detect_model_loader_kind(model_type: str, architectures: Sequence[str]) -> Literal["image_text_to_text", "causal_lm"]`
    - `filter_text_lora_target_modules(module_names: Sequence[str]) -> list[str]` (includes `language_model` attention/linear-attn/MLP projections, strictly excludes `visual` and `mtp`)
    - `compute_yes_no_log_odds(last_logits: torch.Tensor, yes_token_id: int, no_token_id: int) -> torch.Tensor`
    - `run_llm_zero_shot_and_qlora(...)` (4-bit NF4 `bfloat16`, `r=16, alpha=32, dropout=0.05`, `paged_adamw_8bit`, gradient checkpointing, class-weighted Train BCEWithLogitsLoss on `z_tail`, records HF revision, peak CUDA VRAM MB, wall-clock seconds, trainable/total params, outputs `val`/`test` event-day raw logits for both Zero-Shot and QLoRA)
  - `route_b_chronos.py`: `DEFAULT_CHRONOS_MODELS = ("amazon/chronos-2", "amazon/chronos-t5-large")`, `run_chronos_volatility_forecasts(...)` (quarterly expanding rolling context strictly `<= t`, outputs point + `q10/q50/q90` volatility forecasts & coverage)

### Task 5: Cross-Modal Calibration, 2024-Val OOF Fusion & H3 Evaluation (`fusion_and_eval.py`)
- **Files**: Create `src/stock_volatility/phase_b/fusion_and_eval.py`; Test in `tests/test_phase_b.py`.
- **Produces**:
  - `fit_temperature_scaler(logits_val: np.ndarray, y_val: np.ndarray) -> float` & `apply_temperature_scaling(logits: np.ndarray, temperature: float) -> np.ndarray`
  - `fit_2024_val_oof_meta_fusion(val_df, test_df, score_cols, ...) -> dict[str, Any]` (4-quarter OOF CV strictly inside `2024 Val` to select regularization & Temperature Scaling without Train QLoRA leakage, fusing calibrated text/LLM risk scores with Phase A LightGBM predictions for both volatility `rv_hat` and downside tail probability `p_tail`)
  - `evaluate_phase_b_h3_ablation(phase_a_test_df, phase_b_arms, ...) -> dict[str, Any]` (computes 2025 Full QLIKE, Event-Day QLIKE, 2025 Tail PR-AUC, Event-Day PR-AUC, Earnings-Day PR-AUC, Brier, post-cal ECE, nominal 80% coverage where applicable, paired 5-day date-block bootstrap 95% CIs, and `h=5` Bartlett-PSD HLN-DM + Holm-adjusted `p`-values vs Phase A LightGBM and vs `+FinBERT`, rendering `phase_b_comparison.{csv,md,tex}`)

### Task 6: Post-Colab CS482 Final Report Integration & Automated Audit (`report_integration.py`, `scripts/07_integrate_phase_b_report.py`)
- **Files**: Create `src/stock_volatility/phase_b/report_integration.py`, `scripts/07_integrate_phase_b_report.py`; Test in `tests/test_phase_b.py`.
- **Produces**:
  - `render_phase_b_final_tex(phase_a_claims, table1_latex, phase_b_summary, table2_latex) -> str` (expands CS482 draft with Section 6 `Extension: Deep Time-Series & 27B QLoRA News LLM (H3)` + compact Table 2, keeping total main text strictly inside `[4.2, 5.0]` A4 pages)
  - `build_and_audit_phase_b_report(project_root=None, phase_b_dir=None) -> dict[str, Any]` (refuses to run if real Phase B outputs are missing; compiles `report/CS482_GroupX_StockVolatilityLLM.pdf` deterministically via `SOURCE_DATE_EPOCH`; verifies every page A4, `<= 5.0` A4 main pages, all Phase A + Phase B numeric claims, `0` forbidden causal words/placeholders, and writes `report/phase_b_audit.md`)

### Task 7: One-Click Colab Single-A100 CLI, Notebook & `PHASE_B.md` + M7 Milestone Doc & Git Push
- **Files**: Create `scripts/colab_phase_b.py`, `notebooks/phase_b_colab.ipynb`, `PHASE_B.md`, `docs/milestones/M7_phase_b_colab_and_dl_pipeline.md`; Update `docs/milestones/README.md`, `plan_6_stock_volatility/README.md`, root `README.md`.
- **Verification**:
  - `MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync ruff check .` -> `All checks passed!`
  - `MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync pytest -v` -> all Phase A (`97`) + all Phase B tests pass (`0` failed, `0` skipped)
  - `git check-ignore`, zero sibling/frozen-data diffs, secret scan, explicit `git add`, commit & push M7 to `origin/main`.
