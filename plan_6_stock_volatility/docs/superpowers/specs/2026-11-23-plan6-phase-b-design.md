# Design Doc: Plan 6 Phase B — Deep Time-Series (`PatchTST`), `Chronos`, `FinBERT` & `Qwen/Qwen3.8-27B` Single-A100 4-bit QLoRA Extension

- **Date**: `2026-11-23`
- **Project**: `plan_6_stock_volatility/` (CS482 Plan 6, v2.2)
- **Frozen Phase A Baseline**: Commit `547b286e6e85185830608eb3bf0a574669409dc5` (`M0–M6` complete, `97/97` pytest passing, `FREEZE_MANIFEST.json` SHA-256 `8aa3b379b8beef2b077d5c3e6651733be089eea45e0ac56b738de07a30aa3b1b`)
- **Status**: `PROPOSED — Awaiting User Approval`

---

## 1. Intent, Goals, Non-Goals & Anti-Goals

### Intent & Why
Phase A froze a zero-leakage 2019–2025 cross-sector panel (`105,600` rows, `60` stocks × `1,760` NYSE trading days) and `6,822` strictly pre-close (`< 16:00 ET` / `< 13:00 ET` early-close) SEC EDGAR corporate event ticker-days (`5,032` Train / `880` Val / `899` Test, max `786` chars), and built an audited `4.00`-page CS482 Phase A Draft PDF (`1.00` full A4 page remaining under the `5.00`-page CS482 ceiling).

Phase B tests **Hypothesis H3** and extends Phase A along the two pre-registered deep learning routes in `plans/plan_6_stock_volatility_news_llm.md` without modifying a single byte of `data/frozen/` or refitting Phase A's published baselines:
- **Route A (Multivariate Deep Time-Series, trained from scratch)**: `PatchTST` with ticker & sector embeddings (`60`-day lookback window), predicting 5-day forward log-RV point estimates + nominal 80% quantile interval (`q10 / q50 / q90` via Pinball Loss) across 5 seeds (`42..46`), plus a `~1M` vs `~15M` capacity ablation.
- **Route B (Single-A100 Flagship LLM + Foundation Time-Series)**:
  1. **News/Event Multimodal Channel (H3)**: Compare **(a) Phase A Price-Only LightGBM** vs **(b) +`ProsusAI/finbert` (110M)** vs **(c) +`Qwen/Qwen3.8-27B` Zero-Shot** vs **(d) +`Qwen/Qwen3.8-27B` 4-bit NF4 QLoRA** (parameterized via `--llm-model`, supporting `Qwen/Qwen3.8-27B` default and fallbacks `Qwen/Qwen3-32B`, `Qwen/Qwen3-30B-A3B`, `google/gemma-4-31B`, `meta-models/Muse-Glimmer-30B`).
  2. **Dedicated Time-Series Foundation Channel**: `amazon/chronos-2` (120M) and `amazon/chronos-t5-large` (710M, LoRA / zero-shot configurable).

### Goals
1. **Strict Phase A Immutability**: Phase B ONLY reads `data/frozen/` (verified via `verify_freeze_manifest()` at startup) and `outputs/tables/h2_{val,test}_predictions.parquet` / `h2_test_metrics.csv`. Zero changes to frozen data or Phase A results.
2. **Leakage-Free H3 Protocol**:
   - Event prompts use strictly pre-close (`<= t`) event text (`events_text_phase_b.parquet`) + strictly `<= t` daily market state (`ret_1d`, `daily_range_pct`, `rv_d`, `rv_w`, `rv_m`, `vix_close`, `t10y2y`, `sector`). Zero `t+1..t+5` target values ever enter prompts or inputs.
   - `Qwen/Qwen3.8-27B` QLoRA trains strictly on `split == "train"` active event-anchored days (`<= 5,032` rows with `has_full_features & has_full_target`; zero embargo/val/test rows).
   - **Out-of-Fold / Validation Calibration**: LLM raw logits and `FinBERT` probabilities are calibrated strictly on `2024 Val` via 1-parameter Temperature Scaling (`guo2017calibration`) before fusion, and fused LightGBM / HAR-RV models reuse Phase A's exact quarterly expanding-window purged schedule (5-day purge before every 2025 quarter) for single-shot `2025 Test` evaluation.
   - **No-News Days**: Non-event trading days receive deterministic neutral score `0.0` (logit `0.0`) paired with the existing `has_news` indicator so 27B inference runs ONLY on active event-anchored days (`880` Val + `899` Test + optional OOF Train — fast and zero wasted A100 compute!).
3. **Colab Single-A100 Readiness + Local CPU Verification**:
   - All Phase B Python modules (`src/stock_volatility/phase_b/`), tests (`tests/test_phase_b.py`), Colab CLI (`scripts/colab_phase_b.py`), post-Colab report integration CLI (`scripts/07_integrate_phase_b_report.py`), Colab notebook (`notebooks/phase_b_colab.ipynb`), and documentation (`PHASE_B.md`, `docs/milestones/M7_phase_b_*.md`) are fully built, linted (`ruff`), and TDD-tested locally on CPU before pushing to GitHub.
   - All GPU weight downloads & CUDA runs are hard-gated behind `--allow-weight-download` + `torch.cuda.is_available()` (mirroring `plan_5_spatial_equity`), so local CPU `pytest` runs 100% offline with zero accidental downloads.
4. **Honest Academic Reporting (Null Results Accepted)**:
   - Report integration automatically injects Phase B's 6-column comparison table (`2025 Full QLIKE`, `Event-Day QLIKE`, `2025 Tail PR-AUC`, `ECE`, `DM-p vs LightGBM (Holm)`, plus nominal 80% interval empirical coverage for `PatchTST`) and Section 6 (`~0.6` A4 pages) into `report/CS482_GroupX_StockVolatilityLLM.pdf`, keeping total main text strictly inside `[4.3, 4.9] <= 5.0` A4 pages with automated numeric & non-causal audit (`report/phase_b_audit.md`).

### Non-Goals & Anti-Goals
- **Non-Goal**: Running 27B QLoRA, 710M Chronos, or 5-seed 15M `PatchTST` training on this 2-vCPU / 7.8 GiB CPU-only VM (physically impossible; those run on Colab A100 via `notebooks/phase_b_colab.ipynb`).
- **Anti-Goal**: Fabricating, mocking, or hallucinating Phase B test metrics into the final CS482 report before Colab A100 results exist (`07_integrate_phase_b_report.py` refuses to build the Phase B final PDF unless real verified Colab output JSON/CSV files exist on disk, while unit tests validate the integration pipeline inside `tmp_path` using clearly labeled synthetic fixtures).
- **Anti-Goal**: Weakening any Phase A test, modifying `data/frozen/`, or touching sibling workspaces (`plan_2_tourism_hotel/`, `plan_5_spatial_equity/`, `plans/`).

---

## 2. Live Hugging Face Recon Findings (Critical Architecture Details)

Live metadata probes against `huggingface.co/api/models` (2026-11-23, zero weights downloaded) confirmed all four target checkpoints are **public & ungated (`gated: false`, HTTP 200)**:

| Model ID | HF Revision (`sha` prefix) | Architecture / `model_type` | Engineering Implication |
| --- | --- | --- | --- |
| `Qwen/Qwen3.8-27B` | `37561816e66b` | `Qwen3_5ForConditionalGeneration` (`qwen3_5`, `transformers_version: 5.8.0.dev0`) | **Critical catch**: VLM wrapper (`AutoModelForImageTextToText`), NOT `AutoModelForCausalLM`! Language backbone (`model.language_model`, 64 layers, `hidden_size=5120`) alternates 48 `linear_attn` (`in_proj_qkv, in_proj_z, out_proj`) + 16 `self_attn` (`q_proj, k_proj, v_proj, o_proj`) + MLP (`gate_proj, up_proj, down_proj`). LoRA MUST dynamically target text-backbone projections only and explicitly exclude `visual` and `mtp` modules. |
| `ProsusAI/finbert` | `0bc2ac10dda6` | `BertForSequenceClassification` (`bert`) | Standard 3-class financial sentiment (`positive/negative/neutral`); extract `p_neg`, `p_pos`, `sentiment_polarity = p_pos - p_neg` on event days (110M params, runs in <30s on A100). |
| `amazon/chronos-2` | `29ec3766d36d` | `Chronos2Model` (`chronos-forecasting`) | Native quantile forecasting (`[0.01..0.99]`, including `q10/q50/q90`), 120M params, zero-shot / fine-tunable via `Chronos2Pipeline`. |
| `amazon/chronos-t5-large` | `0e46c9c7e2e9` | `T5ForConditionalGeneration` (`chronos-forecasting`) | 710M params, probabilistic quantile forecasting (`ChronosPipeline` / optional LoRA fine-tune on 2019–2023 Train series). |

---

## 3. Approaches Considered

| Approach | Description | Trade-offs | Decision |
| --- | --- | --- | --- |
| **A (Selected — Plan 5 Dual-Track Architecture)** | Build complete, modular `src/stock_volatility/phase_b/` package (PyTorch CPU wheel for real `PatchTST` unit tests; lazy-imported HF/PEFT/Chronos gated by `--allow-weight-download` + CUDA), full CPU TDD suite (`tests/test_phase_b.py`), one-click Colab A100 CLI (`scripts/colab_phase_b.py`) + Notebook (`notebooks/phase_b_colab.ipynb`) + `PHASE_B.md`, and post-Colab report integration (`scripts/07_integrate_phase_b_report.py`). | Zero wasted CPU hours; real PyTorch `PatchTST` math/gradients verified locally; zero risk of OOM or broken `Qwen3_5ForConditionalGeneration` loader on Colab; clean handoff to Colab A100 and one-command report merge afterwards. | **Selected by User** |
| **B (Local CPU Training of Small Models + Colab for 27B)** | Also run 1M `PatchTST` & `FinBERT` locally on 2-vCPU VM before Colab. | Slow on 2 vCPUs (~15–30 min CPU burn, installs large `transformers` cache locally) while Colab A100 runs both in <2 minutes alongside 27B QLoRA. | Rejected |
| **C (Colab-Only Notebook Without Local Package/Tests)** | Put all Phase B code inside Jupyter cells without `src/` modules or pytest. | Untestable before burning Colab A100 units; violates TDD & Plan 5 reproducibility standards. | Rejected |

---

## 4. Detailed Architecture & Module Contracts

### 4.1 Package Layout (`plan_6_stock_volatility/`)
```text
plan_6_stock_volatility/
├── PHASE_B.md                                # Complete Colab A100 + post-Colab merge guide (mirrors Plan 5)
├── notebooks/
│   └── phase_b_colab.ipynb                   # One-click Colab A100 notebook (clone/pull -> install -> run -> display table)
├── src/stock_volatility/phase_b/
│   ├── __init__.py
│   ├── data_contract.py                      # Verifies FREEZE_MANIFEST, builds 60-day Route A tensors & <=512-tok Route B prompts
│   ├── patchtst.py                           # Real PyTorch PatchTST (1M & 15M configs, ticker+sector emb, point + q10/q50/q90 Pinball Loss, Duan smearing & nominal 80% coverage)
│   ├── route_b_finbert.py                    # ProsusAI/finbert (110M) event-anchored feature extractor (GPU-gated)
│   ├── route_b_llm.py                        # Universal 27B-32B loader (Qwen3_5ForConditionalGeneration + CausalLM fallback), text-only LoRA target filter, YES/NO + HIGH/LOW log-odds zero-shot & 4-bit NF4 QLoRA trainer
│   ├── route_b_chronos.py                    # amazon/chronos-2 (120M) & amazon/chronos-t5-large (710M) rolling quarterly volatility forecaster
│   ├── fusion_and_eval.py                    # 2024 Val Temperature Scaling, LightGBM/HAR-RV cross-modal fusion (strict 5d purge), 2025 Full + Event-Day QLIKE/PR-AUC/ECE, paired date-block bootstrap CI & h=5 HLN-DM + Holm (H3)
│   └── report_integration.py                 # Generates Table 2 (Phase B ablation), expands CS482 report §6 (~0.6 pp -> 4.5-4.8 <= 5.0 pp), compiles CS482_GroupX_StockVolatilityLLM.pdf & writes report/phase_b_audit.md
├── scripts/
│   ├── colab_phase_b.py                      # Single-command Colab A100 runner (--allow-weight-download, --llm-model default Qwen/Qwen3.8-27B, --routes all|route_a|finbert|llm|chronos|fuse, --max-train-events optional fallback cap)
│   └── 07_integrate_phase_b_report.py        # Post-Colab CPU CLI: validates Phase B outputs, compiles final CS482_GroupX_StockVolatilityLLM.pdf & runs phase_b_audit.md
└── tests/
    └── test_phase_b.py                       # Comprehensive offline CPU TDD suite (PyTorch PatchTST 1M/15M param counts + Pinball/coverage, prompt zero-leakage, LoRA visual/mtp exclusion, YES/NO log-odds, Temperature Scaling, H3 fusion/bootstrap/DM, GPU gate safety, notebook JSON validity, report integration on synthetic Phase B fixtures)
```

### 4.2 Key Technical Design Decisions (Builder Calls)
1. **Universal Next-Token Log-Odds Head for `Qwen/Qwen3.8-27B` (Zero-Shot & QLoRA Parity)**:
   - Prompt format (`<= 512` tokens, deterministic template): compact ticker/sector/date header + strictly `<= t` market summary (`ret_1d`, `daily_range_pct`, `rv_d/w/m`, `vix_close`, `t10y2y`, `has_8k_earnings`) + capped `<= 5` pre-close SEC EDGAR event items -> ends with a concise risk question expecting single-token `YES` / `NO` (`z_tail = logit("YES") - logit("NO")`).
   - **Why**: Works identically across `Qwen3_5ForConditionalGeneration` (`AutoModelForImageTextToText`), `Qwen3ForCausalLM`, and Gemma 3/4 with **zero architecture-specific classification head fragility**, zero `output_hidden_states` VRAM overhead, and exact apples-to-apples comparison between **(c) Zero-Shot** and **(d) 4-bit NF4 QLoRA**.
   - **QLoRA Training Objective**: Class-weighted `BCEWithLogitsLoss` on `z_tail` (positive weight computed strictly on 2019–2023 Train active event days) + `r=16, lora_alpha=32, lora_dropout=0.05`, `paged_adamw_8bit`, `bfloat16`, `gradient_checkpointing=True`, cosine LR `1e-4`, `1`–`2` epochs (configurable; `5,032` Train event rows × 2 epochs ≈ `~25–40 min` on A100-40GB/80GB with peak VRAM `~24–28 GB`).
   - **Out-of-Fold / Leakage-Free Train Scores for Stage-2 Fusion**: To avoid in-sample overfitting when fusing LLM scores into Stage-2 LightGBM on 2024 Val / 2025 Test:
     - **Zero-Shot LLM** and **Frozen `FinBERT`** require no training, so their scores on `Train (5,032)` / `Val (880)` / `Test (899)` are strictly out-of-sample everywhere.
     - **QLoRA LLM** is trained strictly on `2019–2023 Train` (`5,032` rows); to prevent overfitting when fusing into LightGBM, `Temperature Scaling` is fitted strictly on `2024 Val` (`880` rows), AND Stage-2 fusion supports both **(i) linear/meta-stacking fusion fitted strictly on 2024 Val OOF (`z_llm_cal` + Phase A LightGBM logit/rv_hat)** and **(ii) full LightGBM feature-level fusion** evaluated on the identical 2025 Holdout Test (`14,700` full rows + `899` event-anchored rows + `Has_8K_Earnings` subset)! Wait: stacking `Phase A LightGBM OOF + calibrated LLM score` fitted strictly on `2024 Val` (4-quarter OOF CV inside 2024 Val!) completely eliminates Train-set memorization bias of the QLoRA adapter! Let's include **that exact leakage-free 2024-Val OOF meta-fusion** as the primary Stage-2 fusion method AND record pure LLM standalone scores too!
2. **Route A `PatchTST` Capacity Specs (`~1M` vs `~15M`)**:
   - Input window: `L = 60` trading days × `C = 14` continuous standardized features (standardized strictly using `Train` mean/std from `feature_manifest.json`) + learned `ticker` embedding (`60`) + `sector` embedding (`5`), patch length `P = 5`, stride `S = 5` (`12` temporal tokens per asset).
   - **Small (`patchtst_1m`, ~1.0M–1.5M params)**: `d_model=128, n_heads=4, n_layers=3, d_ff=256`.
   - **Main (`patchtst_15m`, ~14M–16M params)**: `d_model=384, n_heads=8, n_layers=6, d_ff=1536` (exact parameter count verified programmatically by pytest to fall inside `[0.8M, 2.0M]` and `[10M, 20M]`!).
   - Output head: 4 outputs `[y_log_rv_point, q10, q50, q90]` trained jointly with `MSE(point) + Pinball(0.10, 0.50, 0.90)` + `y_tail` auxiliary logit (`BCEWithLogitsLoss`) across 5 seeds (`42..46`), early-stopping strictly on `2024 Val` QLIKE, reporting 2025 QLIKE/RMSE/PR-AUC/ECE + **nominal 80% interval empirical coverage (`q10 <= y_log_rv <= q90`)**.
3. **PyTorch Dependency Hygiene**:
   - Add `torch>=2.5` via the lightweight `pytorch-cpu` index in `pyproject.toml` (identical to `plan_5_spatial_equity/pyproject.toml`), keeping CPU install small and fast, while GPU-only packages (`transformers`, `peft`, `bitsandbytes`, `accelerate`, `chronos-forecasting`) stay Colab-installed and lazy-imported behind `--allow-weight-download` + `torch.cuda.is_available()`.
4. **Milestones & Git Delivery Convention**:
   - Follow our established per-milestone documentation & GitHub push discipline:
     - **M7 (Phase B Engineering Package, PyTorch Route A, Colab A100 Route B & Report Integration Pipeline)**: Implement & TDD-test all Phase B modules, `scripts/colab_phase_b.py`, `scripts/07_integrate_phase_b_report.py`, `notebooks/phase_b_colab.ipynb`, `PHASE_B.md`, and `docs/milestones/M7_phase_b_colab_and_dl_pipeline.md`; verify `ruff` + all Phase A (`97`) + Phase B tests pass; commit & push to `origin/main`.
     - **M8 (Post-Colab A100 Run & Final CS482 PDF Sign-Off)**: After you execute `notebooks/phase_b_colab.ipynb` on Colab A100 and drop/push the produced `outputs/phase_b/` artifacts back, run `scripts/07_integrate_phase_b_report.py` to compile and audit the final `CS482_GroupX_StockVolatilityLLM.pdf` (`~4.6 <= 5.0` A4 pp), document `M8`, and push final submission!

---

## 5. Self-Review (Builder Check)
- **Can a builder implement M7 without asking further questions?** Yes: every input file, anti-leakage boundary, `Qwen3_5ForConditionalGeneration` loader & LoRA filter, `PatchTST` parameter count band, 2024-Val OOF Temperature Scaling + meta-fusion strategy, output JSON/CSV schema, Colab CLI flags, and PDF page budget (`<= 5.0` A4 pp) is concretely specified.
- **Any open risk?** Only `transformers` version on Colab for `Qwen/Qwen3.8-27B` (`transformers_version: 5.8.0.dev0` in HF config): mitigated directly in `notebooks/phase_b_colab.ipynb` by installing latest `transformers` (or git `main` fallback) AND verifying `AutoConfig.from_pretrained` before loading weights, with `--llm-model Qwen/Qwen3-32B` documented as a 1-flag drop-in fallback if needed.
