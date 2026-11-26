# Phase B — Deep Time-Series (`PatchTST`), `Chronos`, `FinBERT` & Colab L4/A100 QLoRA Extension (`google/gemma-4-12B-it` / `Qwen/Qwen3.8-27B`)

Plan 6 lives in `plan_6_stock_volatility/`. Run every command in this file from that folder; every relative path below starts there.

Phase A (`M0`–`M6`) is cryptographically frozen (`data/frozen/FREEZE_MANIFEST.json`, SHA-256 `8aa3b379b8beef2b077d5c3e6651733be089eea45e0ac56b738de07a30aa3b1b`). Phase B **only reads** `data/frozen/` and Phase A's published `outputs/tables/h2_{val,test}_predictions.parquet` — it never rewrites frozen data or refits Phase A econometric/LightGBM baselines.

> **Academic Research & Non-Investment-Advice Notice:** All Phase B volatility and downside tail-risk outputs remain strictly predictive, non-causal CS482 academic benchmarks and **do not constitute investment, financial, or trading advice**.

---

## 1. What Runs & Is Verified Locally on CPU

```bash
cd plan_6_stock_volatility
uv sync --all-groups
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync ruff check .
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync pytest tests/test_phase_b.py -v
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync pytest -v
```

Local CPU verification (`torch==2.15.0+cpu` from `pytorch-cpu` index) validates:
- **Frozen Data Contract & Zero-Leakage Prompts** (`src/stock_volatility/phase_b/data_contract.py`): Verifies `FREEZE_MANIFEST.json`, builds strictly `<= t` 60-day Train-standardized sequences, and constructs strictly pre-close (`< 16:00 ET` / `< 13:00 ET` early-close) SEC EDGAR event-anchored prompts (`4,929` Train / `880` Val / `895` Test modeling-complete scorable event days out of `5,032` / `880` / `899` total split event days; zero embargo rows; zero forward targets leaked).
- **Route A Multivariate `PatchTST`** (`src/stock_volatility/phase_b/patchtst.py`): Real PyTorch `1M` (`1,089,543` params) and `15M` (`14,650,759` params) capacity presets with ticker (`60`) + sector (`5`) embeddings, non-crossing `q10 <= q50 <= q90` Pinball Loss, Duan smearing back-transform, and nominal 80% interval empirical coverage.
- **Route B Multi-GPU Architecture & LoRA Guardrails** (`src/stock_volatility/phase_b/route_b_llm.py`): Live Hugging Face metadata & safetensors header probes confirmed both **`google/gemma-4-12B-it`** (`11.96B`, `sha: 43c9d014b231`, `gated: false`, `Gemma4UnifiedForConditionalGeneration`, `model_type: gemma4_unified`, tied embeddings — fits **Colab L4 24GB** at `15.34 GB` measured peak VRAM with batched `bs=8`) and **`Qwen/Qwen3.8-27B`** (`sha: 37561816e66b`, `gated: false`, `Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5` — fits **Colab A100 40GB/80GB**). `detect_model_loader_kind()` automatically routes multimodal wrappers to `AutoModelForImageTextToText` (with `AutoModelForCausalLM` fallback for text-only models like `Qwen/Qwen3-14B`), `filter_text_lora_target_modules()` restricts LoRA strictly to `model.language_model` linear projections (excluding `visual`/`vision_tower`/`vision_embedder`/`embed_vision`/`audio`/`audio_tower`/`embed_audio`/`mtp`), left-padded `logits_to_keep=1` trims training VRAM, `resolve_hf_model_revision()` records the exact HF snapshot commit SHA, and next-token `logit("YES") - logit("NO")` scoring ensures apples-to-apples Zero-Shot vs. 4-bit NF4 QLoRA parity.
- **2024-Val OOF Calibration, Meta-Fusion & H3 Evaluation** (`src/stock_volatility/phase_b/fusion_and_eval.py`): Fits 1-parameter Temperature Scaling and L2-regularized 4-quarter OOF meta-fusion strictly inside `2024 Val` (preventing in-sample QLoRA memorization bias), evaluating 2025 Full QLIKE, Event-Day QLIKE, Tail PR-AUC (Full / Event / Earnings), ECE, paired date-block bootstrap 95% CIs, and `h=5` HLN-DM + Holm `p`-values.
- **GPU Download Safety Gate**: `scripts/colab_phase_b.py` refuses to download Hugging Face weights or run CUDA unless **both** `--allow-weight-download` is passed **and** `torch.cuda.is_available()` is `True`.

---

## 2. Running Phase B on Colab Single-GPU (`L4 24GB` or `A100 40GB / 80GB`)

Open [`notebooks/phase_b_colab.ipynb`](notebooks/phase_b_colab.ipynb) in Google Colab (select **NVIDIA L4** or **NVIDIA A100** runtime) and run all cells, or run manually from a Colab GPU terminal:

```bash
git clone --branch main https://github.com/Lc-wing/data_science_project.git
cd data_science_project/plan_6_stock_volatility
pip install -q -U "transformers>=5.10" peft bitsandbytes accelerate chronos-forecasting pyarrow lightgbm arch shap
pip install -q -e . --no-deps

export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Auto-selects LLM by VRAM (L4 24GB -> google/gemma-4-12B-it; A100 >=35GB -> Qwen/Qwen3.8-27B):
python scripts/colab_phase_b.py --allow-weight-download --resume --llm-model auto --llm-epochs 1
```

### Useful Colab CLI Flags
- `--llm-model`: Default `auto` (`google/gemma-4-12B-it` on L4 24GB; `Qwen/Qwen3.8-27B` on A100 40GB/80GB). Supported explicit overrides (all ungated):
  - **L4 (24GB VRAM)**: `google/gemma-4-12B-it` (recommended, `15.34 GB` measured peak VRAM at `bs=8`), `google/gemma-4-12B` (base), `Qwen/Qwen3-14B` (`~14–16 GB` peak).
  - **A100 (40GB / 80GB VRAM)**: `Qwen/Qwen3.8-27B`, `Qwen/Qwen3-32B`, `Qwen/Qwen3-30B-A3B`, `google/gemma-4-31B-it`.
- `--resume`: Reuse already-saved Stage 1–4 outputs in `outputs/phase_b/` instead of recomputing completed stages.
- `--llm-train-batch-size`: Micro-batch size for QLoRA fine-tuning (default `8`, automatically adjusting `grad_accum` to preserve effective batch size `>= 8`).
- `--routes`: Run specific subsets (`all`, `route_a`, `finbert`, `llm`, `chronos`), e.g. `--routes finbert llm` while automatically preserving saved artifacts from unselected stages.
- `--max-train-events`: Optional Plan 6 fallback cap if GPU compute units run tight (default uses all `4,929` modeling-complete Train event-anchored days without subsampling).

### Colab Outputs Written to `outputs/phase_b/`
- `outputs/phase_b/phase_b_summary.json` (model revisions, GPU name, peak VRAM GB, elapsed time, and verified H3 metrics)
- `outputs/phase_b/phase_b_comparison.{csv,md,tex}` (unified Phase A vs. Phase B Table 2)
- `outputs/phase_b/{patchtst_1m,patchtst_15m,chronos_2,chronos_t5_large}_test_predictions.parquet`
- `outputs/phase_b/{finbert,llm}_{val,test}_scores.parquet` & `outputs/phase_b/llm_run_metadata.json`

---

## 3. Post-Colab Final CS482 Report Integration (Milestone M8)

Once Colab finishes, extract `phase_b_gpu_outputs.zip` into `plan_6_stock_volatility/outputs/phase_b/` and compile + audit the final CS482 submission PDF:

```bash
cd plan_6_stock_volatility
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync python scripts/07_integrate_phase_b_report.py
```

This verifies `FREEZE_MANIFEST.json`, builds the unified 5-section CS482 final report (`1 Research Question`, `2 Data Collection and Preparation`, `3 Data Analysis` [merging Phase B `PatchTST`, `Chronos`, `FinBERT`, and `QLoRA`], `4 Method Evaluation`, `5 Results and Discussion`) + Table 2 (`phase_b_comparison.tex`) + `18` verified citations (`report/final_references.bib`), compiles `report/CS482_GroupX_StockVolatilityLLM.pdf` deterministically (`SOURCE_DATE_EPOCH`), and writes `report/phase_b_audit.md` verifying:
- Every page exact A4 (`595.28 x 841.89 pts`)
- Main text strictly inside `[3.0, 5.0]` A4 pages (`4.46` A4 main pages across `5` main-text pages; Cover, References, Separate Contribution Statement, and AI Usage Declaration excluded)
- 100% numeric claim traceability across all 51 Phase A claims + 10 Phase B claims (`61/61`)
- Zero forbidden causal words (`impact`, `driver`, `elasticity`, `caused`, `causes`) and zero unresolved placeholders (`??`, `TODO`, `TBD`, `TBD_*`, `NaN`).
