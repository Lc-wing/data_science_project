# Phase B — Deep Time-Series (`PatchTST`), `Chronos`, `FinBERT` & `Qwen/Qwen3.8-27B` Single-A100 QLoRA Extension

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
- **Frozen Data Contract & Zero-Leakage Prompts** (`src/stock_volatility/phase_b/data_contract.py`): Verifies `FREEZE_MANIFEST.json`, builds strictly `<= t` 60-day Train-standardized sequences, constructs strictly pre-close (`< 16:00 ET` / `< 13:00 ET` early-close) SEC EDGAR event-anchored prompts (`5,032` Train / `880` Val / `899` Test active days; zero embargo rows; zero forward targets leaked).
- **Route A Multivariate `PatchTST`** (`src/stock_volatility/phase_b/patchtst.py`): Real PyTorch `1M` (`~1.08M` params) and `15M` (`~14.6M` params) capacity presets with ticker (`60`) + sector (`5`) embeddings, non-crossing `q10 <= q50 <= q90` Pinball Loss, Duan smearing back-transform, and nominal 80% interval empirical coverage.
- **Route B `Qwen/Qwen3.8-27B` Architecture & LoRA Guardrails** (`src/stock_volatility/phase_b/route_b_llm.py`): Live Hugging Face config inspection confirmed `Qwen/Qwen3.8-27B` (`sha: 37561816e66b`, `gated: false`) uses `Qwen3_5ForConditionalGeneration` (`model_type: qwen3_5`, hybrid 48 `linear_attn` + 16 `self_attn` layers). `detect_model_loader_kind()` automatically routes it to `AutoModelForImageTextToText` (with `AutoModelForCausalLM` fallback for text-only checkpoints), `filter_text_lora_target_modules()` restricts LoRA strictly to `model.language_model` linear projections (excluding `visual` and `mtp`), and next-token `logit("YES") - logit("NO")` scoring provides apples-to-apples Zero-Shot vs. 4-bit NF4 QLoRA parity.
- **2024-Val OOF Calibration, Meta-Fusion & H3 Evaluation** (`src/stock_volatility/phase_b/fusion_and_eval.py`): Fits 1-parameter Temperature Scaling and L2-regularized 4-quarter OOF meta-fusion strictly inside `2024 Val` (preventing in-sample QLoRA memorization bias), evaluating 2025 Full QLIKE, Event-Day QLIKE, Tail PR-AUC (Full / Event / Earnings), ECE, paired date-block bootstrap 95% CIs, and `h=5` HLN-DM + Holm `p`-values.
- **GPU Download Safety Gate**: `scripts/colab_phase_b.py` refuses to download Hugging Face weights or run CUDA unless **both** `--allow-weight-download` is passed **and** `torch.cuda.is_available()` is `True`.

---

## 2. Running Phase B on Colab Single-A100 (40GB / 80GB)

Open [`notebooks/phase_b_colab.ipynb`](notebooks/phase_b_colab.ipynb) in Google Colab (select **NVIDIA A100** runtime) and run all cells, or run manually from a Colab A100 terminal:

```bash
git clone --branch main https://github.com/Lc-wing/data_science_project.git
cd data_science_project/plan_6_stock_volatility
pip install -q -U "transformers>=4.57" peft bitsandbytes accelerate chronos-forecasting pyarrow lightgbm arch shap
pip install -q -e . --no-deps

export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Full Phase B run (Route A PatchTST 1M/15M 5-seed + Chronos-2/T5-Large + FinBERT + Qwen/Qwen3.8-27B Zero-Shot & 4-bit QLoRA):
python scripts/colab_phase_b.py --allow-weight-download --llm-model Qwen/Qwen3.8-27B --llm-epochs 1
```

### Useful Colab CLI Flags
- `--llm-model`: Default `Qwen/Qwen3.8-27B` (ungated, `model_type: qwen3_5`). If Colab's PyPI `transformers` reports unrecognized `qwen3_5`, run `pip install -q -U git+https://github.com/huggingface/transformers.git`, or use a drop-in alternative: `Qwen/Qwen3-32B`, `Qwen/Qwen3-30B-A3B`, `google/gemma-4-31B` (requires `HF_TOKEN`), `meta-models/Muse-Glimmer-30B`.
- `--routes`: Run specific subsets (`all`, `route_a`, `finbert`, `llm`, `chronos`), e.g. `--routes finbert llm` for rapid H3 text ablation.
- `--max-train-events`: Optional Plan 6 fallback cap if A100 units run tight (default uses all `5,032` Train event-anchored days — already compact enough for ~25–40 min on A100 without subsampling).

### Colab Outputs Written to `outputs/phase_b/`
- `outputs/phase_b/phase_b_summary.json` (model revisions, GPU name, peak VRAM GB, elapsed time, and verified H3 metrics)
- `outputs/phase_b/phase_b_comparison.{csv,md,tex}` (unified Phase A vs. Phase B Table 2)
- `outputs/phase_b/{patchtst_1m,patchtst_15m,chronos_2,chronos_t5_large}_test_predictions.parquet`
- `outputs/phase_b/{finbert,llm}_{val,test}_scores.parquet` & `outputs/phase_b/llm_run_metadata.json`

---

## 3. Post-Colab Final CS482 Report Integration (Milestone M8)

Once Colab finishes, copy `outputs/phase_b/` from the downloaded `phase_b_a100_outputs.zip` into `plan_6_stock_volatility/outputs/phase_b/` and compile + audit the final CS482 submission PDF:

```bash
cd plan_6_stock_volatility
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run --no-sync python scripts/07_integrate_phase_b_report.py
```

This verifies `FREEZE_MANIFEST.json`, injects Section 6 (`Extension: Deep Time-Series and 27B QLoRA News LLM (H3)`) + Table 2 (`phase_b_comparison.tex`), compiles `report/CS482_GroupX_StockVolatilityLLM.pdf` deterministically (`SOURCE_DATE_EPOCH`), and writes `report/phase_b_audit.md` verifying:
- Every page exact A4 (`595.28 x 841.89 pts`)
- Main text strictly inside `[4.0, 5.0]` A4 pages (target `~4.6` pp, Cover/References/Contribution/AI Declaration excluded)
- 100% numeric claim traceability across all 51 Phase A claims + 9 Phase B claims (`60/60`)
- Zero forbidden causal words (`impact`, `driver`, `elasticity`, `caused`, `causes`) and zero unresolved placeholders (`??`, `TODO`, `TBD`, `TBD_*`, `NaN`).
