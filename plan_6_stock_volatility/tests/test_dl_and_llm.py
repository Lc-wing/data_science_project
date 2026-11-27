"""Comprehensive offline CPU test suite for Section 3.3 Deep Learning & Event-Anchored LLMs."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from pypdf import PdfReader

from stock_volatility import config


def test_dl_datasets_windows_and_zero_target_leakage_prompts(
    block_network: None,
) -> None:
    """Verify frozen manifest check, 60-day PatchTST windows, and zero-leakage event prompts."""
    from stock_volatility.analysis.dl.datasets import (
        DL_CONT_FEATURES,
        build_event_anchored_datasets,
        build_event_prompt,
        build_patchtst_windows,
        load_frozen_dl_inputs,
    )

    bundle = load_frozen_dl_inputs(config.PROJECT_ROOT)
    panel_df = bundle["panel_df"]
    events_df = bundle["events_df"]
    feat_m = bundle["feature_manifest"]

    assert len(DL_CONT_FEATURES) == 14
    assert len(panel_df) == 105_600
    assert len(events_df) == 105_600

    import stock_volatility.analysis as analysis_pkg
    import stock_volatility.analysis.dl as dl_pkg

    assert hasattr(analysis_pkg, "dl")
    assert hasattr(dl_pkg, "patchtst")
    assert hasattr(dl_pkg, "qlora_llm")
    with pytest.raises(AttributeError):
        _ = dl_pkg.nonexistent_submodule

    # 1. Event-anchored datasets strictly exclude embargo and non-event days
    ev_splits = build_event_anchored_datasets(panel_df, events_df)
    assert set(ev_splits.keys()) == {"train", "val", "test"}
    assert len(ev_splits["train"]) == 4929
    assert len(ev_splits["val"]) == 880
    assert len(ev_splits["test"]) == 895
    assert (ev_splits["train"]["date"] <= config.TRAIN_END).all()
    assert (ev_splits["val"]["date"] >= config.VAL_START).all()
    assert (ev_splits["val"]["date"] <= config.VAL_END).all()
    assert (ev_splits["test"]["date"] >= config.TEST_START).all()
    for split_name, df_s in ev_splits.items():
        assert (df_s["has_news"] == 1).all(), f"Non-event row found in {split_name}"
        assert df_s["has_full_features"].all() and df_s["has_full_target"].all()

    # 2. Prompt builder must NEVER leak forward targets even when present on row
    sample_row = ev_splits["train"].iloc[0].to_dict()
    sample_row["rv_5d_fwd"] = 9.876543
    sample_row["y_log_rv"] = 7.654321
    sample_row["cum_ret_5d_fwd"] = -0.432198
    sample_row["y_tail"] = 1
    prompt = build_event_prompt(sample_row)
    assert len(prompt) <= 1800
    assert sample_row["ticker"] in prompt
    assert "YES or NO" in prompt
    for leaked_tok in ("9.8765", "7.6543", "-0.4321", "rv_5d_fwd", "y_log_rv", "cum_ret_5d_fwd"):
        assert leaked_tok not in prompt, f"Target leakage {leaked_tok!r} found in prompt!"

    # 3. 60-day PatchTST windows built strictly <= t on 2 stocks for fast verification
    sub_panel = panel_df[panel_df["ticker"].isin(["AAPL", "JPM"])].copy()
    windows = build_patchtst_windows(sub_panel, feat_m, lookback=60)
    assert set(windows.keys()) == {"train", "val", "test"}
    x_tr = windows["train"]["x_seq"]
    assert x_tr.ndim == 3 and x_tr.shape[1] == 60 and x_tr.shape[2] == len(DL_CONT_FEATURES)
    assert np.isfinite(x_tr).all()
    assert (windows["val"]["dates"] >= config.VAL_START).all()
    assert (windows["test"]["dates"] >= config.TEST_START).all()
    assert set(windows["train"]["ticker_id"].tolist()).issubset(set(range(60)))
    assert set(windows["train"]["sector_id"].tolist()).issubset(set(range(5)))


def test_patchtst_1m_and_15m_param_counts_pinball_coverage_and_training(
    block_network: None,
) -> None:
    """Verify real PyTorch PatchTST 1M/15M capacities, Pinball loss, gradients & mini training."""
    from stock_volatility.analysis.dl.patchtst import (
        PATCHTST_1M_CONFIG,
        PATCHTST_15M_CONFIG,
        MultivariatePatchTST,
        count_trainable_parameters,
        nominal_interval_coverage,
        pinball_loss,
        train_and_predict_patchtst,
    )

    m_1m = MultivariatePatchTST(PATCHTST_1M_CONFIG)
    m_15m = MultivariatePatchTST(PATCHTST_15M_CONFIG)
    n_1m = count_trainable_parameters(m_1m)
    n_15m = count_trainable_parameters(m_15m)
    assert 800_000 <= n_1m <= 2_000_000, f"PatchTST-1M params out of band: {n_1m:,}"
    assert 10_000_000 <= n_15m <= 20_000_000, f"PatchTST-15M params out of band: {n_15m:,}"

    # Forward + Pinball Loss + backward gradient verification on 1M model
    torch.manual_seed(42)
    x = torch.randn(8, 60, 14)
    tid = torch.randint(0, 60, (8,))
    sid = torch.randint(0, 5, (8,))
    y_log = torch.randn(8)
    y_tail = torch.randint(0, 2, (8,)).float()

    out = m_1m(x, tid, sid)
    assert out["point"].shape == (8,)
    assert out["quantiles"].shape == (8, 3)
    assert out["tail_logit"].shape == (8,)
    # Non-crossing quantiles enforced (q10 <= q50 <= q90)
    assert (out["quantiles"][:, 0] <= out["quantiles"][:, 1] + 1e-6).all()
    assert (out["quantiles"][:, 1] <= out["quantiles"][:, 2] + 1e-6).all()

    loss = (
        torch.nn.functional.mse_loss(out["point"], y_log)
        + pinball_loss(out["quantiles"], y_log)
        + torch.nn.functional.binary_cross_entropy_with_logits(out["tail_logit"], y_tail)
    )
    assert torch.isfinite(loss)
    loss.backward()
    grad_norm = sum(float(p.grad.abs().sum()) for p in m_1m.parameters() if p.grad is not None)
    assert grad_norm > 0.0

    # Coverage helper exact checks
    assert nominal_interval_coverage(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([0.5, 2.5, 2.5, 3.5]),
        np.array([1.5, 3.5, 3.5, 4.5]),
    ) == pytest.approx(0.75)

    # Mini end-to-end training smoke test on synthetic windows (2 tiny epochs, 2 seeds on CPU)
    rng = np.random.default_rng(42)

    def _mk_split(n: int, start_date: str) -> dict[str, np.ndarray]:
        dates = np.array([f"{start_date[:8]}{(i % 20) + 1:02d}" for i in range(n)])
        rv = rng.uniform(0.12, 0.45, size=n).astype(np.float32)
        return {
            "x_seq": rng.normal(0.0, 1.0, size=(n, 60, 14)).astype(np.float32),
            "ticker_id": rng.integers(0, 60, size=n, dtype=np.int64),
            "sector_id": rng.integers(0, 5, size=n, dtype=np.int64),
            "y_log_rv": np.log(rv + config.RV_EPS).astype(np.float32),
            "rv_5d_fwd": rv,
            "y_tail": rng.integers(0, 2, size=n, dtype=np.int64),
            "tickers": np.array(["AAPL"] * n),
            "dates": dates,
        }

    tiny_cfg = PATCHTST_1M_CONFIG.with_overrides(d_model=32, n_heads=4, n_layers=1, d_ff=64)
    res = train_and_predict_patchtst(
        windows={
            "train": _mk_split(48, "2023-06-01"),
            "val": _mk_split(24, "2024-06-01"),
            "test": _mk_split(24, "2025-06-01"),
        },
        cfg=tiny_cfg,
        seeds=(42, 43),
        epochs=2,
        batch_size=16,
        device="cpu",
    )
    assert len(res["seed_summaries"]) == 2
    assert len(res["test_pred_df"]) == 24
    assert (res["test_pred_df"]["rv_hat"] > 0).all()
    assert (res["test_pred_df"]["q10_rv"] <= res["test_pred_df"]["q90_rv"] + 1e-6).all()
    assert 0.0 <= res["test_nominal_80_coverage"] <= 1.0


def test_qlora_llm_loader_lora_filter_yes_no_log_odds_and_gpu_guards(
    block_network: None,
) -> None:
    """Verify Qwen3.5 & Gemma4Unified routing, text-only LoRA filter, L4 selector & GPU gates."""
    from stock_volatility.analysis.dl.qlora_llm import (
        DEFAULT_L4_LLM_MODEL,
        DEFAULT_LLM_MODEL,
        compute_yes_no_log_odds,
        detect_model_loader_kind,
        filter_text_lora_target_modules,
        require_cuda_and_download_flag,
        resolve_hf_model_revision,
        select_default_llm_for_vram,
    )

    assert (
        resolve_hf_model_revision("google/gemma-4-12B-it", SimpleNamespace(_commit_hash="43c9d014"))
        == "43c9d014"
    )
    assert (
        resolve_hf_model_revision(
            "google/gemma-4-12B-it",
            SimpleNamespace(_commit_hash=None),
            SimpleNamespace(config=SimpleNamespace(_commit_hash="37561816e66b")),
        )
        == "37561816e66b"
    )
    assert (
        resolve_hf_model_revision("nonexistent-org/nonexistent-model-xyz", SimpleNamespace())
        == "unknown"
    )

    assert DEFAULT_LLM_MODEL == "Qwen/Qwen3.8-27B"
    assert DEFAULT_L4_LLM_MODEL == "google/gemma-4-12B-it"
    assert select_default_llm_for_vram(23.7) == "google/gemma-4-12B-it"
    assert select_default_llm_for_vram(42.5) == "Qwen/Qwen3.8-27B"
    assert (
        detect_model_loader_kind("qwen3_5", ["Qwen3_5ForConditionalGeneration"])
        == "image_text_to_text"
    )
    assert (
        detect_model_loader_kind("gemma4_unified", ["Gemma4UnifiedForConditionalGeneration"])
        == "image_text_to_text"
    )
    assert detect_model_loader_kind("qwen3", ["Qwen3ForCausalLM"]) == "causal_lm"

    sample_modules = [
        "model.visual.blocks.0.attn.qkv",
        "model.visual.merger.linear_fc1",
        "model.vision_tower.layers.0.self_attn.q_proj",
        "model.vision_embedder.layers.0.mlp.gate_proj",
        "model.embed_vision.q_proj",
        "model.audio_tower.layers.0.self_attn.q_proj",
        "model.embed_audio.out_proj",
        "mtp.layers.0.self_attn.q_proj",
        "mtp.layers.0.mlp.gate_proj",
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.out_proj",
        "model.language_model.layers.0.mlp.gate_proj",
        "model.language_model.layers.0.mlp.up_proj",
        "model.language_model.layers.0.mlp.down_proj",
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.self_attn.k_proj",
        "model.language_model.layers.3.self_attn.v_proj",
        "model.language_model.layers.3.self_attn.o_proj",
    ]
    targets = filter_text_lora_target_modules(sample_modules)
    assert len(targets) == 9
    assert all(
        "visual" not in name
        and "vision" not in name
        and "audio" not in name
        and not name.startswith("mtp.")
        for name in targets
    )
    assert "model.language_model.layers.0.linear_attn.in_proj_qkv" in targets
    assert "model.language_model.layers.3.self_attn.q_proj" in targets

    logits = torch.tensor([[1.0, 2.5, 0.5, -0.5], [0.0, -1.0, 0.2, 3.0]], dtype=torch.float32)
    z = compute_yes_no_log_odds(logits, yes_token_id=1, no_token_id=3)
    np.testing.assert_allclose(z.numpy(), np.array([3.0, -4.0]), atol=1e-6)

    with pytest.raises(RuntimeError, match="allow-weight-download"):
        require_cuda_and_download_flag(allow_weight_download=False, component="qwen3_8_27b")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            require_cuda_and_download_flag(allow_weight_download=True, component="qwen3_8_27b")


def test_2024_val_oof_temperature_scaling_meta_fusion_and_h3_evaluation(
    block_network: None,
) -> None:
    """Verify Temperature Scaling, 2024-Val OOF fusion, and 2025 H3 ablation + HLN-DM + Holm."""
    from stock_volatility.analysis.dl.fusion import (
        apply_temperature_scaling,
        evaluate_h3_dl_ablation,
        fit_2024_val_oof_meta_fusion,
        fit_temperature_scaler,
    )

    rng = np.random.default_rng(42)
    z_true = rng.normal(-2.0, 1.0, size=1000)
    p_true = 1.0 / (1.0 + np.exp(-z_true))
    y_sim = rng.binomial(1, p_true)
    z_overconf = z_true * 3.0
    temp = fit_temperature_scaler(z_overconf, y_sim)
    assert temp > 1.5, f"Expected T > 1.5 for overconfident logits, got {temp:.3f}"
    p_cal = apply_temperature_scaling(z_overconf, temp)
    assert np.all((p_cal > 0.0) & (p_cal < 1.0))

    val_base = pd.read_parquet(config.OUTPUTS_PREDICTIONS_DIR / "h2_val_predictions.parquet")
    test_base = pd.read_parquet(config.OUTPUTS_PREDICTIONS_DIR / "h2_test_predictions.parquet")

    def _add_synth_scores(df: pd.DataFrame, seed: int) -> pd.DataFrame:
        r = np.random.default_rng(seed)
        out = df[["ticker", "date", "has_news"]].copy()
        is_ev = out["has_news"].to_numpy() == 1
        y = df["y_tail"].to_numpy(dtype=float)
        finb = np.where(is_ev, 0.4 * (y - 0.1) + r.normal(0.0, 0.5, len(df)), 0.0)
        qwen = np.where(is_ev, 0.9 * (y - 0.1) + r.normal(0.0, 0.5, len(df)), 0.0)
        out["finbert_logit"] = finb
        out["qwen_qlora_logit"] = qwen
        return out

    val_s = _add_synth_scores(val_base, seed=101)
    test_s = _add_synth_scores(test_base, seed=202)

    fused_finb = fit_2024_val_oof_meta_fusion(
        val_base_df=val_base,
        test_base_df=test_base,
        val_text_scores=val_s[["ticker", "date", "finbert_logit"]],
        test_text_scores=test_s[["ticker", "date", "finbert_logit"]],
        score_col="finbert_logit",
        arm_name="lgbm_plus_finbert",
    )
    fused_qwen = fit_2024_val_oof_meta_fusion(
        val_base_df=val_base,
        test_base_df=test_base,
        val_text_scores=val_s[["ticker", "date", "qwen_qlora_logit"]],
        test_text_scores=test_s[["ticker", "date", "qwen_qlora_logit"]],
        score_col="qwen_qlora_logit",
        arm_name="lgbm_plus_qwen_qlora",
    )
    assert len(fused_qwen["test_pred_df"]) == len(test_base)
    assert fused_qwen["val_temperature"] > 0.0
    assert (fused_qwen["test_pred_df"]["rv_hat"] > 0.0).all()
    assert fused_qwen["test_pred_df"]["p_tail"].between(0.0, 1.0).all()

    ablation = evaluate_h3_dl_ablation(
        base_test_df=test_base,
        dl_arms={
            "lgbm_plus_finbert": {
                "label": "LightGBM + FinBERT (110M)",
                "test_pred_df": fused_finb["test_pred_df"],
            },
            "lgbm_plus_qwen_qlora": {
                "label": "LightGBM + Qwen3.8-27B (QLoRA)",
                "test_pred_df": fused_qwen["test_pred_df"],
            },
        },
        n_boot=200,
        seed=42,
    )
    comp_df = ablation["comparison_df"]
    assert len(comp_df) == 3  # Global LightGBM ref + 2 DL/LLM arms
    assert {
        "qlike_full",
        "qlike_event_days",
        "pr_auc_full",
        "pr_auc_event_days",
        "ece",
        "dm_qlike_p_holm",
        "dm_brier_p_holm",
    }.issubset(set(comp_df.columns))
    assert r"\begin{tabular}" in ablation["latex_table"]
    assert "LightGBM + Qwen3.8-27B" in ablation["markdown_table"]


def test_unified_report_builder_refuses_missing_h3_and_builds_valid_pdf_in_tmp(
    tmp_path: Path,
    block_network: None,
) -> None:
    """Verify unified report builder refuses missing H3 outputs and compiles <=5pp A4 PDF."""
    from stock_volatility.report import run_report_build_and_audit

    clone_root = tmp_path / "unified_report_clone"
    for rel_dir in ("data/frozen", "outputs/tables", "outputs/metrics", "outputs/figures"):
        shutil.copytree(config.PROJECT_ROOT / rel_dir, clone_root / rel_dir)
    (clone_root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        config.DATA_RAW_DIR / "universe_60.csv", clone_root / "data" / "raw" / "universe_60.csv"
    )

    # 1. Must refuse to fabricate H3 metrics when h3_dl_summary.json is absent
    (clone_root / "outputs" / "metrics" / "h3_dl_summary.json").unlink()
    with pytest.raises(FileNotFoundError, match="h3_dl_summary.json"):
        run_report_build_and_audit(project_root=clone_root, regenerate_figures=False)

    # 2. Inject synthetic H3 summary (with HF revision) & Table 2 and verify clean <=5pp A4 build
    synth_summary = {
        "schema_version": "1.0.0",
        "milestone": "H3-SyntheticTest",
        "llm_model_id": "Qwen/Qwen3.8-27B",
        "llm_revision": "37561816e66b",
        "gpu_name": "NVIDIA A100-SXM4-40GB (Synthetic Fixture)",
        "peak_vram_gb": 25.4,
        "train_event_days": 4929,
        "val_event_days": 880,
        "test_event_days": 895,
        "patchtst_15m_qlike": 0.2365,
        "patchtst_15m_coverage_80": 0.812,
        "finbert_qlike_event": 0.2410,
        "finbert_pr_auc_event": 0.1480,
        "qwen_zs_pr_auc_event": 0.1510,
        "qwen_qlora_qlike_full": 0.2289,
        "qwen_qlora_qlike_event": 0.2330,
        "qwen_qlora_pr_auc_full": 0.1310,
        "qwen_qlora_pr_auc_event": 0.1625,
        "qwen_qlora_ece": 0.0115,
        "qwen_qlora_dm_p_holm": 0.084,
    }
    (clone_root / "outputs" / "metrics" / "h3_dl_summary.json").write_text(
        json.dumps(synth_summary, indent=2) + "\n", encoding="utf-8"
    )
    synth_tex = (
        r"\begin{table}[!htb]" "\n"
        r"\centering" "\n"
        r"\caption{2025 Holdout Deep Time-Series and Event-Anchored LLM Ablation (H3).}" "\n"
        r"\label{tab:h3_summary}" "\n"
        r"\scriptsize" "\n"
        r"\setlength{\tabcolsep}{3.5pt}" "\n"
        r"\begin{tabular}{lccccc}" "\n"
        r"\toprule" "\n"
        r"Model (Parameters) & Full QLIKE & Event QLIKE & Tail PR-AUC & ECE & DM Holm $p$ \\"
        "\n"
        r"\midrule" "\n"
        r"Global LightGBM (Price-Only) & 0.2306 & 0.2440 & 0.1231 & 0.0102 & Ref. \\" "\n"
        r"PatchTST (15M, 5-seed; 80\% cov=81.2\%) & 0.2365 & 0.2480 & 0.1190 & 0.0140 & 0.210 \\"
        "\n"
        r"LightGBM + FinBERT (110M) & 0.2301 & 0.2410 & 0.1260 & 0.0110 & 0.420 \\" "\n"
        r"\textbf{LightGBM + Qwen3.8-27B (QLoRA)} & \textbf{0.2289} & \textbf{0.2330} & "
        r"\textbf{0.1310} & 0.0115 & 0.084 \\" "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}" "\n"
    )
    (clone_root / "outputs" / "tables" / "table2_h3_ablation.tex").write_text(
        synth_tex, encoding="utf-8"
    )

    res = run_report_build_and_audit(project_root=clone_root, regenerate_figures=False)
    assert res["verdict"] == "PASS", f"Unified final report audit failed: {res}"
    assert res["claims_verified"] == 62  # 51 H1/H2 + 10 H3 + H3_LLM_REV
    assert 4.0 <= res["main_text_pages_est"] <= 5.0
    assert res["main_pages_occupied"] <= 5
    pdf_path = clone_root / "report" / config.REPORT_PDF_NAME
    assert pdf_path.is_file()
    reader = PdfReader(str(pdf_path))
    for page in reader.pages:
        assert abs(float(page.mediabox.width) - 595.28) < 1.5
        assert abs(float(page.mediabox.height) - 841.89) < 1.5


def test_gpu_cli_guard_notebook_and_m7_milestone_docs() -> None:
    """Verify GPU CLI refuses CPU/no-flag run, notebook JSON is valid, and M7 docs tracked."""
    cli = config.PROJECT_ROOT / "scripts" / "06_dl_and_llm_gpu.py"
    assert cli.is_file()
    proc = subprocess.run(
        [sys.executable, str(cli)],
        cwd=config.PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert proc.returncode != 0
    assert "allow-weight-download" in (proc.stdout + proc.stderr)

    # Every unified CLI script must support --help cleanly without running pipelines
    for script_name in (
        "01_download_and_audit.py",
        "02_build_panel_and_freeze.py",
        "03_eda.py",
        "04_panel_har_rv.py",
        "05_forecast_backtest.py",
        "06_dl_and_llm_gpu.py",
        "07_build_report.py",
        "run_pipeline.py",
    ):
        hp = subprocess.run(
            [sys.executable, str(config.PROJECT_ROOT / "scripts" / script_name), "--help"],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        assert hp.returncode == 0, f"{script_name} --help failed: {hp.stderr}"

    # Contradictory --report-only --refreeze must be rejected cleanly
    bad_flags = subprocess.run(
        [
            sys.executable,
            str(config.PROJECT_ROOT / "scripts" / "run_pipeline.py"),
            "--report-only",
            "--refreeze",
        ],
        cwd=config.PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert bad_flags.returncode != 0

    # Subset --routes hydration must preserve saved unselected arms & PatchTST summary fields
    spec = importlib.util.spec_from_file_location("dl_gpu_cli_mod", cli)
    assert spec is not None and spec.loader is not None
    gpu_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gpu_mod)
    assert (
        gpu_mod._format_patchtst_label("PatchTST (15M, 5-seed)", 0.683)
        == "PatchTST (15M, 5-seed; 80% cov=68.3%)"
    )
    hyd_summary: dict[str, object] = {}
    hyd_arms: dict[str, dict[str, object]] = {}
    gpu_mod.hydrate_saved_h3_state(
        outputs_root=config.OUTPUTS_DIR,
        run_set={"finbert", "llm"},
        resolved_llm="google/gemma-4-12B-it",
        summary=hyd_summary,
        arms_for_ablation=hyd_arms,
    )
    assert "patchtst_15m_qlike" in hyd_summary
    assert {"patchtst_1m", "patchtst_15m", "chronos_2", "chronos_t5_large"}.issubset(hyd_arms)

    nb_path = config.PROJECT_ROOT / "notebooks" / "dl_gpu_training_colab.ipynb"
    assert nb_path.is_file()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    nb_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    assert "scripts/06_dl_and_llm_gpu.py" in nb_src
    assert "--allow-weight-download" in nb_src
    assert "Qwen/Qwen3.8-27B" in nb_src
    assert "google/gemma-4-12B-it" in nb_src
    assert "transformers>=5.10" in nb_src
    assert '(cwd / ENTRY).exists() and (cwd.parent / ".git").exists()' in nb_src

    zip_chk = subprocess.run(
        [
            "git",
            "check-ignore",
            "-q",
            "--no-index",
            "plan_6_stock_volatility/dl_gpu_outputs.zip",
        ],
        cwd=config.PROJECT_ROOT.parent,
        check=False,
        timeout=10,
    )
    assert zip_chk.returncode == 0, "dl_gpu_outputs.zip should be gitignored"

    m7_doc = config.PROJECT_ROOT / "docs" / "milestones" / "M7_phase_b_colab_and_dl_pipeline.md"
    assert m7_doc.is_file() and m7_doc.stat().st_size > 2000
    ledger = (config.PROJECT_ROOT / "docs" / "milestones" / "README.md").read_text("utf-8")
    assert "547b286e6e85185830608eb3bf0a574669409dc5" in ledger
    assert "M7_phase_b_colab_and_dl_pipeline.md" in ledger

    repo_root = config.PROJECT_ROOT.parent
    for rel in (
        "plan_6_stock_volatility/notebooks/dl_gpu_training_colab.ipynb",
        "plan_6_stock_volatility/docs/milestones/M7_phase_b_colab_and_dl_pipeline.md",
    ):
        chk = subprocess.run(
            ["git", "check-ignore", "-v", "--no-index", rel],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        assert chk.returncode == 1 and chk.stdout.strip() == "", f"Gitignored: {rel}"


def test_committed_h3_dl_outputs_unified_final_pdf_and_m8_doc(
    block_network: None,
) -> None:
    """Verify committed Colab L4 H3 outputs, unified CS482 PDF (<=5pp, 61 claims), and M8 doc."""
    from stock_volatility.report import audit_compiled_pdf, collect_report_claims

    preds_dir = config.OUTPUTS_PREDICTIONS_DIR
    metrics_dir = config.OUTPUTS_METRICS_DIR
    tables_dir = config.OUTPUTS_TABLES_DIR

    for fn in (
        "patchtst_1m_test_predictions.parquet",
        "patchtst_15m_test_predictions.parquet",
        "chronos_2_test_predictions.parquet",
        "chronos_t5_large_test_predictions.parquet",
        "finbert_val_scores.parquet",
        "finbert_test_scores.parquet",
        "llm_val_scores.parquet",
        "llm_test_scores.parquet",
        "lgbm_plus_finbert_test.parquet",
        "lgbm_plus_llm_zs_test.parquet",
        "lgbm_plus_llm_qlora_test.parquet",
    ):
        p = preds_dir / fn
        assert p.is_file() and p.stat().st_size > 100, f"Missing/empty H3 prediction artifact: {p}"

    for fn in (
        "h3_dl_summary.json",
        "llm_run_metadata.json",
        "patchtst_1m_metrics.json",
        "patchtst_15m_metrics.json",
        "chronos_2_metrics.json",
        "chronos_t5_large_metrics.json",
    ):
        p = metrics_dir / fn
        assert p.is_file() and p.stat().st_size > 100, f"Missing/empty H3 metrics artifact: {p}"

    for ext in ("csv", "md", "tex"):
        p = tables_dir / f"table2_h3_ablation.{ext}"
        assert p.is_file() and p.stat().st_size > 100, f"Missing/empty Table 2 artifact: {p}"

    h3_summary = json.loads((metrics_dir / "h3_dl_summary.json").read_text(encoding="utf-8"))
    assert h3_summary["gpu_name"] == "NVIDIA L4"
    assert h3_summary["llm_model_id"] == "google/gemma-4-12B-it"
    assert h3_summary["train_event_days"] == 4929
    assert h3_summary["val_event_days"] == 880
    assert h3_summary["test_event_days"] == 895
    assert h3_summary["finbert_pr_auc_event"] > 0.13

    comp_df = pd.read_csv(tables_dir / "table2_h3_ablation.csv")
    assert len(comp_df) == 8
    chronos_rows = comp_df[comp_df["arm_id"].str.startswith("chronos_")]
    assert len(chronos_rows) == 2
    assert chronos_rows["pr_auc_full"].isna().all()
    tex_txt = (tables_dir / "table2_h3_ablation.tex").read_text(encoding="utf-8")
    assert "$895$" in tex_txt

    # Single authoritative CS482 submission PDF must pass A4, <=5.0pp, 61-claim & non-causal audit
    pdf_path = config.REPORT_DIR / config.REPORT_PDF_NAME
    audit_md = config.REPORT_DIR / "report_audit.md"
    assert pdf_path.is_file() and pdf_path.stat().st_size > 50_000
    assert (
        audit_md.is_file()
        and "Overall Audit Verdict: **PASS**" in audit_md.read_text(encoding="utf-8")
    )

    manifest = json.loads(
        (config.DATA_FROZEN_DIR / "FREEZE_MANIFEST.json").read_text(encoding="utf-8")
    )
    all_claims = collect_report_claims(config.PROJECT_ROOT)
    assert len(all_claims) == 61
    audit = audit_compiled_pdf(
        pdf_path=pdf_path,
        claims=all_claims,
        generated_at_utc=str(manifest["generated_at_utc"]),
    )
    assert audit["verdict"] == "PASS", f"Final PDF audit failed: {audit}"
    assert 4.0 <= audit["main_text_pages_est"] <= 5.0
    assert audit["main_pages_occupied"] <= 5
    assert audit["claims_verified"] == 61

    # Verify unified 5-section CS482 rubric layout (DL/LLM merged inside Data Analysis)
    report_tex = (config.REPORT_DIR / "report.tex").read_text(encoding="utf-8")
    report_bib = (config.REPORT_DIR / "references.bib").read_text(encoding="utf-8")
    for heading in (
        r"\section{Research Question}",
        r"\section{Data Collection and Preparation}",
        r"\section{Data Analysis}",
        r"\section{Method Evaluation}",
        r"\section{Results and Discussion}",
    ):
        assert heading in report_tex, f"Missing CS482 rubric section {heading}"
    assert r"\section{Extension:" not in report_tex
    sec3_to_sec4 = report_tex.split(r"\section{Data Analysis}", 1)[1].split(
        r"\section{Method Evaluation}", 1
    )[0]
    for tok in ("PatchTST", "chronos-2", "FinBERT", "google/gemma-4-12B-it", "QLoRA"):
        assert tok in sec3_to_sec4, f"Expected H3 {tok!r} merged inside Data Analysis"
    for ai_tok in ("Tools and Purposes", "Independent Verification", "Student Responsibility"):
        assert ai_tok in report_tex
    for bib_key in (
        "nie2023patchtst",
        "ansari2024chronos",
        "araci2019finbert",
        "dettmers2023qlora",
    ):
        assert bib_key in report_bib

    m8_doc = config.PROJECT_ROOT / "docs" / "milestones" / "M8_phase_b_l4_results_and_final_pdf.md"
    assert m8_doc.is_file() and m8_doc.stat().st_size > 2000
    ledger = (config.PROJECT_ROOT / "docs" / "milestones" / "README.md").read_text("utf-8")
    plan6_readme = (config.PROJECT_ROOT / "README.md").read_text("utf-8")
    assert "M8_phase_b_l4_results_and_final_pdf.md" in ledger
    assert "M8_phase_b_l4_results_and_final_pdf.md" in plan6_readme
    assert "CS482_GroupX_StockVolatilityLLM.pdf" in plan6_readme

    chk = subprocess.run(
        [
            "git",
            "check-ignore",
            "-v",
            "--no-index",
            "plan_6_stock_volatility/docs/milestones/M8_phase_b_l4_results_and_final_pdf.md",
            "plan_6_stock_volatility/report/CS482_GroupX_StockVolatilityLLM.pdf",
            "plan_6_stock_volatility/outputs/metrics/h3_dl_summary.json",
        ],
        cwd=config.PROJECT_ROOT.parent,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert chk.returncode == 1 and chk.stdout.strip() == "", f"Gitignored H3 files: {chk.stdout}"
