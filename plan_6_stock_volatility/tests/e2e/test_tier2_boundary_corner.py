"""Tier 2 Opaque-Box E2E Suite: Anti-Leakage, Mathematical, Boundary & Format Invariants."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from pypdf import PdfReader

from stock_volatility import config
from stock_volatility.evaluation import classification_metrics, regression_metrics
from stock_volatility.report import scan_report_text_issues


def test_t2_01_event_alignment_dst_early_close_and_cap_invariants(
    project_root: Path,
    events_phase_b_df: pd.DataFrame,
) -> None:
    """Verify strict 16:00 ET / 13:00 ET early-close alignment and <=5/day event cap."""
    max_cap = int(events_phase_b_df["news_count"].max())
    assert max_cap <= config.MAX_EVENTS_PER_DAY, (
        f"Event cap violated: found news_count={max_cap} > {config.MAX_EVENTS_PER_DAY}"
    )
    assert (events_phase_b_df["news_count"] >= 0).all()
    assert ((events_phase_b_df["has_news"] == 1) == (events_phase_b_df["news_count"] > 0)).all()

    aligned = pd.read_parquet(project_root / "data" / "interim" / "events_aligned.parquet")
    # Any event with aligned_date == et_date must have ET hour strictly < 16 (and < 13 on early)
    same_day = aligned[aligned["aligned_date"] == aligned["et_date"]]
    et_hours = same_day["acceptance_datetime_et"].str.slice(11, 13).astype(int)
    normal_same = same_day[~same_day["is_early_close_day"]]
    early_same = same_day[same_day["is_early_close_day"]]
    assert (et_hours.loc[normal_same.index] < 16).all(), "Post-16:00 ET leakage on same trading day"
    assert (et_hours.loc[early_same.index] < 13).all(), "Post-13:00 ET leakage on early-close day"
    # Every shifted event must land strictly AFTER its ET calendar date
    shifted = aligned[aligned["aligned_date"] != aligned["et_date"]]
    assert (shifted["aligned_date"] > shifted["et_date"]).all()
    assert int(aligned["shifted_due_to_early_close"].sum()) == 2


def test_t2_02_purged_embargo_split_boundaries_and_target_masking(
    panel_df: pd.DataFrame,
    split_metadata: dict[str, Any],
) -> None:
    """Verify zero split/embargo overlap and strict final-5-day target masking in 2025 Test."""
    valid_splits = {"train", "embargo_1", "val", "embargo_2", "test"}
    assert set(panel_df["split"].unique()) == valid_splits

    train_dates = sorted(panel_df.loc[panel_df["split"] == "train", "date"].unique())
    emb1_dates = sorted(panel_df.loc[panel_df["split"] == "embargo_1", "date"].unique())
    val_dates = sorted(panel_df.loc[panel_df["split"] == "val", "date"].unique())
    emb2_dates = sorted(panel_df.loc[panel_df["split"] == "embargo_2", "date"].unique())
    test_dates = sorted(panel_df.loc[panel_df["split"] == "test", "date"].unique())

    assert train_dates[0] == config.START_DATE and train_dates[-1] == config.TRAIN_END, (
        f"Train boundary breach: {train_dates[0]}..{train_dates[-1]}"
    )
    assert val_dates[0] == config.VAL_START and val_dates[-1] == config.VAL_END, (
        f"Val boundary breach: {val_dates[0]}..{val_dates[-1]}"
    )
    assert test_dates[0] == config.TEST_START and test_dates[-1] == config.TEST_END, (
        f"Test boundary breach: {test_dates[0]}..{test_dates[-1]}"
    )
    assert len(emb1_dates) >= config.HORIZON_DAYS and emb1_dates[0] > train_dates[-1]
    assert emb1_dates[-1] < val_dates[0], "Embargo 1 overlaps Val split"
    assert len(emb2_dates) >= config.HORIZON_DAYS and emb2_dates[0] > val_dates[-1]
    assert emb2_dates[-1] < test_dates[0], "Embargo 2 overlaps Test split"

    # Final 5 trading days of 2025 Test must have incomplete targets masked
    tail_5_dates = set(test_dates[-config.HORIZON_DAYS :])
    tail_rows = panel_df[panel_df["date"].isin(tail_5_dates)]
    assert (~tail_rows["has_full_target"]).all(), (
        "Incomplete forward target leakage: final 5 test days have has_full_target=True"
    )
    assert tail_rows["rv_5d_fwd"].isna().all() and tail_rows["y_tail"].isna().all()
    assert split_metadata["test_scorable_trading_days"] == len(test_dates) - config.HORIZON_DAYS


def test_t2_03_independent_target_and_train_only_rv90_recomputation(
    project_root: Path,
    panel_df: pd.DataFrame,
    feature_manifest: dict[str, Any],
) -> None:
    """Independently recompute Parkinson RV, y_tail, neg_ret_1d & Train-only rv90 threshold."""
    # Check neg_ret_1d == min(ret_1d, 0) everywhere non-null
    valid_ret = panel_df["ret_1d"].notna()
    np.testing.assert_allclose(
        panel_df.loc[valid_ret, "neg_ret_1d"].to_numpy(),
        np.minimum(panel_df.loc[valid_ret, "ret_1d"].to_numpy(), 0.0),
        atol=1e-12,
    )

    # Independently recompute Parkinson 5-day forward RV and y_tail on 3 representative stocks
    ohlcv = pd.read_parquet(project_root / "data" / "raw" / "prices" / "ohlcv_daily.parquet")
    park_const = 252.0 / (5.0 * 4.0 * math.log(2.0))
    for ticker in ("AAPL", "JPM", "XOM"):
        raw_s = ohlcv[ohlcv["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        pan_s = panel_df[panel_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        hl_sq = (np.log(raw_s["high"].to_numpy() / raw_s["low"].to_numpy())) ** 2
        adj = raw_s["adj_close"].to_numpy()
        for idx in (40, 500, 1550):
            expected_rv = math.sqrt(park_const * float(np.sum(hl_sq[idx + 1 : idx + 6])))
            expected_ret = float(adj[idx + 5] / adj[idx] - 1.0)
            expected_tail = int(expected_ret < config.TAIL_RETURN_THRESHOLD)
            np.testing.assert_allclose(pan_s.loc[idx, "rv_5d_fwd"], expected_rv, rtol=1e-10)
            assert int(pan_s.loc[idx, "y_tail"]) == expected_tail

    # Recompute Train-only 90th-percentile rv_5d_fwd threshold strictly on Train slice
    train_mod = panel_df[
        (panel_df["split"] == "train") & panel_df["has_full_features"] & panel_df["has_full_target"]
    ]
    recomputed_rv90 = float(np.quantile(train_mod["rv_5d_fwd"].to_numpy(), 0.90))
    recorded_rv90 = float(feature_manifest["train_preprocessing"]["rv90_threshold"])
    np.testing.assert_allclose(recorded_rv90, recomputed_rv90, rtol=1e-10)


def test_t2_04_independent_m4_metric_and_holm_recomputation(
    project_root: Path,
    test_pred_df: pd.DataFrame,
) -> None:
    """Independently recompute QLIKE/RMSE/MAE/PR-AUC/ROC-AUC/Brier/ECE & Holm bounds."""
    metrics_df = pd.read_csv(project_root / "outputs" / "tables" / "h2_test_metrics.csv")
    rv_true = test_pred_df["rv_5d_fwd"].to_numpy(dtype=float)
    y_true = test_pred_df["y_tail"].to_numpy(dtype=int)

    model_cols = {
        "persistence": ("rv_hat_persistence", "p_tail_persistence"),
        "gjr_garch": ("rv_hat_gjr_garch", "p_tail_gjr_garch"),
        "har_rv": ("rv_hat_har_rv", "p_tail_har_rv"),
        "lgbm": ("rv_hat_lgbm", "p_tail_lgbm_cal"),
    }
    for m_key, (rv_col, p_col) in model_cols.items():
        rv_hat = test_pred_df[rv_col].to_numpy(dtype=float)
        p_hat = test_pred_df[p_col].to_numpy(dtype=float)
        assert np.isfinite(rv_hat).all() and (rv_hat > 0).all()
        assert np.isfinite(p_hat).all() and (p_hat >= 0.0).all() and (p_hat <= 1.0).all()

        rm = regression_metrics(rv_true, rv_hat)
        cm = classification_metrics(y_true, p_hat)
        row = metrics_df[metrics_df["model"] == m_key].iloc[0]
        np.testing.assert_allclose(row["qlike"], rm["qlike"], atol=1e-6)
        np.testing.assert_allclose(row["rmse"], rm["rmse"], atol=1e-6)
        np.testing.assert_allclose(row["mae"], rm["mae"], atol=1e-6)
        np.testing.assert_allclose(row["pr_auc"], cm["pr_auc"], atol=1e-6)
        np.testing.assert_allclose(row["roc_auc"], cm["roc_auc"], atol=1e-6)
        np.testing.assert_allclose(row["brier"], cm["brier"], atol=1e-6)
        np.testing.assert_allclose(row["ece"], cm["ece"], atol=1e-6)
        assert row["qlike_ci95_low"] <= row["qlike_ci95_high"]
        assert row["pr_auc_ci95_low"] <= row["pr_auc_ci95_high"]

    # Check Holm monotonicity across both H1 and H2 DM tables (qlike, se, brier x 3 = 9 rows)
    dm_df = pd.read_csv(project_root / "outputs" / "tables" / "h2_dm_tests.csv")
    assert len(dm_df) == 9
    assert set(dm_df["loss_metric"].unique()) == {"qlike", "se", "brier"}
    assert (dm_df["p_holm"] >= dm_df["p_value"] - 1e-12).all()
    assert ((dm_df["p_holm"] >= 0.0) & (dm_df["p_holm"] <= 1.0)).all()

    h1 = json.loads(
        (project_root / "outputs" / "metrics" / "h1_regression.json").read_text(encoding="utf-8")
    )
    for item in h1["confirmatory_family_holm"]:
        assert item["p_holm"] >= item["p_value"] - 1e-12
        assert 0.0 <= item["p_holm"] <= 1.0


def test_t2_05_treeshap_additivity_and_non_causal_contract(project_root: Path) -> None:
    """Verify Train-background TreeSHAP additivity < 1e-6 and non-causal framing."""
    shap_audit = json.loads(
        (project_root / "outputs" / "metrics" / "shap_audit.json").read_text(encoding="utf-8")
    )
    tol = float(shap_audit["additivity_tolerance"])
    assert tol <= 1e-6
    assert shap_audit["classifier_max_additivity_error"] < tol
    assert shap_audit["regressor_max_additivity_error"] < tol
    assert shap_audit["background_split"] == "train"
    assert shap_audit["background_date_range"][1] <= config.TRAIN_END
    assert shap_audit["classifier_classes"] == [0, 1]
    assert shap_audit["classifier_positive_class_index"] == 1
    assert "non-causal" in shap_audit["non_causal_framing"].lower()


def test_t2_06_figures_300_dpi_high_res_and_non_blank(project_root: Path) -> None:
    """Verify Figure 1 & Figure 2 PNGs are 300 DPI (pHYs chunk), high-res, and non-blank."""
    for stem in ("fig1_volatility_regimes_and_forecasts", "fig2_shap_and_calibration"):
        png_path = project_root / "outputs" / "figures" / f"{stem}.png"
        raw = png_path.read_bytes()
        phys_idx = raw.find(b"pHYs")
        assert phys_idx != -1, f"Missing pHYs DPI metadata in {stem}.png"
        x_ppm = int.from_bytes(raw[phys_idx + 4 : phys_idx + 8], "big")
        dpi = x_ppm * 0.0254
        assert dpi >= 299.0, f"DPI below 300 in {stem}.png: got {dpi:.2f}"

        img = mpimg.imread(png_path)
        h, w = img.shape[:2]
        assert w >= 2000 and h >= 1000, f"Low pixel resolution in {stem}.png: {w}x{h}"
        assert float(np.std(img[..., :3])) > 0.05, f"Blank or near-blank figure: {stem}.png"


def test_t2_07_pdf_every_page_a4_budget_and_zero_forbidden_words(project_root: Path) -> None:
    """Verify EVERY page of CS482 PDF is A4, main text is in [3.0, 5.0] pp, and zero causal hits."""
    pdf_path = project_root / "report" / config.REPORT_PDF_NAME
    reader = PdfReader(str(pdf_path))
    assert len(reader.pages) >= 5, f"Too few PDF pages: {len(reader.pages)}"

    bad_pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        w = float(page.mediabox.width)
        h = float(page.mediabox.height)
        if abs(w - 595.28) >= 1.5 or abs(h - 841.89) >= 1.5:
            bad_pages.append(f"page {i} is not A4 ({w:.2f}x{h:.2f} pts)")
    assert not bad_pages, f"Non-A4 pages detected in {pdf_path.name}: {'; '.join(bad_pages)}"

    page_texts = [p.extract_text() or "" for p in reader.pages]
    ref_idx = next(
        (i for i in range(1, len(page_texts)) if re.search(r"(?m)^References\b", page_texts[i])),
        None,
    )
    assert ref_idx is not None, "Missing References section in PDF"
    main_pages = ref_idx - 1
    assert 3 <= main_pages <= 5, f"Main text occupies {main_pages} pages, outside CS482 [3, 5]"

    # Scan cover + main text + all declaration pages after References for forbidden causal terms
    scan_chunks = [page_texts[0], *page_texts[1:ref_idx]]
    in_decl = False
    for idx in range(ref_idx, len(page_texts)):
        txt = page_texts[idx]
        if in_decl:
            scan_chunks.append(txt)
        elif "Separate Contribution Statement" in txt:
            in_decl = True
            scan_chunks.append(txt.split("Separate Contribution Statement", 1)[1])
        elif "AI Usage Declaration" in txt:
            in_decl = True
            scan_chunks.append(txt.split("AI Usage Declaration", 1)[1])

    issues = scan_report_text_issues("\n".join(scan_chunks))
    assert issues == [], f"Forbidden wording or unresolved artifacts found in PDF: {issues}"
