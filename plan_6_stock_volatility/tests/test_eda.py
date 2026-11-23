"""Unit, determinism, and committed-output contract tests for Milestone M3 EDA."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_volatility import config

# ---------------------------------------------------------------------------
# Layer 1: Pure Offline Unit Tests on Synthetic Panels
# ---------------------------------------------------------------------------


def _make_synthetic_panel() -> pd.DataFrame:
    """Create a minimal synthetic multi-sector panel covering Train, Val, and Test."""
    dates = [
        "2019-06-03",
        "2020-03-16",
        "2022-06-13",
        "2023-05-01",
        "2024-06-03",
        "2025-04-07",
    ]
    splits = ["train", "train", "train", "train", "val", "test"]
    vix_vals = [15.0, 45.0, 30.0, 17.0, 14.0, 35.0]
    rows = []
    for ticker, sector in [("AAPL", "Technology"), ("JNJ", "Healthcare")]:
        for idx, (dt, sp, vix) in enumerate(zip(dates, splits, vix_vals, strict=True)):
            ret = -0.04 if idx % 2 == 1 else 0.02
            rv = 0.45 if vix >= 25.0 else 0.18
            rows.append(
                {
                    "ticker": ticker,
                    "sector": sector,
                    "sub_group": "Test",
                    "date": dt,
                    "split": sp,
                    "has_full_features": True,
                    "has_full_target": True,
                    "rv_5d_fwd": rv,
                    "y_log_rv": float(np.log(rv + config.RV_EPS)),
                    "rv_5d_cc_fwd": rv * 0.95,
                    "cum_ret_5d_fwd": -0.07 if idx in (1, 5) else 0.01,
                    "y_tail": 1 if idx in (1, 5) else 0,
                    "y_tail_rv90": 1 if rv >= 0.37 else 0,
                    "rv_d": rv,
                    "rv_w": rv,
                    "rv_m": rv,
                    "log_rv_d": float(np.log(rv + config.RV_EPS)),
                    "log_rv_w": float(np.log(rv + config.RV_EPS)),
                    "log_rv_m": float(np.log(rv + config.RV_EPS)),
                    "ret_1d": ret,
                    "neg_ret_1d": min(ret, 0.0),
                    "ret_5d": ret,
                    "vol_ratio_20d": 1.2,
                    "daily_range_pct": 0.02,
                    "vix_close": vix,
                    "t10y2y": 0.10,
                    "spy_ret_1d": ret * 0.5,
                    "has_news": 1 if idx in (1, 4, 5) else 0,
                    "news_count": 2 if idx in (1, 4, 5) else 0,
                    "has_8k_earnings": 1 if idx == 4 else 0,
                    "news_source": "sec_edgar" if idx in (1, 4, 5) else "none",
                }
            )
    return pd.DataFrame(rows)


def test_eda_tables_pure_functions_and_vix_train_only_threshold(block_network: None) -> None:
    """Verify EDA builders compute exact numerators/denominators and Train-only VIX median."""
    from stock_volatility.eda import (
        compute_coverage_consistency,
        compute_leverage_asymmetry,
        compute_regimes_by_sector,
        compute_vix_news_tail_incidence,
    )

    df = _make_synthetic_panel()

    regimes = compute_regimes_by_sector(df)
    assert {"covid_shock_2020", "rate_hike_bear_2022", "ai_cycle_test_2025", "full_2019_2025"} <= (
        set(regimes["regime"])
    )
    full_all = regimes[(regimes["regime"] == "full_2019_2025") & (regimes["sector"] == "ALL")]
    assert int(full_all["n_obs"].iloc[0]) == 12
    assert int(full_all["y_tail_pos"].iloc[0]) == 4
    assert np.isclose(full_all["y_tail_rate"].iloc[0], 4 / 12)

    lev = compute_leverage_asymmetry(df)
    assert {"ALL", "Technology", "Healthcare"}.issubset(set(lev["sector"]))
    assert {"ret_lt_0", "ret_ge_0"}.issubset(set(lev["ret_bucket"]))

    cov = compute_coverage_consistency(df)
    all_cov = cov[cov["sector"] == "ALL"].iloc[0]
    assert int(all_cov["event_days_1923"]) == 2
    assert int(all_cov["event_days_2425"]) == 4

    # Train daily VIX values are [15, 45, 30, 17] -> median = 23.5 (must ignore Val=14, Test=35)
    vix_tbl, vix_thresh = compute_vix_news_tail_incidence(df)
    assert np.isclose(vix_thresh, 23.5)
    overall_hi_ev = vix_tbl[
        (vix_tbl["scope"] == "overall")
        & (vix_tbl["vix_regime"] == "High_VIX")
        & (vix_tbl["event_status"] == "Has_Event")
    ].iloc[0]
    assert int(overall_hi_ev["n_obs"]) == 4
    assert int(overall_hi_ev["y_tail_pos"]) == 4
    assert np.isclose(overall_hi_ev["y_tail_rate"], 1.0)


def test_eda_pipeline_writes_deterministic_outputs_to_tmp(
    tmp_path: Path, block_network: None
) -> None:
    """Verify run_eda_pipeline runs offline from frozen panel and writes all 6 outputs."""
    from stock_volatility.eda import run_eda_pipeline

    out_tables = tmp_path / "tables"
    out_metrics = tmp_path / "metrics"
    summary_1 = run_eda_pipeline(
        outputs_tables_dir=out_tables,
        outputs_metrics_dir=out_metrics,
    )
    hashes_1 = {
        p.name: config.sha256_file(p)
        for p in sorted(list(out_tables.glob("*")) + list(out_metrics.glob("*")))
    }
    summary_2 = run_eda_pipeline(
        outputs_tables_dir=out_tables,
        outputs_metrics_dir=out_metrics,
    )
    hashes_2 = {
        p.name: config.sha256_file(p)
        for p in sorted(list(out_tables.glob("*")) + list(out_metrics.glob("*")))
    }
    assert summary_1 == summary_2
    assert hashes_1 == hashes_2
    assert len(hashes_1) == 6


# ---------------------------------------------------------------------------
# Layer 2: Committed M3 EDA Artifacts Contract Tests
# ---------------------------------------------------------------------------


def test_committed_m3_eda_outputs_and_non_causal_wording(block_network: None) -> None:
    """Verify committed M3 EDA CSVs, JSON, and Markdown match frozen panel invariants."""
    expected_tables = (
        "eda_regimes_by_sector.csv",
        "eda_leverage_asymmetry.csv",
        "eda_coverage_consistency.csv",
        "eda_vix_news_tail_incidence.csv",
        "eda_summary.md",
    )
    for name in expected_tables:
        p = config.OUTPUTS_TABLES_DIR / name
        assert p.is_file(), f"Missing M3 EDA table: {p}"

    summary_json_path = config.OUTPUTS_METRICS_DIR / "eda_summary.json"
    assert summary_json_path.is_file()
    summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
    assert summary["modeling_total_rows"] == 104_040
    assert summary["modeling_total_days"] == 1734
    assert 15.0 < summary["vix_train_median_threshold"] < 25.0

    regimes = pd.read_csv(config.OUTPUTS_TABLES_DIR / "eda_regimes_by_sector.csv")
    full_all = regimes[(regimes["regime"] == "full_2019_2025") & (regimes["sector"] == "ALL")]
    assert int(full_all["n_obs"].iloc[0]) == 104_040
    # 2020 COVID shock mean RV must exceed 2024 calm market mean RV
    rv_2020 = float(
        regimes[(regimes["regime"] == "covid_shock_2020") & (regimes["sector"] == "ALL")][
            "rv_5d_fwd_mean"
        ].iloc[0]
    )
    rv_2024 = float(
        regimes[(regimes["regime"] == "ai_cycle_val_2024") & (regimes["sector"] == "ALL")][
            "rv_5d_fwd_mean"
        ].iloc[0]
    )
    assert rv_2020 > rv_2024

    cov = pd.read_csv(config.OUTPUTS_TABLES_DIR / "eda_coverage_consistency.csv")
    all_cov = cov[cov["sector"] == "ALL"].iloc[0]
    assert int(all_cov["active_tickers_1923"]) == 60
    assert int(all_cov["active_tickers_2425"]) == 60

    vix_tbl = pd.read_csv(config.OUTPUTS_TABLES_DIR / "eda_vix_news_tail_incidence.csv")
    assert {"n_obs", "y_tail_pos", "y_tail_rate", "ci95_low", "ci95_high"}.issubset(
        set(vix_tbl.columns)
    )
    assert (vix_tbl["n_obs"] > 0).all()

    md_text = (config.OUTPUTS_TABLES_DIR / "eda_summary.md").read_text(encoding="utf-8").lower()
    for forbidden in ("caused", "causal impact", "drivers", "elasticity"):
        assert forbidden not in md_text, (
            f"Forbidden causal phrasing {forbidden!r} in eda_summary.md"
        )
