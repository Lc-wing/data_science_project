"""Unit, econometric equivalence, Holm/Wald, and committed-output tests for M3 H1 regression."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from stock_volatility import config

# ---------------------------------------------------------------------------
# Layer 1: Pure Offline Econometric Verification Tests
# ---------------------------------------------------------------------------


def test_holm_bonferroni_adjustment_exact_values(block_network: None) -> None:
    """Verify step-down Holm-Bonferroni adjusted p-values match textbook values."""
    from stock_volatility.regression import holm_bonferroni_adjust

    raw = [0.01, 0.04, 0.03, 0.005]
    # Sorted: 0.005 (*4=0.02), 0.01 (*3=0.03), 0.03 (*2=0.06), 0.04 (*1=0.04 -> max=0.06)
    adj = holm_bonferroni_adjust(raw)
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02], atol=1e-12)


def test_driscoll_kraay_se_matches_statsmodels_hac_groupsum(block_network: None) -> None:
    """Verify Stock-FE Driscoll-Kraay SE matches statsmodels hac-groupsum up to exact FE dof."""
    from stock_volatility.regression import fit_stock_fe_driscoll_kraay

    rng = np.random.default_rng(42)
    n_entities, n_dates = 10, 60
    tickers = [f"T{i:02d}" for i in range(n_entities)]
    dates = pd.bdate_range("2020-01-02", periods=n_dates).strftime("%Y-%m-%d").tolist()

    # Common time shock induces cross-sectional correlation (why Driscoll-Kraay matters)
    common_shock = rng.normal(0.0, 0.5, size=n_dates)
    entity_fe = rng.normal(0.0, 1.0, size=n_entities)

    rows = []
    for ei, tk in enumerate(tickers):
        for ti, dt in enumerate(dates):
            x1 = rng.normal(0.0, 1.0) + 0.3 * common_shock[ti]
            x2 = rng.normal(0.0, 1.0)
            y = entity_fe[ei] + 0.8 * x1 - 0.5 * x2 + common_shock[ti] + rng.normal(0.0, 0.2)
            rows.append({"ticker": tk, "date": dt, "y": y, "x1": x1, "x2": x2})
    df = pd.DataFrame(rows)

    res = fit_stock_fe_driscoll_kraay(df, y_col="y", x_cols=["x1", "x2"], maxlags=5)
    assert np.isclose(res["beta"][0], 0.8, atol=0.15)
    assert np.isclose(res["beta"][1], -0.5, atol=0.15)

    # Cross-check against statsmodels OLS(cov_type="hac-groupsum") on demeaned data
    df_s = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    y_dm = df_s["y"] - df_s.groupby("ticker")["y"].transform("mean")
    X_dm = df_s[["x1", "x2"]].copy()
    for c in ("x1", "x2"):
        X_dm[c] = df_s[c] - df_s.groupby("ticker")[c].transform("mean")
    date_codes = pd.Categorical(df_s["date"], categories=dates, ordered=True).codes
    sm_res = sm.OLS(y_dm.to_numpy(), X_dm.to_numpy()).fit(
        cov_type="hac-groupsum",
        cov_kwds={"time": date_codes, "maxlags": 5, "use_correction": True},
    )

    # Beta must match to machine precision
    assert np.allclose(res["beta"], sm_res.params, atol=1e-10)
    # statsmodels hac-groupsum returns unscaled sandwich; our DK applies exact FE + time dof
    n_obs = len(df_s)
    k = 2
    expected_ratio = np.sqrt(
        (n_obs / (n_obs - k - n_entities)) * (n_dates / (n_dates - 1))
    )
    observed_ratios = res["se"] / sm_res.bse
    assert np.allclose(observed_ratios, expected_ratio, rtol=1e-10)


def test_h1_pipeline_runs_offline_and_is_deterministic(
    tmp_path: Path, block_network: None
) -> None:
    """Verify run_h1_regression_pipeline runs offline from frozen panel deterministically."""
    from stock_volatility.regression import run_h1_regression_pipeline

    out_tables = tmp_path / "tables"
    out_metrics = tmp_path / "metrics"
    r1 = run_h1_regression_pipeline(
        outputs_tables_dir=out_tables,
        outputs_metrics_dir=out_metrics,
    )
    h1 = {
        p.name: config.sha256_file(p)
        for p in sorted(list(out_tables.glob("*")) + list(out_metrics.glob("*")))
    }
    r2 = run_h1_regression_pipeline(
        outputs_tables_dir=out_tables,
        outputs_metrics_dir=out_metrics,
    )
    h2 = {
        p.name: config.sha256_file(p)
        for p in sorted(list(out_tables.glob("*")) + list(out_metrics.glob("*")))
    }
    assert r1 == r2
    assert h1 == h2
    assert len(h1) == 4


# ---------------------------------------------------------------------------
# Layer 2: Committed M3 H1 Regression Outputs & Milestone Doc Contracts
# ---------------------------------------------------------------------------


def test_committed_m3_h1_regression_outputs_and_contracts(block_network: None) -> None:
    """Verify committed H1 main table, sensitivities, JSON metrics, and Holm confirmatory family."""
    main_csv = config.OUTPUTS_TABLES_DIR / "h1_panel_har_rv_main.csv"
    sens_csv = config.OUTPUTS_TABLES_DIR / "h1_panel_har_rv_sensitivities.csv"
    md_path = config.OUTPUTS_TABLES_DIR / "h1_summary.md"
    json_path = config.OUTPUTS_METRICS_DIR / "h1_regression.json"
    for p in (main_csv, sens_csv, md_path, json_path):
        assert p.is_file(), f"Missing H1 output: {p}"

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["main_model"]["n_obs"] == 104_040
    assert payload["main_model"]["n_entities"] == 60
    assert payload["main_model"]["n_dates"] == 1734
    assert payload["main_model"]["bartlett_bandwidth_L"] == 7
    assert payload["main_model"]["within_r2"] > 0.40

    # Confirmatory family must have all 5 pre-registered hypotheses with Holm-adjusted p-values
    conf = {row["test_id"]: row for row in payload["confirmatory_family_holm"]}
    expected_ids = {
        "neg_ret_1d",
        "vix_close",
        "tech_x_neg_ret_1d_vs_healthcare",
        "tech_x_vix_close_vs_healthcare",
        "joint_5sector_interaction_wald",
    }
    assert set(conf.keys()) == expected_ids
    # Leverage overall (neg_ret_1d < 0 => higher vol after drops) & VIX > 0 must be significant
    assert conf["neg_ret_1d"]["estimate"] < 0
    assert conf["neg_ret_1d"]["p_holm"] < 0.01
    assert conf["vix_close"]["estimate"] > 0
    assert conf["vix_close"]["p_holm"] < 0.001
    assert conf["joint_5sector_interaction_wald"]["p_holm"] < 0.001

    main_df = pd.read_csv(main_csv)
    assert {"term", "coef", "se_dk", "t_stat", "p_value", "ci95_low", "ci95_high"}.issubset(
        set(main_df.columns)
    )
    assert len(main_df) >= 13

    sens_df = pd.read_csv(sens_csv)
    assert {
        "train_only_2019_2023",
        "close_to_close_rv_5d_cc_fwd",
        "newey_west_entity_clustered_se",
    }.issubset(set(sens_df["specification"]))

    md_text = md_path.read_text(encoding="utf-8").lower()
    for forbidden in ("caused", "causal impact", "drivers", "elasticity"):
        assert forbidden not in md_text, f"Forbidden causal phrasing {forbidden!r} in h1_summary.md"


def test_m3_milestone_delivery_doc_complete_linked_and_not_gitignored() -> None:
    """Verify M3 milestone delivery doc exists, records deliverables + AI log, & is tracked."""
    repo_root = config.PROJECT_ROOT.parent
    milestones_dir = config.PROJECT_ROOT / "docs" / "milestones"
    m3_doc = milestones_dir / "M3_eda_and_h1_panel_har_rv.md"
    assert m3_doc.is_file(), f"Missing M3 milestone doc: {m3_doc}"
    assert m3_doc.stat().st_size > 2000

    m3_text = m3_doc.read_text(encoding="utf-8")
    for token in (
        "3c9bba99b09c53a838d787a60b0b7ebb9c4fb025",
        "eda_regimes_by_sector.csv",
        "eda_leverage_asymmetry.csv",
        "eda_coverage_consistency.csv",
        "eda_vix_news_tail_incidence.csv",
        "h1_panel_har_rv_main.csv",
        "h1_panel_har_rv_sensitivities.csv",
        "h1_regression.json",
        "Driscoll",
        "Holm",
        "03_eda.py",
        "04_panel_har_rv.py",
    ):
        assert token in m3_text, f"Missing {token!r} in M3_eda_and_h1_panel_har_rv.md"

    ledger_text = (milestones_dir / "README.md").read_text(encoding="utf-8")
    assert "3c9bba99b09c53a838d787a60b0b7ebb9c4fb025" in ledger_text
    assert "M3_eda_and_h1_panel_har_rv.md" in ledger_text

    plan6_readme = (config.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/milestones/M3_eda_and_h1_panel_har_rv.md" in plan6_readme

    proc = subprocess.run(
        [
            "git",
            "check-ignore",
            "-v",
            "--no-index",
            "plan_6_stock_volatility/docs/milestones/M3_eda_and_h1_panel_har_rv.md",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert proc.returncode == 1, f"M3 milestone doc must NOT be gitignored:\n{proc.stdout}"
    assert proc.stdout.strip() == ""
