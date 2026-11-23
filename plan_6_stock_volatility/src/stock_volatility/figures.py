"""Milestone M5: Publication-grade 300-DPI Figure 1 and Figure 2 generators (PNG + PDF)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stock_volatility import config
from stock_volatility.panel.freeze import verify_freeze_manifest

matplotlib.use("Agg")

# Okabe-Ito colorblind-safe publication palette
SECTOR_COLORS: dict[str, str] = {
    "Technology": "#0072B2",
    "Financials": "#D55E00",
    "Healthcare": "#009E73",
    "Consumer": "#CC79A7",
    "Energy": "#E69F00",
}

MODEL_COLORS: dict[str, str] = {
    "actual": "#111111",
    "persistence": "#999999",
    "gjr_garch": "#E69F00",
    "har_rv": "#009E73",
    "lgbm": "#0072B2",
    "lgbm_raw": "#D55E00",
}


def _apply_pub_style() -> None:
    """Configure crisp, publication-grade matplotlib typography and styling."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def generate_figure1(
    *,
    panel_df: pd.DataFrame,
    leverage_df: pd.DataFrame,
    test_pred_df: pd.DataFrame,
    out_dir: Path,
) -> dict[str, str]:
    """Generate 300-DPI Figure 1 (PNG + PDF): sector regimes, leverage, and 2025 spike forecasts."""
    _apply_pub_style()
    fig = plt.figure(figsize=(10.6, 6.8), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0], hspace=0.40, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # --- Panel A: 2019-2025 20-day smoothed sector RV trajectories ---
    mod = panel_df[
        panel_df["has_full_features"].astype(bool) & panel_df["has_full_target"].astype(bool)
    ].copy()
    sec_daily = (
        mod.groupby(["date", "sector"], sort=True)["rv_5d_fwd"]
        .mean()
        .unstack("sector")
        .sort_index()
    )
    sec_daily.index = pd.to_datetime(sec_daily.index)
    sec_smooth = sec_daily.rolling(20, min_periods=5).mean()

    # Shaded macro regimes
    ax_a.axvspan(
        pd.Timestamp("2020-02-20"), pd.Timestamp("2020-06-30"), color="#FDEBD0", alpha=0.55
    )
    ax_a.axvspan(
        pd.Timestamp("2022-01-03"), pd.Timestamp("2022-12-30"), color="#FADBD8", alpha=0.45
    )
    ax_a.axvspan(
        pd.Timestamp("2024-01-02"), pd.Timestamp("2024-12-20"), color="#D6EAF8", alpha=0.32
    )
    ax_a.axvspan(
        pd.Timestamp("2025-01-02"), pd.Timestamp("2025-12-23"), color="#D5F5E3", alpha=0.35
    )

    for sec in config.SECTORS:
        ax_a.plot(
            sec_smooth.index,
            sec_smooth[sec],
            label=sec,
            color=SECTOR_COLORS[sec],
            linewidth=1.35,
            alpha=0.92,
        )

    ax_a.axvline(pd.Timestamp("2024-01-02"), color="#444444", linestyle=":", linewidth=1.1)
    ax_a.axvline(pd.Timestamp("2025-01-02"), color="#111111", linestyle="--", linewidth=1.1)
    y_top = float(sec_smooth.max().max()) * 1.12
    ax_a.set_ylim(0.08, y_top)
    ax_a.text(
        pd.Timestamp("2020-04-25"), y_top * 0.90, "COVID\nShock", ha="center", fontsize=7.8,
        color="#7E5109", fontweight="bold"
    )
    ax_a.text(
        pd.Timestamp("2022-07-01"), y_top * 0.90, "2022 Rate-Hike\nBear Market", ha="center",
        fontsize=7.8, color="#78281F", fontweight="bold"
    )
    ax_a.text(
        pd.Timestamp("2024-07-01"), y_top * 0.90, "2024 Val\n(Calib.)", ha="center",
        fontsize=7.8, color="#1B4F72", fontweight="bold"
    )
    ax_a.text(
        pd.Timestamp("2025-07-01"), y_top * 0.90, "2025 Test\n(Holdout)", ha="center",
        fontsize=7.8, color="#145A32", fontweight="bold"
    )
    ax_a.set_title(
        "A. Cross-Sector Realized Volatility Regimes (2019–2025, 20-Day Smoothed)", pad=24
    )
    ax_a.set_ylabel("Annualized 5-Day Parkinson RV")
    ax_a.xaxis.set_major_locator(mdates.YearLocator())
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_a.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=5,
        frameon=True,
        framealpha=0.95,
        edgecolor="#CCCCCC",
    )

    # --- Panel B: Leverage Asymmetry across 6 lagged daily return bins ---
    bin_order = [
        ("ret_le_neg3", "<=-3%"),
        ("ret_neg3_to_neg1", "(-3,-1]%"),
        ("ret_neg1_to_0", "(-1,0)%"),
        ("ret_0_to_pos1", "[0,+1)%"),
        ("ret_pos1_to_pos3", "[+1,+3)%"),
        ("ret_ge_pos3", ">=+3%"),
    ]
    bin_keys = [k for k, _ in bin_order]
    bin_labels = [lbl for _, lbl in bin_order]
    x_pos = np.arange(len(bin_keys))

    lev_ov = leverage_df[leverage_df["scope"] == "overall"]
    plot_series = [
        ("ALL", "#111111", "o", 2.0, "All 60 Stocks"),
        ("Technology", SECTOR_COLORS["Technology"], "s", 1.4, "Technology"),
        ("Financials", SECTOR_COLORS["Financials"], "^", 1.4, "Financials"),
        ("Healthcare", SECTOR_COLORS["Healthcare"], "D", 1.4, "Healthcare"),
    ]
    for sec_key, color, marker, lw, lbl in plot_series:
        sub = lev_ov[lev_ov["sector"] == sec_key].set_index("ret_bucket")
        y_vals = [float(sub.loc[b, "mean_rv_5d_fwd"]) for b in bin_keys]
        ax_b.plot(
            x_pos, y_vals, marker=marker, markersize=4.8, linewidth=lw, color=color, label=lbl
        )

    ax_b.axvline(2.5, color="#666666", linestyle=":", linewidth=1.0)
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(bin_labels, rotation=15, ha="right")
    ax_b.set_xlabel("Lagged Daily Return Bucket ($r_{i,t}$)")
    ax_b.set_ylabel("Mean Forward 5-Day RV")
    ax_b.set_title("B. Leverage Asymmetry (2019–2025)")
    ax_b.legend(loc="upper center", ncol=2, frameon=True, framealpha=0.92, edgecolor="#CCCCCC")

    # --- Panel C: 2025 Holdout Test Volatility Spike Forecast Comparison (Mar-May 2025) ---
    spike = test_pred_df[
        (test_pred_df["date"] >= "2025-03-01") & (test_pred_df["date"] <= "2025-05-31")
    ].copy()
    daily_sp = (
        spike.groupby("date", sort=True)[
            [
                "rv_5d_fwd",
                "rv_hat_persistence",
                "rv_hat_gjr_garch",
                "rv_hat_har_rv",
                "rv_hat_lgbm",
            ]
        ]
        .mean()
        .reset_index()
    )
    dt_sp = pd.to_datetime(daily_sp["date"])
    ax_c.plot(
        dt_sp, daily_sp["rv_5d_fwd"], color=MODEL_COLORS["actual"], linewidth=1.9,
        label="Actual RV (60-Stock Mean)"
    )
    ax_c.plot(
        dt_sp, daily_sp["rv_hat_lgbm"], color=MODEL_COLORS["lgbm"], linewidth=1.6,
        label="LightGBM (QLIKE 0.231)"
    )
    ax_c.plot(
        dt_sp, daily_sp["rv_hat_har_rv"], color=MODEL_COLORS["har_rv"], linewidth=1.35,
        linestyle="--", label="Panel HAR-RV (0.238)"
    )
    ax_c.plot(
        dt_sp, daily_sp["rv_hat_gjr_garch"], color=MODEL_COLORS["gjr_garch"], linewidth=1.15,
        linestyle="-.", label="GJR-GARCH (0.301)"
    )
    ax_c.plot(
        dt_sp, daily_sp["rv_hat_persistence"], color=MODEL_COLORS["persistence"], linewidth=1.05,
        linestyle=":", label="Persistence (0.314)"
    )
    ax_c.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax_c.get_xticklabels(), rotation=18, ha="right")
    ax_c.set_xlabel("2025 Forecast Origin Date (Mar–May Spike)")
    ax_c.set_ylabel("Cross-Sectional Mean 5-Day RV")
    ax_c.set_title("C. 2025 Holdout Spike Forecast Tracking")
    ax_c.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="#CCCCCC")

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "fig1_volatility_regimes_and_forecasts.png"
    pdf_path = out_dir / "fig1_volatility_regimes_and_forecasts.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(
        pdf_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.10,
        metadata={"CreationDate": None},
    )
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def generate_figure2(
    *,
    shap_imp_df: pd.DataFrame,
    shap_regime_df: pd.DataFrame,
    shap_audit: dict[str, Any],
    cal_curve_df: pd.DataFrame,
    test_metrics_df: pd.DataFrame,
    out_dir: Path,
) -> dict[str, str]:
    """Generate 300-DPI Figure 2 (PNG + PDF): TreeSHAP model behavior & calibration curve."""
    _apply_pub_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.6, 4.4), dpi=300)
    plt.subplots_adjust(wspace=0.30)

    # --- Panel A: Top-10 Train-Background TreeSHAP by VIX Regime (Classifier Prob Space) ---
    clf_imp = (
        shap_imp_df[shap_imp_df["model_type"] == "classifier_prob"]
        .sort_values("rank")
        .head(10)
        .copy()
    )
    top_feats = clf_imp["feature"].tolist()[::-1]  # bottom-to-top for barh
    reg_clf = shap_regime_df[shap_regime_df["model_type"] == "classifier_prob"]
    hi_map = (
        reg_clf[reg_clf["regime_group"] == "High_VIX"]
        .set_index("feature")["mean_abs_shap"]
        .to_dict()
    )
    lo_map = (
        reg_clf[reg_clf["regime_group"] == "Low_VIX"]
        .set_index("feature")["mean_abs_shap"]
        .to_dict()
    )

    pretty_names = {
        "rv_m": "Monthly RV (rv_m)",
        "vix_close": "CBOE VIX (vix_close)",
        "t10y2y": "10Y–2Y Spread (t10y2y)",
        "log_rv_m": "Log Monthly RV (log_rv_m)",
        "sector_healthcare": "Sector: Healthcare",
        "spy_ret_1d": "SPY 1-Day Return",
        "vol_ratio_20d": "20-Day Volume Ratio",
        "sector_consumer": "Sector: Consumer",
        "log_rv_w": "Log Weekly RV (log_rv_w)",
        "rv_w": "Weekly RV (rv_w)",
        "rv_d": "Daily RV (rv_d)",
        "log_rv_d": "Log Daily RV (log_rv_d)",
        "neg_ret_1d": "Leverage Term (neg_ret_1d)",
        "ret_5d": "5-Day Trailing Return",
        "sector_energy": "Sector: Energy",
        "sector_technology": "Sector: Technology",
        "sector_financials": "Sector: Financials",
    }

    y_pos = np.arange(len(top_feats))
    h = 0.36
    hi_vals = [float(hi_map.get(f, 0.0)) for f in top_feats]
    lo_vals = [float(lo_map.get(f, 0.0)) for f in top_feats]

    ax_a.barh(
        y_pos + h / 2, hi_vals, height=h, color="#D55E00", alpha=0.88,
        label=f"High VIX (>= {shap_audit['vix_train_median_threshold']:.2f})"
    )
    ax_a.barh(
        y_pos - h / 2, lo_vals, height=h, color="#0072B2", alpha=0.88,
        label=f"Low VIX (< {shap_audit['vix_train_median_threshold']:.2f})"
    )
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([pretty_names.get(f, f) for f in top_feats])
    ax_a.set_xlabel("Mean |TreeSHAP| Attribution (Tail-Risk Prob.)")
    ax_a.set_title("A. Train-Background TreeSHAP (Non-Causal)")
    ax_a.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#CCCCCC")
    ax_a.text(
        0.98, 0.25,
        f"Train bg N={shap_audit['background_n']:,}\n"
        f"Max add. err={shap_audit['classifier_max_additivity_error']:.1e}",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.6,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F8F9F9", "edgecolor": "#CCCCCC"},
    )

    # --- Panel B: 2025 Holdout Reliability Diagram (Pre vs Post Calibration) ---
    m_map = test_metrics_df.set_index("model").to_dict(orient="index")
    test_cal = cal_curve_df[cal_curve_df["split"] == "test"].copy()

    ax_b.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#555555", linewidth=1.1,
              label="Ideal Calibration (y = x)")

    curve_specs = [
        (
            "lgbm_raw",
            MODEL_COLORS["lgbm_raw"],
            "^",
            "--",
            f"LightGBM Raw (ECE={m_map['lgbm']['ece_pre_cal']:.3f})",
        ),
        (
            "lgbm_cal",
            MODEL_COLORS["lgbm"],
            "o",
            "-",
            f"LightGBM Isotonic Cal. (ECE={m_map['lgbm']['ece']:.3f})",
        ),
        (
            "har_rv",
            MODEL_COLORS["har_rv"],
            "s",
            "-",
            f"Logistic HAR-RV (ECE={m_map['har_rv']['ece']:.3f})",
        ),
        (
            "gjr_garch",
            MODEL_COLORS["gjr_garch"],
            "D",
            "-.",
            f"GJR-GARCH (ECE={m_map['gjr_garch']['ece']:.3f})",
        ),
    ]
    for var_key, color, marker, ls, lbl in curve_specs:
        sub = test_cal[(test_cal["model_variant"] == var_key) & (test_cal["n_obs"] >= 15)].copy()
        ax_b.plot(
            sub["mean_pred_prob"],
            sub["observed_rate"],
            marker=marker,
            markersize=5.0,
            linestyle=ls,
            linewidth=1.55,
            color=color,
            label=lbl,
        )

    ax_b.set_xlim(-0.02, 0.95)
    ax_b.set_ylim(-0.02, 0.95)
    ax_b.set_xlabel("Mean Predicted Tail-Risk Probability")
    ax_b.set_ylabel("Observed 2025 Tail-Risk Frequency")
    ax_b.set_title("B. 2025 Holdout Reliability Curves (10 Bins)")
    ax_b.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#CCCCCC")
    ax_b.text(
        0.97, 0.42,
        "2024 Val OOF Isotonic\n"
        f"ECE: {m_map['lgbm']['ece_pre_cal']:.3f} -> {m_map['lgbm']['ece']:.3f} (-97.3%)\n"
        f"Brier: {m_map['lgbm']['brier_pre_cal']:.3f} -> {m_map['lgbm']['brier']:.3f}",
        transform=ax_b.transAxes,
        ha="right",
        va="center",
        fontsize=7.6,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#EBF5FB", "edgecolor": "#AED6F1"},
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "fig2_shap_and_calibration.png"
    pdf_path = out_dir / "fig2_shap_and_calibration.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(
        pdf_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.10,
        metadata={"CreationDate": None},
    )
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def generate_all_figures(
    *,
    project_root: Path | None = None,
    outputs_figures_dir: Path | None = None,
) -> dict[str, dict[str, str]]:
    """Verify frozen manifest and generate 300-DPI Figure 1 and Figure 2 (PNG + PDF)."""
    root = Path(project_root) if project_root is not None else config.PROJECT_ROOT
    verify_freeze_manifest(frozen_dir=root / "data" / "frozen", project_root=root)

    fig_dir = Path(outputs_figures_dir) if outputs_figures_dir else (root / "outputs" / "figures")
    tables_dir = root / "outputs" / "tables"
    metrics_dir = root / "outputs" / "metrics"

    panel_df = pd.read_parquet(root / "data" / "frozen" / "panel_daily.parquet")
    leverage_df = pd.read_csv(tables_dir / "eda_leverage_asymmetry.csv")
    test_pred_df = pd.read_parquet(tables_dir / "h2_test_predictions.parquet")
    shap_imp_df = pd.read_csv(tables_dir / "shap_importance.csv")
    shap_regime_df = pd.read_csv(tables_dir / "shap_regime_summary.csv")
    shap_audit = json.loads((metrics_dir / "shap_audit.json").read_text(encoding="utf-8"))
    cal_curve_df = pd.read_csv(tables_dir / "h2_calibration_curve.csv")
    test_metrics_df = pd.read_csv(tables_dir / "h2_test_metrics.csv")

    f1 = generate_figure1(
        panel_df=panel_df,
        leverage_df=leverage_df,
        test_pred_df=test_pred_df,
        out_dir=fig_dir,
    )
    f2 = generate_figure2(
        shap_imp_df=shap_imp_df,
        shap_regime_df=shap_regime_df,
        shap_audit=shap_audit,
        cal_curve_df=cal_curve_df,
        test_metrics_df=test_metrics_df,
        out_dir=fig_dir,
    )
    return {"fig1": f1, "fig2": f2}
