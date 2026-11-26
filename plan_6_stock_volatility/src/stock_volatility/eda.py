"""Milestone M3: Cross-cycle EDA tables, leverage asymmetry, coverage, and VIX x event incidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_volatility import config
from stock_volatility.panel.freeze import verify_freeze_manifest

REGIME_WINDOWS: tuple[tuple[str, str, str, str], ...] = (
    ("pre_covid_2019", "2019 Pre-COVID Expansion", "2019-01-02", "2019-12-31"),
    ("covid_shock_2020", "2020 COVID-19 Shock & Recovery", "2020-01-02", "2020-12-31"),
    ("rebound_2021", "2021 Post-COVID Rebound", "2021-01-02", "2021-12-31"),
    ("rate_hike_bear_2022", "2022 Rate-Hike Bear Market", "2022-01-02", "2022-12-31"),
    ("disinflation_2023", "2023 Disinflation & AI Rally", "2023-01-02", "2023-12-31"),
    ("ai_cycle_val_2024", "2024 AI Expansion (Validation)", "2024-01-02", "2024-12-31"),
    ("ai_cycle_test_2025", "2025 AI & Policy Vol (Test)", "2025-01-02", "2025-12-31"),
    ("full_2019_2025", "2019-2025 Full Sample", "2019-01-02", "2025-12-31"),
)

RET_BUCKETS: tuple[tuple[str, str], ...] = (
    ("ret_lt_0", "ret_1d < 0 (Negative Return)"),
    ("ret_ge_0", "ret_1d >= 0 (Non-Negative Return)"),
    ("ret_le_neg3", "ret_1d <= -3%"),
    ("ret_neg3_to_neg1", "-3% < ret_1d <= -1%"),
    ("ret_neg1_to_0", "-1% < ret_1d < 0%"),
    ("ret_0_to_pos1", "0% <= ret_1d < +1%"),
    ("ret_pos1_to_pos3", "+1% <= ret_1d < +3%"),
    ("ret_ge_pos3", "ret_1d >= +3%"),
)


def _wilson_ci95(k: int, n: int) -> tuple[float, float]:
    """Compute exact Wilson score 95% confidence interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _modeling_slice(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with complete <=t features and complete t+1..t+5 targets."""
    mask = panel_df["has_full_features"].astype(bool) & panel_df["has_full_target"].astype(bool)
    return panel_df.loc[mask].copy()


def _sector_order(df: pd.DataFrame) -> list[str]:
    """Return canonical sector order filtered to sectors present in df."""
    present = set(df["sector"].unique())
    ordered = [s for s in config.SECTORS if s in present]
    extras = sorted(present - set(ordered))
    return ["ALL", *ordered, *extras]


def compute_regimes_by_sector(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize 2019-2025 volatility regimes and tail-risk rates by regime and sector."""
    df = _modeling_slice(panel_df)
    sectors = _sector_order(df)
    records: list[dict[str, Any]] = []

    for regime_id, label, start_d, end_d in REGIME_WINDOWS:
        reg_df = df[(df["date"] >= start_d) & (df["date"] <= end_d)]
        if reg_df.empty:
            continue
        for sec in sectors:
            sub = reg_df if sec == "ALL" else reg_df[reg_df["sector"] == sec]
            if sub.empty:
                continue
            n_obs = int(len(sub))
            y_tail_pos = int(sub["y_tail"].astype(int).sum())
            y_rv90_pos = int(sub["y_tail_rv90"].astype(int).sum())
            ev_days = int(sub["has_news"].astype(int).sum())
            records.append(
                {
                    "regime": regime_id,
                    "regime_label": label,
                    "start_date": str(sub["date"].min()),
                    "end_date": str(sub["date"].max()),
                    "sector": sec,
                    "n_obs": n_obs,
                    "n_days": int(sub["date"].nunique()),
                    "rv_5d_fwd_mean": float(sub["rv_5d_fwd"].mean()),
                    "rv_5d_fwd_median": float(sub["rv_5d_fwd"].median()),
                    "rv_5d_fwd_p90": float(sub["rv_5d_fwd"].quantile(0.90)),
                    "vix_mean": float(sub["vix_close"].mean()),
                    "y_tail_pos": y_tail_pos,
                    "y_tail_rate": float(y_tail_pos / n_obs),
                    "y_tail_rv90_pos": y_rv90_pos,
                    "y_tail_rv90_rate": float(y_rv90_pos / n_obs),
                    "event_day_count": ev_days,
                    "event_day_rate": float(ev_days / n_obs),
                }
            )
    return pd.DataFrame(records)


def _ret_bucket_masks(ret: pd.Series) -> dict[str, pd.Series]:
    """Return boolean masks for binary and 6-bin daily return buckets."""
    return {
        "ret_lt_0": ret < 0.0,
        "ret_ge_0": ret >= 0.0,
        "ret_le_neg3": ret <= -0.03,
        "ret_neg3_to_neg1": (ret > -0.03) & (ret <= -0.01),
        "ret_neg1_to_0": (ret > -0.01) & (ret < 0.0),
        "ret_0_to_pos1": (ret >= 0.0) & (ret < 0.01),
        "ret_pos1_to_pos3": (ret >= 0.01) & (ret < 0.03),
        "ret_ge_pos3": ret >= 0.03,
    }


def compute_leverage_asymmetry(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Quantify forward 5-day RV and tail-risk asymmetry across lagged daily return buckets."""
    df = _modeling_slice(panel_df)
    sectors = _sector_order(df)
    scopes: list[tuple[str, pd.DataFrame]] = [("overall", df)]
    for sp in ("train", "val", "test"):
        part = df[df["split"] == sp]
        if not part.empty:
            scopes.append((sp, part))

    records: list[dict[str, Any]] = []
    for scope_name, scope_df in scopes:
        for sec in sectors:
            sub = scope_df if sec == "ALL" else scope_df[scope_df["sector"] == sec]
            if sub.empty:
                continue
            masks = _ret_bucket_masks(sub["ret_1d"])
            for bucket_id, bucket_label in RET_BUCKETS:
                b_df = sub[masks[bucket_id]]
                if b_df.empty:
                    continue
                n_obs = int(len(b_df))
                y_pos = int(b_df["y_tail"].astype(int).sum())
                records.append(
                    {
                        "scope": scope_name,
                        "sector": sec,
                        "ret_bucket": bucket_id,
                        "ret_bucket_label": bucket_label,
                        "n_obs": n_obs,
                        "mean_ret_1d": float(b_df["ret_1d"].mean()),
                        "mean_rv_5d_fwd": float(b_df["rv_5d_fwd"].mean()),
                        "median_rv_5d_fwd": float(b_df["rv_5d_fwd"].median()),
                        "mean_y_log_rv": float(b_df["y_log_rv"].mean()),
                        "y_tail_pos": y_pos,
                        "y_tail_rate": float(y_pos / n_obs),
                    }
                )
    return pd.DataFrame(records)


def compute_coverage_consistency(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Audit 2019-2023 vs 2024-2025 aligned corporate-event coverage consistency across sectors."""
    df = panel_df.copy()
    sectors = _sector_order(df)
    p1 = df[df["date"] <= "2023-12-31"]
    p2 = df[df["date"] >= "2024-01-01"]

    records: list[dict[str, Any]] = []
    for sec in sectors:
        s1 = p1 if sec == "ALL" else p1[p1["sector"] == sec]
        s2 = p2 if sec == "ALL" else p2[p2["sector"] == sec]
        n1 = int(len(s1))
        n2 = int(len(s2))
        ev_days_1 = int(s1["has_news"].astype(int).sum())
        ev_days_2 = int(s2["has_news"].astype(int).sum())
        tot_ev_1 = int(s1["news_count"].astype(int).sum())
        tot_ev_2 = int(s2["news_count"].astype(int).sum())
        earn_1 = int(s1["has_8k_earnings"].astype(int).sum())
        earn_2 = int(s2["has_8k_earnings"].astype(int).sum())
        n_tk_1 = max(1, int(s1["ticker"].nunique()))
        n_tk_2 = max(1, int(s2["ticker"].nunique()))
        rate_1 = ev_days_1 / n1 if n1 else 0.0
        rate_2 = ev_days_2 / n2 if n2 else 0.0
        per_yr_1 = tot_ev_1 / (5.0 * n_tk_1)
        per_yr_2 = tot_ev_2 / (2.0 * n_tk_2)
        active_1 = int(s1.loc[s1["has_news"] == 1, "ticker"].nunique())
        active_2 = int(s2.loc[s2["has_news"] == 1, "ticker"].nunique())
        records.append(
            {
                "sector": sec,
                "n_stock_days_1923": n1,
                "event_days_1923": ev_days_1,
                "event_day_rate_1923": float(rate_1),
                "total_events_1923": tot_ev_1,
                "events_per_stock_year_1923": float(per_yr_1),
                "earnings_days_1923": earn_1,
                "active_tickers_1923": active_1,
                "n_stock_days_2425": n2,
                "event_days_2425": ev_days_2,
                "event_day_rate_2425": float(rate_2),
                "total_events_2425": tot_ev_2,
                "events_per_stock_year_2425": float(per_yr_2),
                "earnings_days_2425": earn_2,
                "active_tickers_2425": active_2,
                "event_day_rate_ratio_2425_vs_1923": float(rate_2 / rate_1) if rate_1 > 0 else 0.0,
                "events_per_year_ratio_2425_vs_1923": (
                    float(per_yr_2 / per_yr_1) if per_yr_1 > 0 else 0.0
                ),
            }
        )
    return pd.DataFrame(records)


def compute_vix_news_tail_incidence(
    panel_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """Compare tail-risk incidence across Train-median High/Low VIX and event-day status."""
    df = _modeling_slice(panel_df)
    train_df = df[df["split"] == "train"]
    if train_df.empty:
        raise ValueError("Cannot compute Train-only VIX median threshold on empty Train slice")

    vix_train_daily = train_df.groupby("date")["vix_close"].first()
    vix_threshold = float(vix_train_daily.median())

    df["vix_regime"] = np.where(df["vix_close"] >= vix_threshold, "High_VIX", "Low_VIX")
    scopes: list[tuple[str, pd.DataFrame]] = [("overall", df)]
    for sp in ("train", "val", "test"):
        part = df[df["split"] == sp]
        if not part.empty:
            scopes.append((sp, part))

    records: list[dict[str, Any]] = []
    for scope_name, scope_df in scopes:
        for vix_reg in ("Low_VIX", "High_VIX"):
            v_df = scope_df[scope_df["vix_regime"] == vix_reg]
            if v_df.empty:
                continue
            status_masks = [
                ("No_Event", v_df["has_news"] == 0),
                ("Has_Event", v_df["has_news"] == 1),
                ("Has_8K_Earnings", v_df["has_8k_earnings"] == 1),
            ]
            for status_name, mask in status_masks:
                cell = v_df[mask]
                if cell.empty:
                    continue
                n_obs = int(len(cell))
                y_tail_pos = int(cell["y_tail"].astype(int).sum())
                y_rv90_pos = int(cell["y_tail_rv90"].astype(int).sum())
                ci_lo, ci_hi = _wilson_ci95(y_tail_pos, n_obs)
                records.append(
                    {
                        "scope": scope_name,
                        "vix_regime": vix_reg,
                        "vix_threshold": vix_threshold,
                        "event_status": status_name,
                        "n_obs": n_obs,
                        "y_tail_pos": y_tail_pos,
                        "y_tail_rate": float(y_tail_pos / n_obs),
                        "ci95_low": float(ci_lo),
                        "ci95_high": float(ci_hi),
                        "y_tail_rv90_pos": y_rv90_pos,
                        "y_tail_rv90_rate": float(y_rv90_pos / n_obs),
                        "rv_5d_fwd_mean": float(cell["rv_5d_fwd"].mean()),
                        "rv_5d_fwd_median": float(cell["rv_5d_fwd"].median()),
                    }
                )
    return pd.DataFrame(records), vix_threshold


def render_eda_summary_markdown(
    *,
    generated_at_utc: str,
    summary_payload: dict[str, Any],
    regimes_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    vix_news_df: pd.DataFrame,
) -> str:
    """Render descriptive, strictly non-causal M3 EDA summary Markdown."""
    lines: list[str] = [
        "# Milestone M3 — Cross-Cycle Exploratory Data Analysis (2019–2025)",
        "",
        f"- **Provenance Timestamp (UTC)**: `{generated_at_utc}`",
        f"- **Modeling-Complete Sample**: `{summary_payload['modeling_total_rows']}` stock-days "
        f"(`{summary_payload['modeling_total_days']}` trading days × `60` stocks)",
        f"- **Train-Only High/Low VIX Threshold**: "
        f"`VIX >= {summary_payload['vix_train_median_threshold']:.4f}` "
        "(2019–2023 Train daily median, zero lookahead)",
        "- **Scope Note**: All summaries below are descriptive and associational only; "
        "no causal claims are implied.",
        "",
        "---",
        "",
        "## 1. Volatility Regimes Across Market Cycles (`ALL` 60 Stocks)",
        "",
        "| Regime | Date Range | Days | Obs (`N`) | Mean `rv_5d_fwd` | Median `rv_5d_fwd` | "
        "Mean `VIX` | `y_tail` Pos / `N` (%) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    all_reg = regimes_df[regimes_df["sector"] == "ALL"]
    for row in all_reg.to_dict(orient="records"):
        lines.append(
            f"| `{row['regime']}` | `{row['start_date']}`..`{row['end_date']}` | "
            f"`{row['n_days']}` | `{row['n_obs']}` | `{row['rv_5d_fwd_mean']:.4f}` | "
            f"`{row['rv_5d_fwd_median']:.4f}` | `{row['vix_mean']:.2f}` | "
            f"`{row['y_tail_pos']}/{row['n_obs']}` (`{row['y_tail_rate']*100:.2f}%`) |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Leverage Asymmetry Summary (`ret_1d < 0` vs `ret_1d >= 0`)",
            "",
            f"- **Overall Mean `rv_5d_fwd` (`ret_1d < 0` vs `ret_1d >= 0`)**: "
            f"`{summary_payload['leverage_overall']['mean_rv_neg']:.4f}` vs "
            f"`{summary_payload['leverage_overall']['mean_rv_pos']:.4f}` "
            f"(ratio `{summary_payload['leverage_overall']['ratio_neg_over_pos']:.3f}`)",
            f"- **Extreme Move Asymmetry (`ret_1d <= -3%` vs `ret_1d >= +3%`)**: "
            f"`{summary_payload['leverage_overall']['mean_rv_le_neg3']:.4f}` vs "
            f"`{summary_payload['leverage_overall']['mean_rv_ge_pos3']:.4f}` "
            f"(ratio `{summary_payload['leverage_overall']['ratio_le_neg3_over_ge_pos3']:.3f}`)",
            "",
            "---",
            "",
            "## 3. Cross-Period Corporate-Event Coverage Consistency (2019–2023 vs 2024–2025)",
            "",
            "| Sector | 19–23 Event Days / `N` (%) | 24–25 Event Days / `N` (%) | "
            "Rate Ratio | Events/Stock-Yr (19–23 / 24–25) | Active Tickers (19–23 / 24–25) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in coverage_df.to_dict(orient="records"):
        lines.append(
            f"| `{row['sector']}` | `{row['event_days_1923']}/{row['n_stock_days_1923']}` "
            f"(`{row['event_day_rate_1923']*100:.2f}%`) | "
            f"`{row['event_days_2425']}/{row['n_stock_days_2425']}` "
            f"(`{row['event_day_rate_2425']*100:.2f}%`) | "
            f"`{row['event_day_rate_ratio_2425_vs_1923']:.4f}` | "
            f"`{row['events_per_stock_year_1923']:.2f}` / "
            f"`{row['events_per_stock_year_2425']:.2f}` | "
            f"`{row['active_tickers_1923']}` / `{row['active_tickers_2425']}` |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Tail-Risk Incidence by VIX Regime × Event Status (`overall` 2019–2025)",
            "",
            "| VIX Regime | Event Status | `y_tail` Pos / `N` | Incidence Rate (Wilson 95% CI) | "
            "Mean `rv_5d_fwd` | `y_tail_rv90` Pos / `N` (%) |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    ov_vn = vix_news_df[vix_news_df["scope"] == "overall"]
    for row in ov_vn.to_dict(orient="records"):
        lines.append(
            f"| `{row['vix_regime']}` | `{row['event_status']}` | "
            f"`{row['y_tail_pos']}/{row['n_obs']}` | "
            f"`{row['y_tail_rate']*100:.2f}%` "
            f"[`{row['ci95_low']*100:.2f}%`, `{row['ci95_high']*100:.2f}%`] | "
            f"`{row['rv_5d_fwd_mean']:.4f}` | "
            f"`{row['y_tail_rv90_pos']}/{row['n_obs']}` (`{row['y_tail_rv90_rate']*100:.2f}%`) |"
        )
    lines.append("")
    return "\n".join(lines)


def run_eda_pipeline(
    *,
    project_root: Path | None = None,
    outputs_tables_dir: Path | None = None,
    outputs_metrics_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify frozen manifest and deterministically generate all M3 EDA outputs."""
    root = Path(project_root) if project_root is not None else config.PROJECT_ROOT
    frozen_dir = root / "data" / "frozen"
    manifest = verify_freeze_manifest(frozen_dir=frozen_dir, project_root=root)
    generated_at_utc = str(manifest["generated_at_utc"])

    tables_dir = Path(outputs_tables_dir) if outputs_tables_dir else (root / "outputs" / "tables")
    metrics_dir = (
        Path(outputs_metrics_dir) if outputs_metrics_dir else (root / "outputs" / "metrics")
    )
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    panel_df = pd.read_parquet(frozen_dir / "panel_daily.parquet")
    modeling_df = _modeling_slice(panel_df)

    regimes_df = compute_regimes_by_sector(panel_df)
    leverage_df = compute_leverage_asymmetry(panel_df)
    coverage_df = compute_coverage_consistency(panel_df)
    vix_news_df, vix_thresh = compute_vix_news_tail_incidence(panel_df)

    # Extract concise summary indicators for JSON & report rendering
    lev_ov = leverage_df[(leverage_df["scope"] == "overall") & (leverage_df["sector"] == "ALL")]
    lev_map = lev_ov.set_index("ret_bucket")["mean_rv_5d_fwd"].to_dict()
    mean_rv_neg = float(lev_map.get("ret_lt_0", float("nan")))
    mean_rv_pos = float(lev_map.get("ret_ge_0", float("nan")))
    mean_rv_le_neg3 = float(lev_map.get("ret_le_neg3", float("nan")))
    mean_rv_ge_pos3 = float(lev_map.get("ret_ge_pos3", float("nan")))

    all_cov = coverage_df[coverage_df["sector"] == "ALL"].iloc[0].to_dict()
    reg_all_map = (
        regimes_df[regimes_df["sector"] == "ALL"]
        .set_index("regime")[["rv_5d_fwd_mean", "rv_5d_fwd_median", "vix_mean", "y_tail_rate"]]
        .to_dict(orient="index")
    )

    summary_payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "milestone": "M3",
        "generated_at_utc": generated_at_utc,
        "modeling_total_rows": int(len(modeling_df)),
        "modeling_total_days": int(modeling_df["date"].nunique()),
        "vix_train_median_threshold": vix_thresh,
        "regimes_all": reg_all_map,
        "leverage_overall": {
            "mean_rv_neg": mean_rv_neg,
            "mean_rv_pos": mean_rv_pos,
            "ratio_neg_over_pos": (
                float(mean_rv_neg / mean_rv_pos) if mean_rv_pos > 0.0 else float("nan")
            ),
            "mean_rv_le_neg3": mean_rv_le_neg3,
            "mean_rv_ge_pos3": mean_rv_ge_pos3,
            "ratio_le_neg3_over_ge_pos3": (
                float(mean_rv_le_neg3 / mean_rv_ge_pos3)
                if mean_rv_ge_pos3 > 0.0
                else float("nan")
            ),
        },
        "coverage_all": {
            k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
            for k, v in all_cov.items()
            if k != "sector"
        },
        "vix_news_overall": (
            vix_news_df[vix_news_df["scope"] == "overall"].to_dict(orient="records")
        ),
    }

    regimes_df.to_csv(tables_dir / "eda_regimes_by_sector.csv", index=False, lineterminator="\n")
    leverage_df.to_csv(tables_dir / "eda_leverage_asymmetry.csv", index=False, lineterminator="\n")
    coverage_df.to_csv(
        tables_dir / "eda_coverage_consistency.csv", index=False, lineterminator="\n"
    )
    vix_news_df.to_csv(
        tables_dir / "eda_vix_news_tail_incidence.csv", index=False, lineterminator="\n"
    )

    md_text = render_eda_summary_markdown(
        generated_at_utc=generated_at_utc,
        summary_payload=summary_payload,
        regimes_df=regimes_df,
        coverage_df=coverage_df,
        vix_news_df=vix_news_df,
    )
    (tables_dir / "eda_summary.md").write_text(md_text, encoding="utf-8")
    (metrics_dir / "eda_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary_payload
