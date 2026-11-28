"""Milestone M3: H1 Stock-FE Extended Panel HAR-RV with Driscoll-Kraay SE, Holm & Sensitivities."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from stock_volatility import config
from stock_volatility.panel.freeze import verify_freeze_manifest

REFERENCE_SECTOR: str = "Healthcare"
NON_REF_SECTORS: tuple[str, ...] = ("Technology", "Financials", "Consumer", "Energy")
CORE_HAR_COLS: tuple[str, ...] = ("log_rv_d", "log_rv_w", "log_rv_m", "neg_ret_1d", "vix_close")


def driscoll_kraay_bandwidth(n_dates: int) -> int:
    """Compute pre-registered Bartlett bandwidth L = max(5, floor(4 * (T / 100)^(2/9)))."""
    if n_dates < 2:
        raise ValueError("n_dates must be >= 2")
    return max(5, int(math.floor(4.0 * ((n_dates / 100.0) ** (2.0 / 9.0)))))


def holm_bonferroni_adjust(p_values: Sequence[float]) -> list[float]:
    """Compute step-down Holm-Bonferroni adjusted p-values preserving original ordering."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(float(p) for p in p_values), key=lambda pair: pair[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        scaled = min(1.0, p_val * (m - rank))
        running_max = max(running_max, scaled)
        adjusted[orig_idx] = running_max
    return adjusted


def _within_demean(
    df: pd.DataFrame,
    *,
    entity_col: str,
    y_col: str,
    x_cols: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply exact Frisch-Waugh-Lovell within-entity demeaning for stock fixed effects."""
    y_dm = (df[y_col] - df.groupby(entity_col)[y_col].transform("mean")).to_numpy(dtype=float)
    x_mat = np.column_stack(
        [
            (df[col] - df.groupby(entity_col)[col].transform("mean")).to_numpy(dtype=float)
            for col in x_cols
        ]
    )
    return y_dm, x_mat


def fit_stock_fe_driscoll_kraay(
    df: pd.DataFrame,
    *,
    y_col: str,
    x_cols: Sequence[str],
    entity_col: str = "ticker",
    time_col: str = "date",
    maxlags: int | None = None,
    cov_mode: str = "driscoll_kraay",
) -> dict[str, Any]:
    """Fit Stock-FE linear panel model with Bartlett Driscoll-Kraay or entity-clustered NW SE."""
    work = df.sort_values([time_col, entity_col], kind="mergesort").reset_index(drop=True)
    y_dm, X_dm = _within_demean(work, entity_col=entity_col, y_col=y_col, x_cols=x_cols)

    n_obs, k_params = X_dm.shape
    n_entities = int(work[entity_col].nunique())
    if n_obs <= k_params + n_entities:
        raise ValueError(
            f"Insufficient degrees of freedom: n_obs={n_obs} <= k_params+n_entities="
            f"{k_params + n_entities}"
        )
    if cov_mode == "newey_west_entity" and n_entities < 2:
        raise ValueError("newey_west_entity requires at least 2 entities")
    dates = sorted(work[time_col].unique().tolist())
    n_dates = len(dates)
    lag_l = int(maxlags) if maxlags is not None else driscoll_kraay_bandwidth(n_dates)

    XtX_inv = np.linalg.inv(X_dm.T @ X_dm)
    beta = XtX_inv @ (X_dm.T @ y_dm)
    resid = y_dm - X_dm @ beta

    tss_within = float(np.sum(y_dm**2))
    rss = float(np.sum(resid**2))
    within_r2 = 1.0 - (rss / tss_within) if tss_within > 0.0 else float("nan")

    date_codes = pd.Categorical(work[time_col], categories=dates, ordered=True).codes
    scores = X_dm * resid[:, None]

    if cov_mode == "driscoll_kraay":
        H = np.zeros((n_dates, k_params), dtype=float)
        np.add.at(H, date_codes, scores)
        S = H.T @ H
        for lag in range(1, lag_l + 1):
            weight = 1.0 - lag / (lag_l + 1.0)
            gamma = H[lag:].T @ H[:-lag]
            S += weight * (gamma + gamma.T)
        dof_scale = (n_obs / (n_obs - k_params - n_entities)) * (n_dates / (n_dates - 1.0))
        cov_mat = dof_scale * (XtX_inv @ S @ XtX_inv)
        df_t = n_dates - 1
    elif cov_mode == "newey_west_entity":
        # Per-entity Bartlett Newey-West HAC summed across independent entities (no cross-stock
        # contemporaneous covariance) with entity-clustered small-sample adjustment
        S = np.zeros((k_params, k_params), dtype=float)
        for _, grp_idx in work.groupby(entity_col, sort=True).indices.items():
            idx_sorted = grp_idx[np.argsort(date_codes[grp_idx])]
            h_i = scores[idx_sorted]
            S += h_i.T @ h_i
            t_i = len(idx_sorted)
            eff_l = min(lag_l, t_i - 1)
            for lag in range(1, eff_l + 1):
                weight = 1.0 - lag / (lag_l + 1.0)
                gamma = h_i[lag:].T @ h_i[:-lag]
                S += weight * (gamma + gamma.T)
        dof_scale = (n_obs / (n_obs - k_params - n_entities)) * (
            n_entities / max(1.0, n_entities - 1.0)
        )
        cov_mat = dof_scale * (XtX_inv @ S @ XtX_inv)
        df_t = n_entities - 1
    else:
        raise ValueError(f"Unsupported cov_mode: {cov_mode!r}")

    se = np.sqrt(np.maximum(np.diag(cov_mat), 0.0))
    t_stat = np.where(se > 0.0, beta / np.where(se > 0.0, se, 1.0), np.nan)
    p_val = 2.0 * stats.t.sf(np.abs(t_stat), df=df_t)
    t_crit = float(stats.t.ppf(0.975, df=df_t))
    ci_lo = beta - t_crit * se
    ci_hi = beta + t_crit * se

    return {
        "x_cols": list(x_cols),
        "beta": beta,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_val,
        "ci95_low": ci_lo,
        "ci95_high": ci_hi,
        "cov": cov_mat,
        "n_obs": n_obs,
        "n_entities": n_entities,
        "n_dates": n_dates,
        "k_params": k_params,
        "df_t": df_t,
        "bartlett_bandwidth_L": lag_l,
        "cov_mode": cov_mode,
        "within_r2": float(within_r2),
        "rmse_within": float(math.sqrt(rss / n_obs)),
    }


def _add_sector_interaction_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Construct Healthcare-reference sector x (neg_ret_1d, vix_close) interaction regressors."""
    out = df.copy()
    x_cols = list(CORE_HAR_COLS)
    for sec in NON_REF_SECTORS:
        ind = (out["sector"] == sec).astype(float)
        c_neg = f"{sec.lower()}_x_neg_ret_1d"
        c_vix = f"{sec.lower()}_x_vix_close"
        out[c_neg] = ind * out["neg_ret_1d"]
        out[c_vix] = ind * out["vix_close"]
        x_cols.extend([c_neg, c_vix])
    return out, x_cols


def _linear_contrast(
    fit_res: dict[str, Any],
    weights: dict[str, float],
) -> dict[str, float]:
    """Evaluate c'beta, SE, t-stat, two-sided p-value, and 95% CI from covariance matrix."""
    x_cols: list[str] = fit_res["x_cols"]
    c = np.zeros(len(x_cols), dtype=float)
    for col, w in weights.items():
        c[x_cols.index(col)] = float(w)
    est = float(c @ fit_res["beta"])
    var = float(c @ fit_res["cov"] @ c)
    se = math.sqrt(max(var, 0.0))
    t_stat = est / se if se > 0.0 else float("nan")
    df_t = int(fit_res["df_t"])
    p_val = float(2.0 * stats.t.sf(abs(t_stat), df=df_t))
    t_crit = float(stats.t.ppf(0.975, df=df_t))
    return {
        "estimate": est,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_val,
        "ci95_low": est - t_crit * se,
        "ci95_high": est + t_crit * se,
    }


def _joint_wald_test(
    fit_res: dict[str, Any],
    restricted_cols: Sequence[str],
) -> dict[str, float]:
    """Compute exact joint Wald chi2 and F statistics for H0: R*beta = 0."""
    x_cols: list[str] = fit_res["x_cols"]
    q = len(restricted_cols)
    R = np.zeros((q, len(x_cols)), dtype=float)
    for row_idx, col in enumerate(restricted_cols):
        R[row_idx, x_cols.index(col)] = 1.0
    r_beta = R @ fit_res["beta"]
    r_cov = R @ fit_res["cov"] @ R.T
    wald_chi2 = float(r_beta.T @ np.linalg.solve(r_cov, r_beta))
    f_stat = wald_chi2 / q
    df_denom = int(fit_res["df_t"])
    p_chi2 = float(stats.chi2.sf(wald_chi2, df=q))
    p_f = float(stats.f.sf(f_stat, dfn=q, dfd=df_denom))
    return {
        "wald_chi2": wald_chi2,
        "f_stat": f_stat,
        "df_num": float(q),
        "df_denom": float(df_denom),
        "p_value_chi2": p_chi2,
        "p_value_f": p_f,
    }


def _evaluate_confirmatory_family(fit_res: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate all 5 pre-registered H1 confirmatory tests with Holm-Bonferroni adjustment."""
    # Equal-weight (1/5) 5-sector average marginal effect for neg_ret_1d & vix_close
    ame_neg_weights = {"neg_ret_1d": 1.0}
    ame_vix_weights = {"vix_close": 1.0}
    for sec in NON_REF_SECTORS:
        ame_neg_weights[f"{sec.lower()}_x_neg_ret_1d"] = 0.2
        ame_vix_weights[f"{sec.lower()}_x_vix_close"] = 0.2

    c_neg = _linear_contrast(fit_res, ame_neg_weights)
    c_vix = _linear_contrast(fit_res, ame_vix_weights)
    c_tech_neg = _linear_contrast(fit_res, {"technology_x_neg_ret_1d": 1.0})
    c_tech_vix = _linear_contrast(fit_res, {"technology_x_vix_close": 1.0})

    interaction_cols = [
        f"{sec.lower()}_x_{var}"
        for sec in NON_REF_SECTORS
        for var in ("neg_ret_1d", "vix_close")
    ]
    wald = _joint_wald_test(fit_res, interaction_cols)

    df_t = int(fit_res["df_t"])
    t_bonf = float(stats.t.ppf(1.0 - 0.05 / (2.0 * 5.0), df=df_t))

    entries: list[dict[str, Any]] = [
        {
            "test_id": "neg_ret_1d",
            "description": (
                "Overall 5-sector average leverage association (slope on min(ret_1d, 0))"
            ),
            "statistic_type": "t",
            **c_neg,
            "ci95_fwer_low": c_neg["estimate"] - t_bonf * c_neg["se"],
            "ci95_fwer_high": c_neg["estimate"] + t_bonf * c_neg["se"],
        },
        {
            "test_id": "vix_close",
            "description": "Overall 5-sector average VIX association (slope per +1 VIX point)",
            "statistic_type": "t",
            **c_vix,
            "ci95_fwer_low": c_vix["estimate"] - t_bonf * c_vix["se"],
            "ci95_fwer_high": c_vix["estimate"] + t_bonf * c_vix["se"],
        },
        {
            "test_id": "tech_x_neg_ret_1d_vs_healthcare",
            "description": "Technology vs Healthcare differential leverage slope",
            "statistic_type": "t",
            **c_tech_neg,
            "ci95_fwer_low": c_tech_neg["estimate"] - t_bonf * c_tech_neg["se"],
            "ci95_fwer_high": c_tech_neg["estimate"] + t_bonf * c_tech_neg["se"],
        },
        {
            "test_id": "tech_x_vix_close_vs_healthcare",
            "description": "Technology vs Healthcare differential VIX slope",
            "statistic_type": "t",
            **c_tech_vix,
            "ci95_fwer_low": c_tech_vix["estimate"] - t_bonf * c_tech_vix["se"],
            "ci95_fwer_high": c_tech_vix["estimate"] + t_bonf * c_tech_vix["se"],
        },
        {
            "test_id": "joint_5sector_interaction_wald",
            "description": "Joint Wald test of all 8 sector x (neg_ret_1d, vix_close) interactions",
            "statistic_type": "wald_F",
            "estimate": wald["wald_chi2"],
            "se": float("nan"),
            "t_stat": wald["f_stat"],
            "p_value": wald["p_value_f"],
            "p_value_chi2": wald["p_value_chi2"],
            "df_num": int(wald["df_num"]),
            "df_denom": int(wald["df_denom"]),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "ci95_fwer_low": float("nan"),
            "ci95_fwer_high": float("nan"),
        },
    ]

    p_holm = holm_bonferroni_adjust([e["p_value"] for e in entries])
    for entry, ph in zip(entries, p_holm, strict=True):
        entry["p_holm"] = float(ph)
        entry["significant_holm_05"] = bool(ph < 0.05)
    return entries


def _build_main_table_df(
    fit_res: dict[str, Any],
    confirmatory_family: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build main H1 coefficient table including regressors and 5 sector-specific total slopes."""
    conf_by_id = {row["test_id"]: row for row in confirmatory_family}
    rows: list[dict[str, Any]] = []
    for idx, term in enumerate(fit_res["x_cols"]):
        p_holm = float("nan")
        if term == "technology_x_neg_ret_1d":
            p_holm = conf_by_id["tech_x_neg_ret_1d_vs_healthcare"]["p_holm"]
        elif term == "technology_x_vix_close":
            p_holm = conf_by_id["tech_x_vix_close_vs_healthcare"]["p_holm"]
        rows.append(
            {
                "row_type": "regressor",
                "term": term,
                "coef": float(fit_res["beta"][idx]),
                "se_dk": float(fit_res["se"][idx]),
                "t_stat": float(fit_res["t_stat"][idx]),
                "p_value": float(fit_res["p_value"][idx]),
                "p_holm_confirmatory": p_holm,
                "ci95_low": float(fit_res["ci95_low"][idx]),
                "ci95_high": float(fit_res["ci95_high"][idx]),
            }
        )

    # Append equal-weight 5-sector average marginal effects (confirmatory H1 terms)
    for cid in ("neg_ret_1d", "vix_close"):
        c = conf_by_id[cid]
        rows.append(
            {
                "row_type": "confirmatory_ame",
                "term": f"ame_5sector_{cid}",
                "coef": float(c["estimate"]),
                "se_dk": float(c["se"]),
                "t_stat": float(c["t_stat"]),
                "p_value": float(c["p_value"]),
                "p_holm_confirmatory": float(c["p_holm"]),
                "ci95_low": float(c["ci95_low"]),
                "ci95_high": float(c["ci95_high"]),
            }
        )

    # Append total sector-specific slopes for all 5 sectors
    for sec in config.SECTORS:
        for base_var in ("neg_ret_1d", "vix_close"):
            weights = {base_var: 1.0}
            if sec != REFERENCE_SECTOR:
                weights[f"{sec.lower()}_x_{base_var}"] = 1.0
            c = _linear_contrast(fit_res, weights)
            rows.append(
                {
                    "row_type": "sector_total_slope",
                    "term": f"total_slope_{sec.lower()}_{base_var}",
                    "coef": float(c["estimate"]),
                    "se_dk": float(c["se"]),
                    "t_stat": float(c["t_stat"]),
                    "p_value": float(c["p_value"]),
                    "p_holm_confirmatory": float("nan"),
                    "ci95_low": float(c["ci95_low"]),
                    "ci95_high": float(c["ci95_high"]),
                }
            )
    return pd.DataFrame(rows)


def _summarize_sensitivity(
    spec_name: str,
    spec_label: str,
    fit_res: dict[str, Any],
    *,
    has_interactions: bool = True,
) -> list[dict[str, Any]]:
    """Extract compact comparison rows across key H1 terms for a sensitivity specification."""
    if has_interactions:
        conf = _evaluate_confirmatory_family(fit_res)
        conf_map = {r["test_id"]: r for r in conf}
        key_items = [
            ("ame_neg_ret_1d", conf_map["neg_ret_1d"]),
            ("ame_vix_close", conf_map["vix_close"]),
            ("tech_x_neg_ret_1d_vs_healthcare", conf_map["tech_x_neg_ret_1d_vs_healthcare"]),
            ("tech_x_vix_close_vs_healthcare", conf_map["tech_x_vix_close_vs_healthcare"]),
            ("joint_5sector_interaction_wald", conf_map["joint_5sector_interaction_wald"]),
        ]
    else:
        c_neg = _linear_contrast(fit_res, {"neg_ret_1d": 1.0})
        c_vix = _linear_contrast(fit_res, {"vix_close": 1.0})
        key_items = [
            ("pooled_neg_ret_1d", {**c_neg, "p_holm": float("nan")}),
            ("pooled_vix_close", {**c_vix, "p_holm": float("nan")}),
        ]

    rows: list[dict[str, Any]] = []
    for term_name, item in key_items:
        rows.append(
            {
                "specification": spec_name,
                "specification_label": spec_label,
                "cov_mode": fit_res["cov_mode"],
                "n_obs": fit_res["n_obs"],
                "n_dates": fit_res["n_dates"],
                "bartlett_bandwidth_L": fit_res["bartlett_bandwidth_L"],
                "within_r2": fit_res["within_r2"],
                "term": term_name,
                "estimate": float(item["estimate"]),
                "se": float(item["se"]),
                "stat": float(item["t_stat"]),
                "p_value": float(item["p_value"]),
                "p_holm": float(item["p_holm"]),
                "ci95_low": float(item["ci95_low"]),
                "ci95_high": float(item["ci95_high"]),
            }
        )
    return rows


def render_h1_summary_markdown(
    *,
    generated_at_utc: str,
    payload: dict[str, Any],
    main_df: pd.DataFrame,
    sens_df: pd.DataFrame,
    date_range: tuple[str, str] = ("2019-02-01", "2025-12-23"),
) -> str:
    """Render associational H1 Panel HAR-RV report Markdown (strictly non-causal wording)."""
    m = payload["main_model"]
    d_min, d_max = date_range
    lines: list[str] = [
        "# Milestone M3 — H1 Stock-FE Extended Panel HAR-RV Regression",
        "",
        f"- **Provenance Timestamp (UTC)**: `{generated_at_utc}`",
        f"- **Sample**: `{m['n_obs']}` stock-days (`{m['n_entities']}` stocks × `{m['n_dates']}` "
        f"consecutive NYSE trading days, `{d_min}`..`{d_max}`)",
        f"- **Estimator**: Within-entity Stock-FE OLS with Bartlett Driscoll–Kraay SE "
        f"(bandwidth `L = {m['bartlett_bandwidth_L']}`, `df = {m['df_t']}`); "
        f"Within $R^2 = {m['within_r2']:.4f}$",
        f"- **Reference Sector**: `{REFERENCE_SECTOR}` (defensive benchmark)",
        "- **Sign Convention**: `neg_ret_1d = min(ret_1d, 0) <= 0`; a **negative** coefficient on "
        "`neg_ret_1d` indicates that larger negative returns are associated with **higher** "
        "forward 5-day log realized volatility (leverage association).",
        "- **Scope Boundary**: Associational econometric evidence only; no causal claims.",
        "",
        "---",
        "",
        "## 1. Pre-Registered H1 Confirmatory Family (Holm–Bonferroni Adjusted)",
        "",
        "| Test ID | Estimate / Stat | DK SE | `t` / `F` | Raw `p` | Holm `p` | 95% CI | "
        "Holm Sig (`α=0.05`) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload["confirmatory_family_holm"]:
        se_val = row["se"]
        ci_lo_val = row["ci95_low"]
        se_str = "—" if (se_val is None or math.isnan(se_val)) else f"`{se_val:.5f}`"
        ci_str = (
            "—"
            if (ci_lo_val is None or math.isnan(ci_lo_val))
            else f"[`{ci_lo_val:+.4f}`, `{row['ci95_high']:+.4f}`]"
        )
        lines.append(
            f"| `{row['test_id']}` | `{row['estimate']:+.5f}` | {se_str} | "
            f"`{row['t_stat']:+.2f}` | `{row['p_value']:.3e}` | `{row['p_holm']:.3e}` | "
            f"{ci_str} | `{'YES' if row['significant_holm_05'] else 'NO'}` |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Main Stock-FE Extended Panel HAR-RV Coefficients (Driscoll–Kraay SE)",
            "",
            "| Term | Coef | DK SE | `t` | `p` | 95% CI |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    reg_rows = main_df[main_df["row_type"] == "regressor"]
    for row in reg_rows.to_dict(orient="records"):
        lines.append(
            f"| `{row['term']}` | `{row['coef']:+.5f}` | `{row['se_dk']:.5f}` | "
            f"`{row['t_stat']:+.2f}` | `{row['p_value']:.3e}` | "
            f"[`{row['ci95_low']:+.5f}`, `{row['ci95_high']:+.5f}`] |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Pre-Specified Sensitivity Checks (Zero Pretest Switching)",
            "",
            "| Specification | `N` | Within $R^2$ | Term | Estimate | SE | `p` | Holm `p` |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sens_df.to_dict(orient="records"):
        se_s = "—" if pd.isna(row["se"]) else f"`{row['se']:.5f}`"
        ph_s = "—" if pd.isna(row["p_holm"]) else f"`{row['p_holm']:.3e}`"
        lines.append(
            f"| `{row['specification']}` | `{row['n_obs']}` | `{row['within_r2']:.4f}` | "
            f"`{row['term']}` | `{row['estimate']:+.5f}` | {se_s} | "
            f"`{row['p_value']:.3e}` | {ph_s} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_h1_regression_pipeline(
    *,
    project_root: Path | None = None,
    outputs_tables_dir: Path | None = None,
    outputs_metrics_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify frozen manifest and execute H1 Stock-FE Extended Panel HAR-RV + sensitivities."""
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
    full_df = panel_df[
        panel_df["has_full_features"].astype(bool) & panel_df["has_full_target"].astype(bool)
    ].copy()
    full_df, inter_cols = _add_sector_interaction_columns(full_df)
    full_df["y_log_rv_cc"] = np.log(full_df["rv_5d_cc_fwd"].to_numpy(dtype=float) + config.RV_EPS)

    # 1. Main H1 model (full 2019-2025 sample, Parkinson log-RV, Driscoll-Kraay SE)
    main_fit = fit_stock_fe_driscoll_kraay(
        full_df,
        y_col="y_log_rv",
        x_cols=inter_cols,
        cov_mode="driscoll_kraay",
    )
    conf_family = _evaluate_confirmatory_family(main_fit)
    main_df = _build_main_table_df(main_fit, conf_family)

    # 2. Pre-registered sensitivities (no pretest switching — all reported side-by-side)
    train_df = full_df[full_df["split"] == "train"].copy()
    sens_train = fit_stock_fe_driscoll_kraay(
        train_df, y_col="y_log_rv", x_cols=inter_cols, cov_mode="driscoll_kraay"
    )
    sens_cc = fit_stock_fe_driscoll_kraay(
        full_df, y_col="y_log_rv_cc", x_cols=inter_cols, cov_mode="driscoll_kraay"
    )
    sens_nw_ent = fit_stock_fe_driscoll_kraay(
        full_df,
        y_col="y_log_rv",
        x_cols=inter_cols,
        maxlags=main_fit["bartlett_bandwidth_L"],
        cov_mode="newey_west_entity",
    )
    sens_pooled = fit_stock_fe_driscoll_kraay(
        full_df, y_col="y_log_rv", x_cols=CORE_HAR_COLS, cov_mode="driscoll_kraay"
    )
    sens_controls = fit_stock_fe_driscoll_kraay(
        full_df,
        y_col="y_log_rv",
        x_cols=[*inter_cols, "t10y2y", "spy_ret_1d", "has_news", "has_8k_earnings"],
        cov_mode="driscoll_kraay",
    )

    sens_rows: list[dict[str, Any]] = []
    sens_rows.extend(
        _summarize_sensitivity(
            "train_only_2019_2023",
            "Pre-registered sensitivity 1: 2019-2023 Train-only (DK SE)",
            sens_train,
        )
    )
    sens_rows.extend(
        _summarize_sensitivity(
            "close_to_close_rv_5d_cc_fwd",
            "Pre-registered sensitivity 2: Close-to-close log RV target (DK SE)",
            sens_cc,
        )
    )
    sens_rows.extend(
        _summarize_sensitivity(
            "newey_west_entity_clustered_se",
            "Pre-registered sensitivity 3: Entity-clustered Newey-West HAC SE",
            sens_nw_ent,
        )
    )
    sens_rows.extend(
        _summarize_sensitivity(
            "pooled_har_rv_no_interactions",
            "Supplementary: Pooled Stock-FE HAR-RV without sector interactions (DK SE)",
            sens_pooled,
            has_interactions=False,
        )
    )
    sens_rows.extend(
        _summarize_sensitivity(
            "extended_with_macro_and_events",
            "Supplementary: Adding t10y2y, spy_ret_1d, has_news, has_8k_earnings (DK SE)",
            sens_controls,
        )
    )
    sens_df = pd.DataFrame(sens_rows)

    def _clean_json(val: Any) -> Any:
        if isinstance(val, float) and math.isnan(val):
            return None
        if isinstance(val, dict):
            return {k: _clean_json(v) for k, v in val.items()}
        if isinstance(val, list):
            return [_clean_json(v) for v in val]
        return val

    payload: dict[str, Any] = _clean_json(
        {
            "schema_version": "1.0.0",
            "milestone": "M3",
            "generated_at_utc": generated_at_utc,
            "reference_sector": REFERENCE_SECTOR,
            "main_model": {
                "n_obs": main_fit["n_obs"],
                "n_entities": main_fit["n_entities"],
                "n_dates": main_fit["n_dates"],
                "k_params": main_fit["k_params"],
                "df_t": main_fit["df_t"],
                "bartlett_bandwidth_L": main_fit["bartlett_bandwidth_L"],
                "within_r2": main_fit["within_r2"],
                "rmse_within": main_fit["rmse_within"],
                "coefficients": main_df[main_df["row_type"] == "regressor"].to_dict(
                    orient="records"
                ),
                "sector_total_slopes": main_df[
                    main_df["row_type"] == "sector_total_slope"
                ].to_dict(orient="records"),
            },
            "confirmatory_family_holm": conf_family,
            "sensitivities_summary": {
                "train_only_2019_2023": {
                    "n_obs": sens_train["n_obs"],
                    "n_dates": sens_train["n_dates"],
                    "bartlett_bandwidth_L": sens_train["bartlett_bandwidth_L"],
                    "within_r2": sens_train["within_r2"],
                },
                "close_to_close_rv_5d_cc_fwd": {
                    "n_obs": sens_cc["n_obs"],
                    "within_r2": sens_cc["within_r2"],
                },
                "newey_west_entity_clustered_se": {
                    "n_obs": sens_nw_ent["n_obs"],
                    "df_t": sens_nw_ent["df_t"],
                    "within_r2": sens_nw_ent["within_r2"],
                },
                "pooled_har_rv_no_interactions": {
                    "n_obs": sens_pooled["n_obs"],
                    "within_r2": sens_pooled["within_r2"],
                },
                "extended_with_macro_and_events": {
                    "n_obs": sens_controls["n_obs"],
                    "within_r2": sens_controls["within_r2"],
                },
            },
        }
    )

    main_df.to_csv(tables_dir / "h1_panel_har_rv_main.csv", index=False, lineterminator="\n")
    sens_df.to_csv(
        tables_dir / "h1_panel_har_rv_sensitivities.csv", index=False, lineterminator="\n"
    )
    md_text = render_h1_summary_markdown(
        generated_at_utc=generated_at_utc,
        payload=payload,
        main_df=main_df,
        sens_df=sens_df,
        date_range=(str(full_df["date"].min()), str(full_df["date"].max())),
    )
    (tables_dir / "h1_summary.md").write_text(md_text, encoding="utf-8")
    (metrics_dir / "h1_regression.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload
