"""Milestone M4: QLIKE/RMSE/MAE, PR-AUC/ROC-AUC/Brier/ECE, date-block bootstrap & h=5 HLN-DM."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from stock_volatility import config
from stock_volatility.regression import holm_bonferroni_adjust

RV_FLOOR: float = 1e-4


def qlike_loss_array(rv_true: np.ndarray, rv_hat: np.ndarray) -> np.ndarray:
    """Compute elementwise variance-scale QLIKE loss: s2/s2_hat - ln(s2/s2_hat) - 1."""
    true_arr = np.maximum(np.asarray(rv_true, dtype=float), RV_FLOOR)
    hat_arr = np.maximum(np.asarray(rv_hat, dtype=float), RV_FLOOR)
    ratio = (true_arr**2) / (hat_arr**2)
    return ratio - np.log(ratio) - 1.0


def regression_metrics(rv_true: np.ndarray, rv_hat: np.ndarray) -> dict[str, float]:
    """Compute variance-scale QLIKE and annualized-scale RMSE & MAE."""
    t_arr = np.asarray(rv_true, dtype=float)
    h_arr = np.maximum(np.asarray(rv_hat, dtype=float), RV_FLOOR)
    err = t_arr - h_arr
    return {
        "qlike": float(np.mean(qlike_loss_array(t_arr, h_arr))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
    }


def calibration_bins_table(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute uniform [0, 1] reliability curve bin statistics."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_pred, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n_total = len(y)
    rows: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        cnt = int(mask.sum())
        if cnt > 0:
            mean_p = float(p[mask].mean())
            obs_rate = float(y[mask].mean())
            abs_gap = abs(obs_rate - mean_p)
            weight = cnt / n_total
        else:
            mean_p = float("nan")
            obs_rate = float("nan")
            abs_gap = 0.0
            weight = 0.0
        rows.append(
            {
                "bin_index": i,
                "bin_low": lo,
                "bin_high": hi,
                "n_obs": cnt,
                "weight": float(weight),
                "mean_pred_prob": mean_p,
                "observed_rate": obs_rate,
                "abs_gap": float(abs_gap),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) over uniform [0, 1] probability bins."""
    bins_df = calibration_bins_table(y_true, p_pred, n_bins=n_bins)
    return float(np.sum(bins_df["weight"] * bins_df["abs_gap"]))


def classification_metrics(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute PR-AUC (primary), ROC-AUC, Brier Score, and 10-bin ECE."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(p_pred, dtype=float), 0.0, 1.0)
    has_both = len(np.unique(y)) == 2
    pr_auc = float(average_precision_score(y, p)) if has_both else float("nan")
    roc_auc = float(roc_auc_score(y, p)) if has_both else float("nan")
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p, n_bins=n_bins),
    }


def date_block_bootstrap_ci(
    dates: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    *,
    block_len: int = 5,
    n_boot: int = 1000,
    seed: int = config.RANDOM_SEED,
) -> tuple[float, float]:
    """Compute percentile 95% CI by resampling contiguous date blocks (keeping all stocks/day)."""
    dates_arr = np.asarray(dates)
    unique_dates = np.unique(dates_arr)
    unique_dates.sort()
    n_dates = len(unique_dates)
    if n_dates < block_len:
        raise ValueError(f"Need at least {block_len} unique dates, got {n_dates}")

    date_to_rows: list[np.ndarray] = [np.flatnonzero(dates_arr == d) for d in unique_dates]
    n_blocks = int(math.ceil(n_dates / block_len))
    max_start = n_dates - block_len + 1
    rng = np.random.default_rng(seed)

    boot_vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, max_start, size=n_blocks)
        sampled_date_indices: list[int] = []
        for s in starts:
            sampled_date_indices.extend(range(int(s), int(s) + block_len))
        sampled_date_indices = sampled_date_indices[:n_dates]
        row_idx = np.concatenate([date_to_rows[i] for i in sampled_date_indices])
        boot_vals[b] = float(metric_fn(row_idx))

    finite = boot_vals[np.isfinite(boot_vals)]
    if len(finite) == 0:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return (float(lo), float(hi))


def hln_dm_test(
    loss_diff_by_date: np.ndarray,
    *,
    horizon: int = config.HORIZON_DAYS,
) -> dict[str, float]:
    """Compute multi-step HLN-modified Diebold-Mariano test with Bartlett PSD NW variance.

    Convention: `d_t = L(model_A) - L(model_B)`; negative `dm_hln_stat` means model A has
    LOWER loss (better) than model B.
    """
    d = np.asarray(loss_diff_by_date, dtype=float)
    d = d[np.isfinite(d)]
    n_t = len(d)
    if n_t <= horizon + 1:
        raise ValueError(f"Need > {horizon + 1} time points for h={horizon} HLN-DM, got {n_t}")

    d_bar = float(np.mean(d))
    d_c = d - d_bar
    max_lag = max(0, horizon - 1)

    gamma0 = float(np.dot(d_c, d_c) / n_t)
    lrv = gamma0
    for k in range(1, max_lag + 1):
        weight = 1.0 - k / (max_lag + 1.0)  # Bartlett weights guarantee PSD >= 0
        gamma_k = float(np.dot(d_c[k:], d_c[:-k]) / n_t)
        lrv += 2.0 * weight * gamma_k
    lrv = max(lrv, 1e-18)

    se = math.sqrt(lrv / n_t)
    dm_raw = d_bar / se
    hln_num = n_t + 1.0 - 2.0 * horizon + (horizon * (horizon - 1.0)) / n_t
    hln_factor = math.sqrt(max(hln_num, 1.0) / n_t)
    dm_hln = dm_raw * hln_factor
    df = n_t - 1
    p_val = float(2.0 * stats.t.sf(abs(dm_hln), df=df))
    return {
        "n_dates": float(n_t),
        "horizon": float(horizon),
        "max_lag": float(max_lag),
        "mean_diff": d_bar,
        "long_run_var": lrv,
        "se": se,
        "dm_raw_stat": dm_raw,
        "hln_factor": hln_factor,
        "dm_hln_stat": dm_hln,
        "df": float(df),
        "p_value": p_val,
    }


def evaluate_holdout_predictions(
    test_pred_df: pd.DataFrame,
    *,
    val_pred_df: pd.DataFrame | None = None,
    block_len: int = 5,
    n_boot: int = 1000,
    seed: int = config.RANDOM_SEED,
) -> dict[str, Any]:
    """Run full M4 2025 Holdout evaluation: metrics + bootstrap CIs, breakdowns, HLN-DM & ECE."""
    df = test_pred_df.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    dates_arr = df["date"].to_numpy()
    rv_true = df["rv_5d_fwd"].to_numpy(dtype=float)
    y_tail = df["y_tail"].to_numpy(dtype=int)

    model_specs = [
        ("persistence", "rv_hat_persistence", "p_tail_persistence"),
        ("gjr_garch", "rv_hat_gjr_garch", "p_tail_gjr_garch"),
        ("har_rv", "rv_hat_har_rv", "p_tail_har_rv"),
        ("lgbm", "rv_hat_lgbm", "p_tail_lgbm_cal"),
    ]

    # 1. Overall 2025 Test metrics with date-block bootstrap 95% CIs
    metric_rows: list[dict[str, Any]] = []
    for model_name, rv_col, p_col in model_specs:
        rv_hat = df[rv_col].to_numpy(dtype=float)
        p_hat = df[p_col].to_numpy(dtype=float)
        rm = regression_metrics(rv_true, rv_hat)
        cm = classification_metrics(y_tail, p_hat)

        ql_lo, ql_hi = date_block_bootstrap_ci(
            dates_arr,
            lambda idx, rh=rv_hat: float(np.mean(qlike_loss_array(rv_true[idx], rh[idx]))),
            block_len=block_len,
            n_boot=n_boot,
            seed=seed,
        )
        rmse_lo, rmse_hi = date_block_bootstrap_ci(
            dates_arr,
            lambda idx, rh=rv_hat: float(np.sqrt(np.mean((rv_true[idx] - rh[idx]) ** 2))),
            block_len=block_len,
            n_boot=n_boot,
            seed=seed,
        )
        mae_lo, mae_hi = date_block_bootstrap_ci(
            dates_arr,
            lambda idx, rh=rv_hat: float(np.mean(np.abs(rv_true[idx] - rh[idx]))),
            block_len=block_len,
            n_boot=n_boot,
            seed=seed,
        )
        pr_lo, pr_hi = date_block_bootstrap_ci(
            dates_arr,
            lambda idx, ph=p_hat: (
                float(average_precision_score(y_tail[idx], ph[idx]))
                if len(np.unique(y_tail[idx])) == 2
                else float("nan")
            ),
            block_len=block_len,
            n_boot=n_boot,
            seed=seed,
        )
        roc_lo, roc_hi = date_block_bootstrap_ci(
            dates_arr,
            lambda idx, ph=p_hat: (
                float(roc_auc_score(y_tail[idx], ph[idx]))
                if len(np.unique(y_tail[idx])) == 2
                else float("nan")
            ),
            block_len=block_len,
            n_boot=n_boot,
            seed=seed,
        )
        brier_lo, brier_hi = date_block_bootstrap_ci(
            dates_arr,
            lambda idx, ph=p_hat: float(brier_score_loss(y_tail[idx], ph[idx])),
            block_len=block_len,
            n_boot=n_boot,
            seed=seed,
        )
        ece_lo, ece_hi = date_block_bootstrap_ci(
            dates_arr,
            lambda idx, ph=p_hat: expected_calibration_error(y_tail[idx], ph[idx], n_bins=10),
            block_len=block_len,
            n_boot=n_boot,
            seed=seed,
        )

        row_dict: dict[str, Any] = {
            "model": model_name,
            "n_obs": int(len(df)),
            "n_dates": int(df["date"].nunique()),
            "qlike": rm["qlike"],
            "qlike_ci95_low": ql_lo,
            "qlike_ci95_high": ql_hi,
            "rmse": rm["rmse"],
            "rmse_ci95_low": rmse_lo,
            "rmse_ci95_high": rmse_hi,
            "mae": rm["mae"],
            "mae_ci95_low": mae_lo,
            "mae_ci95_high": mae_hi,
            "pr_auc": cm["pr_auc"],
            "pr_auc_ci95_low": pr_lo,
            "pr_auc_ci95_high": pr_hi,
            "roc_auc": cm["roc_auc"],
            "roc_auc_ci95_low": roc_lo,
            "roc_auc_ci95_high": roc_hi,
            "brier": cm["brier"],
            "brier_ci95_low": brier_lo,
            "brier_ci95_high": brier_hi,
            "ece": cm["ece"],
            "ece_ci95_low": ece_lo,
            "ece_ci95_high": ece_hi,
        }
        if model_name == "lgbm":
            raw_cm = classification_metrics(y_tail, df["p_tail_lgbm_raw"].to_numpy(dtype=float))
            row_dict["brier_pre_cal"] = raw_cm["brier"]
            row_dict["ece_pre_cal"] = raw_cm["ece"]
        else:
            row_dict["brier_pre_cal"] = cm["brier"]
            row_dict["ece_pre_cal"] = cm["ece"]
        metric_rows.append(row_dict)
    metrics_df = pd.DataFrame(metric_rows)

    # 2. Sector breakdown
    sector_rows: list[dict[str, Any]] = []
    for sec in config.SECTORS:
        sub = df[df["sector"] == sec]
        for model_name, rv_col, p_col in model_specs:
            rm = regression_metrics(
                sub["rv_5d_fwd"].to_numpy(dtype=float), sub[rv_col].to_numpy(dtype=float)
            )
            cm = classification_metrics(
                sub["y_tail"].to_numpy(dtype=int), sub[p_col].to_numpy(dtype=float)
            )
            sector_rows.append(
                {
                    "sector": sec,
                    "model": model_name,
                    "n_obs": int(len(sub)),
                    "y_tail_pos": int(sub["y_tail"].astype(int).sum()),
                    **rm,
                    **cm,
                }
            )
    by_sector_df = pd.DataFrame(sector_rows)

    # 3. Event-day breakdown (No_Event, Has_Event, Has_8K_Earnings)
    event_masks = [
        ("No_Event", df["has_news"] == 0),
        ("Has_Event", df["has_news"] == 1),
        ("Has_8K_Earnings", df["has_8k_earnings"] == 1),
    ]
    event_rows: list[dict[str, Any]] = []
    for grp_name, mask in event_masks:
        sub = df[mask]
        for model_name, rv_col, p_col in model_specs:
            rm = regression_metrics(
                sub["rv_5d_fwd"].to_numpy(dtype=float), sub[rv_col].to_numpy(dtype=float)
            )
            cm = classification_metrics(
                sub["y_tail"].to_numpy(dtype=int), sub[p_col].to_numpy(dtype=float)
            )
            event_rows.append(
                {
                    "event_group": grp_name,
                    "model": model_name,
                    "n_obs": int(len(sub)),
                    "y_tail_pos": int(sub["y_tail"].astype(int).sum()),
                    **rm,
                    **cm,
                }
            )
    by_event_df = pd.DataFrame(event_rows)

    # 4. Multi-step h=5 HLN-DM tests with Holm-Bonferroni correction within each loss family
    dm_rows: list[dict[str, Any]] = []
    baselines = [
        ("persistence", "rv_hat_persistence", "p_tail_persistence"),
        ("gjr_garch", "rv_hat_gjr_garch", "p_tail_gjr_garch"),
        ("har_rv", "rv_hat_har_rv", "p_tail_har_rv"),
    ]
    loss_defs: list[tuple[str, Callable[[pd.DataFrame, str, str], np.ndarray]]] = [
        (
            "qlike",
            lambda d_df, c_a, c_b: (
                qlike_loss_array(d_df["rv_5d_fwd"].to_numpy(), d_df[c_a].to_numpy())
                - qlike_loss_array(d_df["rv_5d_fwd"].to_numpy(), d_df[c_b].to_numpy())
            ),
        ),
        (
            "se",
            lambda d_df, c_a, c_b: (
                (d_df["rv_5d_fwd"].to_numpy() - d_df[c_a].to_numpy()) ** 2
                - (d_df["rv_5d_fwd"].to_numpy() - d_df[c_b].to_numpy()) ** 2
            ),
        ),
        (
            "brier",
            lambda d_df, c_a, c_b: (
                (d_df["y_tail"].to_numpy(dtype=float) - d_df[c_a].to_numpy(dtype=float)) ** 2
                - (d_df["y_tail"].to_numpy(dtype=float) - d_df[c_b].to_numpy(dtype=float)) ** 2
            ),
        ),
    ]

    for loss_name, diff_fn in loss_defs:
        family_rows: list[dict[str, Any]] = []
        for base_name, rv_b_col, p_b_col in baselines:
            if loss_name == "brier":
                col_a, col_b = "p_tail_lgbm_cal", p_b_col
            else:
                col_a, col_b = "rv_hat_lgbm", rv_b_col
            row_diffs = diff_fn(df, col_a, col_b)
            daily_diff = (
                pd.DataFrame({"date": df["date"], "d": row_diffs})
                .groupby("date", sort=True)["d"]
                .mean()
                .to_numpy(dtype=float)
            )
            dm_res = hln_dm_test(daily_diff, horizon=config.HORIZON_DAYS)
            family_rows.append(
                {
                    "loss_metric": loss_name,
                    "challenger_model": "lgbm",
                    "baseline_model": base_name,
                    **dm_res,
                    "challenger_better": bool(dm_res["mean_diff"] < 0.0),
                }
            )
        p_holm = holm_bonferroni_adjust([r["p_value"] for r in family_rows])
        for row_item, ph in zip(family_rows, p_holm, strict=True):
            row_item["p_holm"] = float(ph)
            row_item["significant_holm_05"] = bool(ph < 0.05)
            dm_rows.append(row_item)
    dm_df = pd.DataFrame(dm_rows)

    # 5. Calibration curve bins (2025 Test pre- vs post-calibration + baselines)
    cal_variants = [
        ("test", "lgbm_raw", df["y_tail"].to_numpy(dtype=int), df["p_tail_lgbm_raw"].to_numpy()),
        ("test", "lgbm_cal", df["y_tail"].to_numpy(dtype=int), df["p_tail_lgbm_cal"].to_numpy()),
        ("test", "har_rv", df["y_tail"].to_numpy(dtype=int), df["p_tail_har_rv"].to_numpy()),
        ("test", "gjr_garch", df["y_tail"].to_numpy(dtype=int), df["p_tail_gjr_garch"].to_numpy()),
    ]
    if val_pred_df is not None:
        cal_variants.extend(
            [
                (
                    "val",
                    "lgbm_raw",
                    val_pred_df["y_tail"].to_numpy(dtype=int),
                    val_pred_df["p_tail_lgbm_raw"].to_numpy(),
                ),
                (
                    "val",
                    "lgbm_cal",
                    val_pred_df["y_tail"].to_numpy(dtype=int),
                    val_pred_df["p_tail_lgbm_cal"].to_numpy(),
                ),
            ]
        )
    cal_parts: list[pd.DataFrame] = []
    for split_name, variant_name, y_v, p_v in cal_variants:
        b_df = calibration_bins_table(y_v, p_v, n_bins=10)
        b_df.insert(0, "model_variant", variant_name)
        b_df.insert(0, "split", split_name)
        cal_parts.append(b_df)
    cal_curve_df = pd.concat(cal_parts, ignore_index=True)

    return {
        "metrics_df": metrics_df,
        "by_sector_df": by_sector_df,
        "by_event_df": by_event_df,
        "dm_df": dm_df,
        "cal_curve_df": cal_curve_df,
        "summary_json": {
            "schema_version": "1.0.0",
            "milestone": "M4",
            "test_scorable_rows": int(len(df)),
            "test_scorable_days": int(df["date"].nunique()),
            "test_tail_positive_count": int(y_tail.sum()),
            "test_tail_base_rate": float(np.mean(y_tail)),
            "bootstrap_config": {
                "type": "date_block",
                "block_len": block_len,
                "B": n_boot,
                "seed": seed,
            },
            "dm_config": {
                "method": "HLN_modified_DM_Bartlett_PSD",
                "horizon": config.HORIZON_DAYS,
                "max_lag": config.HORIZON_DAYS - 1,
                "multiple_testing": "Holm-Bonferroni_within_loss_family",
            },
            "test_metrics": metrics_df.to_dict(orient="records"),
            "dm_tests": dm_df.to_dict(orient="records"),
        },
    }
