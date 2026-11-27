"""Milestone M4: Purged quarterly expanding folds & Persistence, GJR-GARCH(1,1), and HAR-RV."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stock_volatility import config
from stock_volatility.analysis.panel_har_rv import CORE_HAR_COLS, NON_REF_SECTORS
from stock_volatility.evaluation import RV_FLOOR

MODEL_FEATURES: tuple[str, ...] = (
    "rv_d",
    "rv_w",
    "rv_m",
    "log_rv_d",
    "log_rv_w",
    "log_rv_m",
    "ret_1d",
    "neg_ret_1d",
    "ret_5d",
    "vol_ratio_20d",
    "daily_range_pct",
    "vix_close",
    "t10y2y",
    "spy_ret_1d",
    "has_news",
    "news_count",
    "has_8k_earnings",
    "sector_technology",
    "sector_financials",
    "sector_healthcare",
    "sector_consumer",
    "sector_energy",
)


def duan_smearing_backtransform(
    y_log_pred: np.ndarray,
    train_residuals: np.ndarray,
    *,
    rv_eps: float = config.RV_EPS,
    rv_floor: float = RV_FLOOR,
) -> tuple[float, np.ndarray]:
    """Back-transform log(RV + eps) predictions via Train-fold Duan smearing estimator."""
    smear = float(np.mean(np.exp(np.asarray(train_residuals, dtype=float))))
    rv_hat = np.maximum(np.exp(np.asarray(y_log_pred, dtype=float)) * smear - rv_eps, rv_floor)
    return smear, rv_hat


def gjr_garch_5d_annualized_vol(
    *,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    last_ret_pct: float,
    last_var_pct2: float,
    horizon: int = config.HORIZON_DAYS,
) -> tuple[float, float]:
    """Compute closed-form h-day GJR-GARCH(1,1) annualized volatility and h-day return std dev."""
    ind_neg = 1.0 if last_ret_pct < 0.0 else 0.0
    h1 = max(omega + (alpha + gamma * ind_neg) * (last_ret_pct**2) + beta * last_var_pct2, 1e-8)
    phi = min(max(alpha + 0.5 * gamma + beta, 0.0), 0.9995)

    var_sum_pct2 = h1
    h_prev = h1
    for _ in range(2, horizon + 1):
        h_next = max(omega + phi * h_prev, 1e-8)
        var_sum_pct2 += h_next
        h_prev = h_next

    var_sum_dec = var_sum_pct2 / 10_000.0
    sigma_h_dec = math.sqrt(max(var_sum_dec, 1e-12))
    rv_ann = max(math.sqrt((config.TRADING_DAYS_PER_YEAR / float(horizon)) * var_sum_dec), RV_FLOOR)
    return rv_ann, sigma_h_dec


def build_quarterly_expanding_folds(
    panel_df: pd.DataFrame,
    *,
    horizon_days: int = config.HORIZON_DAYS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build Stage 1 (Val 2024) and Stage 2 (2025 Q1..Q4) purged expanding-window folds.

    Every fold excludes `embargo_1` and `embargo_2` from fitting and enforces a strict
    `>= horizon_days` trading-day purge so the last fitting date's `t+1..t+5` target window
    strictly precedes the evaluation window start date.
    """
    all_dates = sorted(panel_df["date"].astype(str).unique().tolist())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    val_eval_dates = sorted(
        panel_df.loc[
            (panel_df["split"] == "val")
            & panel_df["has_full_features"]
            & panel_df["has_full_target"],
            "date",
        ]
        .unique()
        .tolist()
    )
    val_fit_end = config.TRAIN_END
    val_purge = date_to_idx[val_eval_dates[0]] - date_to_idx[val_fit_end] - 1
    if val_purge < horizon_days:
        raise ValueError(f"Insufficient purge in val_2024: {val_purge} < {horizon_days}")
    val_fold: dict[str, Any] = {
        "fold_id": "val_2024",
        "fit_end_date": val_fit_end,
        "eval_start_date": val_eval_dates[0],
        "eval_end_date": val_eval_dates[-1],
        "n_eval_days": len(val_eval_dates),
        "purge_trading_days": int(val_purge),
        "allowed_fit_splits": ("train",),
    }

    test_scorable_dates = sorted(
        panel_df.loc[
            (panel_df["split"] == "test")
            & panel_df["has_full_features"]
            & panel_df["has_full_target"],
            "date",
        ]
        .unique()
        .tolist()
    )
    q_series = pd.Series(test_scorable_dates)
    q_labels = pd.to_datetime(q_series).dt.to_period("Q").astype(str)

    test_folds: list[dict[str, Any]] = []
    for q_label in sorted(q_labels.unique().tolist()):
        q_dates = q_series[q_labels == q_label].tolist()
        q_start = q_dates[0]
        q_start_idx = date_to_idx[q_start]
        if q_label == "2025Q1":
            fit_end_date = config.VAL_END
            allowed_splits: tuple[str, ...] = ("train", "val")
        else:
            # Strict 5-day purge: fit_end_idx + 5 < q_start_idx => fit_end_idx = q_start_idx - 6
            fit_end_idx = q_start_idx - (horizon_days + 1)
            fit_end_date = all_dates[fit_end_idx]
            allowed_splits = ("train", "val", "test")
        purge_days = q_start_idx - date_to_idx[fit_end_date] - 1
        if purge_days < horizon_days:
            raise ValueError(f"Insufficient purge in {q_label}: {purge_days} < {horizon_days}")
        test_folds.append(
            {
                "fold_id": q_label,
                "fit_end_date": fit_end_date,
                "eval_start_date": q_start,
                "eval_end_date": q_dates[-1],
                "n_eval_days": len(q_dates),
                "purge_trading_days": int(purge_days),
                "allowed_fit_splits": allowed_splits,
            }
        )
    return val_fold, test_folds


def slice_fold_data(
    panel_df: pd.DataFrame,
    fold: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice strictly leak-free fit rows and evaluation rows for a given expanding fold."""
    fit_mask = (
        panel_df["split"].isin(fold["allowed_fit_splits"])
        & (panel_df["date"] <= fold["fit_end_date"])
        & panel_df["has_full_features"].astype(bool)
        & panel_df["has_full_target"].astype(bool)
    )
    eval_mask = (
        (panel_df["date"] >= fold["eval_start_date"])
        & (panel_df["date"] <= fold["eval_end_date"])
        & panel_df["has_full_features"].astype(bool)
        & panel_df["has_full_target"].astype(bool)
    )
    fit_df = (
        panel_df.loc[fit_mask]
        .sort_values(["date", "ticker"], kind="mergesort")
        .reset_index(drop=True)
    )
    eval_df = (
        panel_df.loc[eval_mask]
        .sort_values(["date", "ticker"], kind="mergesort")
        .reset_index(drop=True)
    )
    return fit_df, eval_df


def _build_har_design(df: pd.DataFrame) -> np.ndarray:
    """Build 13-column extended HAR-RV regressor matrix (matching H1)."""
    cols = [df[c].to_numpy(dtype=float) for c in CORE_HAR_COLS]
    for sec in NON_REF_SECTORS:
        ind = (df["sector"] == sec).to_numpy(dtype=float)
        cols.append(ind * df["neg_ret_1d"].to_numpy(dtype=float))
        cols.append(ind * df["vix_close"].to_numpy(dtype=float))
    return np.column_stack(cols)


def fit_predict_har_rv_fold(
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit Stock-FE Predictive HAR-RV (+ Duan smearing) and Logistic HAR-RV on one fold."""
    X_fit = _build_har_design(fit_df)
    y_fit = fit_df["y_log_rv"].to_numpy(dtype=float)

    # Demean within each ticker for Stock-FE OLS
    tickers_fit = fit_df["ticker"].to_numpy()
    unique_tks = np.unique(tickers_fit)
    y_dm = np.empty_like(y_fit)
    X_dm = np.empty_like(X_fit)
    for tk in unique_tks:
        m = tickers_fit == tk
        y_dm[m] = y_fit[m] - y_fit[m].mean()
        X_dm[m] = X_fit[m] - X_fit[m].mean(axis=0)

    beta = np.linalg.lstsq(X_dm, y_dm, rcond=None)[0]
    alpha_map: dict[str, float] = {}
    for tk in unique_tks:
        m = tickers_fit == tk
        alpha_map[str(tk)] = float(y_fit[m].mean() - (X_fit[m].mean(axis=0) @ beta))
    global_alpha = float(np.mean(list(alpha_map.values())))

    # Compute fit residuals for Train-fold Duan smearing
    fit_alpha = np.array([alpha_map[str(tk)] for tk in tickers_fit], dtype=float)
    fit_resid = y_fit - (fit_alpha + X_fit @ beta)

    X_eval = _build_har_design(eval_df)
    eval_alpha = np.array(
        [alpha_map.get(str(tk), global_alpha) for tk in eval_df["ticker"].to_numpy()], dtype=float
    )
    y_eval_log = eval_alpha + X_eval @ beta
    smear, rv_hat_eval = duan_smearing_backtransform(y_eval_log, fit_resid)

    # Logistic HAR-RV probabilistic baseline (standardized strictly on fit fold)
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(C=1.0, max_iter=1000, random_state=config.RANDOM_SEED)),
        ]
    )
    clf.fit(fit_df[list(MODEL_FEATURES)], fit_df["y_tail"].astype(int))
    p_tail_eval = clf.predict_proba(eval_df[list(MODEL_FEATURES)])[:, 1]

    meta = {
        "duan_smearing_factor": smear,
        "n_fit_rows": int(len(fit_df)),
        "n_eval_rows": int(len(eval_df)),
    }
    return rv_hat_eval, p_tail_eval, meta


def fit_predict_gjr_garch_fold(
    panel_df: pd.DataFrame,
    fold: dict[str, Any],
    eval_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Fit per-stock GJR-GARCH(1,1) strictly <= fit_end_date and filter forward causally."""
    fit_end = fold["fit_end_date"]
    eval_end = fold["eval_end_date"]
    eval_keys = set(zip(eval_df["ticker"].astype(str), eval_df["date"].astype(str), strict=True))

    pred_map: dict[tuple[str, str], tuple[float, float]] = {}
    conv_records: list[dict[str, Any]] = []

    for ticker, grp in panel_df.sort_values(["ticker", "date"], kind="mergesort").groupby(
        "ticker", sort=True
    ):
        g = grp[grp["date"] <= eval_end].reset_index(drop=True)
        fit_g = g[(g["date"] <= fit_end) & g["ret_1d"].notna()]
        r_fit_pct = fit_g["ret_1d"].to_numpy(dtype=float) * 100.0
        fallback_used = False

        am = arch_model(r_fit_pct, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="normal")
        try:
            res = am.fit(disp="off", show_warning=False)
            conv_flag = int(res.convergence_flag)
            omega = float(res.params["omega"])
            alpha = float(res.params["alpha[1]"])
            gamma = float(res.params["gamma[1]"])
            beta = float(res.params["beta[1]"])
            var_cur = float(res.conditional_volatility[-1] ** 2)
            converged = conv_flag == 0 and bool(
                np.isfinite([omega, alpha, gamma, beta, var_cur]).all()
            )
        except Exception:
            conv_flag = -1
            converged = False

        if not converged:
            # Explicit documented fallback to RiskMetrics EWMA if optimizer fails
            fallback_used = True
            omega, alpha, gamma, beta = 0.0, 0.06, 0.0, 0.94
            var_cur = max(float(np.var(r_fit_pct)), 1e-8)

        persistence = alpha + 0.5 * gamma + beta
        conv_records.append(
            {
                "fold_id": fold["fold_id"],
                "ticker": str(ticker),
                "fit_end_date": fit_end,
                "n_returns": int(len(r_fit_pct)),
                "convergence_flag": conv_flag,
                "converged": bool(converged),
                "fallback_used": bool(fallback_used),
                "omega": omega,
                "alpha": alpha,
                "gamma": gamma,
                "beta": beta,
                "persistence": persistence,
            }
        )

        # Causally filter conditional variance day-by-day past fit_end_date using ONLY <= t returns
        last_fit_ret_pct = float(r_fit_pct[-1])
        post_g = g[g["date"] > fit_end]
        prev_ret_pct = last_fit_ret_pct
        for row_date, ret_dec in zip(
            post_g["date"].astype(str), post_g["ret_1d"].to_numpy(dtype=float), strict=True
        ):
            ind_neg = 1.0 if prev_ret_pct < 0.0 else 0.0
            var_cur = max(
                omega + (alpha + gamma * ind_neg) * (prev_ret_pct**2) + beta * var_cur,
                1e-8,
            )
            ret_pct = float(ret_dec * 100.0)
            if (str(ticker), row_date) in eval_keys:
                rv_hat, sig5 = gjr_garch_5d_annualized_vol(
                    omega=omega,
                    alpha=alpha,
                    gamma=gamma,
                    beta=beta,
                    last_ret_pct=ret_pct,
                    last_var_pct2=var_cur,
                )
                p_tail = float(stats.norm.cdf(config.TAIL_RETURN_THRESHOLD / sig5))
                pred_map[(str(ticker), row_date)] = (rv_hat, p_tail)
            prev_ret_pct = ret_pct

    rv_out = np.empty(len(eval_df), dtype=float)
    p_out = np.empty(len(eval_df), dtype=float)
    for idx, (tk, dt) in enumerate(
        zip(eval_df["ticker"].astype(str), eval_df["date"].astype(str), strict=True)
    ):
        rv_val, p_val = pred_map[(tk, dt)]
        rv_out[idx] = rv_val
        p_out[idx] = p_val
    return rv_out, p_out, conv_records


def run_baselines_all_folds(
    panel_df: pd.DataFrame,
) -> dict[str, Any]:
    """Run Persistence, per-stock GJR-GARCH(1,1), and Predictive HAR-RV on Val and 2025 Test."""
    val_fold, test_folds = build_quarterly_expanding_folds(panel_df)
    all_conv: list[dict[str, Any]] = []
    fold_audits: list[dict[str, Any]] = []

    def _run_one_fold(fold: dict[str, Any]) -> pd.DataFrame:
        fit_df, eval_df = slice_fold_data(panel_df, fold)
        base_rate = float(fit_df["y_tail"].astype(int).mean())
        rv_pers = np.maximum(eval_df["rv_w"].to_numpy(dtype=float), RV_FLOOR)
        p_pers = np.full(len(eval_df), base_rate, dtype=float)

        rv_garch, p_garch, conv_rows = fit_predict_gjr_garch_fold(panel_df, fold, eval_df)
        all_conv.extend(conv_rows)

        rv_har, p_har, har_meta = fit_predict_har_rv_fold(fit_df, eval_df)
        fold_audits.append(
            {
                **{k: v for k, v in fold.items() if k != "allowed_fit_splits"},
                "allowed_fit_splits": list(fold["allowed_fit_splits"]),
                "n_fit_rows": int(len(fit_df)),
                "n_eval_rows": int(len(eval_df)),
                "persistence_base_rate": base_rate,
                "har_duan_smearing": har_meta["duan_smearing_factor"],
                "garch_converged_count": int(sum(r["converged"] for r in conv_rows)),
            }
        )

        out = eval_df[
            ["ticker", "sector", "date", "has_news", "has_8k_earnings", "rv_5d_fwd", "y_tail"]
        ].copy()
        out["fold_id"] = fold["fold_id"]
        out["y_tail"] = out["y_tail"].astype(int)
        out["rv_hat_persistence"] = rv_pers
        out["rv_hat_gjr_garch"] = rv_garch
        out["rv_hat_har_rv"] = rv_har
        out["p_tail_persistence"] = p_pers
        out["p_tail_gjr_garch"] = p_garch
        out["p_tail_har_rv"] = p_har
        return out

    val_df = _run_one_fold(val_fold)
    test_parts = [_run_one_fold(f) for f in test_folds]
    test_df = (
        pd.concat(test_parts, ignore_index=True)
        .sort_values(["date", "ticker"], kind="mergesort")
        .reset_index(drop=True)
    )
    conv_df = pd.DataFrame(all_conv)
    return {
        "val_fold": val_fold,
        "test_folds": test_folds,
        "val_pred_df": val_df,
        "test_pred_df": test_df,
        "garch_convergence_df": conv_df,
        "baselines_summary": {
            "schema_version": "1.0.0",
            "milestone": "M4",
            "total_garch_fits": int(len(conv_df)),
            "garch_converged_fits": int(conv_df["converged"].sum()),
            "garch_convergence_rate": float(conv_df["converged"].mean()),
            "folds": fold_audits,
        },
    }
