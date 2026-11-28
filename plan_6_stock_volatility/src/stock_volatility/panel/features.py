"""Strictly zero-leakage target construction, <=t feature engineering, and Train-only scaling."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from stock_volatility import config

SECTOR_DUMMY_COLUMNS: tuple[str, ...] = (
    "sector_technology",
    "sector_financials",
    "sector_healthcare",
    "sector_consumer",
    "sector_energy",
)

CONTINUOUS_FEATURE_COLUMNS: tuple[str, ...] = (
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
)

EVENT_FEATURE_COLUMNS: tuple[str, ...] = (
    "has_news",
    "news_count",
    "has_8k_earnings",
    "news_source",
)

TARGET_COLUMNS: tuple[str, ...] = (
    "rv_5d_fwd",
    "y_log_rv",
    "rv_5d_cc_fwd",
    "cum_ret_5d_fwd",
    "y_tail",
    "y_tail_rv90",
)

PANEL_DAILY_COLUMNS: tuple[str, ...] = (
    "ticker",
    "sector",
    "sub_group",
    "date",
    "split",
    "has_full_features",
    "has_full_target",
    *TARGET_COLUMNS,
    *CONTINUOUS_FEATURE_COLUMNS,
    *EVENT_FEATURE_COLUMNS,
    *SECTOR_DUMMY_COLUMNS,
)

CORE_REQUIRED_FEATURE_COLS: tuple[str, ...] = (
    *CONTINUOUS_FEATURE_COLUMNS,
    "has_news",
    "news_count",
    "has_8k_earnings",
)

_PARKINSON_DENOM: float = 4.0 * math.log(2.0)


def fit_train_preprocessing_params(
    panel_df: pd.DataFrame,
    *,
    winsor_lower_q: float = 0.005,
    winsor_upper_q: float = 0.995,
    rv_tail_q: float = 0.90,
) -> dict[str, Any]:
    """Fit Winsorization bounds, z-score stats, and RV 90th quantile STRICTLY on Train rows."""
    train_mask = (
        (panel_df["split"] == "train")
        & panel_df["has_full_features"]
        & panel_df["rv_5d_fwd"].notna()
    )
    train_sub = panel_df.loc[train_mask]
    if train_sub.empty:
        raise ValueError("Cannot fit Train preprocessing parameters on empty Train slice")

    rv90_threshold = float(train_sub["rv_5d_fwd"].quantile(rv_tail_q))
    feature_stats: dict[str, dict[str, float]] = {}
    for col in CONTINUOUS_FEATURE_COLUMNS:
        series = pd.to_numeric(train_sub[col], errors="coerce").dropna()
        if series.empty:
            raise ValueError(f"Train feature column {col!r} has zero non-null values")
        p_low = float(series.quantile(winsor_lower_q))
        p_high = float(series.quantile(winsor_upper_q))
        clipped = series.clip(lower=p_low, upper=p_high)
        std_raw = float(series.std(ddof=0))
        std_win = float(clipped.std(ddof=0))
        feature_stats[col] = {
            "p005": p_low,
            "p995": p_high,
            "mean": float(series.mean()),
            "std": std_raw if std_raw > 0.0 else 1.0,
            "winsor_mean": float(clipped.mean()),
            "winsor_std": std_win if std_win > 0.0 else 1.0,
            "min": float(series.min()),
            "max": float(series.max()),
        }

    return {
        "fitted_split": "train",
        "fitted_row_count": int(len(train_sub)),
        "fitted_date_min": str(train_sub["date"].min()),
        "fitted_date_max": str(train_sub["date"].max()),
        "winsor_lower_q": winsor_lower_q,
        "winsor_upper_q": winsor_upper_q,
        "rv_tail_q": rv_tail_q,
        "rv90_threshold": rv90_threshold,
        "continuous_features": feature_stats,
    }


def apply_train_winsor_and_scale(
    df: pd.DataFrame,
    params: dict[str, Any],
    *,
    columns: tuple[str, ...] = CONTINUOUS_FEATURE_COLUMNS,
    standardize: bool = True,
) -> pd.DataFrame:
    """Apply frozen Train-fitted Winsorization (and optional z-scoring) without refitting."""
    out = df.copy()
    feat_stats = params["continuous_features"]
    for col in columns:
        st = feat_stats[col]
        vals = pd.to_numeric(out[col], errors="coerce").clip(lower=st["p005"], upper=st["p995"])
        if standardize:
            vals = (vals - st["winsor_mean"]) / st["winsor_std"]
        out[col] = vals
    return out


def _compute_ticker_series(group: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Compute strictly-past (<=t) price features and forward (t+1..t+h) targets for one ticker."""
    g = group.sort_values("date", kind="mergesort").reset_index(drop=True).copy()
    high = g["high"].to_numpy(dtype=float)
    low = g["low"].to_numpy(dtype=float)
    close = g["close"].to_numpy(dtype=float)
    adj_close = g["adj_close"].to_numpy(dtype=float)
    volume = g["volume"].to_numpy(dtype=float)

    if (high < low).any() or (low <= 0).any() or (close <= 0).any() or (adj_close <= 0).any():
        raise ValueError(f"Invalid OHLCV values encountered for ticker {g['ticker'].iloc[0]}")

    ann = float(config.TRADING_DAYS_PER_YEAR)
    eps = float(config.RV_EPS)

    hl_sq = pd.Series(np.log(high / low) ** 2, index=g.index)
    adj_s = pd.Series(adj_close, index=g.index)
    vol_s = pd.Series(volume, index=g.index)

    # Strictly <= t features
    rv_d = np.sqrt((ann / _PARKINSON_DENOM) * hl_sq)
    rv_w = np.sqrt((ann / (5.0 * _PARKINSON_DENOM)) * hl_sq.rolling(5, min_periods=5).sum())
    rv_m = np.sqrt((ann / (22.0 * _PARKINSON_DENOM)) * hl_sq.rolling(22, min_periods=22).sum())

    ret_1d = adj_s.pct_change(1)
    log_ret_1d = np.log(adj_s / adj_s.shift(1))
    neg_ret_1d = ret_1d.clip(upper=0.0)
    ret_5d = adj_s.pct_change(5)
    vol_ratio_20d = vol_s / vol_s.rolling(20, min_periods=20).mean()
    daily_range_pct = pd.Series((high - low) / close, index=g.index)

    # Strictly future t+1..t+h targets
    fwd_hl_sq_sum = sum(hl_sq.shift(-k) for k in range(1, horizon_days + 1))
    rv_5d_fwd = np.sqrt((ann / (float(horizon_days) * _PARKINSON_DENOM)) * fwd_hl_sq_sum)
    y_log_rv = np.log(rv_5d_fwd + eps)

    fwd_lr_sq_sum = sum(log_ret_1d.shift(-k) ** 2 for k in range(1, horizon_days + 1))
    rv_5d_cc_fwd = np.sqrt((ann / float(horizon_days)) * fwd_lr_sq_sum)

    cum_ret_5d_fwd = adj_s.shift(-horizon_days) / adj_s - 1.0
    y_tail_arr = np.where(
        cum_ret_5d_fwd.notna(),
        (cum_ret_5d_fwd < float(config.TAIL_RETURN_THRESHOLD)).astype(int),
        np.nan,
    )

    g["rv_5d_fwd"] = rv_5d_fwd.to_numpy(dtype=float)
    g["y_log_rv"] = y_log_rv.to_numpy(dtype=float)
    g["rv_5d_cc_fwd"] = rv_5d_cc_fwd.to_numpy(dtype=float)
    g["cum_ret_5d_fwd"] = cum_ret_5d_fwd.to_numpy(dtype=float)
    g["y_tail"] = pd.Series(y_tail_arr, index=g.index, dtype="Int8")

    g["rv_d"] = rv_d.to_numpy(dtype=float)
    g["rv_w"] = rv_w.to_numpy(dtype=float)
    g["rv_m"] = rv_m.to_numpy(dtype=float)
    g["log_rv_d"] = np.log(rv_d + eps).to_numpy(dtype=float)
    g["log_rv_w"] = np.log(rv_w + eps).to_numpy(dtype=float)
    g["log_rv_m"] = np.log(rv_m + eps).to_numpy(dtype=float)
    g["ret_1d"] = ret_1d.to_numpy(dtype=float)
    g["neg_ret_1d"] = neg_ret_1d.to_numpy(dtype=float)
    g["ret_5d"] = ret_5d.to_numpy(dtype=float)
    g["vol_ratio_20d"] = vol_ratio_20d.to_numpy(dtype=float)
    g["daily_range_pct"] = daily_range_pct.to_numpy(dtype=float)
    return g


def build_panel_features_and_targets(
    *,
    ohlcv_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    daily_events_df: pd.DataFrame,
    split_df: pd.DataFrame,
    horizon_days: int = config.HORIZON_DAYS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the full modeling panel with <=t predictors, t+1..t+5 targets, and Train params."""
    parts = [
        _compute_ticker_series(grp, horizon_days=horizon_days)
        for _, grp in ohlcv_df.groupby("ticker", sort=True)
    ]
    panel = pd.concat(parts, ignore_index=True)

    # Merge deterministic sector/sub_group metadata
    uni_meta = universe_df[["ticker", "sector", "sub_group"]].drop_duplicates("ticker")
    panel = panel.merge(uni_meta, on="ticker", how="left", validate="many_to_one")
    if panel["sector"].isna().any():
        raise ValueError("Missing sector metadata after joining universe_df")

    # Merge canonical split label
    panel = panel.merge(
        split_df[["date", "split"]].drop_duplicates("date"),
        on="date",
        how="left",
        validate="many_to_one",
    )
    if panel["split"].isna().any():
        raise ValueError("Missing split label after joining split_df")

    # Merge macro predictors (strictly <= t)
    panel = panel.merge(
        macro_df[["date", "vix_close", "t10y2y", "spy_ret_1d"]].drop_duplicates("date"),
        on="date",
        how="left",
        validate="many_to_one",
    )

    # Merge pre-close corporate event features (strictly < 16:00 ET / 13:00 ET early close)
    panel = panel.merge(
        daily_events_df[
            ["ticker", "date", "has_news", "news_count", "has_8k_earnings", "news_source"]
        ],
        on=["ticker", "date"],
        how="left",
        validate="one_to_one",
    )
    panel["has_news"] = panel["has_news"].fillna(0).astype(int)
    panel["news_count"] = panel["news_count"].fillna(0).astype(int)
    panel["has_8k_earnings"] = panel["has_8k_earnings"].fillna(0).astype(int)
    panel["news_source"] = panel["news_source"].fillna("none").astype(str)

    # Sector dummies (5 GICS sectors)
    for sector in config.SECTORS:
        col_name = f"sector_{sector.lower()}"
        panel[col_name] = (panel["sector"] == sector).astype(int)

    # Compute completeness flags (prior to Train-only y_tail_rv90)
    feat_valid = np.ones(len(panel), dtype=bool)
    for col in CORE_REQUIRED_FEATURE_COLS:
        vals = pd.to_numeric(panel[col], errors="coerce").to_numpy(dtype=float)
        feat_valid &= np.isfinite(vals)
    panel["has_full_features"] = feat_valid

    base_target_valid = (
        np.isfinite(panel["rv_5d_fwd"].to_numpy(dtype=float))
        & np.isfinite(panel["y_log_rv"].to_numpy(dtype=float))
        & np.isfinite(panel["rv_5d_cc_fwd"].to_numpy(dtype=float))
        & np.isfinite(panel["cum_ret_5d_fwd"].to_numpy(dtype=float))
        & panel["y_tail"].notna().to_numpy(dtype=bool)
    )
    panel["has_full_target"] = base_target_valid

    # Fit Train-only preprocessing & 90th-percentile RV threshold strictly on Train rows
    train_params = fit_train_preprocessing_params(panel)
    rv90_thresh = float(train_params["rv90_threshold"])

    y_tail_rv90_arr = np.where(
        panel["has_full_target"].to_numpy(dtype=bool),
        (panel["rv_5d_fwd"].to_numpy(dtype=float) >= rv90_thresh).astype(int),
        np.nan,
    )
    panel["y_tail_rv90"] = pd.Series(y_tail_rv90_arr, index=panel.index, dtype="Int8")

    panel = (
        panel[list(PANEL_DAILY_COLUMNS)]
        .sort_values(["ticker", "date"], kind="mergesort")
        .reset_index(drop=True)
    )
    return panel, train_params
