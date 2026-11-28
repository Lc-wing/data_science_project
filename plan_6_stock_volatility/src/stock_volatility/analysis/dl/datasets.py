"""Frozen sequence/text dataset loader, 60-day PatchTST windows & zero-leakage prompt builder."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_volatility import config
from stock_volatility.panel.freeze import verify_freeze_manifest

# 14 strictly <= t continuous predictors used by multivariate PatchTST (standardized on Train only)
DL_CONT_FEATURES: tuple[str, ...] = (
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

TICKER_TO_ID: dict[str, int] = {t: i for i, t in enumerate(config.UNIVERSE_TICKERS)}
SECTOR_TO_ID: dict[str, int] = {s: i for i, s in enumerate(config.SECTORS)}


def load_frozen_dl_inputs(project_root: Path | None = None) -> dict[str, Any]:
    """Verify `FREEZE_MANIFEST.json` SHA-256 hashes and load frozen panel + event-text inputs."""
    root = Path(project_root) if project_root is not None else config.PROJECT_ROOT
    frozen_dir = root / "data" / "frozen"
    manifest = verify_freeze_manifest(frozen_dir=frozen_dir, project_root=root)

    panel_df = pd.read_parquet(frozen_dir / "panel_daily.parquet")
    events_df = pd.read_parquet(frozen_dir / "events_text_phase_b.parquet")
    feature_manifest = json.loads((frozen_dir / "feature_manifest.json").read_text("utf-8"))
    split_metadata = json.loads((frozen_dir / "split_metadata.json").read_text("utf-8"))
    analysis_spec = json.loads((frozen_dir / "analysis_spec.json").read_text("utf-8"))

    return {
        "project_root": root,
        "manifest": manifest,
        "panel_df": panel_df,
        "events_df": events_df,
        "feature_manifest": feature_manifest,
        "split_metadata": split_metadata,
        "analysis_spec": analysis_spec,
    }


def build_event_anchored_datasets(
    panel_df: pd.DataFrame,
    events_phase_b_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build Train/Val/Test event-anchored tables (`has_news == 1`, modeling-complete only).

    Embargo days (`embargo_1`, `embargo_2`), warmup rows (`has_full_features == False`), and
    incomplete final-5-day targets (`has_full_target == False`) are strictly excluded.
    """
    merged = panel_df.merge(
        events_phase_b_df[
            ["ticker", "date", "forms", "item_codes", "titles_capped_5", "text_concat_capped_5"]
        ],
        on=["ticker", "date"],
        how="inner",
        validate="one_to_one",
    )
    active = merged[
        (merged["has_news"] == 1) & merged["has_full_features"] & merged["has_full_target"]
    ].copy()

    splits: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        sub = (
            active[active["split"] == split_name]
            .sort_values(["date", "ticker"], kind="mergesort")
            .reset_index(drop=True)
        )
        splits[split_name] = sub
    return splits


def build_event_prompt(
    row: Mapping[str, Any],
    *,
    max_event_chars: int = 1100,
) -> str:
    """Build a strictly `<= t` corporate-event risk brief ending in a single-token YES/NO query.

    Uses ONLY pre-close (`< 16:00 ET` / `< 13:00 ET` early-close) SEC EDGAR filings and strictly
    `<= t` daily market indicators. Never accesses forward target columns (`rv_5d_fwd`, `y_log_rv`,
    `cum_ret_5d_fwd`, `y_tail`, `y_tail_rv90`).
    """
    ticker = str(row["ticker"])
    sector = str(row["sector"])
    date_str = str(row["date"])
    ret_1d = float(row["ret_1d"]) * 100.0
    ret_5d = float(row["ret_5d"]) * 100.0
    rng_pct = float(row["daily_range_pct"]) * 100.0
    rv_d = float(row["rv_d"])
    rv_w = float(row["rv_w"])
    rv_m = float(row["rv_m"])
    vix = float(row["vix_close"])
    spread = float(row["t10y2y"])
    n_cnt = int(row.get("news_count", 0))
    has_earn = int(row.get("has_8k_earnings", 0))
    forms = str(row.get("forms", "") or "")
    items = str(row.get("item_codes", "") or "")
    event_text = str(row.get("text_concat_capped_5", "") or "").strip()
    if len(event_text) > max_event_chars:
        event_text = event_text[: max_event_chars - 3].rstrip() + "..."

    return (
        "You are an academic financial risk classifier evaluating U.S. equity downside tail risk.\n"
        f"Stock: {ticker} ({sector}) | Trading Date (t): {date_str}\n"
        f"Market State (<= t close): 1d_ret={ret_1d:+.2f}%, 5d_ret={ret_5d:+.2f}%, "
        f"range={rng_pct:.2f}%, RV_d={rv_d:.3f}, RV_w={rv_w:.3f}, RV_m={rv_m:.3f}, "
        f"VIX={vix:.2f}, 10Y-2Y={spread:+.2f}%\n"
        f"Pre-Close SEC EDGAR Disclosures (count={n_cnt}, earnings_8k_202={has_earn}, "
        f"forms={forms}, items={items}):\n"
        f"{event_text}\n"
        "Question: Based strictly on information through date t close, will the stock experience "
        "a 5-day forward downside tail-risk drop (< -5.0% cumulative return over t+1..t+5)?\n"
        "Answer YES or NO:"
    )


def build_patchtst_windows(
    panel_df: pd.DataFrame,
    feature_manifest: dict[str, Any],
    *,
    lookback: int = 60,
    cont_features: tuple[str, ...] = DL_CONT_FEATURES,
) -> dict[str, dict[str, np.ndarray]]:
    """Construct strictly `<= t` lookback windows standardized ONLY with Train statistics.

    Each window for `(ticker, t)` covers trading indices `[idx - lookback + 1 .. idx]` strictly
    within the same ticker's continuous NYSE history, using Train Winsorization bounds and Train
    mean/std from `feature_manifest.json` (zero Val/Test leakage).
    """
    if lookback < 5:
        raise ValueError(f"lookback must be >= 5, got {lookback}")

    cont_stats = feature_manifest["train_preprocessing"]["continuous_features"]
    p005 = np.array([float(cont_stats[c]["p005"]) for c in cont_features], dtype=np.float32)
    p995 = np.array([float(cont_stats[c]["p995"]) for c in cont_features], dtype=np.float32)
    mean = np.array([float(cont_stats[c]["winsor_mean"]) for c in cont_features], dtype=np.float32)
    std = np.maximum(
        np.array([float(cont_stats[c]["winsor_std"]) for c in cont_features], dtype=np.float32),
        1e-6,
    )

    buckets: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for ticker, grp in panel_df.groupby("ticker", sort=True):
        g = grp.sort_values("date", kind="mergesort").reset_index(drop=True)
        raw_mat = g[list(cont_features)].to_numpy(dtype=np.float32)
        # Forward-fill warmup NaNs inside a single stock (only affects first ~21 days of 2019),
        # then clip & standardize strictly using Train-only Winsorization & mean/std.
        valid_mask = np.isfinite(raw_mat)
        norm_mat = np.where(valid_mask, np.clip(raw_mat, p005, p995), mean)
        norm_mat = (norm_mat - mean) / std

        tid = TICKER_TO_ID[str(ticker)]
        sid = SECTOR_TO_ID[str(g["sector"].iloc[0])]
        splits = g["split"].to_numpy()
        has_feat = g["has_full_features"].to_numpy(dtype=bool)
        has_tgt = g["has_full_target"].to_numpy(dtype=bool)
        y_log = g["y_log_rv"].to_numpy(dtype=np.float32)
        rv_fwd = g["rv_5d_fwd"].to_numpy(dtype=np.float32)
        y_tail = g["y_tail"].fillna(0).to_numpy(dtype=np.int64)
        dates = g["date"].astype(str).to_numpy()

        for idx in range(lookback - 1, len(g)):
            sp = str(splits[idx])
            if sp not in buckets:
                continue
            if not (has_feat[idx] and has_tgt[idx]):
                continue
            # Require all 60 days in window to have complete features (strictly >= warmup)
            if not has_feat[idx - lookback + 1 : idx + 1].all():
                continue
            buckets[sp].append(
                {
                    "x_seq": norm_mat[idx - lookback + 1 : idx + 1],
                    "ticker_id": tid,
                    "sector_id": sid,
                    "y_log_rv": y_log[idx],
                    "rv_5d_fwd": rv_fwd[idx],
                    "y_tail": y_tail[idx],
                    "ticker": str(ticker),
                    "date": str(dates[idx]),
                }
            )

    out: dict[str, dict[str, np.ndarray]] = {}
    for sp, items in buckets.items():
        if not items:
            out[sp] = {
                "x_seq": np.empty((0, lookback, len(cont_features)), dtype=np.float32),
                "ticker_id": np.empty((0,), dtype=np.int64),
                "sector_id": np.empty((0,), dtype=np.int64),
                "y_log_rv": np.empty((0,), dtype=np.float32),
                "rv_5d_fwd": np.empty((0,), dtype=np.float32),
                "y_tail": np.empty((0,), dtype=np.int64),
                "tickers": np.empty((0,), dtype=object),
                "dates": np.empty((0,), dtype=object),
            }
            continue
        out[sp] = {
            "x_seq": np.stack([it["x_seq"] for it in items], axis=0).astype(np.float32),
            "ticker_id": np.array([it["ticker_id"] for it in items], dtype=np.int64),
            "sector_id": np.array([it["sector_id"] for it in items], dtype=np.int64),
            "y_log_rv": np.array([it["y_log_rv"] for it in items], dtype=np.float32),
            "rv_5d_fwd": np.array([it["rv_5d_fwd"] for it in items], dtype=np.float32),
            "y_tail": np.array([it["y_tail"] for it in items], dtype=np.int64),
            "tickers": np.array([it["ticker"] for it in items], dtype=object),
            "dates": np.array([it["date"] for it in items], dtype=object),
        }
    return out
