"""Purged and 5-day embargoed chronological split assignment and leakage verification."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from stock_volatility import config

SPLIT_ORDER: tuple[str, ...] = ("train", "embargo_1", "val", "embargo_2", "test")


def assign_split_label(
    date_str: str,
    *,
    start_date: str = config.START_DATE,
    train_end: str = config.TRAIN_END,
    val_start: str = config.VAL_START,
    val_end: str = config.VAL_END,
    test_start: str = config.TEST_START,
    test_end: str = config.TEST_END,
) -> str:
    """Assign an ISO date string to its pre-registered purged+embargoed split."""
    if date_str < start_date or date_str > test_end:
        raise ValueError(f"Date {date_str!r} outside [{start_date}, {test_end}]")
    if date_str <= train_end:
        return "train"
    if date_str < val_start:
        return "embargo_1"
    if date_str <= val_end:
        return "val"
    if date_str < test_start:
        return "embargo_2"
    return "test"


def assign_purged_splits(trading_dates: Sequence[str]) -> pd.DataFrame:
    """Return a sorted DataFrame `[date, split]` covering the canonical NYSE calendar."""
    cal = sorted({str(d) for d in trading_dates})
    if not cal:
        raise ValueError("trading_dates cannot be empty")
    return pd.DataFrame(
        {
            "date": cal,
            "split": [assign_split_label(d) for d in cal],
        }
    )


def validate_purged_embargo_splits(
    panel_df: pd.DataFrame,
    trading_dates: Sequence[str],
    *,
    horizon_days: int = config.HORIZON_DAYS,
) -> dict[str, Any]:
    """Formally verify zero target-window overlap across Train, Val, and Test boundaries."""
    cal = sorted({str(d) for d in trading_dates})
    date_to_idx = {d: i for i, d in enumerate(cal)}

    split_by_date = (
        panel_df[["date", "split"]]
        .drop_duplicates()
        .sort_values("date", kind="mergesort")
        .reset_index(drop=True)
    )
    if split_by_date["date"].duplicated().any():
        raise ValueError("Conflicting split labels found for the same date")

    splits_present = set(split_by_date["split"])
    if splits_present != set(SPLIT_ORDER):
        raise ValueError(f"Expected splits {set(SPLIT_ORDER)}, got {splits_present}")

    def _dates_for(name: str) -> list[str]:
        return split_by_date.loc[split_by_date["split"] == name, "date"].tolist()

    train_dates = _dates_for("train")
    emb1_dates = _dates_for("embargo_1")
    val_dates = _dates_for("val")
    emb2_dates = _dates_for("embargo_2")
    test_dates = _dates_for("test")

    if len(emb1_dates) < horizon_days:
        raise ValueError(
            f"embargo_1 has {len(emb1_dates)} trading days, need >= {horizon_days}"
        )
    if len(emb2_dates) < horizon_days:
        raise ValueError(
            f"embargo_2 has {len(emb2_dates)} trading days, need >= {horizon_days}"
        )

    # Verify last Train date's t+horizon_days target strictly precedes VAL_START
    train_last_idx = date_to_idx[train_dates[-1]]
    train_last_target_date = cal[train_last_idx + horizon_days]
    if train_last_target_date >= val_dates[0]:
        raise ValueError(
            f"Train target window leaks into Val: {train_last_target_date} >= {val_dates[0]}"
        )

    # Verify last Val date's t+horizon_days target strictly precedes TEST_START
    val_last_idx = date_to_idx[val_dates[-1]]
    val_last_target_date = cal[val_last_idx + horizon_days]
    if val_last_target_date >= test_dates[0]:
        raise ValueError(
            f"Val target window leaks into Test: {val_last_target_date} >= {test_dates[0]}"
        )

    # Compute per-split day and row breakdowns
    split_summary: dict[str, dict[str, Any]] = {}
    for name in SPLIT_ORDER:
        sub = panel_df[panel_df["split"] == name]
        full_feat = sub[sub["has_full_features"]]
        full_both = sub[sub["has_full_features"] & sub["has_full_target"]]
        sub_dates = sorted(sub["date"].unique().tolist())
        both_dates = sorted(full_both["date"].unique().tolist())
        split_summary[name] = {
            "start_date": sub_dates[0],
            "end_date": sub_dates[-1],
            "trading_days": len(sub_dates),
            "total_rows": int(len(sub)),
            "feature_complete_trading_days": int(full_feat["date"].nunique()),
            "feature_complete_rows": int(len(full_feat)),
            "modeling_complete_trading_days": len(both_dates),
            "modeling_complete_rows": int(len(full_both)),
            "modeling_start_date": both_dates[0] if both_dates else None,
            "modeling_end_date": both_dates[-1] if both_dates else None,
        }

    test_scorable_days = split_summary["test"]["modeling_complete_trading_days"]
    test_incomplete_tail_days = split_summary["test"]["trading_days"] - test_scorable_days
    if test_incomplete_tail_days != horizon_days:
        raise ValueError(
            f"Expected {horizon_days} incomplete tail test days, got {test_incomplete_tail_days}"
        )

    return {
        "schema_version": "1.0.0",
        "horizon_days": horizon_days,
        "total_canonical_trading_days": len(cal),
        "total_panel_rows": int(len(panel_df)),
        "embargo_1_trading_days": len(emb1_dates),
        "embargo_2_trading_days": len(emb2_dates),
        "train_last_date": train_dates[-1],
        "train_last_target_date": train_last_target_date,
        "val_first_date": val_dates[0],
        "val_last_date": val_dates[-1],
        "val_last_target_date": val_last_target_date,
        "test_first_date": test_dates[0],
        "test_last_date": test_dates[-1],
        "test_scorable_trading_days": test_scorable_days,
        "test_incomplete_tail_days": test_incomplete_tail_days,
        "zero_target_overlap_verified": True,
        "splits": split_summary,
    }
