"""The tier × month panel (plan section 2.2), built from the raw snapshots only."""

from __future__ import annotations

import calendar

import numpy as np
import pandas as pd

from macao_tourism.config import (
    MAIN_SAMPLE_PERIODS,
    PANEL_END,
    PANEL_START,
    TIERS,
    period_label,
)
from macao_tourism.ingest import dsec, holidays, smg

DERIVED_DECIMALS = 10

MONTH_COLUMNS: tuple[str, ...] = (
    "period",
    "year",
    "month",
    "days_in_month",
    "period_label",
    "post",
    "in_main_sample",
    "visitors_total",
    "visitors_overnight",
    "visitors_same_day",
    "visitors_mainland",
    "visitors_hong_kong",
    "visitors_taiwan",
    "package_tour_visitors",
    "log_visitors_overnight",
    "same_day_share_pct",
    "international_share_pct",
    "package_tour_share_pct",
    "cn_holiday_days",
    "cn_golden_week_days",
    "cn_holiday_days_at_origin",
    "cn_golden_week_days_at_origin",
    "typhoon_days",
)
TIER_COLUMNS: tuple[str, ...] = (
    "occupancy_pct",
    "rooms",
    "log_rooms",
    "establishments",
    "guests",
)
PANEL_COLUMNS: tuple[str, ...] = ("tier", *MONTH_COLUMNS[:1], *TIER_COLUMNS, *MONTH_COLUMNS[1:])
COUNT_COLUMNS: tuple[str, ...] = (
    "visitors_total",
    "visitors_overnight",
    "visitors_same_day",
    "visitors_mainland",
    "visitors_hong_kong",
    "visitors_taiwan",
    "package_tour_visitors",
)


def panel_months() -> list[str]:
    return [str(m) for m in pd.period_range(PANEL_START, PANEL_END, freq="M")]


def logit_pct(occupancy_pct: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    """Logit of an occupancy rate given in percent. Defined for values strictly in (0, 100)."""
    share = np.asarray(occupancy_pct, dtype=float) / 100.0
    if np.any((share <= 0.0) | (share >= 1.0)):
        raise ValueError("logit is undefined for occupancy of 0% or 100%")
    result = np.log(share / (1.0 - share))
    return _like(occupancy_pct, result)


def inv_logit_pct(value: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    """Back-transform a logit to an occupancy rate in percent."""
    result = 100.0 / (1.0 + np.exp(-np.asarray(value, dtype=float)))
    return _like(value, result)


def _like(template: object, result: np.ndarray) -> pd.Series | np.ndarray | float:
    if isinstance(template, pd.Series):
        return pd.Series(result, index=template.index, name=template.name)
    return float(result) if np.ndim(result) == 0 else result


def build_months() -> pd.DataFrame:
    """One row per month of the panel window with the month-level variables."""
    months = panel_months()
    dsec.verify_registry(dsec.load_indicator_meta())
    long = dsec.load_tsdb_long()
    wide = long.pivot(index="period", columns="key", values="value").reindex(months)

    frame = pd.DataFrame({"period": months})
    frame["year"] = [int(m[:4]) for m in months]
    frame["month"] = [int(m[5:7]) for m in months]
    frame["days_in_month"] = [
        calendar.monthrange(y, m)[1] for y, m in zip(frame["year"], frame["month"], strict=True)
    ]
    frame["period_label"] = [period_label(m) for m in months]
    frame["post"] = (frame["period_label"] == "post_pandemic").astype(int)
    frame["in_main_sample"] = frame["period_label"].isin(MAIN_SAMPLE_PERIODS).astype(int)
    for column in COUNT_COLUMNS:
        frame[column] = wide[column].to_numpy()

    total = frame["visitors_total"]
    greater_china = frame[["visitors_mainland", "visitors_hong_kong", "visitors_taiwan"]].sum(
        axis=1
    )
    frame["log_visitors_overnight"] = np.log(frame["visitors_overnight"])
    frame["same_day_share_pct"] = 100.0 * frame["visitors_same_day"] / total
    frame["international_share_pct"] = 100.0 * (1.0 - greater_china / total)
    frame["package_tour_share_pct"] = 100.0 * frame["package_tour_visitors"] / total

    periods = holidays.load_holiday_periods()
    holiday_days = holidays.monthly_holiday_days(periods, PANEL_START, PANEL_END)
    frame["cn_holiday_days"] = holiday_days["cn_holiday_days"].to_numpy()
    frame["cn_golden_week_days"] = holiday_days["cn_golden_week_days"].to_numpy()
    at_origin = holidays.monthly_holiday_days_at_origin(
        holidays.load_schedule_versions(), PANEL_START, PANEL_END
    )
    frame["cn_holiday_days_at_origin"] = at_origin["cn_holiday_days_at_origin"].to_numpy()
    frame["cn_golden_week_days_at_origin"] = at_origin["cn_golden_week_days_at_origin"].to_numpy()

    events = smg.load_signal_events()
    daily = smg.daily_signals(smg.signal_intervals(events))
    signals = smg.monthly_signal_days(daily, PANEL_START, PANEL_END)
    frame["typhoon_days"] = signals["typhoon_days_signal8plus"].to_numpy()

    for column in COUNT_COLUMNS:
        frame[column] = _as_int(frame[column], column)
    derived = [
        "log_visitors_overnight",
        "same_day_share_pct",
        "international_share_pct",
        "package_tour_share_pct",
    ]
    frame[derived] = frame[derived].round(DERIVED_DECIMALS)
    return frame[list(MONTH_COLUMNS)]


def build_panel(months: pd.DataFrame) -> pd.DataFrame:
    """One row per star class and month: the tier variables next to the month variables."""
    long = dsec.load_tsdb_long()
    wide = long.pivot(index="period", columns="key", values="value").reindex(months["period"])
    blocks = []
    for tier in TIERS:
        block = months.copy()
        block.insert(0, "tier", tier)
        block["occupancy_pct"] = wide[f"occupancy_{tier}"].to_numpy()
        block["rooms"] = _as_int(wide[f"rooms_{tier}"].reset_index(drop=True), f"rooms_{tier}")
        block["log_rooms"] = np.log(block["rooms"]).round(DERIVED_DECIMALS)
        block["establishments"] = _as_int(
            wide[f"establishments_{tier}"].reset_index(drop=True), f"establishments_{tier}"
        )
        block["guests"] = _as_int(wide[f"guests_{tier}"].reset_index(drop=True), f"guests_{tier}")
        blocks.append(block)
    panel = pd.concat(blocks, ignore_index=True)
    panel["_tier_order"] = panel["tier"].map({tier: k for k, tier in enumerate(TIERS)})
    panel = panel.sort_values(["period", "_tier_order"]).drop(columns="_tier_order")
    return panel[list(PANEL_COLUMNS)].reset_index(drop=True)


def _as_int(series: pd.Series, name: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{name} has missing months in the panel window")
    if not np.allclose(series, series.round()):
        raise ValueError(f"{name} is not a whole number in every month")
    return series.round().astype("int64")


def validate_panel(months: pd.DataFrame, panel: pd.DataFrame) -> None:
    """Stop if the panel is not the complete, consistent table the analysis assumes."""
    expected_months = panel_months()
    if months["period"].tolist() != expected_months:
        raise ValueError("Month table does not cover the panel window exactly once")
    if len(panel) != len(expected_months) * len(TIERS):
        raise ValueError("Panel is not balanced")
    if panel.duplicated(["tier", "period"]).any():
        raise ValueError("Duplicate (tier, period) rows in the panel")
    for name, frame in (("months", months), ("panel", panel)):
        missing = frame.columns[frame.isna().any()].tolist()
        if missing:
            raise ValueError(f"Missing values in {name}: {missing}")
    if not (months["visitors_same_day"] + months["visitors_overnight"]).equals(
        months["visitors_total"]
    ):
        raise ValueError("Same-day plus overnight visitors does not equal total visitors")
    for column in ("same_day_share_pct", "international_share_pct", "package_tour_share_pct"):
        if not months[column].between(0.0, 100.0).all():
            raise ValueError(f"{column} is outside [0, 100]")
    if not ((panel["occupancy_pct"] > 0.0) & (panel["occupancy_pct"] < 100.0)).all():
        raise ValueError("Occupancy must lie strictly between 0 and 100 for the logit analysis")
    if (panel["rooms"] <= 0).any() or (months["visitors_overnight"] <= 0).any():
        raise ValueError("Logarithm of a non-positive count")
    if (months["cn_holiday_days_at_origin"] > months["cn_holiday_days"]).any():
        raise ValueError("More holiday days known ahead than in the final schedule")
    if (months["cn_holiday_days"] > months["days_in_month"]).any() or (
        months["typhoon_days"] > months["days_in_month"]
    ).any():
        raise ValueError("More holiday or typhoon days than days in the month")


def variable_summary(months: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Range and completeness of every numeric variable over the whole window.

    Deliberately not split by period or star class: the freeze is written before any
    comparison the hypotheses are about.
    """
    rows = []
    for level, frame, columns in (
        ("tier_month", panel, TIER_COLUMNS),
        ("month", months, [c for c in MONTH_COLUMNS if c not in ("period", "period_label")]),
    ):
        for column in columns:
            values = frame[column].astype(float)
            rows.append(
                {
                    "variable": column,
                    "level": level,
                    "n": int(values.notna().sum()),
                    "missing": int(values.isna().sum()),
                    "min": round(float(values.min()), 4),
                    "mean": round(float(values.mean()), 4),
                    "max": round(float(values.max()), 4),
                }
            )
    return pd.DataFrame(rows)
