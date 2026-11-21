"""Exploratory description of the frozen panel: STL, change points, and period summaries.

Nothing computed here enters a regression or a forecast. The settings below that the
analysis specification leaves open were fixed before the first run and were not tuned.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import ruptures
from statsmodels.tsa.seasonal import STL

from macao_tourism.config import MAIN_SAMPLE_PERIODS, TIERS

STL_DEFAULT_SEASONAL_WINDOW = 7
STL_PERIODIC_WINDOW = 100_001
STL_BLOCKS: tuple[str, ...] = MAIN_SAMPLE_PERIODS
PELT_COST = "l2"
PELT_MIN_SEGMENT_MONTHS = 12
PELT_MAIN_MULTIPLIER = 1.0
PELT_PENALTY_MULTIPLIERS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
MAD_TO_SD = 1.4826
SUMMARY_PERIODS: tuple[str, ...] = ("pre_pandemic", "pandemic", "post_pandemic")
DECIMALS = 6


def stl_fit(values: np.ndarray, period: int, robust: bool, periodic: bool = False) -> Any:
    """STL fit; `periodic` holds the seasonal pattern constant over the years of the series."""
    if periodic:
        return STL(
            values, period=period, robust=robust, seasonal=STL_PERIODIC_WINDOW, seasonal_deg=0
        ).fit()
    return STL(values, period=period, robust=robust, seasonal=STL_DEFAULT_SEASONAL_WINDOW).fit()


def stl_components(panel: pd.DataFrame, period: int, robust: bool) -> pd.DataFrame:
    """Trend, seasonal, and remainder of occupancy per star class.

    One fit per contiguous block of the main sample (pre-pandemic, post-pandemic), with a
    seasonal pattern that is constant inside the block. The pandemic months are not
    decomposed: a single fit across them distorts the months next to them.
    """
    parts = []
    for tier in TIERS:
        for block in STL_BLOCKS:
            rows = panel[(panel["tier"] == tier) & (panel["period_label"] == block)]
            rows = rows.sort_values("period")
            fit = stl_fit(rows["occupancy_pct"].to_numpy(dtype=float), period, robust, True)
            parts.append(
                pd.DataFrame(
                    {
                        "tier": tier,
                        "block": block,
                        "period": rows["period"].to_numpy(),
                        "month": rows["month"].to_numpy(),
                        "occupancy_pct": rows["occupancy_pct"].to_numpy(dtype=float),
                        "trend": fit.trend,
                        "seasonal": fit.seasonal,
                        "remainder": fit.resid,
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)


def _strength(component: pd.Series, remainder: pd.Series) -> float:
    """1 − Var(remainder) / Var(component + remainder), floored at 0."""
    return max(0.0, 1.0 - remainder.var(ddof=1) / (component + remainder).var(ddof=1))


def stl_summary(components: pd.DataFrame) -> pd.DataFrame:
    """Strength of trend and seasonality, and the seasonal profile, per star class and block."""
    rows = []
    for (tier, block), part in components.groupby(["tier", "block"], sort=False):
        profile = part.groupby("month")["seasonal"].mean()
        rows.append(
            {
                "tier": tier,
                "block": block,
                "months": len(part),
                "trend_strength": _strength(part["trend"], part["remainder"]),
                "seasonal_strength": _strength(part["seasonal"], part["remainder"]),
                "seasonal_peak_month": int(profile.idxmax()),
                "seasonal_trough_month": int(profile.idxmin()),
                "seasonal_range_pp": float(profile.max() - profile.min()),
                "remainder_sd_pp": float(part["remainder"].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows).round(DECIMALS)


def seasonal_profiles(components: pd.DataFrame) -> pd.DataFrame:
    """Mean seasonal component by calendar month, per star class and block."""
    profile = components.groupby(["tier", "block", "month"], sort=False)["seasonal"].mean()
    return profile.rename("seasonal_pp").reset_index().round(DECIMALS)


def stl_full_series_check(panel: pd.DataFrame, period: int, robust: bool) -> pd.DataFrame:
    """What one STL fit over all months, pandemic included, does to the months around it.

    This was the first run of the step. It is kept as a check and is the reason the
    decomposition is fitted per block.
    """
    rows = []
    for tier in TIERS:
        part = panel[panel["tier"] == tier].sort_values("period").reset_index(drop=True)
        fit = stl_fit(part["occupancy_pct"].to_numpy(dtype=float), period, robust)
        seasonal = pd.Series(np.abs(fit.seasonal))
        post = part["period_label"] == "post_pandemic"
        worst = seasonal[post].idxmax()
        rows.append(
            {
                "tier": tier,
                "max_abs_seasonal_pre_pandemic_pp": float(
                    seasonal[part["period_label"] == "pre_pandemic"].max()
                ),
                "max_abs_seasonal_post_pandemic_pp": float(seasonal[post].max()),
                "month_of_post_pandemic_max": part["period"].iloc[worst],
                "occupancy_pct_that_month": float(part["occupancy_pct"].iloc[worst]),
                "seasonal_that_month_pp": float(fit.seasonal[worst]),
                "remainder_that_month_pp": float(fit.resid[worst]),
            }
        )
    return pd.DataFrame(rows).round(DECIMALS)


def seasonally_adjusted(values: np.ndarray, period: int, robust: bool) -> np.ndarray:
    return values - stl_fit(values, period, robust).seasonal


def noise_sd(values: np.ndarray, period: int) -> float:
    """Robust scale of the series around its level: MAD of the seasonal differences / √2."""
    diff = values[period:] - values[:-period]
    mad = float(np.median(np.abs(diff - np.median(diff))))
    return MAD_TO_SD * mad / math.sqrt(2)


def pelt_penalty(values: np.ndarray, period: int, multiplier: float) -> float:
    return multiplier * 2.0 * noise_sd(values, period) ** 2 * math.log(len(values))


def pelt_breaks(values: np.ndarray, penalty: float) -> list[int]:
    """Positions where a new segment starts, for a piecewise-constant mean."""
    algo = ruptures.Pelt(model=PELT_COST, min_size=PELT_MIN_SEGMENT_MONTHS, jump=1)
    return [int(i) for i in algo.fit(values.reshape(-1, 1)).predict(pen=penalty)[:-1]]


def _run(name: str, values: np.ndarray, months: pd.Series, period: int, multiplier: float):
    penalty = pelt_penalty(values, period, multiplier)
    breaks = pelt_breaks(values, penalty)
    return {
        "series": name,
        "penalty_multiplier": multiplier,
        "noise_sd_pp": noise_sd(values, period),
        "penalty": penalty,
        "n_change_points": len(breaks),
        "change_points": ";".join(months.iloc[breaks]),
    }


def change_point_runs(
    months: pd.DataFrame, series: list[str], period: int, robust: bool
) -> pd.DataFrame:
    """PELT on each share series: seasonally adjusted at every penalty, and unadjusted once.

    The main run is the seasonally adjusted series at the main penalty multiplier.
    """
    ordered = months.sort_values("period").reset_index(drop=True)
    rows = []
    for name in series:
        raw = ordered[name].to_numpy(dtype=float)
        adjusted = seasonally_adjusted(raw, period, robust)
        for multiplier in PELT_PENALTY_MULTIPLIERS:
            run = _run(name, adjusted, ordered["period"], period, multiplier)
            rows.append(
                {"input": "seasonally_adjusted", "main": multiplier == PELT_MAIN_MULTIPLIER, **run}
            )
        run = _run(name, raw, ordered["period"], period, PELT_MAIN_MULTIPLIER)
        rows.append({"input": "unadjusted", "main": False, **run})
    return pd.DataFrame(rows).round(DECIMALS)


def segments(months: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Segments of each share series under the main penalty, with the mean share in each."""
    ordered = months.sort_values("period").reset_index(drop=True)
    rows = []
    for run in runs[runs["main"]].itertuples():
        starts = [m for m in str(run.change_points).split(";") if m]
        first_months = [ordered["period"].iloc[0], *starts]
        for number, first in enumerate(first_months, start=1):
            later = first_months[number] if number < len(first_months) else None
            inside = ordered[
                (ordered["period"] >= first) & ((ordered["period"] < later) if later else True)
            ]
            rows.append(
                {
                    "series": run.series,
                    "segment": number,
                    "first_month": first,
                    "last_month": inside["period"].iloc[-1],
                    "months": len(inside),
                    "mean_share_pct": float(inside[run.series].mean()),
                    "starts_in_period": inside["period_label"].iloc[0],
                }
            )
    return pd.DataFrame(rows).round(DECIMALS)


def visitor_mix_by_period(months: pd.DataFrame, series: list[str]) -> pd.DataFrame:
    """Mean of each share, and mean monthly arrivals, in each period of the plan."""
    rows = []
    for label in SUMMARY_PERIODS:
        inside = months[months["period_label"] == label]
        row: dict[str, Any] = {"period_label": label, "months": len(inside)}
        row.update({f"mean_{name}": float(inside[name].mean()) for name in series})
        row["mean_visitors_total"] = float(inside["visitors_total"].mean())
        row["mean_visitors_overnight"] = float(inside["visitors_overnight"].mean())
        rows.append(row)
    return pd.DataFrame(rows).round(DECIMALS)


def occupancy_by_period(panel: pd.DataFrame) -> pd.DataFrame:
    """Occupancy per star class and period, with its correlation with log overnight visitors.

    The correlation is a description of the scatter plot. No test is attached to it; the
    confirmatory tests are those of the regression step.
    """
    rows = []
    for tier in TIERS:
        for label in SUMMARY_PERIODS:
            inside = panel[(panel["tier"] == tier) & (panel["period_label"] == label)]
            occupancy = inside["occupancy_pct"]
            rows.append(
                {
                    "tier": tier,
                    "period_label": label,
                    "months": len(inside),
                    "occupancy_mean": float(occupancy.mean()),
                    "occupancy_sd": float(occupancy.std(ddof=1)),
                    "occupancy_min": float(occupancy.min()),
                    "occupancy_max": float(occupancy.max()),
                    "pearson_r_log_overnight": float(
                        occupancy.corr(inside["log_visitors_overnight"])
                    ),
                }
            )
    return pd.DataFrame(rows).round(DECIMALS)


def settings(period: int, robust: bool) -> dict[str, Any]:
    return {
        "stl_occupancy": {
            "period": period,
            "robust": robust,
            "fits": "one fit per star class and block (pre-pandemic, post-pandemic)",
            "seasonal": "periodic (constant inside a block)",
            "other": "statsmodels defaults",
            "pandemic_months": "not decomposed",
        },
        "pelt": {
            "input": (
                "share in percent minus its STL seasonal component (one fit over all months, "
                f"period {period}, robust, seasonal window {STL_DEFAULT_SEASONAL_WINDOW})"
            ),
            "cost": f"{PELT_COST} (piecewise-constant mean)",
            "min_segment_months": PELT_MIN_SEGMENT_MONTHS,
            "penalty": "multiplier × 2 × noise_sd² × ln(n)",
            "noise_sd": "1.4826 × MAD of the 12-month differences / √2",
            "main_multiplier": PELT_MAIN_MULTIPLIER,
            "multipliers_reported": list(PELT_PENALTY_MULTIPLIERS),
            "also_reported": "the same run on the shares without seasonal adjustment",
        },
        "left_open_by_the_specification": (
            "The specification names STL (period 12, robust) on occupancy and PELT on the "
            "three shares. The months of each STL fit, the seasonal window, the seasonal "
            "adjustment before PELT, the cost, the minimum segment length, and the penalty "
            "are choices of this step."
        ),
        "changed_after_the_first_run": [
            (
                "STL on occupancy was first fitted once per star class over all 201 months "
                f"with seasonal window {STL_DEFAULT_SEASONAL_WINDOW}. That fit pushed the "
                "pandemic collapse into the seasonal component and the remainder of the "
                "post-pandemic months (see eda_stl_full_series_check.csv). It was replaced by "
                "one fit per block with a periodic seasonal pattern."
            ),
            (
                "The PELT settings were not changed. A run on the shares without seasonal "
                "adjustment was added to the sensitivity table."
            ),
        ],
    }
