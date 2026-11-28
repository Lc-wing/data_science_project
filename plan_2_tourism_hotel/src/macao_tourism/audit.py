"""Day-1 data audit (plan section 2.1): coverage, definitions, and cross-checks."""

from __future__ import annotations

from typing import Any

import pandas as pd

from macao_tourism.config import (
    INDICATORS,
    PANEL_END,
    PANEL_START,
    PERIODS,
    TIERS,
    period_label,
)

__all__ = ["PERIODS", "period_label"]

TIER_MEASURES: tuple[str, ...] = ("occupancy", "rooms", "establishments", "guests")


def panel_months() -> list[str]:
    return [str(m) for m in pd.period_range(PANEL_START, PANEL_END, freq="M")]


def wide(long: pd.DataFrame) -> pd.DataFrame:
    return long.pivot(index="period", columns="key", values="value").sort_index()


def coverage_table(long: pd.DataFrame, registry: list[dict[str, Any]]) -> pd.DataFrame:
    """First and last month, and gaps inside the panel window, for every series."""
    months = panel_months()
    info = {row["key"]: row for row in registry}
    rows = []
    for ind in INDICATORS:
        series = long[long["key"] == ind.key]
        valued = series.dropna(subset=["value"])
        in_window = valued[valued["period"].isin(months)]
        missing = sorted(set(months) - set(in_window["period"]))
        rows.append(
            {
                "key": ind.key,
                "indicator_id": ind.indicator_id,
                "dsec_path": info[ind.key]["path_en"],
                "unit": ind.unit_en,
                "first_month": valued["period"].min(),
                "last_month": valued["period"].max(),
                "months_with_value": int(len(valued)),
                "panel_months_with_value": int(len(in_window)),
                "panel_months_missing": len(missing),
                "missing_span": f"{missing[0]}..{missing[-1]}" if missing else "",
                "zero_by_remark": int(series["zero_by_remark"].sum()),
                "revised_flag_months": int(series["revised"].sum()),
                "latest_dsec_update": series["dsec_last_update"].max(),
            }
        )
    return pd.DataFrame(rows)


def _max_abs_gap(frame: pd.DataFrame, parts: list[str], total: str) -> dict[str, Any]:
    both = frame[[*parts, total]].dropna()
    gap = (both[parts].sum(axis=1) - both[total]).abs()
    return {
        "months_compared": int(len(both)),
        "months_exact": int((gap == 0).sum()),
        "max_abs_gap": float(gap.max()),
        "month_of_max_gap": str(gap.idxmax()),
    }


def identity_checks(long: pd.DataFrame) -> dict[str, Any]:
    """Accounting identities that hold if the right series were downloaded."""
    frame = wide(long).reindex(panel_months())
    tiers = list(TIERS)
    checks = {
        "same_day_plus_overnight_equals_total": _max_abs_gap(
            frame, ["visitors_same_day", "visitors_overnight"], "visitors_total"
        ),
        "sea_plus_land_plus_air_equals_total": _max_abs_gap(
            frame, ["visitors_by_sea", "visitors_by_land", "visitors_by_air"], "visitors_total"
        ),
        "tier_rooms_sum_equals_hotel_rooms": _max_abs_gap(
            frame, [f"rooms_{t}" for t in tiers], "rooms_hotels"
        ),
        "tier_establishments_sum_equals_hotels": _max_abs_gap(
            frame, [f"establishments_{t}" for t in tiers], "establishments_hotels"
        ),
        "tier_guests_sum_equals_hotel_guests": _max_abs_gap(
            frame, [f"guests_{t}" for t in tiers], "guests_hotels"
        ),
    }
    origin = frame[["visitors_mainland", "visitors_hong_kong", "visitors_taiwan"]].sum(axis=1)
    share = 1.0 - origin / frame["visitors_total"]
    occupancy = frame[[f"occupancy_{t}" for t in tiers]]
    checks["international_share_within_0_1"] = {
        "min": float(share.min()),
        "max": float(share.max()),
        "ok": bool(share.between(0.0, 1.0).all()),
    }
    checks["tier_occupancy_within_0_100"] = {
        "min": float(occupancy.min().min()),
        "max": float(occupancy.max().max()),
        "ok": bool(((occupancy >= 0) & (occupancy <= 100)).all().all()),
    }
    return checks


def release_crosscheck(long: pd.DataFrame, printed: pd.DataFrame) -> pd.DataFrame:
    """Time Series Database value next to the value printed in the latest Excel release."""
    merged = printed.merge(long[["key", "period", "value"]], on=["key", "period"], how="left")
    merged = merged.rename(columns={"value": "tsdb_value"})
    # A printed figure in thousands with one decimal can differ from the unit count by 50.
    tolerance = merged["xlsx_scale"].map(lambda scale: 0.5 * scale / 10 if scale > 1 else 0.0)
    merged["xlsx_in_units"] = merged["xlsx_value"] * merged["xlsx_scale"]
    merged["abs_diff"] = (merged["xlsx_in_units"] - merged["tsdb_value"]).abs()
    merged["tolerance"] = tolerance + 1e-6
    merged["match"] = merged["abs_diff"] <= merged["tolerance"]
    return merged


def tier_continuity(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Is each star class observed every month, and where does its supply jump?

    Returns a per-tier summary and the list of months in which the number of
    operating establishments of a tier changed (the events a reclassification or an
    opening/closure would produce).
    """
    months = panel_months()
    frame = wide(long).reindex(months)
    summary_rows, change_rows = [], []
    for tier in TIERS:
        row: dict[str, Any] = {"tier": tier}
        for measure in TIER_MEASURES:
            series = frame[f"{measure}_{tier}"]
            row[f"{measure}_months"] = int(series.notna().sum())
        establishments = frame[f"establishments_{tier}"]
        rooms = frame[f"rooms_{tier}"]
        for label, (start, end) in PERIODS.items():
            window = establishments[(establishments.index >= start) & (establishments.index <= end)]
            row[f"establishments_{label}"] = f"{int(window.min())}-{int(window.max())}"
        step = establishments.diff()
        room_change = rooms.pct_change() * 100.0
        row["months_establishment_count_changed"] = int((step.fillna(0) != 0).sum())
        row["largest_monthly_room_change_pct"] = round(float(room_change.abs().max()), 1)
        row["month_of_largest_room_change"] = str(room_change.abs().idxmax())
        row["occupancy_min"] = float(frame[f"occupancy_{tier}"].min())
        row["occupancy_max"] = float(frame[f"occupancy_{tier}"].max())
        summary_rows.append(row)
        for month in step.index[step.fillna(0) != 0]:
            change_rows.append(
                {
                    "tier": tier,
                    "period": month,
                    "establishments_before": int(establishments.shift(1)[month]),
                    "establishments_after": int(establishments[month]),
                    "rooms_before": int(rooms.shift(1)[month]),
                    "rooms_after": int(rooms[month]),
                    "rooms_change_pct": round(float(room_change[month]), 1),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(change_rows)


def supply_jumps(long: pd.DataFrame, threshold_pct: float = 10.0) -> pd.DataFrame:
    """Tier-months in which the number of guest rooms moved by at least ``threshold_pct``."""
    frame = wide(long).reindex(panel_months())
    rows = []
    for tier in TIERS:
        rooms = frame[f"rooms_{tier}"]
        establishments = frame[f"establishments_{tier}"]
        change = rooms.pct_change() * 100.0
        for month in change.index[change.abs() >= threshold_pct]:
            rows.append(
                {
                    "tier": tier,
                    "period": month,
                    "period_label": period_label(month),
                    "rooms_before": int(rooms.shift(1)[month]),
                    "rooms_after": int(rooms[month]),
                    "rooms_change_pct": round(float(change[month]), 1),
                    "establishments_before": int(establishments.shift(1)[month]),
                    "establishments_after": int(establishments[month]),
                }
            )
    return pd.DataFrame(rows)


def offsetting_reclassifications(changes: pd.DataFrame) -> pd.DataFrame:
    """Months in which one tier lost establishments while another gained them."""
    rows = []
    for month, group in changes.groupby("period"):
        delta = group["establishments_after"] - group["establishments_before"]
        if (delta > 0).any() and (delta < 0).any():
            rows.append(
                {
                    "period": month,
                    "moves": "; ".join(
                        f"{tier} {before}->{after} est., rooms {rb}->{ra}"
                        for tier, before, after, rb, ra in zip(
                            group["tier"],
                            group["establishments_before"],
                            group["establishments_after"],
                            group["rooms_before"],
                            group["rooms_after"],
                            strict=True,
                        )
                    ),
                }
            )
    return pd.DataFrame(rows, columns=["period", "moves"])


def remark_table(long: pd.DataFrame) -> pd.DataFrame:
    """Every DSEC footnote attached to a value in the panel window, with its span."""
    noted = long[(long["remarks"] != "") & (long["period"] >= PANEL_START)]
    rows = []
    for (key, remark), group in noted.groupby(["key", "remarks"]):
        rows.append(
            {
                "key": key,
                "remark": remark,
                "months": int(len(group)),
                "first_month": group["period"].min(),
                "last_month": group["period"].max(),
            }
        )
    return pd.DataFrame(rows, columns=["key", "remark", "months", "first_month", "last_month"])


def period_counts() -> dict[str, int]:
    months = panel_months()
    counts = {
        label: sum(start <= month <= end for month in months)
        for label, (start, end) in PERIODS.items()
    }
    counts["all_months"] = len(months)
    return counts
