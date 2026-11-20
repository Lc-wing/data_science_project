"""MAUP comparison of H3 resolution 8 against resolution 9.

Gini, the Theil within/between decomposition, and Global Moran's I are copied
from the H1 table so the sensitivity uses the same population-weighted
statistics as the confirmatory analysis. Deltas are resolution 9 minus
resolution 8.
"""

from __future__ import annotations

import pandas as pd


def maup_r8_vs_r9(h1: pd.DataFrame) -> pd.DataFrame:
    """One row per threshold and facility type present at both resolutions."""
    required = {"resolution", "threshold_min", "facility_type"}
    missing = required.difference(h1.columns)
    if missing:
        raise ValueError(f"H1 table is missing {sorted(missing)}")
    rows: list[dict[str, object]] = []
    grouped = h1.groupby(["threshold_min", "facility_type"], sort=True)
    for (threshold, facility), group in grouped:
        res8 = group[group["resolution"].astype(str) == "res8"]
        res9 = group[group["resolution"].astype(str) == "res9"]
        if res8.empty or res9.empty:
            continue
        left = res8.iloc[0]
        right = res9.iloc[0]
        rows.append(
            {
                "threshold_min": int(threshold),
                "facility_type": facility,
                "gini_r8": float(left["gini"]),
                "gini_r9": float(right["gini"]),
                "gini_delta_r9_minus_r8": float(right["gini"] - left["gini"]),
                "theil_r8": float(left["theil_total"]),
                "theil_r9": float(right["theil_total"]),
                "theil_delta_r9_minus_r8": float(right["theil_total"] - left["theil_total"]),
                "theil_within_r8": float(left["theil_within"]),
                "theil_within_r9": float(right["theil_within"]),
                "theil_within_delta_r9_minus_r8": float(
                    right["theil_within"] - left["theil_within"]
                ),
                "theil_between_r8": float(left["theil_between"]),
                "theil_between_r9": float(right["theil_between"]),
                "theil_between_delta_r9_minus_r8": float(
                    right["theil_between"] - left["theil_between"]
                ),
                "moran_i_r8": float(left["moran_i"]),
                "moran_i_r9": float(right["moran_i"]),
                "moran_delta_r9_minus_r8": float(right["moran_i"] - left["moran_i"]),
            }
        )
    return pd.DataFrame(rows)
