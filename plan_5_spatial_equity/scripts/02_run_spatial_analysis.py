#!/usr/bin/env python3
"""Milestone M2 CLI: Gaussian 2SFCA, H1–H3 inference, and the r8/r9 MAUP check.

Hypotheses and models are pre-specified. H3 always writes OLS with HC3 standard
errors, spreg ML_Error, and spreg ML_Lag for the same outcome. Language in the
diagnostics file stays associational.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from macao_equity.analysis.inequality import (  # noqa: E402
    bootstrap_gini_theil,
    bootstrap_indices,
    holm_bonferroni,
    lorenz_points,
    paired_gini_diff_bootstrap,
)
from macao_equity.analysis.maup import maup_r8_vs_r9  # noqa: E402
from macao_equity.analysis.regression import fit_prespecified_models, ols_hc3  # noqa: E402
from macao_equity.analysis.sfca import (  # noqa: E402
    TRADITIONAL_FACILITY_TYPES,
    composite_accessibility,
    gaussian_2sfca,
    prepare_walk_network,
)
from macao_equity.analysis.spatial_stats import (  # noqa: E402
    bridge_queen_weights,
    global_moran,
    island_safe_knn_weights,
    knn_weights,
    lisa_fdr,
    weights_diagnostics,
)
from macao_equity.config import (  # noqa: E402
    DATA_FROZEN_DIR,
    DATA_INTERIM_DIR,
    ISLANDS_PARISH_IDS,
    PENINSULA_PARISH_IDS,
    RANDOM_SEED,
)

FACILITY_TYPES = ("park", "health_centre", "library", "bus_stop", "ev_charger")
ACCESSIBILITY_TYPES = (*FACILITY_TYPES, "composite")
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
H1_COLUMNS = [
    "resolution",
    "threshold_min",
    "facility_type",
    "gini",
    "gini_ci_low",
    "gini_ci_high",
    "theil_total",
    "theil_within",
    "theil_between",
    "theil_total_ci_low",
    "theil_total_ci_high",
    "theil_between_share",
    "moran_i",
    "moran_p",
    "n_ll_fdr",
    "n_hh_fdr",
    "mean_acc",
    "median_acc",
    "zero_acc_pct",
]
H2_COLUMNS = [
    "threshold_min",
    "comparison",
    "gini_ev",
    "gini_other",
    "gini_diff",
    "ci_low",
    "ci_high",
    "p_boot_raw",
    "p_holm",
    "n_boot",
    "seed",
]
H3_COLUMNS = [
    "threshold_min",
    "outcome",
    "weights_spec",
    "model_type",
    "term",
    "coef",
    "se",
    "ci_low",
    "ci_high",
    "p_value",
    "r2_or_pseudo_r2",
    "aic",
    "resid_moran_i",
    "resid_moran_p",
    "n_obs",
]


def _landmass(parish_ids: pd.Series) -> np.ndarray:
    labels = np.full(len(parish_ids), "islands", dtype=object)
    peninsula = parish_ids.isin(PENINSULA_PARISH_IDS).to_numpy()
    islands = parish_ids.isin(ISLANDS_PARISH_IDS).to_numpy()
    labels[peninsula] = "peninsula"
    labels[islands] = "islands"
    if not np.all(peninsula | islands):
        unknown = sorted(set(parish_ids[~peninsula & ~islands].astype(str)))
        raise ValueError(f"parish ids outside the two landmasses: {unknown}")
    return labels


def _xy(frame: gpd.GeoDataFrame) -> np.ndarray:
    if frame.geom_type.iloc[0] == "Point":
        return np.column_stack([frame.geometry.x, frame.geometry.y])
    centroids = frame.geometry.centroid
    return np.column_stack([centroids.x, centroids.y])


def _zone_pop_weighted_xy(zones: gpd.GeoDataFrame, cells_r9: gpd.GeoDataFrame) -> np.ndarray:
    """Population-weighted centroid per zone from populated H3 r9 cells (fallback: geometry)."""
    fallback = _xy(zones)
    coords: list[list[float]] = []
    for idx, zid in enumerate(zones["zone_id"].astype(str).tolist()):
        sub = cells_r9[cells_r9["zone_id"].astype(str) == zid]
        weights = sub["pop_total"].to_numpy(dtype=float) if len(sub) else np.array([], dtype=float)
        if weights.size > 0 and float(weights.sum()) > 0.0:
            cx = float(np.average(sub["centroid_x"].to_numpy(dtype=float), weights=weights))
            cy = float(np.average(sub["centroid_y"].to_numpy(dtype=float), weights=weights))
            coords.append([cx, cy])
        else:
            coords.append([float(fallback[idx, 0]), float(fallback[idx, 1])])
    return np.asarray(coords, dtype=float)


def _attach_accessibility(
    frame: gpd.GeoDataFrame,
    minutes: np.ndarray,
    facilities: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Append 10- and 15-minute 2SFCA columns. Base columns are left unchanged."""
    out = frame.copy()
    population = out["pop_total"].to_numpy(dtype=float)
    facility_type = facilities["facility_type"].astype(str).to_numpy()
    capacity = facilities["capacity"].to_numpy(dtype=float)
    by_cutoff: dict[int, dict[str, np.ndarray]] = {10: {}, 15: {}}
    for cutoff in (10, 15):
        for facility in FACILITY_TYPES:
            mask = facility_type == facility
            access, _ratios = gaussian_2sfca(population, capacity[mask], minutes[:, mask], cutoff)
            out[f"acc{cutoff}_{facility}"] = access
            by_cutoff[cutoff][facility] = access
        ev_mask = facility_type == "ev_charger"
        unweighted, _ratios = gaussian_2sfca(
            population,
            np.ones(int(ev_mask.sum()), dtype=float),
            minutes[:, ev_mask],
            cutoff,
        )
        out[f"acc{cutoff}_ev_unweighted"] = unweighted
        out[f"acc{cutoff}_composite"] = composite_accessibility(by_cutoff[cutoff])
    return out


def _attach_lisa(
    frame: gpd.GeoDataFrame,
    weights: object,
    permutations: int,
    seed: int,
) -> gpd.GeoDataFrame:
    out = frame.copy()
    for cutoff in (10, 15):
        for facility in ACCESSIBILITY_TYPES:
            metric = f"acc{cutoff}_{facility}"
            local = lisa_fdr(
                out[metric].to_numpy(dtype=float),
                weights,
                permutations=permutations,
                seed=seed,
            )
            out[f"lisa_i_{metric}"] = local["i"]
            out[f"lisa_p_{metric}"] = local["p"]
            out[f"lisa_q_fdr_{metric}"] = local["q_fdr"]
            out[f"lisa_cluster_{metric}"] = local["cluster"]
    return out


def _h1_rows(
    resolution: str,
    frame: gpd.GeoDataFrame,
    weights: object,
    indices: np.ndarray,
    permutations: int,
    seed: int,
) -> list[dict[str, object]]:
    population = frame["pop_total"].to_numpy(dtype=float)
    parishes = frame["parish_id"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for cutoff in (10, 15):
        for facility in ACCESSIBILITY_TYPES:
            column = f"acc{cutoff}_{facility}"
            values = frame[column].to_numpy(dtype=float)
            inequality = bootstrap_gini_theil(values, population, parishes, indices)
            moran_i, moran_p = global_moran(
                values, weights, permutations=permutations, seed=seed
            )
            clusters = frame[f"lisa_cluster_{column}"].astype(str)
            rows.append(
                {
                    "resolution": resolution,
                    "threshold_min": cutoff,
                    "facility_type": facility,
                    "gini": inequality["gini"],
                    "gini_ci_low": inequality["gini_ci_low"],
                    "gini_ci_high": inequality["gini_ci_high"],
                    "theil_total": inequality["theil_total"],
                    "theil_within": inequality["theil_within"],
                    "theil_between": inequality["theil_between"],
                    "theil_total_ci_low": inequality["theil_total_ci_low"],
                    "theil_total_ci_high": inequality["theil_total_ci_high"],
                    "theil_between_share": inequality["theil_between_share"],
                    "moran_i": moran_i,
                    "moran_p": moran_p,
                    "n_ll_fdr": int((clusters == "LL").sum()),
                    "n_hh_fdr": int((clusters == "HH").sum()),
                    "mean_acc": float(values.mean()),
                    "median_acc": float(np.median(values)),
                    "zero_acc_pct": float(100.0 * np.mean(values <= 1e-12)),
                }
            )
    return rows


def _h2_rows(frame: gpd.GeoDataFrame, indices: np.ndarray, seed: int) -> list[dict[str, object]]:
    population = frame["pop_total"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for cutoff in (10, 15):
        values = {
            facility: frame[f"acc{cutoff}_{facility}"].to_numpy(dtype=float)
            for facility in ("ev_charger", *TRADITIONAL_FACILITY_TYPES)
        }
        family = paired_gini_diff_bootstrap(
            values,
            population,
            TRADITIONAL_FACILITY_TYPES,
            indices,
            seed=seed,
        )
        for row in family:
            rows.append({"threshold_min": cutoff, **row})
    adjusted = holm_bonferroni([float(row["p_boot_raw"]) for row in rows])
    for row, p_holm in zip(rows, adjusted, strict=True):
        row["p_holm"] = p_holm
    return rows


def _h3_rows(
    zones: gpd.GeoDataFrame,
    permutations: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, float], dict[str, dict[str, int | str]]]:
    predictors = zones[["elderly_share", "log_pop_density"]].to_numpy(dtype=float)
    names = ["elderly_share", "log_pop_density"]
    vif = ols_hc3(zones["acc15_composite"].to_numpy(dtype=float), predictors, names)["vif"]
    weight_bank = {
        "knn4": knn_weights(zones, k=4),
        "bridge_queen": bridge_queen_weights(zones),
    }
    weight_info = {
        spec: {
            "definition": (
                "row-standardized k=4 nearest neighbors"
                if spec == "knn4"
                else "queen contiguity with centroid bridges between components"
            ),
            **weights_diagnostics(weights),
        }
        for spec, weights in weight_bank.items()
    }
    rows: list[dict[str, object]] = []
    for cutoff in (10, 15):
        for facility in ACCESSIBILITY_TYPES:
            outcome = f"acc{cutoff}_{facility}"
            y = zones[outcome].to_numpy(dtype=float)
            for spec, weights in weight_bank.items():
                fitted = fit_prespecified_models(
                    y,
                    predictors,
                    names,
                    weights,
                    permutations=permutations,
                    seed=seed,
                )
                for row in fitted:
                    rows.append(
                        {
                            "threshold_min": cutoff,
                            "outcome": outcome,
                            "weights_spec": spec,
                            **row,
                        }
                    )
    return rows, {key: float(value) for key, value in vif.items()}, weight_info  # type: ignore[attr-defined]


def _lorenz_rows(frame: gpd.GeoDataFrame) -> list[dict[str, object]]:
    population = frame["pop_total"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for cutoff in (10, 15):
        for facility in ACCESSIBILITY_TYPES:
            pop_share, acc_share = lorenz_points(
                frame[f"acc{cutoff}_{facility}"].to_numpy(dtype=float),
                population,
            )
            for p_value, q_value in zip(pop_share, acc_share, strict=True):
                rows.append(
                    {
                        "threshold_min": cutoff,
                        "facility_type": facility,
                        "cum_pop_share": float(p_value),
                        "cum_acc_share": float(q_value),
                    }
                )
    return rows


def _eda_rows(frame: gpd.GeoDataFrame) -> list[dict[str, object]]:
    names = [f"acc{cutoff}_{facility}" for cutoff in (10, 15) for facility in FACILITY_TYPES]
    names.extend(
        [
            "acc10_composite",
            "acc15_composite",
            "acc10_ev_unweighted",
            "acc15_ev_unweighted",
            "pop_total",
            "pop_density",
            "elderly_share",
        ]
    )
    rows: list[dict[str, object]] = []
    for name in names:
        values = np.sort(frame[name].to_numpy(dtype=float))
        rows.append(
            {
                "variable": name,
                "n_cells": int(values.size),
                "min": float(values[0]),
                "median": float(np.median(values)),
                "mean": float(values.mean()),
                "max": float(values[-1]),
                "std": float(values.std(ddof=0)),
            }
        )
    return rows


def _write_gpkg(frame: gpd.GeoDataFrame, path: Path, layer: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    frame.to_file(path, driver="GPKG", layer=layer)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run(n_boot: int, permutations: int, seed: int) -> None:
    """Compute the M2 contract tables and interim geopackages from frozen M1 inputs."""
    cells_r9 = gpd.read_file(DATA_FROZEN_DIR / "cells_base_res9.gpkg")
    cells_r8 = gpd.read_file(DATA_FROZEN_DIR / "cells_base_res8.gpkg")
    zones = gpd.read_file(DATA_FROZEN_DIR / "zones_23.gpkg").sort_values("zone_id")
    zones = zones.reset_index(drop=True)
    facilities = gpd.read_file(DATA_FROZEN_DIR / "facilities_clean.gpkg")
    graph = nx.read_graphml(PROJECT_ROOT / "data" / "raw" / "osm_walk_graph.graphml")
    network = prepare_walk_network(graph)
    supply_xy = _xy(facilities)

    enriched: dict[str, gpd.GeoDataFrame] = {}
    for label, frame in (("r9", cells_r9), ("r8", cells_r8), ("zones", zones)):
        print(f"2SFCA {label}: {len(frame)} demand units", flush=True)
        demand_xy = _zone_pop_weighted_xy(frame, cells_r9) if label == "zones" else _xy(frame)
        minutes = network.minutes(demand_xy, supply_xy, max_minutes=15.0)
        enriched[label] = _attach_accessibility(frame, minutes, facilities)

    for label in ("r9", "r8", "zones"):
        print(f"LISA {label}", flush=True)
        weights = island_safe_knn_weights(
            enriched[label], _landmass(enriched[label]["parish_id"]), k=4
        )
        enriched[label] = _attach_lisa(enriched[label], weights, permutations, seed)

    print("H1 inequality and H2 paired bootstrap", flush=True)
    idx_r9 = bootstrap_indices(len(enriched["r9"]), n_boot, seed)
    idx_r8 = bootstrap_indices(len(enriched["r8"]), n_boot, seed)
    w_r9 = island_safe_knn_weights(enriched["r9"], _landmass(enriched["r9"]["parish_id"]), k=4)
    w_r8 = island_safe_knn_weights(enriched["r8"], _landmass(enriched["r8"]["parish_id"]), k=4)
    h1 = pd.DataFrame(
        _h1_rows("res9", enriched["r9"], w_r9, idx_r9, permutations, seed)
        + _h1_rows("res8", enriched["r8"], w_r8, idx_r8, permutations, seed)
    )
    h2 = pd.DataFrame(_h2_rows(enriched["r9"], idx_r9, seed))

    print("H3 zone regressions", flush=True)
    h3_records, vif, weight_info = _h3_rows(enriched["zones"], permutations, seed)
    h3 = pd.DataFrame(h3_records)
    diagnostics = {
        "n_obs": 23,
        "seed": seed,
        "n_boot": n_boot,
        "permutations": permutations,
        "inference": (
            "Cross-sectional associations only. Coefficients describe conditional "
            "associations between zone accessibility and the two pre-specified covariates."
        ),
        "model_policy": (
            "OLS_HC3, SEM_ML, and SLM_ML are pre-specified and reported side by side "
            "for every outcome. Diagnostic p-values do not select a model."
        ),
        "predictors": ["elderly_share", "log_pop_density"],
        "outcomes": [
            f"acc{cutoff}_{facility}"
            for cutoff in (10, 15)
            for facility in ACCESSIBILITY_TYPES
        ],
        "vif": vif,
        "weights": weight_info,
    }
    maup = maup_r8_vs_r9(h1)
    lorenz = pd.DataFrame(_lorenz_rows(enriched["r9"]))
    eda = pd.DataFrame(_eda_rows(enriched["r9"]))

    _write_gpkg(
        enriched["r9"],
        DATA_INTERIM_DIR / "cells_r9_with_2sfca_lisa.gpkg",
        "cells_r9_with_2sfca_lisa",
    )
    _write_gpkg(
        enriched["r8"],
        DATA_INTERIM_DIR / "cells_r8_with_2sfca.gpkg",
        "cells_r8_with_2sfca",
    )
    _write_gpkg(
        enriched["zones"],
        DATA_INTERIM_DIR / "zones_23_with_2sfca.gpkg",
        "zones_23_with_2sfca",
    )
    _write_csv(h1[H1_COLUMNS], TABLES_DIR / "h1_inequality_lisa.csv")
    _write_csv(h2[H2_COLUMNS], TABLES_DIR / "h2_gini_diff_bootstrap.csv")
    _write_csv(h3[H3_COLUMNS], TABLES_DIR / "h3_spatial_regression.csv")
    _write_csv(maup, TABLES_DIR / "maup_r8_vs_r9.csv")
    _write_csv(eda, TABLES_DIR / "eda_summary.csv")
    _write_csv(lorenz, TABLES_DIR / "lorenz_curves.csv")
    diag_path = TABLES_DIR / "h3_regression_diagnostics.json"
    diag_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {diag_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Plan 5 Phase A milestone M2.")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    run(n_boot=args.n_boot, permutations=args.permutations, seed=args.seed)


if __name__ == "__main__":
    main()
