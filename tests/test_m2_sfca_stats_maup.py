"""Milestone M2 contract tests: Gaussian 2SFCA, H1–H3, and MAUP.

Scientific checks are pre-specified. H3 must report OLS-HC3, SEM, and SLM
side by side; a diagnostic p-value must not drop a model.
"""

from __future__ import annotations

import json
import math

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from macao_equity.analysis.inequality import (
    bh_fdr,
    bootstrap_gini_theil,
    bootstrap_indices,
    holm_bonferroni,
    lorenz_points,
    paired_gini_diff_bootstrap,
    weighted_gini,
    weighted_theil_t,
)
from macao_equity.analysis.maup import maup_r8_vs_r9
from macao_equity.analysis.regression import fit_prespecified_models, ols_hc3
from macao_equity.analysis.sfca import (
    composite_accessibility,
    gaussian_2sfca,
    gaussian_weight,
    travel_minutes_from_graph,
)
from macao_equity.analysis.spatial_stats import (
    bridge_queen_weights,
    global_moran,
    island_safe_knn_weights,
    knn_weights,
    lisa_fdr,
    weights_diagnostics,
)
from macao_equity.config import (
    ANALYSIS_CRS,
    DATA_FROZEN_DIR,
    DATA_INTERIM_DIR,
    ISLANDS_PARISH_IDS,
    PENINSULA_PARISH_IDS,
    PROJECT_ROOT,
    RANDOM_SEED,
    WALK_SPEED_M_PER_MIN,
    WGS84_CRS,
    make_transformer,
)

TABLES = PROJECT_ROOT / "outputs" / "tables"
FACILITY_TYPES = ("park", "health_centre", "library", "bus_stop", "ev_charger")
TRADITIONAL = ("park", "health_centre", "library", "bus_stop")
H1_COLS = (
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
)
H2_COLS = (
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
)
H3_COLS = (
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
)
ACC_TYPES = (*FACILITY_TYPES, "composite", "ev_unweighted")


def _reference_hc3(
    y: np.ndarray, x_cols: list[np.ndarray], names: list[str]
) -> dict[str, dict[str, float]]:
    """Independent HC3 sandwich (MacKinnon–White), matching the e2e oracle."""
    y = np.asarray(y, dtype=float)
    x = np.column_stack([np.ones(len(y)), *x_cols])
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta
    h = np.sum((x @ xtx_inv) * x, axis=1)
    meat = (x * (resid**2 / np.maximum((1.0 - h) ** 2, 1e-18))[:, None]).T @ x
    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    keys = ["const", *names]
    return {
        "coefs": dict(zip(keys, beta, strict=True)),
        "se": dict(zip(keys, se, strict=True)),
    }


def test_gaussian_kernel_is_one_at_origin_and_zero_at_cutoff() -> None:
    cutoff = 10.0
    weights = gaussian_weight(np.array([0.0, 5.0, 10.0, 12.0, np.inf]), cutoff)
    assert weights[0] == pytest.approx(1.0)
    assert 0.0 < weights[1] < 1.0
    assert weights[2] == 0.0
    assert weights[3] == 0.0
    assert weights[4] == 0.0


def test_2sfca_zero_demand_guard_and_exact_supply_conservation() -> None:
    population = np.array([0.0, 10.0, 4.0])
    capacity = np.array([5.0, 7.0])
    # Facility 0 only meets the zero-population cell. Facility 1 meets cells 1 and 2.
    distance = np.array(
        [
            [1.0, 100.0],
            [100.0, 2.0],
            [100.0, 4.0],
        ]
    )
    acc, ratios = gaussian_2sfca(population, capacity, distance, cutoff_min=10.0)
    assert ratios[0] == 0.0
    assert ratios[1] > 0.0
    assert acc[0] == 0.0
    allocated = float(np.sum(acc * population))
    assert allocated == pytest.approx(capacity[1])
    assert allocated <= float(capacity.sum()) + 1e-9


def test_2sfca_wider_cutoff_keeps_support_and_allocated_supply() -> None:
    population = np.array([5.0, 8.0, 3.0])
    capacity = np.array([4.0, 9.0])
    distance = np.array(
        [
            [2.0, 12.0],
            [3.0, 14.0],
            [40.0, 6.0],
        ]
    )
    acc10, _ = gaussian_2sfca(population, capacity, distance, cutoff_min=10.0)
    acc15, ratios15 = gaussian_2sfca(population, capacity, distance, cutoff_min=15.0)
    assert np.all((acc10 <= 1e-12) | (acc15 > 0.0))
    supply10 = float(np.sum(acc10 * population))
    supply15 = float(np.sum(acc15 * population))
    assert supply15 >= supply10 - 1e-8
    reachable = float(capacity[ratios15 > 0.0].sum())
    assert supply15 == pytest.approx(reachable)


def test_capacity_weighted_ev_dominates_unit_weights_on_the_same_support() -> None:
    population = np.array([6.0, 4.0, 9.0])
    distance = np.array(
        [
            [1.0, 8.0, 30.0],
            [6.0, 2.0, 30.0],
            [4.0, 5.0, 1.5],
        ]
    )
    capacity = np.array([2.0, 5.0, 3.0])
    weighted, _ = gaussian_2sfca(population, capacity, distance, cutoff_min=15.0)
    unweighted, _ = gaussian_2sfca(population, np.ones(3), distance, cutoff_min=15.0)
    assert np.all((weighted > 1e-12) == (unweighted > 1e-12))
    assert np.all(weighted + 1e-9 >= unweighted)
    assert float(np.max(np.abs(weighted - unweighted))) > 1e-9


def test_composite_is_minmax_mean_of_four_public_facilities() -> None:
    access = {
        "park": np.array([0.0, 10.0]),
        "health_centre": np.array([5.0, 5.0]),
        "library": np.array([1.0, 3.0]),
        "bus_stop": np.array([2.0, 2.0]),
    }
    composite = composite_accessibility(access)
    # park -> [0, 1]; constants -> 0; library -> [0, 1]; mean of four.
    assert composite == pytest.approx(np.array([0.0, 0.5]))
    assert float(composite.min()) >= 0.0


def test_weighted_gini_known_values() -> None:
    assert weighted_gini(np.array([3.0, 3.0, 3.0]), np.array([1.0, 2.0, 4.0])) == 0.0
    assert weighted_gini(np.array([0.0, 1.0]), np.array([1.0, 1.0])) == pytest.approx(0.5)
    assert weighted_gini(np.array([0.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0])) == pytest.approx(
        2.0 / 3.0
    )


def test_theil_t_exact_additivity_with_zero_ln_zero() -> None:
    values = np.array([0.0, 2.0, 2.0])
    weights = np.array([1.0, 1.0, 1.0])
    groups = np.array(["A", "A", "B"])
    parts = weighted_theil_t(values, weights, groups)
    assert parts["total"] == pytest.approx(parts["within"] + parts["between"])
    assert parts["total"] == pytest.approx(math.log(1.5))
    assert parts["between_share"] == pytest.approx(parts["between"] / parts["total"])
    # A second group of pure zeros must not emit NaN from 0 ln 0.
    parts_zero = weighted_theil_t(
        np.array([0.0, 0.0, 4.0]),
        np.array([2.0, 2.0, 2.0]),
        np.array(["A", "A", "B"]),
    )
    assert all(math.isfinite(v) for v in parts_zero.values())
    assert parts_zero["total"] == pytest.approx(parts_zero["within"] + parts_zero["between"])


def test_bootstrap_ci_is_reproducible_and_brackets_the_estimate() -> None:
    rng = np.random.default_rng(0)
    values = rng.uniform(0.0, 5.0, size=40)
    weights = rng.uniform(1.0, 3.0, size=40)
    groups = np.array(["P01", "P02", "P03", "P06"] * 10)
    indices = bootstrap_indices(40, n_boot=400, seed=RANDOM_SEED)
    first = bootstrap_gini_theil(values, weights, groups, indices)
    second = bootstrap_gini_theil(values, weights, groups, indices)
    assert first["gini"] == pytest.approx(weighted_gini(values, weights))
    assert first["gini_ci_low"] < first["gini_ci_high"]
    assert first["gini_ci_low"] - 1e-9 <= first["gini"] <= first["gini_ci_high"] + 1e-9
    assert first["theil_total"] == pytest.approx(first["theil_within"] + first["theil_between"])
    assert first["gini_ci_low"] == pytest.approx(second["gini_ci_low"])
    assert indices.shape == (400, 40)


def test_paired_bootstrap_uses_identical_cells_and_holm() -> None:
    rng = np.random.default_rng(1)
    n = 36
    weights = rng.uniform(1.0, 4.0, size=n)
    base = rng.uniform(0.0, 2.0, size=n)
    values = {
        "ev_charger": base * 3.0,
        "park": base,
        "health_centre": np.sqrt(base),
        "library": base * 0.2,
        "bus_stop": rng.uniform(0.2, 0.8, size=n),
    }
    indices = bootstrap_indices(n, n_boot=300, seed=RANDOM_SEED)
    rows = paired_gini_diff_bootstrap(
        values,
        weights,
        comparisons=TRADITIONAL,
        indices=indices,
    )
    assert [row["comparison"] for row in rows] == [f"ev_charger_vs_{ft}" for ft in TRADITIONAL]
    expected_p = holm_bonferroni([row["p_boot_raw"] for row in rows])
    for row, p_holm in zip(rows, expected_p, strict=True):
        assert row["gini_diff"] == pytest.approx(row["gini_ev"] - row["gini_other"])
        assert row["ci_low"] < row["ci_high"]
        assert row["ci_low"] - 1e-9 <= row["gini_diff"] <= row["ci_high"] + 1e-9
        assert 0.0 <= row["p_boot_raw"] <= p_holm <= 1.0
        assert row["p_holm"] == pytest.approx(p_holm)
        assert row["n_boot"] == 300
        assert row["seed"] == RANDOM_SEED
    # Identical accessibility series produce a paired difference of exactly zero.
    tied = dict(values)
    tied["park"] = values["ev_charger"].copy()
    tied_rows = paired_gini_diff_bootstrap(
        tied, weights, comparisons=("park",), indices=indices
    )
    assert tied_rows[0]["gini_diff"] == pytest.approx(0.0)


def test_bh_fdr_and_holm_match_hand_calculated_adjustments() -> None:
    qvals = bh_fdr([0.001, 0.01, 0.2, 0.5])
    assert qvals == pytest.approx([0.004, 0.02, 4.0 / 3.0 * 0.2, 0.5])
    assert all(q >= p for q, p in zip(qvals, [0.001, 0.01, 0.2, 0.5], strict=True))
    holm = holm_bonferroni([0.01, 0.04, 0.03, 0.20])
    assert holm == pytest.approx([0.04, 0.09, 0.09, 0.20])


def test_travel_minutes_add_snap_and_respect_disconnected_components() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("a", x=113.55, y=22.19)
    graph.add_node("b", x=113.551, y=22.19)
    graph.add_node("c", x=113.60, y=22.19)
    graph.add_edge("a", "b", length=80.0)
    graph.add_edge("b", "a", length=80.0)
    transformer = make_transformer(WGS84_CRS, ANALYSIS_CRS)
    ax, ay = transformer.transform(113.55, 22.19)
    bx, by = transformer.transform(113.551, 22.19)
    cx, cy = transformer.transform(113.60, 22.19)
    minutes = travel_minutes_from_graph(
        np.array([[ax, ay]]),
        np.array([[bx, by], [cx, cy]]),
        graph,
        max_minutes=15.0,
    )
    assert minutes.shape == (1, 2)
    assert minutes[0, 0] == pytest.approx(80.0 / WALK_SPEED_M_PER_MIN, abs=0.05)
    assert not np.isfinite(minutes[0, 1])


def test_island_safe_knn_has_no_isolates_and_does_not_cross_the_harbour() -> None:
    peninsula = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0], [50.0, 50.0]])
    islands = peninsula + np.array([10_000.0, 0.0])
    coords = np.vstack([peninsula, islands])
    frame = gpd.GeoDataFrame(
        {
            "landmass": ["peninsula"] * 5 + ["islands"] * 5,
            "geometry": [Point(float(x), float(y)) for x, y in coords],
        },
        geometry="geometry",
        crs=ANALYSIS_CRS,
    )
    weights = island_safe_knn_weights(frame, frame["landmass"].to_numpy(), k=4)
    info = weights_diagnostics(weights)
    assert info["n_islands"] == 0
    assert info["min_neighbors"] >= 1
    for origin, neighbors in weights.neighbors.items():
        for dest in neighbors:
            assert frame.loc[origin, "landmass"] == frame.loc[dest, "landmass"]


def test_bridge_queen_removes_islands_by_linking_components() -> None:
    squares = []
    ids = []
    for row in range(2):
        for col in range(2):
            squares.append(box(col, row, col + 1, row + 1))
            ids.append(f"near-{row}{col}")
    for row in range(2):
        for col in range(2):
            squares.append(box(100 + col, row, 101 + col, row + 1))
            ids.append(f"far-{row}{col}")
    frame = gpd.GeoDataFrame(
        {"zone_id": ids, "geometry": squares}, geometry="geometry", crs=ANALYSIS_CRS
    )
    weights = bridge_queen_weights(frame)
    info = weights_diagnostics(weights)
    assert info["n_islands"] == 0
    assert info["n_components"] == 1
    assert info["min_neighbors"] >= 1


def test_ols_hc3_matches_independent_sandwich_and_vif() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    elderly = rng.uniform(0.05, 0.30, size=23)
    log_density = rng.normal(8.0, 0.4, size=23)
    y = 0.4 - 0.8 * elderly + 0.15 * log_density + rng.normal(0.0, 0.02, size=23)
    fit = ols_hc3(y, np.column_stack([elderly, log_density]), ["elderly_share", "log_pop_density"])
    ref = _reference_hc3(y, [elderly, log_density], ["elderly_share", "log_pop_density"])
    for term in ("const", "elderly_share", "log_pop_density"):
        assert fit["coefs"][term] == pytest.approx(ref["coefs"][term], abs=1e-8)
        assert fit["se"][term] == pytest.approx(ref["se"][term], abs=1e-8)
        assert fit["se"][term] > 0.0
    assert fit["vif"]["elderly_share"] >= 1.0
    assert fit["vif"]["log_pop_density"] >= 1.0
    assert 0.0 <= fit["r2"] <= 1.0


def test_h3_reports_three_models_side_by_side_without_pretest_switching() -> None:
    rng = np.random.default_rng(7)
    coords = rng.normal(size=(30, 2))
    frame = gpd.GeoDataFrame(
        {"geometry": [Point(float(x), float(y)) for x, y in coords]},
        geometry="geometry",
        crs=ANALYSIS_CRS,
    )
    weights = knn_weights(frame, k=4)
    elderly = rng.uniform(0.05, 0.4, size=30)
    log_density = rng.normal(5.0, 0.5, size=30)
    # Spatially smooth outcome so a Lagrange-multiplier test would prefer a spatial model.
    y = 1.0 + elderly - 0.2 * log_density
    y = y + 0.3 * y[np.array([weights.neighbors[i][0] for i in range(30)])]
    rows = fit_prespecified_models(
        y,
        np.column_stack([elderly, log_density]),
        ["elderly_share", "log_pop_density"],
        weights,
        permutations=199,
        seed=RANDOM_SEED,
    )
    models = {row["model_type"] for row in rows}
    assert models == {"OLS_HC3", "SEM_ML", "SLM_ML"}
    by_model: dict[str, set[str]] = {}
    for row in rows:
        by_model.setdefault(row["model_type"], set()).add(row["term"])
        assert row["se"] > 0.0
        assert row["ci_low"] < row["coef"] < row["ci_high"]
        width = (row["ci_high"] - row["ci_low"]) / (2.0 * row["se"])
        assert 1.85 <= width <= 2.35
        assert 0.0 <= row["p_value"] <= 1.0
        assert 0.0 < row["resid_moran_p"] <= 1.0
        assert row["n_obs"] == 30
    for terms in by_model.values():
        assert "elderly_share" in terms and "log_pop_density" in terms
    assert any("lambda" in term for term in by_model["SEM_ML"])
    assert any(term == "rho" for term in by_model["SLM_ML"])


def test_lisa_fdr_labels_respect_quadrants_and_q_values() -> None:
    xs, ys = np.meshgrid(np.arange(6), np.arange(6))
    coords = np.column_stack([xs.ravel(), ys.ravel()])
    frame = gpd.GeoDataFrame(
        {"geometry": [Point(float(x), float(y)) for x, y in coords]},
        geometry="geometry",
        crs=ANALYSIS_CRS,
    )
    values = np.array([float(x + y) for x, y in coords])
    weights = knn_weights(frame, k=4)
    lisa = lisa_fdr(values, weights, permutations=199, seed=RANDOM_SEED)
    moran_i, moran_p = global_moran(values, weights, permutations=199, seed=RANDOM_SEED)
    assert -1.5 <= moran_i <= 1.5
    assert 1.0 / 200.0 - 1e-12 <= moran_p <= 1.0
    assert np.all((lisa["p"] >= 0.0) & (lisa["p"] <= 1.0))
    assert np.all(lisa["q_fdr"] + 1e-12 >= lisa["p"])
    for local_i, q_value, cluster in zip(lisa["i"], lisa["q_fdr"], lisa["cluster"], strict=True):
        assert cluster in {"HH", "LL", "LH", "HL", "NS"}
        if cluster != "NS":
            assert q_value <= 0.05 + 1e-9
        if q_value > 0.05 + 1e-9:
            assert cluster == "NS"
        if cluster in {"HH", "LL"}:
            assert local_i >= -1e-9
        elif cluster in {"HL", "LH"}:
            assert local_i <= 1e-9


def test_lorenz_curve_area_matches_gini_and_maup_table_copies_both_scales() -> None:
    values = np.array([0.0, 1.0, 2.0, 4.0, 8.0])
    weights = np.array([1.0, 1.0, 2.0, 1.0, 3.0])
    pop_share, acc_share = lorenz_points(values, weights)
    assert pop_share[0] == 0.0 and acc_share[0] == 0.0
    assert pop_share[-1] == pytest.approx(1.0)
    assert acc_share[-1] == pytest.approx(1.0)
    area = float(
        np.sum(0.5 * np.diff(pop_share) * (acc_share[1:] + acc_share[:-1]))
    )
    assert 1.0 - 2.0 * area == pytest.approx(weighted_gini(values, weights), abs=1e-9)
    h1 = pd.DataFrame(
        [
            {
                "resolution": "res9",
                "threshold_min": 15,
                "facility_type": "park",
                "gini": 0.2,
                "theil_total": 0.05,
                "theil_within": 0.03,
                "theil_between": 0.02,
                "moran_i": 0.4,
            },
            {
                "resolution": "res8",
                "threshold_min": 15,
                "facility_type": "park",
                "gini": 0.1,
                "theil_total": 0.04,
                "theil_within": 0.01,
                "theil_between": 0.03,
                "moran_i": 0.2,
            },
        ]
    )
    maup = maup_r8_vs_r9(h1)
    assert list(maup["facility_type"]) == ["park"]
    assert maup.loc[0, "gini_r9"] == pytest.approx(0.2)
    assert maup.loc[0, "gini_delta_r9_minus_r8"] == pytest.approx(0.1)
    assert maup.loc[0, "theil_between_r9"] == pytest.approx(0.02)
    assert maup.loc[0, "moran_i_r8"] == pytest.approx(0.2)


def _load_table(name: str) -> pd.DataFrame:
    path = TABLES / name
    assert path.is_file() and path.stat().st_size > 0, f"missing contracted table {path}"
    return pd.read_csv(path)


def test_contract_artifacts_match_project_md_schemas_and_invariants() -> None:
    r9 = gpd.read_file(DATA_INTERIM_DIR / "cells_r9_with_2sfca_lisa.gpkg")
    r8 = gpd.read_file(DATA_INTERIM_DIR / "cells_r8_with_2sfca.gpkg")
    zones = gpd.read_file(DATA_INTERIM_DIR / "zones_23_with_2sfca.gpkg")
    base_r9 = gpd.read_file(DATA_FROZEN_DIR / "cells_base_res9.gpkg")
    base_r8 = gpd.read_file(DATA_FROZEN_DIR / "cells_base_res8.gpkg")
    base_zones = gpd.read_file(DATA_FROZEN_DIR / "zones_23.gpkg")
    facilities = gpd.read_file(DATA_FROZEN_DIR / "facilities_clean.gpkg")

    for layer in (r9, r8, zones):
        assert layer.crs is not None and layer.crs.to_epsg() == 32649
        for kind in ACC_TYPES:
            for cutoff in (10, 15):
                column = f"acc{cutoff}_{kind}"
                assert column in layer.columns
                values = layer[column].to_numpy(dtype=float)
                assert np.isfinite(values).all()
                if kind != "composite":
                    assert np.min(values) >= 0.0

    assert r9["cell_id"].tolist() == base_r9["cell_id"].tolist()
    assert np.allclose(r9["pop_total"], base_r9["pop_total"])
    assert r8["cell_id"].tolist() == base_r8["cell_id"].tolist()
    assert zones["zone_id"].tolist() == sorted(base_zones["zone_id"].tolist())
    assert len(zones) == 23

    lisa_metrics = [
        c.removeprefix("lisa_cluster_") for c in r9.columns if c.startswith("lisa_cluster_")
    ]
    assert len(lisa_metrics) >= 12
    for metric in lisa_metrics:
        clusters = r9[f"lisa_cluster_{metric}"].astype(str)
        pvals = r9[f"lisa_p_{metric}"].to_numpy(dtype=float)
        qvals = r9[f"lisa_q_fdr_{metric}"].to_numpy(dtype=float)
        local_i = r9[f"lisa_i_{metric}"].to_numpy(dtype=float)
        assert set(clusters).issubset({"HH", "LL", "LH", "HL", "NS"})
        assert np.all(qvals + 1e-9 >= pvals)
        assert np.all((clusters == "NS") | (qvals <= 0.05 + 1e-9))
        assert np.all((qvals <= 0.05 + 1e-9) | (clusters == "NS"))
        assert np.all(local_i[clusters.isin(["HH", "LL"])] >= -1e-9)
        assert np.all(local_i[clusters.isin(["HL", "LH"])] <= 1e-9)

    pop9 = r9["pop_total"].to_numpy(dtype=float)
    supply = facilities.groupby("facility_type")["capacity"].sum().to_dict()
    n_ev = int((facilities["facility_type"] == "ev_charger").sum())
    for cutoff in (10, 15):
        for facility in FACILITY_TYPES:
            reached = float(np.sum(pop9 * r9[f"acc{cutoff}_{facility}"].to_numpy(dtype=float)))
            ratio = reached / float(supply[facility])
            floor = 0.50 if facility == "ev_charger" else 0.70
            assert floor <= ratio <= 1.001
        unw = r9[f"acc{cutoff}_ev_unweighted"].to_numpy(dtype=float)
        reached_unweighted = float(np.sum(pop9 * unw))
        assert 0.0 < reached_unweighted <= n_ev * 1.001
        weighted = r9[f"acc{cutoff}_ev_charger"].to_numpy(dtype=float)
        unweighted = r9[f"acc{cutoff}_ev_unweighted"].to_numpy(dtype=float)
        assert np.all((weighted > 1e-12) == (unweighted > 1e-12))
        assert np.all(weighted + 1e-8 >= unweighted)
        for facility in (*FACILITY_TYPES, "ev_unweighted"):
            a10 = r9[f"acc10_{facility}"].to_numpy(dtype=float)
            a15 = r9[f"acc15_{facility}"].to_numpy(dtype=float)
            assert np.all((a10 <= 1e-12) | (a15 > 0.0))
            assert float(np.sum(pop9 * a15)) >= float(np.sum(pop9 * a10)) - 1e-4

    h1 = _load_table("h1_inequality_lisa.csv")
    assert list(h1.columns) == list(H1_COLS)
    keys = {
        (row.resolution, int(row.threshold_min), row.facility_type) for row in h1.itertuples()
    }
    for resolution in ("res9", "res8"):
        for cutoff in (10, 15):
            for facility in (*FACILITY_TYPES, "composite"):
                assert (resolution, cutoff, facility) in keys
    for row in h1.itertuples():
        assert 0.0 <= row.gini <= 1.0
        assert 0.0 <= row.gini_ci_low < row.gini_ci_high <= 1.0
        assert row.gini_ci_low - 0.015 <= row.gini <= row.gini_ci_high + 0.015
        assert abs(row.theil_total - (row.theil_within + row.theil_between)) < 1e-6
        assert 0.0 <= row.theil_total_ci_low <= row.theil_total_ci_high
        assert row.theil_total_ci_low - 0.02 <= row.theil_total <= row.theil_total_ci_high + 0.02
        assert 0.001 - 1e-9 <= row.moran_p <= 1.0
        assert 0.0 <= row.zero_acc_pct < 100.0
    for layer, resolution in ((r9, "res9"), (r8, "res8")):
        pops = layer["pop_total"].to_numpy(dtype=float)
        parishes = layer["parish_id"].astype(str).to_numpy()
        subset = h1[h1["resolution"] == resolution]
        for row in subset.itertuples():
            column = f"acc{int(row.threshold_min)}_{row.facility_type}"
            if column not in layer.columns or row.facility_type == "composite":
                continue
            vals = layer[column].to_numpy(dtype=float)
            assert row.gini == pytest.approx(weighted_gini(vals, pops), abs=1e-6)
            parts = weighted_theil_t(vals, pops, parishes)
            assert row.theil_total == pytest.approx(parts["total"], abs=1e-6)
            assert row.theil_within == pytest.approx(parts["within"], abs=1e-6)
            assert row.theil_between == pytest.approx(parts["between"], abs=1e-6)

    h2 = _load_table("h2_gini_diff_bootstrap.csv")
    assert list(h2.columns) == list(H2_COLS)
    assert len(h2) == 8
    assert set(h2["n_boot"]) == {2000}
    assert set(h2["seed"]) == {RANDOM_SEED}
    seen = {(int(r.threshold_min), r.comparison) for r in h2.itertuples()}
    for cutoff in (10, 15):
        for facility in TRADITIONAL:
            assert (cutoff, f"ev_charger_vs_{facility}") in seen
    h1_r9 = h1[h1["resolution"] == "res9"]
    gini_lookup = {
        (int(r.threshold_min), r.facility_type): r.gini for r in h1_r9.itertuples()
    }
    for row in h2.itertuples():
        facility = row.comparison.removeprefix("ev_charger_vs_")
        assert row.gini_diff == pytest.approx(row.gini_ev - row.gini_other, abs=1e-8)
        assert row.gini_ev == pytest.approx(gini_lookup[(int(row.threshold_min), "ev_charger")])
        assert row.gini_other == pytest.approx(gini_lookup[(int(row.threshold_min), facility)])
        assert row.ci_low < row.ci_high
        assert 0.0 <= row.p_boot_raw <= row.p_holm <= 1.0
    holm = holm_bonferroni(h2["p_boot_raw"].tolist())
    assert np.allclose(h2["p_holm"].to_numpy(dtype=float), holm, atol=1e-8)

    h3 = _load_table("h3_spatial_regression.csv")
    assert list(h3.columns) == list(H3_COLS)
    assert set(h3["n_obs"]) == {23}
    groups: dict[tuple[str, str, str], set[str]] = {}
    for row in h3.itertuples():
        key = (str(row.threshold_min), row.outcome, row.weights_spec)
        groups.setdefault(key, set()).add(row.model_type)
        assert row.se > 0.0
        assert row.ci_low < row.coef < row.ci_high
        width = (row.ci_high - row.ci_low) / (2.0 * row.se)
        assert 1.85 <= width <= 2.35
        assert 0.0 < row.resid_moran_p <= 1.0
    for models in groups.values():
        assert models == {"OLS_HC3", "SEM_ML", "SLM_ML"}
    assert {"knn4", "bridge_queen"}.issubset(set(h3["weights_spec"]))
    sem_terms = set(h3.loc[h3["model_type"] == "SEM_ML", "term"])
    slm_terms = set(h3.loc[h3["model_type"] == "SLM_ML", "term"])
    assert any("lambda" in term for term in sem_terms)
    assert any(term == "rho" for term in slm_terms)

    outcome = "acc15_composite"
    block = h3[
        (h3["threshold_min"] == 15)
        & (h3["outcome"] == outcome)
        & (h3["weights_spec"] == "knn4")
        & (h3["model_type"] == "OLS_HC3")
    ]
    fit = ols_hc3(
        zones[outcome].to_numpy(dtype=float),
        zones[["elderly_share", "log_pop_density"]].to_numpy(dtype=float),
        ["elderly_share", "log_pop_density"],
    )
    for row in block.itertuples():
        if row.term in fit["coefs"]:
            assert row.coef == pytest.approx(fit["coefs"][row.term], abs=1e-6)
            assert row.se == pytest.approx(fit["se"][row.term], abs=1e-6)

    diagnostics_path = TABLES / "h3_regression_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["vif"]["elderly_share"] >= 1.0
    assert diagnostics["vif"]["log_pop_density"] >= 1.0
    assert diagnostics["weights"]["knn4"]["n_islands"] == 0
    assert diagnostics["weights"]["bridge_queen"]["n_islands"] == 0
    blob = json.dumps(diagnostics).lower()
    assert "cause" not in blob and "elasticity" not in blob

    maup = _load_table("maup_r8_vs_r9.csv")
    assert {"gini_r8", "gini_r9", "theil_r8", "theil_r9", "moran_i_r8", "moran_i_r9"}.issubset(
        maup.columns
    )
    assert set(FACILITY_TYPES).issubset(set(maup["facility_type"]))
    for row in maup.itertuples():
        ref9 = h1[
            (h1["resolution"] == "res9")
            & (h1["threshold_min"] == row.threshold_min)
            & (h1["facility_type"] == row.facility_type)
        ].iloc[0]
        ref8 = h1[
            (h1["resolution"] == "res8")
            & (h1["threshold_min"] == row.threshold_min)
            & (h1["facility_type"] == row.facility_type)
        ].iloc[0]
        assert row.gini_r9 == pytest.approx(ref9.gini)
        assert row.gini_r8 == pytest.approx(ref8.gini)
        assert row.theil_r9 == pytest.approx(ref9.theil_total)
        assert row.moran_i_r8 == pytest.approx(ref8.moran_i)

    lorenz = _load_table("lorenz_curves.csv")
    assert {"threshold_min", "facility_type", "cum_pop_share", "cum_acc_share"}.issubset(
        lorenz.columns
    )
    for (cutoff, facility), curve in lorenz.groupby(["threshold_min", "facility_type"]):
        p = curve["cum_pop_share"].to_numpy(dtype=float)
        q = curve["cum_acc_share"].to_numpy(dtype=float)
        assert p[0] <= 0.05 and q[0] <= 0.05
        assert abs(p[-1] - 1.0) <= 1e-6 and abs(q[-1] - 1.0) <= 1e-6
        assert np.all(q <= p + 1e-4)
        area = float(np.sum(0.5 * np.diff(p) * (q[1:] + q[:-1])))
        table_gini = float(
            h1_r9[
                (h1_r9["threshold_min"] == int(cutoff)) & (h1_r9["facility_type"] == facility)
            ]["gini"].iloc[0]
        )
        assert abs((1.0 - 2.0 * area) - table_gini) <= 0.02

    eda = _load_table("eda_summary.csv")
    assert {"min", "median", "mean", "max", "std"}.issubset({c.lower() for c in eda.columns})
    text = " ".join(eda.astype(str).to_numpy().ravel()).lower()
    for facility in FACILITY_TYPES:
        assert facility in text
    assert "pop" in text
    assert len(eda) >= 10

    landmass = np.where(
        r9["parish_id"].isin(PENINSULA_PARISH_IDS),
        "peninsula",
        np.where(r9["parish_id"].isin(ISLANDS_PARISH_IDS), "islands", "other"),
    )
    assert set(np.unique(landmass)) <= {"peninsula", "islands"}
    safe = island_safe_knn_weights(r9, landmass, k=4)
    assert weights_diagnostics(safe)["n_islands"] == 0
    for origin, neighbors in safe.neighbors.items():
        for dest in neighbors:
            assert landmass[origin] == landmass[dest]
