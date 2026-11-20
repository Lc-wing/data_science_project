"""Tier 3 E2E Tests: 12 pairwise cross-artifact and independent pure-math oracle tests."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from e2e_lib.contracts import (
    ALL_ACC_COLS,
    CELLS_BASE_R8_REQUIRED_COLS,
    CELLS_BASE_R9_REQUIRED_COLS,
    FACILITY_TYPES,
    FROZEN_CORE_HASHED_FILES,
    GLOBAL_SEED,
    GRAPH_EDGES_REQUIRED_COLS,
    ML_METRICS,
    ML_MODEL_NAMES,
    POI_TEXTS_REQUIRED_COLS,
    POP_TOL_H3_65PLUS,
    POP_TOL_H3_TOTAL,
    SHA256_RE,
    SPATIAL_FOLDS_REQUIRED_COLS,
    normalize_resolution_label,
)
from e2e_lib.oracle import (
    average_precision_score,
    bh_fdr,
    brier_score,
    ols_hc3_and_vif,
    pearson_corr,
    point_distance_to_polygons,
    roc_auc_score,
    top20_hit_rates,
    weighted_gini,
    weighted_theil_t,
)
from e2e_lib.readers import (
    GpkgLayer,
    PdfInfo,
    as_bool,
    as_float,
    file_sha256,
    read_csv_strict,
)


def _extract_freeze_file_map(manifest: dict[str, Any]) -> dict[str, str]:
    raw_map: dict[str, Any] = {}
    for key in ("files", "checksums", "sha256", "artifacts"):
        val = manifest.get(key)
        if isinstance(val, dict):
            raw_map = val
            break
    if not raw_map:
        raw_map = {
            k: v
            for k, v in manifest.items()
            if isinstance(v, str) and SHA256_RE.match(v.strip().lower())
        }
    out: dict[str, str] = {}
    for rel, val in raw_map.items():
        if isinstance(val, dict):
            digest = str(val.get("sha256", "")).strip().lower()
        else:
            digest = str(val).strip().lower()
        norm_rel = rel if rel.startswith("data/") else f"data/frozen/{rel}"
        out[norm_rel] = digest
    return out


def _find_lorenz_cols(headers: list[str]) -> tuple[str, str]:
    pop_col = next(
        (h for h in headers if h.lower() in {"cum_pop_share", "pop_share", "cum_pop", "p", "x"}),
        None,
    )
    acc_col = next(
        (h for h in headers if h.lower() in {"cum_acc_share", "acc_share", "cum_acc", "q", "y"}),
        None,
    )
    assert pop_col and acc_col
    return pop_col, acc_col


def _find_oof_cols(headers: list[str]) -> tuple[str, dict[str, str]]:
    y_col = next((h for h in headers if h.lower() in {"has_ev_charger", "y_true", "target"}), None)
    assert y_col
    model_cols: dict[str, str] = {}
    for m_name in ML_MODEL_NAMES:
        candidates = [f"prob_{m_name}", f"pred_{m_name}", f"{m_name}_prob", m_name]
        found = next((h for h in headers if h.lower() in candidates), None)
        if found is None:
            token = m_name.split("_")[0]
            found = next((h for h in headers if token in h.lower() and h != y_col), None)
        assert found
        model_cols[m_name] = found
    return y_col, model_cols


def _match_h2_facility(comparison: str) -> str | None:
    low = comparison.lower()
    if "park" in low:
        return "park"
    if "health" in low:
        return "health_centre"
    if "librar" in low:
        return "library"
    if "bus" in low:
        return "bus_stop"
    return None


def _extract_floats(text: str) -> list[float]:
    return [float(m.group(0)) for m in re.finditer(r"-?\d+\.\d+", text)]


def _assert_layers_verbatim_match(
    base_layer: GpkgLayer,
    enriched_layer: GpkgLayer,
    cols_to_compare: tuple[str, ...],
) -> None:
    base_by_id = {
        str(r["cell_id"]): (r, g) for r, g in zip(base_layer.rows, base_layer.geoms, strict=True)
    }
    enr_by_id = {
        str(r["cell_id"]): (r, g)
        for r, g in zip(enriched_layer.rows, enriched_layer.geoms, strict=True)
    }
    assert set(base_by_id.keys()) == set(enr_by_id.keys()), (
        f"cell_id mismatch between {base_layer.path.name} and {enriched_layer.path.name}"
    )
    for cid, (b_row, b_geom) in base_by_id.items():
        e_row, e_geom = enr_by_id[cid]
        assert b_geom.centroid is not None and e_geom.centroid is not None
        assert (
            math.hypot(
                b_geom.centroid[0] - e_geom.centroid[0],
                b_geom.centroid[1] - e_geom.centroid[1],
            )
            < 1e-5
        )
        assert abs(b_geom.area - e_geom.area) < 1e-3
        for col in cols_to_compare:
            b_val = b_row[col]
            e_val = e_row[col]
            if isinstance(b_val, (int, float)) or isinstance(e_val, (int, float)):
                assert abs(as_float(b_val) - as_float(e_val)) <= 1e-9, (
                    f"Column {col} drifted on {cid}: {b_val} vs {e_val}"
                )
            else:
                assert str(b_val) == str(e_val), (
                    f"Column {col} drifted on {cid}: {b_val} vs {e_val}"
                )


def test_t3_01_frozen_cells_preserve_m1_base_verbatim(
    project_root: Path,
    cells_r9: GpkgLayer,
    cells_r8: GpkgLayer,
    load_gpkg: Callable[..., GpkgLayer],
) -> None:
    """M1 (cells_base_res9/8) <-> M2 (interim) <-> M4 (cells/cells_res8) verbatim match."""
    base_r9_path = project_root / "data/frozen/cells_base_res9.gpkg"
    if base_r9_path.is_file():
        base_r9 = load_gpkg(base_r9_path)
        _assert_layers_verbatim_match(base_r9, cells_r9, CELLS_BASE_R9_REQUIRED_COLS)

    interim_r9_path = project_root / "data/interim/cells_r9_with_2sfca_lisa.gpkg"
    if interim_r9_path.is_file():
        interim_r9 = load_gpkg(interim_r9_path)
        _assert_layers_verbatim_match(
            interim_r9,
            cells_r9,
            (*CELLS_BASE_R9_REQUIRED_COLS, *ALL_ACC_COLS),
        )

    base_r8_path = project_root / "data/frozen/cells_base_res8.gpkg"
    if base_r8_path.is_file():
        base_r8 = load_gpkg(base_r8_path)
        _assert_layers_verbatim_match(base_r8, cells_r8, CELLS_BASE_R8_REQUIRED_COLS)

    interim_r8_path = project_root / "data/interim/cells_r8_with_2sfca.gpkg"
    if interim_r8_path.is_file():
        interim_r8 = load_gpkg(interim_r8_path)
        _assert_layers_verbatim_match(
            interim_r8,
            cells_r8,
            (*CELLS_BASE_R8_REQUIRED_COLS, *ALL_ACC_COLS),
        )


def test_t3_02_spatial_hierarchy_pop_and_parish_rollup(
    zones_23: GpkgLayer,
    parishes_7: GpkgLayer,
    cells_m1_r9: GpkgLayer,
    cells_m1_r8: GpkgLayer,
) -> None:
    cells_r9 = cells_m1_r9
    cells_r8 = cells_m1_r8
    """zones_23.gpkg <-> parishes_7.gpkg <-> cells.gpkg <-> cells_res8.gpkg pop rollup."""
    zone_to_parish = {str(r["zone_id"]): str(r["parish_id"]) for r in zones_23.rows}
    parish_ids = {str(r["parish_id"]) for r in parishes_7.rows}
    assert set(zone_to_parish.values()) == parish_ids

    # Roll up zones_23 by parish_id and compare to parishes_7
    z_rollup: dict[str, list[float]] = {pid: [0.0, 0.0, 0.0] for pid in parish_ids}
    for r in zones_23.rows:
        pid = str(r["parish_id"])
        z_rollup[pid][0] += as_float(r["pop_total"])
        z_rollup[pid][1] += as_float(r["pop_65plus"])
        z_rollup[pid][2] += as_float(r["area_km2"])

    for r in parishes_7.rows:
        pid = str(r["parish_id"])
        p_tot, p_65, p_area = z_rollup[pid]
        assert abs(as_float(r["pop_total"]) - p_tot) / max(p_tot, 1.0) <= 1e-3
        assert abs(as_float(r["pop_65plus"]) - p_65) / max(p_65, 1.0) <= 1e-3
        assert abs(as_float(r["area_km2"]) - p_area) / max(p_area, 1e-3) <= 5e-3

    # Foreign-key integrity on cells_r9 and cells_r8
    for label, layer in (("cells_r9", cells_r9), ("cells_r8", cells_r8)):
        for r in layer.rows:
            zid = str(r["zone_id"])
            pid = str(r["parish_id"])
            assert zid in zone_to_parish, f"{label} references unknown zone_id {zid}"
            assert zone_to_parish[zid] == pid, (
                f"{label} cell {r['cell_id']} maps zone {zid} to parish "
                f"{pid} != {zone_to_parish[zid]}"
            )

    # Multi-layer population conservation agreement
    z_pop = sum(as_float(r["pop_total"]) for r in zones_23.rows)
    p_pop = sum(as_float(r["pop_total"]) for r in parishes_7.rows)
    r9_pop = sum(as_float(r["pop_total"]) for r in cells_r9.rows)
    r8_pop = sum(as_float(r["pop_total"]) for r in cells_r8.rows)
    for other in (p_pop, r9_pop, r8_pop):
        assert abs(other - z_pop) / z_pop <= POP_TOL_H3_TOTAL

    z_65 = sum(as_float(r["pop_65plus"]) for r in zones_23.rows)
    for layer in (parishes_7, cells_r9, cells_r8):
        l_65 = sum(as_float(r["pop_65plus"]) for r in layer.rows)
        assert abs(l_65 - z_65) / z_65 <= POP_TOL_H3_65PLUS


def test_t3_03_facilities_to_cells_ev_counts_and_distances(
    facilities: GpkgLayer,
    lrt_stations: GpkgLayer,
    cells_m1_r9: GpkgLayer,
) -> None:
    cells_r9 = cells_m1_r9
    """facilities_clean.gpkg + lrt_stations.gpkg <-> cells.gpkg EV counts/capacity & distances."""
    ev_pts: list[tuple[float, float, int]] = []
    bus_pts: list[tuple[float, float]] = []
    for r, g in zip(facilities.rows, facilities.geoms, strict=True):
        assert g.point_xy is not None
        ft = str(r["facility_type"])
        if ft == "ev_charger" and not as_bool(r["in_zona_a"]):
            ev_pts.append((g.point_xy[0], g.point_xy[1], int(round(as_float(r["capacity"])))))
        elif ft == "bus_stop":
            bus_pts.append(g.point_xy)

    lrt_pts = [g.point_xy for g in lrt_stations.geoms if g.point_xy is not None]
    assert ev_pts and bus_pts and lrt_pts

    # Assign each non-Zona-A EV station to the containing or nearest H3 res-9 cell polygon
    recomp_cnt = [0] * len(cells_r9.rows)
    recomp_cap = [0] * len(cells_r9.rows)
    for ex, ey, ecap in ev_pts:
        best_idx = min(
            range(len(cells_r9.geoms)),
            key=lambda idx: point_distance_to_polygons(ex, ey, cells_r9.geoms[idx].polygons),
        )
        min_d = point_distance_to_polygons(ex, ey, cells_r9.geoms[best_idx].polygons)
        if min_d <= 150.0:
            recomp_cnt[best_idx] += 1
            recomp_cap[best_idx] += ecap

    obs_cnt = [int(r["ev_station_count"]) for r in cells_r9.rows]
    obs_cap = [int(r["ev_charger_capacity"]) for r in cells_r9.rows]
    obs_has = [int(r["has_ev_charger"]) for r in cells_r9.rows]
    assert sum(obs_cnt) > 0 and abs(sum(obs_cnt) - sum(recomp_cnt)) <= 4
    assert abs(sum(obs_cap) - sum(recomp_cap)) / max(sum(obs_cap), 1) <= 0.10
    matches_has = sum(1 for o, rc in zip(obs_has, recomp_cnt, strict=True) if o == int(rc > 0))
    assert matches_has / len(obs_has) >= 0.95

    # Nearest LRT and nearest bus stop distance correlation with cell centroids
    recomp_lrt_d: list[float] = []
    recomp_bus_d: list[float] = []
    obs_lrt_d: list[float] = []
    obs_bus_d: list[float] = []
    for r in cells_r9.rows:
        cx, cy = as_float(r["centroid_x"]), as_float(r["centroid_y"])
        recomp_lrt_d.append(min(math.hypot(cx - lx, cy - ly) for lx, ly in lrt_pts))
        recomp_bus_d.append(min(math.hypot(cx - bx, cy - by) for bx, by in bus_pts))
        obs_lrt_d.append(as_float(r["dist_nearest_lrt_m"]))
        obs_bus_d.append(as_float(r["dist_nearest_bus_m"]))
    assert pearson_corr(recomp_lrt_d, obs_lrt_d) >= 0.90
    assert pearson_corr(recomp_bus_d, obs_bus_d) >= 0.85


def test_t3_04_graph_folds_poi_metadata_four_way_join(
    cells_r9: GpkgLayer,
    spatial_folds: tuple[list[str], list[dict[str, str]]],
    split_metadata: dict[str, Any],
    graph_edges: tuple[list[str], list[dict[str, str]]],
    poi_texts: tuple[list[str], list[dict[str, str]]],
    feature_manifest: dict[str, Any],
    h2_table: tuple[list[str], list[dict[str, str]]],
    ml_baselines: dict[str, Any],
    provenance_info: tuple[list[Path], Any],
    load_json: Callable[[str | Path], Any],
) -> None:
    """5-way join cells.gpkg <-> spatial_folds <-> split_metadata <-> graph_edges <-> poi_texts."""
    cell_block = {str(r["cell_id"]): int(r["spatial_block"]) for r in cells_r9.rows}
    cell_ev = {str(r["cell_id"]): int(r["has_ev_charger"]) for r in cells_r9.rows}

    _, fold_rows = spatial_folds
    fold_block = {r["cell_id"]: int(float(r["spatial_block"])) for r in fold_rows}
    assert fold_block == cell_block

    _, poi_rows = poi_texts
    assert {r["cell_id"] for r in poi_rows} == set(cell_block.keys())

    _, edge_rows = graph_edges
    for r in edge_rows:
        u, v = r["src_cell_id"], r["dst_cell_id"]
        assert u in cell_block and v in cell_block
        assert as_bool(r["same_block"]) == (cell_block[u] == cell_block[v])

    blocks_meta = (
        split_metadata.get("blocks")
        or split_metadata.get("per_block")
        or split_metadata.get("folds")
    )
    assert blocks_meta is not None
    for blk in range(5):
        b_info = (
            blocks_meta[str(blk)]
            if isinstance(blocks_meta, dict) and str(blk) in blocks_meta
            else blocks_meta[blk]
        )
        cids = [cid for cid, b in cell_block.items() if b == blk]
        pos_cnt = sum(cell_ev[cid] for cid in cids)
        assert int(b_info["n_cells"]) == len(cids)
        assert int(b_info["n_pos_ev"]) == pos_cnt
        assert int(b_info["n_neg_ev"]) == len(cids) - pos_cnt

    # Cross-check seed == 42 across all 4 seeded artifacts
    _, h2_rows = h2_table
    assert feature_manifest["random_seed"] == GLOBAL_SEED
    assert split_metadata.get("seed", split_metadata.get("random_seed")) == GLOBAL_SEED
    assert ml_baselines["seed"] == GLOBAL_SEED
    assert {int(float(r["seed"])) for r in h2_rows} == {GLOBAL_SEED}

    # Dual-path data_provenance.json agreement when both exist
    prov_paths, prov_primary = provenance_info
    if len(prov_paths) == 2:
        prov_secondary = load_json(prov_paths[1])
        assert prov_primary == prov_secondary


def test_t3_05_recompute_h1_gini_theil_lisa_fdr_from_cells(
    cells_r9: GpkgLayer,
    cells_r8: GpkgLayer,
    h1_table: tuple[list[str], list[dict[str, str]]],
) -> None:
    """Independent recomputation of pop-weighted Gini, parish Theil's T, BH-FDR q, LISA counts."""
    _, h1_rows = h1_table
    by_res = {"res9": cells_r9, "res8": cells_r8}

    checked = 0
    for r in h1_rows:
        res = normalize_resolution_label(r["resolution"])
        if res not in by_res:
            continue
        layer = by_res[res]
        thr = int(float(r["threshold_min"]))
        ft = r["facility_type"].strip()
        col = f"acc{thr}_{ft}"
        if col not in layer.columns:
            continue

        pops = [as_float(row["pop_total"]) for row in layer.rows]
        parishes = [str(row["parish_id"]) for row in layer.rows]
        vals = [as_float(row[col]) for row in layer.rows]

        # Only recompute Gini/Theil on non-negative columns
        if min(vals) >= -1e-12:
            g_oracle = weighted_gini(vals, pops)
            t_oracle = weighted_theil_t(vals, pops, parishes)
            assert abs(as_float(r["gini"]) - g_oracle) <= 1e-3, (
                f"H1 Gini mismatch on ({res}, {col}): {r['gini']} vs {g_oracle:.6f}"
            )
            assert abs(as_float(r["theil_total"]) - t_oracle["total"]) <= 1e-4
            assert abs(as_float(r["theil_within"]) - t_oracle["within"]) <= 1e-4
            assert abs(as_float(r["theil_between"]) - t_oracle["between"]) <= 1e-4
            if t_oracle["total"] > 1e-6:
                assert abs(as_float(r["theil_between_share"]) - t_oracle["between_share"]) <= 1e-4

        # zero_acc_pct check (either cell-% or pop-%)
        n_zero = sum(1 for v in vals if v <= 1e-12)
        cell_zero_pct = 100.0 * n_zero / len(vals)
        pop_zero_pct = (
            100.0 * sum(p for v, p in zip(vals, pops, strict=True) if v <= 1e-12) / sum(pops)
        )
        rep_zero = as_float(r["zero_acc_pct"])
        assert min(abs(rep_zero - cell_zero_pct), abs(rep_zero - pop_zero_pct)) <= 0.1

        # LISA BH-FDR & n_ll_fdr / n_hh_fdr on res9
        p_col = f"lisa_p_{col}"
        q_col = f"lisa_q_fdr_{col}"
        c_col = f"lisa_cluster_{col}"
        if res == "res9" and c_col in layer.columns:
            pvals = [as_float(row[p_col]) for row in layer.rows]
            qvals_obs = [as_float(row[q_col]) for row in layer.rows]
            qvals_exp = bh_fdr(pvals)
            max_q_err = max(abs(a - b) for a, b in zip(qvals_obs, qvals_exp, strict=True))
            assert max_q_err <= 1e-5, f"BH-FDR q-value mismatch on {col}: max_err={max_q_err}"
            n_ll_obs = sum(1 for row in layer.rows if str(row[c_col]) == "LL")
            n_hh_obs = sum(1 for row in layer.rows if str(row[c_col]) == "HH")
            assert int(float(r["n_ll_fdr"])) == n_ll_obs
            assert int(float(r["n_hh_fdr"])) == n_hh_obs
        checked += 1
    assert checked >= 10


def test_t3_06_h1_lorenz_h2_maup_four_table_consistency(
    h1_table: tuple[list[str], list[dict[str, str]]],
    lorenz_table: tuple[list[str], list[dict[str, str]]],
    h2_table: tuple[list[str], list[dict[str, str]]],
    maup_table: tuple[list[str], list[dict[str, str]]],
) -> None:
    """h1_inequality_lisa.csv <-> lorenz_curves.csv <-> h2_gini_diff <-> maup_r8_vs_r9.csv."""
    _, h1_rows = h1_table
    h1_lookup = {
        (
            normalize_resolution_label(r["resolution"]),
            int(float(r["threshold_min"])),
            r["facility_type"].strip(),
        ): r
        for r in h1_rows
    }

    # 1. Lorenz trapezoid vs H1 res9 Gini
    l_headers, l_rows = lorenz_table
    p_col, q_col = _find_lorenz_cols(l_headers)
    curves: dict[tuple[int, str], list[tuple[float, float]]] = defaultdict(list)
    for r in l_rows:
        curves[(int(float(r.get("threshold_min", "15"))), r["facility_type"].strip())].append(
            (as_float(r[p_col]), as_float(r[q_col]))
        )
    for (thr, ft), pts in curves.items():
        if ("res9", thr, ft) in h1_lookup:
            area = sum(
                0.5 * (pts[i][0] - pts[i - 1][0]) * (pts[i][1] + pts[i - 1][1])
                for i in range(1, len(pts))
            )
            g_trapz = 1.0 - 2.0 * area
            g_h1 = as_float(h1_lookup[("res9", thr, ft)]["gini"])
            assert abs(g_trapz - g_h1) <= 2e-3

    # 2. H2 gini_ev and gini_other vs H1 res9
    _, h2_rows = h2_table
    for r in h2_rows:
        thr = int(float(r["threshold_min"]))
        ft = _match_h2_facility(r["comparison"])
        assert ft is not None
        assert (
            abs(as_float(r["gini_ev"]) - as_float(h1_lookup[("res9", thr, "ev_charger")]["gini"]))
            <= 1e-6
        )
        assert (
            abs(as_float(r["gini_other"]) - as_float(h1_lookup[("res9", thr, ft)]["gini"])) <= 1e-6
        )

    # 3. MAUP r8 & r9 vs H1
    _, m_rows = maup_table
    for r in m_rows:
        thr = int(float(r.get("threshold_min", "15")))
        ft = r["facility_type"].strip()
        h1_r8 = h1_lookup[("res8", thr, ft)]
        h1_r9 = h1_lookup[("res9", thr, ft)]
        if "gini_r8" in r and "gini_r9" in r:
            assert abs(as_float(r["gini_r8"]) - as_float(h1_r8["gini"])) <= 1e-5
            assert abs(as_float(r["gini_r9"]) - as_float(h1_r9["gini"])) <= 1e-5
        if "theil_r8" in r and "theil_r9" in r:
            assert abs(as_float(r["theil_r8"]) - as_float(h1_r8["theil_total"])) <= 1e-5
            assert abs(as_float(r["theil_r9"]) - as_float(h1_r9["theil_total"])) <= 1e-5
        if "moran_i_r8" in r and "moran_i_r9" in r:
            assert abs(as_float(r["moran_i_r8"]) - as_float(h1_r8["moran_i"])) <= 1e-5
            assert abs(as_float(r["moran_i_r9"]) - as_float(h1_r9["moran_i"])) <= 1e-5


def test_t3_07_recompute_h3_ols_hc3_and_vif_from_zones(
    zones_23_with_2sfca: GpkgLayer,
    h3_table: tuple[list[str], list[dict[str, str]]],
    h3_diagnostics: dict[str, Any],
) -> None:
    """Independent pure-Python OLS + HC3 sandwich SE & VIF from zones_23_with_2sfca.gpkg."""
    _, h3_rows = h3_table
    ols_rows = [
        r
        for r in h3_rows
        if r["model_type"].strip() == "OLS_HC3"
        and r["outcome"].strip() in zones_23_with_2sfca.columns
    ]
    assert ols_rows, "No OLS_HC3 rows match zones_23_with_2sfca outcome columns"

    x_eld = [as_float(z["elderly_share"]) for z in zones_23_with_2sfca.rows]
    x_lpd = [as_float(z["log_pop_density"]) for z in zones_23_with_2sfca.rows]

    by_outcome: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in ols_rows:
        by_outcome[r["outcome"].strip()].append(r)

    computed_vif: dict[str, float] = {}
    for outcome_col, rows_grp in by_outcome.items():
        y = [as_float(z[outcome_col]) for z in zones_23_with_2sfca.rows]
        oracle_res = ols_hc3_and_vif(y, [x_eld, x_lpd], ["elderly_share", "log_pop_density"])
        computed_vif = oracle_res["vif"]
        for r in rows_grp:
            term = r["term"].strip().lower()
            if term in {"const", "intercept"}:
                term_key = "const"
            elif term in {"elderly_share", "log_pop_density"}:
                term_key = term
            else:
                continue
            assert abs(as_float(r["coef"]) - oracle_res["coefs"][term_key]) <= 1e-4, (
                f"H3 OLS_HC3 coef mismatch on {outcome_col}.{term}: "
                f"{r['coef']} vs {oracle_res['coefs'][term_key]}"
            )
            assert abs(as_float(r["se"]) - oracle_res["se_hc3"][term_key]) <= 1e-4, (
                f"H3 OLS_HC3 SE mismatch on {outcome_col}.{term}: "
                f"{r['se']} vs {oracle_res['se_hc3'][term_key]}"
            )
            assert abs(as_float(r["r2_or_pseudo_r2"]) - float(oracle_res["r2"])) <= 1e-4

    vif_json = h3_diagnostics.get("vif", {})
    assert isinstance(vif_json, dict) and vif_json
    for pred in ("elderly_share", "log_pop_density"):
        assert abs(as_float(vif_json[pred]) - computed_vif[pred]) <= 1e-3


def test_t3_08_recompute_ml_oof_and_fold_metrics(
    cells_r9: GpkgLayer,
    feature_manifest: dict[str, Any],
    ml_oof_table: tuple[list[str], list[dict[str, str]]],
    ml_cv_table: tuple[list[str], list[dict[str, str]]],
    ml_baselines: dict[str, Any],
) -> None:
    """Independent recomputation of OOF and per-fold ROC-AUC, PR-AUC, Brier, Top-20% Hit Rate."""
    assert int(ml_baselines["n_cells"]) == len(cells_r9.rows)
    assert int(ml_baselines["n_pos"]) == sum(int(r["has_ev_charger"]) for r in cells_r9.rows)
    assert list(ml_baselines["feature_columns"]) == list(feature_manifest["ml_features"])

    cell_meta = {
        str(r["cell_id"]): (int(r["spatial_block"]), int(r["has_ev_charger"]))
        for r in cells_r9.rows
    }
    oof_headers, oof_rows = ml_oof_table
    y_col, model_cols = _find_oof_cols(oof_headers)
    assert len(oof_rows) == len(cell_meta)

    for r in oof_rows:
        cid = r["cell_id"]
        assert cid in cell_meta
        exp_blk, exp_y = cell_meta[cid]
        assert int(float(r["spatial_block"])) == exp_blk
        assert int(float(r[y_col])) == exp_y

    y_all = [int(float(r[y_col])) for r in oof_rows]
    blks_all = [int(float(r["spatial_block"])) for r in oof_rows]

    for m_name in ML_MODEL_NAMES:
        m_json = ml_baselines["models"][m_name]
        probs = [as_float(r[model_cols[m_name]]) for r in oof_rows]
        assert abs(roc_auc_score(y_all, probs) - as_float(m_json["oof_roc_auc"])) <= 1e-4
        assert abs(average_precision_score(y_all, probs) - as_float(m_json["oof_pr_auc"])) <= 1e-4
        assert abs(brier_score(y_all, probs) - as_float(m_json["oof_brier"])) <= 1e-4
        rec20, prec20 = top20_hit_rates(y_all, probs)
        rep20 = as_float(m_json["oof_top20_hit_rate"])
        assert min(abs(rep20 - rec20), abs(rep20 - prec20)) <= 0.02

        fold_list = m_json["fold_metrics"]
        for fm in fold_list:
            b_id = int(fm.get("fold", fm.get("spatial_block", fm.get("block", 0))))
            idx_b = [i for i, b in enumerate(blks_all) if b == b_id]
            yb = [y_all[i] for i in idx_b]
            pb = [probs[i] for i in idx_b]
            assert abs(roc_auc_score(yb, pb) - as_float(fm["roc_auc"])) <= 1e-4
            assert abs(average_precision_score(yb, pb) - as_float(fm["pr_auc"])) <= 1e-4
            assert abs(brier_score(yb, pb) - as_float(fm["brier"])) <= 1e-4

    # Cross-check ml_cv_results.csv against phase_a_ml_baselines.json
    cv_headers, cv_rows = ml_cv_table
    m_col = next((h for h in cv_headers if "model" in h.lower()), cv_headers[0])
    cv_by_model = {r[m_col].strip(): r for r in cv_rows}
    for m_name in ML_MODEL_NAMES:
        assert m_name in cv_by_model
        row = cv_by_model[m_name]
        m_json = ml_baselines["models"][m_name]
        for met in ML_METRICS:
            for prefix in ("mean", "std", "oof"):
                col_key = f"{prefix}_{met}"
                if col_key in row:
                    assert abs(as_float(row[col_key]) - as_float(m_json[col_key])) <= 1e-4


def test_t3_09_freeze_manifest_sha256_cryptographic_match(
    project_root: Path,
    freeze_manifest: dict[str, Any],
) -> None:
    """FREEZE_MANIFEST.json <-> live hashlib.sha256 cryptographic verification on frozen files."""
    fmap = _extract_freeze_file_map(freeze_manifest)
    for req_rel in FROZEN_CORE_HASHED_FILES:
        assert req_rel in fmap, (
            f"Required frozen deliverable {req_rel} missing from FREEZE_MANIFEST.json"
        )
    for rel, expected_digest in fmap.items():
        assert SHA256_RE.match(expected_digest)
        target_path = project_root / rel
        assert target_path.is_file(), f"Frozen file {rel} missing on disk"
        actual_digest = file_sha256(target_path)
        assert actual_digest == expected_digest, (
            f"Cryptographic SHA-256 mismatch on {rel}: "
            f"disk={actual_digest} != manifest={expected_digest}"
        )


def test_t3_10_table1_csv_md_tex_numeric_consistency(
    project_root: Path,
    h1_table: tuple[list[str], list[dict[str, str]]],
    h3_table: tuple[list[str], list[dict[str, str]]],
    ml_baselines: dict[str, Any],
) -> None:
    """h1, h3, phase_a_ml_baselines.json <-> table1_summary.{csv,md,tex} numeric consistency."""
    csv_txt = (project_root / "outputs/tables/table1_summary.csv").read_text(encoding="utf-8")
    md_txt = (project_root / "outputs/tables/table1_summary.md").read_text(encoding="utf-8")
    tex_txt = (project_root / "outputs/tables/table1_summary.tex").read_text(encoding="utf-8")

    csv_nums = _extract_floats(csv_txt)
    md_nums = _extract_floats(md_txt)
    tex_nums = _extract_floats(tex_txt)
    assert len(csv_nums) >= 8 and len(md_nums) >= 8 and len(tex_nums) >= 8

    # Every float in table1_summary.csv must appear in .md and .tex
    for val in csv_nums:
        assert min(abs(val - m) for m in md_nums) <= 1e-4, f"{val} missing from table1_summary.md"
        assert min(abs(val - t) for t in tex_nums) <= 1e-4, f"{val} missing from table1_summary.tex"

    # Match upstream H1 15m Gini, H3 N=23 OLS elderly_share coef, and ML OOF ROC-AUC/PR-AUC/Brier
    _, h1_rows = h1_table
    for r in h1_rows:
        if (
            normalize_resolution_label(r["resolution"]) == "res9"
            and int(float(r["threshold_min"])) == 15
            and r["facility_type"].strip() in FACILITY_TYPES
        ):
            g_val = as_float(r["gini"])
            assert min(abs(v - g_val) for v in csv_nums) <= 0.005, (
                f"H1 15m {r['facility_type']} Gini {g_val:.4f} missing from Table 1"
            )

    _, h3_rows = h3_table
    h3_ols_eld = next(
        as_float(r["coef"])
        for r in h3_rows
        if int(float(r["threshold_min"])) == 15
        and r["model_type"].strip() == "OLS_HC3"
        and r["term"].strip().lower() == "elderly_share"
    )
    assert min(abs(v - h3_ols_eld) for v in csv_nums) <= 0.005

    for m_name in ("logistic_regression", "random_forest"):
        auc = as_float(ml_baselines["models"][m_name]["oof_roc_auc"])
        assert min(abs(v - auc) for v in csv_nums) <= 0.005


def test_t3_11_report_tex_pdf_audit_numeric_traceability(
    cells_r9: GpkgLayer,
    h1_table: tuple[list[str], list[dict[str, str]]],
    ml_baselines: dict[str, Any],
    report_tex_text: str,
    report_pdf_info: PdfInfo,
    go_nogo_audit_text: str,
) -> None:
    """Frozen data + M2/M3 outputs <-> phase_a_draft.tex <-> compiled PDF <-> go_nogo_audit.md."""
    _, h1_rows = h1_table
    ev_gini_15 = next(
        as_float(r["gini"])
        for r in h1_rows
        if normalize_resolution_label(r["resolution"]) == "res9"
        and int(float(r["threshold_min"])) == 15
        and r["facility_type"].strip() == "ev_charger"
    )
    rf_auc = as_float(ml_baselines["models"]["random_forest"]["oof_roc_auc"])
    n_cells = len(cells_r9.rows)

    for doc_name, doc_txt in (
        ("phase_a_draft.tex", report_tex_text),
        ("PDF main_text", report_pdf_info.main_text),
        ("go_nogo_audit.md", go_nogo_audit_text),
    ):
        assert "681,293" in doc_txt or "681293" in doc_txt, f"681,293 missing from {doc_name}"
        assert "82,751" in doc_txt or "82751" in doc_txt, f"82,751 missing from {doc_name}"
        assert str(n_cells) in doc_txt, f"H3 r9 n_cells={n_cells} missing from {doc_name}"
        doc_floats = _extract_floats(doc_txt)
        assert min(abs(v - ev_gini_15) for v in doc_floats) <= 0.005, (
            f"15m EV Gini {ev_gini_15:.3f} missing from {doc_name}"
        )
        assert min(abs(v - rf_auc) for v in doc_floats) <= 0.005, (
            f"RF OOF ROC-AUC {rf_auc:.3f} missing from {doc_name}"
        )

    # Verify go_nogo_audit.md main-text page count matches actual PDF fractional page count
    page_m = re.search(r"(\d+\.\d+)\s*(?:a4\s*)?pages", go_nogo_audit_text, re.I)
    if page_m:
        audit_pages = float(page_m.group(1))
        assert abs(audit_pages - report_pdf_info.main_fractional_pages) <= 0.25, (
            f"go_nogo_audit.md page count {audit_pages} != "
            f"PDF {report_pdf_info.main_fractional_pages:.2f}"
        )


def test_t3_12_data_dictionary_covers_frozen_schema(
    project_root: Path,
    cells_r9: GpkgLayer,
    feature_manifest: dict[str, Any],
) -> None:
    """Frozen deliverables and feature_manifest <-> data_dictionary.md schema coverage."""
    dd_text = (project_root / "data/frozen/data_dictionary.md").read_text(encoding="utf-8")
    for col in cells_r9.columns:
        assert col in dd_text, f"cells.gpkg column {col!r} undocumented in data_dictionary.md"
    for col in GRAPH_EDGES_REQUIRED_COLS:
        assert col in dd_text, f"graph_edges.csv column {col!r} undocumented in data_dictionary.md"
    for col in SPATIAL_FOLDS_REQUIRED_COLS:
        assert col in dd_text, (
            f"spatial_folds.csv column {col!r} undocumented in data_dictionary.md"
        )
    for col in POI_TEXTS_REQUIRED_COLS:
        assert col in dd_text, (
            f"cell_poi_texts.csv column {col!r} undocumented in data_dictionary.md"
        )
    for feat in feature_manifest["ml_features"]:
        assert feat in dd_text, f"ml_features column {feat!r} undocumented in data_dictionary.md"
    for prefix in ("acc10_", "acc15_", "lisa_"):
        assert prefix in dd_text, (
            f"Anti-leakage prefix {prefix!r} undocumented in data_dictionary.md"
        )
    assert len(read_csv_strict(project_root / "data/frozen/graph_edges.csv")[1]) > 0
