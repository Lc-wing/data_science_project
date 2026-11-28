"""Frozen 5 geographic spatial CV blocks and 10-min pedestrian graph edges (A06)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from macao_equity.config import (
    ANALYSIS_CRS,
    CUTOFF_10_MIN,
    PENINSULA_PARISH_IDS,
    RANDOM_SEED,
    WALK_SPEED_M_PER_MIN,
    WGS84_CRS,
    make_transformer,
)


def build_spatial_folds(
    cells_r9_gdf: gpd.GeoDataFrame,
    out_folds_csv: Path,
    seed: int = RANDOM_SEED,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Assign deterministic 5 geographic CV blocks via Island-Aware KMeans (Peninsula=3, Islands=2).

    Canonicalizes block labels 0..4 sorted by descending mean centroid_y (North -> South)
    so block IDs are 100% invariant to KMeans label permutation.
    """
    out = cells_r9_gdf.sort_values("cell_id").reset_index(drop=True).copy()
    is_pen = out["parish_id"].isin(PENINSULA_PARISH_IDS).to_numpy()

    raw_labels = np.full(len(out), -1, dtype=int)

    pen_idx = np.where(is_pen)[0]
    isl_idx = np.where(~is_pen)[0]
    if len(pen_idx) < 3 or len(isl_idx) < 2:
        raise ValueError("Insufficient Peninsula or Islands cells for 3+2 spatial blocking")

    pen_xy = out.loc[pen_idx, ["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    isl_xy = out.loc[isl_idx, ["centroid_x", "centroid_y"]].to_numpy(dtype=float)

    km_pen = KMeans(n_clusters=3, random_state=seed, n_init=20)
    km_isl = KMeans(n_clusters=2, random_state=seed, n_init=20)

    raw_labels[pen_idx] = km_pen.fit_predict(pen_xy)
    raw_labels[isl_idx] = 3 + km_isl.fit_predict(isl_xy)

    out["_raw_block"] = raw_labels
    mean_y_by_raw = out.groupby("_raw_block")["centroid_y"].mean().to_dict()
    ordered_raw = sorted(mean_y_by_raw.keys(), key=lambda r: (-mean_y_by_raw[r], r))
    raw_to_canonical = {raw_id: canon_id for canon_id, raw_id in enumerate(ordered_raw)}

    out["spatial_block"] = out["_raw_block"].map(raw_to_canonical).astype(int)
    out = out.drop(columns=["_raw_block"])

    blocks_meta: list[dict[str, Any]] = []
    for blk_id in range(5):
        sub = out[out["spatial_block"] == blk_id]
        n_cells = int(len(sub))
        n_pos = int(sub["has_ev_charger"].sum())
        n_neg = int(n_cells - n_pos)
        if n_pos < 1 or n_neg < 1:
            raise ValueError(
                f"Spatial block {blk_id} violates positive/negative class guard: "
                f"n_pos={n_pos}, n_neg={n_neg}"
            )
        parishes = sorted(sub["parish_id"].unique().tolist())
        landmass = (
            "Macao Peninsula"
            if all(p in PENINSULA_PARISH_IDS for p in parishes)
            else "Taipa-Cotai-Coloane"
        )
        blocks_meta.append(
            {
                "block_id": blk_id,
                "landmass": landmass,
                "parishes": parishes,
                "n_cells": n_cells,
                "n_pos_ev": n_pos,
                "n_neg_ev": n_neg,
                "prevalence": round(float(n_pos / n_cells), 4),
                "mean_centroid_x": round(float(sub["centroid_x"].mean()), 2),
                "mean_centroid_y": round(float(sub["centroid_y"].mean()), 2),
            }
        )

    folds_df = out[["cell_id", "spatial_block"]].copy()
    out_folds_csv.parent.mkdir(parents=True, exist_ok=True)
    folds_df.to_csv(out_folds_csv, index=False)
    return out, folds_df, blocks_meta


def build_10min_pedestrian_graph(
    cells_r9_gdf: gpd.GeoDataFrame,
    G_walk: nx.MultiDiGraph,
    out_edges_csv: Path,
    cutoff_min: float = CUTOFF_10_MIN,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build symmetric 10-min pedestrian walk graph edges between populated H3 res-9 cells.

    Includes centroid-to-nearest-node snap walk time at 4.5 km/h (75.0 m/min) and guards
    same-snap-node cell pairs with ``max(snap_u + snap_v, euclid_uv)`` so ``walk_time_min > 0``
    strictly for all distinct cell pairs.
    """
    tr_4326_to_32649 = make_transformer(WGS84_CRS, ANALYSIS_CRS)
    node_ids = list(G_walk.nodes)
    xs = [float(G_walk.nodes[n]["x"]) for n in node_ids]
    ys = [float(G_walk.nodes[n]["y"]) for n in node_ids]
    ux, uy = tr_4326_to_32649.transform(xs, ys)
    node_xy = np.column_stack([ux, uy])
    tree = cKDTree(node_xy)

    cells_sorted = cells_r9_gdf.sort_values("cell_id").reset_index(drop=True)
    cell_ids = cells_sorted["cell_id"].astype(str).tolist()
    cxs = cells_sorted["centroid_x"].to_numpy(dtype=float)
    cys = cells_sorted["centroid_y"].to_numpy(dtype=float)
    blocks = cells_sorted["spatial_block"].to_numpy(dtype=int)

    snap_dists_m, snap_idxs = tree.query(np.column_stack([cxs, cys]))
    snap_dists_m = snap_dists_m.astype(float)
    snap_nodes = [node_ids[int(i)] for i in snap_idxs]

    snap_under_300_pct = float(100.0 * np.mean(snap_dists_m < 300.0))
    if snap_under_300_pct < 95.0:
        raise ValueError(
            f"Walk-graph snap rate < 300 m is {snap_under_300_pct:.2f}% (< 95.0% requirement)"
        )

    cutoff_m = cutoff_min * WALK_SPEED_M_PER_MIN
    unique_snap_nodes = sorted(set(snap_nodes))
    dijkstra_cache: dict[Any, dict[Any, float]] = {}
    for sn in unique_snap_nodes:
        dijkstra_cache[sn] = nx.single_source_dijkstra_path_length(
            G_walk, sn, cutoff=cutoff_m, weight="length"
        )

    edge_rows: list[dict[str, Any]] = []
    same_node_pair_count = 0
    n_cells = len(cell_ids)

    for i in range(n_cells):
        sd_i = float(snap_dists_m[i])
        if sd_i >= cutoff_m:
            continue
        sn_i = snap_nodes[i]
        reach_i = dijkstra_cache[sn_i]

        for j in range(i + 1, n_cells):
            sd_j = float(snap_dists_m[j])
            if sd_j >= cutoff_m:
                continue
            sn_j = snap_nodes[j]

            if sn_i == sn_j:
                euclid_m = math.hypot(float(cxs[i] - cxs[j]), float(cys[i] - cys[j]))
                total_m = max(sd_i + sd_j, euclid_m)
                same_node_pair_count += 1
            elif sn_j in reach_i:
                net_ij = float(reach_i[sn_j])
                # Symmetrize network distance in case of tiny floating-point asymmetry
                net_ji = float(dijkstra_cache[sn_j].get(sn_i, net_ij))
                net_m = 0.5 * (net_ij + net_ji)
                total_m = sd_i + net_m + sd_j
            else:
                continue

            walk_time_min = total_m / WALK_SPEED_M_PER_MIN
            if 0.0 < walk_time_min <= cutoff_min:
                weight = 1.0 / walk_time_min
                same_blk = bool(blocks[i] == blocks[j])
                u = cell_ids[i]
                v = cell_ids[j]
                edge_rows.append(
                    {
                        "src_cell_id": u,
                        "dst_cell_id": v,
                        "walk_time_min": walk_time_min,
                        "weight": weight,
                        "same_block": same_blk,
                    }
                )
                edge_rows.append(
                    {
                        "src_cell_id": v,
                        "dst_cell_id": u,
                        "walk_time_min": walk_time_min,
                        "weight": weight,
                        "same_block": same_blk,
                    }
                )

    edges_df = pd.DataFrame(
        edge_rows,
        columns=["src_cell_id", "dst_cell_id", "walk_time_min", "weight", "same_block"],
    )
    edges_df = edges_df.sort_values(["src_cell_id", "dst_cell_id"]).reset_index(drop=True)

    if (edges_df["src_cell_id"] == edges_df["dst_cell_id"]).any():
        raise ValueError("Self-loop detected in graph_edges.csv")
    if (edges_df["walk_time_min"] <= 0.0).any() or (edges_df["walk_time_min"] > cutoff_min).any():
        raise ValueError("walk_time_min out of (0, cutoff_min] bounds")

    out_edges_csv.parent.mkdir(parents=True, exist_ok=True)
    edges_df.to_csv(out_edges_csv, index=False)

    cross_block_edges = int((~edges_df["same_block"]).sum())
    total_edges = int(len(edges_df))
    graph_audit = {
        "cutoff_min": cutoff_min,
        "cutoff_m": cutoff_m,
        "walk_speed_m_per_min": WALK_SPEED_M_PER_MIN,
        "snap_under_300m_pct": round(snap_under_300_pct, 2),
        "snap_dist_median_m": round(float(np.median(snap_dists_m)), 2),
        "snap_dist_p95_m": round(float(np.percentile(snap_dists_m, 95)), 2),
        "snap_dist_max_m": round(float(np.max(snap_dists_m)), 2),
        "total_directed_edges": total_edges,
        "total_undirected_pairs": total_edges // 2,
        "same_snap_node_pairs_guarded": same_node_pair_count,
        "min_walk_time_min": round(float(edges_df["walk_time_min"].min()), 4)
        if total_edges
        else None,
        "max_walk_time_min": round(float(edges_df["walk_time_min"].max()), 4)
        if total_edges
        else None,
        "cross_block_edge_count": cross_block_edges,
        "cross_block_edge_share": round(cross_block_edges / total_edges, 4)
        if total_edges
        else 0.0,
    }
    return edges_df, graph_audit


def write_split_metadata(
    out_json_path: Path,
    cells_r9_gdf: gpd.GeoDataFrame,
    blocks_meta: list[dict[str, Any]],
    graph_audit: dict[str, Any],
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Write canonical data/frozen/split_metadata.json."""
    n_cells = int(len(cells_r9_gdf))
    n_pos = int(cells_r9_gdf["has_ev_charger"].sum())
    n_neg = n_cells - n_pos
    meta = {
        "method": "island_aware_kmeans",
        "description": (
            "Island-Aware Geographic KMeans on EPSG:32649 cell centroids "
            "(Macao Peninsula k=3, Taipa-Cotai-Coloane k=2), "
            "canonically ordered North->South (0..4)"
        ),
        "seed": seed,
        "n_init": 20,
        "n_blocks": 5,
        "peninsula_k": 3,
        "islands_k": 2,
        "n_cells_total": n_cells,
        "n_pos_ev_total": n_pos,
        "n_neg_ev_total": n_neg,
        "prevalence_total": round(float(n_pos / n_cells), 4),
        "cross_block_edge_count": graph_audit["cross_block_edge_count"],
        "cross_block_edge_share": graph_audit["cross_block_edge_share"],
        "total_directed_edges": graph_audit["total_directed_edges"],
        "total_undirected_pairs": graph_audit["total_undirected_pairs"],
        "blocks": blocks_meta,
    }
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta
