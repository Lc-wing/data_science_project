"""H3 v4 res-9 & res-8 building-level census allocation, 11 leak-free ML features, and POI text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import h3
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid

from macao_equity.config import (
    ANALYSIS_CRS,
    ANALYSIS_PREFIX_EXCLUDED,
    COARSE_GEOGROUP_AREA_M2,
    DSEC_TERRESTRIAL_POP_65PLUS,
    DSEC_TERRESTRIAL_POP_TOTAL,
    EV_REGEX_EXT,
    EV_REGEX_SPEC,
    EXCLUDED_TARGET_AND_AUDIT,
    ML_FEATURES,
    PARISH_INFO,
    RANDOM_SEED,
    WGS84_CRS,
    ZONE_TO_PARISH,
    make_transformer,
    reproject_geom,
)

RES_BLDG_TYPES = frozenset(
    {
        "apartments",
        "residential",
        "house",
        "detached",
        "dormitory",
        "terrace",
        "semidetached_house",
        "bungalow",
    }
)

COM_BLDG_TYPES = frozenset(
    {
        "commercial",
        "retail",
        "office",
        "hotel",
        "supermarket",
        "mall",
        "kiosk",
        "shop",
    }
)

# Strictly non-EV commercial POI classes from Geofabrik gis_osm_pois_free / gis_osm_pois_a_free
NON_EV_COM_POI_CLASSES = frozenset(
    {
        "restaurant",
        "cafe",
        "fast_food",
        "bar",
        "pub",
        "food_court",
        "biergarten",
        "supermarket",
        "convenience",
        "clothes",
        "hairdresser",
        "bakery",
        "beauty_shop",
        "jewelry",
        "mall",
        "department_store",
        "shoe_shop",
        "optician",
        "books",
        "gift_shop",
        "furniture_shop",
        "mobile_phone_shop",
        "computer_shop",
        "electronics",
        "florist",
        "chemist",
        "laundry",
        "doityourself",
        "toy_shop",
        "sports_shop",
        "stationery",
        "kiosk",
        "butcher",
        "greengrocer",
        "beverages",
        "alcohol",
        "tea",
        "travel_agent",
        "newsagent",
        "copyshop",
        "tailor",
        "outdoor_shop",
        "bicycle_shop",
        "musical_instrument",
        "photo",
        "video_shop",
        "vending_machine",
        "hotel",
        "guest_house",
        "hostel",
        "motel",
        "bank",
        "atm",
        "pharmacy",
        "cinema",
        "theatre",
        "nightclub",
        "casino",
    }
)


def _extract_polygonal(geom: Any) -> Polygon | MultiPolygon:
    """Extract valid polygonal parts from a geometry or GeometryCollection."""
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    if geom.geom_type == "GeometryCollection":
        parts = [
            g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon") and g.area > 0.0
        ]
        if not parts:
            raise ValueError("GeometryCollection contains no positive-area polygons")
        merged = parts[0] if len(parts) == 1 else shapely.union_all(parts)
        return _extract_polygonal(merged)
    raise ValueError(f"Expected polygonal geometry, got {geom.geom_type}")


def refine_coarse_dsec_buildings(
    buildings_pop_gdf: gpd.GeoDataFrame,
    gpkg_path: Path,
    coarse_threshold_m2: float = COARSE_GEOGROUP_AREA_M2,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Refine coarse DSEC statistical geogroupings (> 100,000 m²) using OSM building footprints.

    3,721 / 3,733 (99.68%) DSEC populated records are compact urban buildings/estates (<= 0.10 km²).
    Only 12 coarse statistical disclosure geogroupings exceed 100,000 m² (e.g., OID=5322 covering
    the entire Macao International Airport runway + Pac On + MUST, 3.352 km²). Intersecting only
    those 12 coarse polygons with actual OSM building footprints prevents spawning unwalkable
    hexagons over airport runways/forests while preserving 100.0000% of every building record's
    official population.
    """
    osm_bldgs = gpd.read_file(gpkg_path, layer="gis_osm_buildings_a_free")
    if osm_bldgs.crs is None:
        osm_bldgs = osm_bldgs.set_crs(WGS84_CRS)
    osm_bldgs = osm_bldgs.to_crs(ANALYSIS_CRS)
    osm_bldgs["geometry"] = osm_bldgs.geometry.apply(
        lambda g: make_valid(g) if not g.is_valid else g
    )
    sindex = osm_bldgs.sindex

    refined_geoms: list[Any] = []
    coarse_refined_ids: list[int] = []
    coarse_fallback_ids: list[int] = []

    for _, row in buildings_pop_gdf.iterrows():
        geom = row.geometry
        oid = int(row["objectid"])
        if float(geom.area) <= coarse_threshold_m2:
            refined_geoms.append(geom)
            continue

        cand_idx = list(sindex.query(geom, predicate="intersects"))
        if not cand_idx:
            refined_geoms.append(geom)
            coarse_fallback_ids.append(oid)
            continue

        sub_osm = osm_bldgs.iloc[cand_idx]
        clipped_parts = []
        for o_geom in sub_osm.geometry:
            inter = geom.intersection(o_geom)
            if not inter.is_empty and inter.area > 0.0:
                if not inter.is_valid:
                    inter = make_valid(inter)
                clipped_parts.append(inter)

        if not clipped_parts:
            refined_geoms.append(geom)
            coarse_fallback_ids.append(oid)
            continue

        unioned = shapely.union_all(clipped_parts)
        if not unioned.is_valid:
            unioned = make_valid(unioned)
        if unioned.is_empty or unioned.area <= 1.0:
            refined_geoms.append(geom)
            coarse_fallback_ids.append(oid)
        else:
            refined_geoms.append(_extract_polygonal(unioned))
            coarse_refined_ids.append(oid)

    out = buildings_pop_gdf.copy()
    out["geometry"] = refined_geoms
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=ANALYSIS_CRS)

    audit = {
        "coarse_threshold_m2": coarse_threshold_m2,
        "total_populated_buildings": int(len(buildings_pop_gdf)),
        "coarse_geogroupings_refined_count": len(coarse_refined_ids),
        "coarse_geogroupings_refined_oids": sorted(coarse_refined_ids),
        "coarse_geogroupings_fallback_oids": sorted(coarse_fallback_ids),
    }
    return out, audit


def build_candidate_h3_grid(
    refined_buildings_gdf: gpd.GeoDataFrame,
    resolution: int,
) -> gpd.GeoDataFrame:
    """Seed and construct a complete covering H3 v4 polygon grid in EPSG:32649.

    Combines ``h3.geo_to_cells``, representative points, exterior vertices, and a 1-ring
    ``h3.grid_disk(cid, 1)`` halo so 100% of every building polygon's area is covered.
    """
    tr_32649_to_4326 = make_transformer(ANALYSIS_CRS, WGS84_CRS)
    tr_4326_to_32649 = make_transformer(WGS84_CRS, ANALYSIS_CRS)

    seeds: set[str] = set()
    for geom_32649 in refined_buildings_gdf.geometry:
        geom_4326 = reproject_geom(geom_32649, tr_32649_to_4326)
        if not geom_4326.is_valid:
            geom_4326 = _extract_polygonal(make_valid(geom_4326))

        try:
            seeds.update(h3.geo_to_cells(geom_4326.__geo_interface__, resolution))
        except Exception:  # noqa: BLE001
            pass

        rp = geom_4326.representative_point()
        seeds.add(h3.latlng_to_cell(rp.y, rp.x, resolution))

        polys = [geom_4326] if geom_4326.geom_type == "Polygon" else list(geom_4326.geoms)
        for poly in polys:
            for lng, lat in poly.exterior.coords:
                seeds.add(h3.latlng_to_cell(lat, lng, resolution))

    cand_cells = {nb for cid in seeds for nb in h3.grid_disk(cid, 1)}

    rows: list[dict[str, Any]] = []
    for cid in sorted(cand_cells):
        # h3 v4 cell_to_boundary returns ((lat, lng), ...) -> swap to (lng, lat)
        boundary_latlng = h3.cell_to_boundary(cid)
        poly_4326 = Polygon([(lng, lat) for lat, lng in boundary_latlng])
        poly_32649 = reproject_geom(poly_4326, tr_4326_to_32649)
        if not poly_32649.is_valid:
            poly_32649 = _extract_polygonal(make_valid(poly_32649))
        rows.append({"cell_id": cid, "geometry": poly_32649})

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=ANALYSIS_CRS)


def _assign_zone_and_parish(
    cells_gdf: gpd.GeoDataFrame,
    zones_23_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Link each populated H3 cell to its primary DSEC zone and official parish (zero NaNs)."""
    c_sub = cells_gdf[["cell_id", "geometry"]].copy()
    z_sub = zones_23_gdf[["zone_id", "zone_name", "geometry"]].copy()

    inter = gpd.overlay(c_sub, z_sub, how="intersection", keep_geom_type=False)
    inter["inter_area"] = inter.geometry.area
    inter = inter[inter["inter_area"] > 0.0].copy()
    inter = inter.sort_values(
        ["cell_id", "inter_area", "zone_id"], ascending=[True, False, True]
    )
    best_zone = inter.drop_duplicates(subset=["cell_id"], keep="first")[
        ["cell_id", "zone_id", "zone_name"]
    ]

    merged = cells_gdf.merge(best_zone, on="cell_id", how="left")

    # Fallback via nearest zone polygon if any shoreline cell had zero area overlap
    missing_mask = merged["zone_id"].isna()
    if missing_mask.any():
        missing_cells = merged.loc[missing_mask, ["cell_id", "geometry"]].copy()
        nearest = gpd.sjoin_nearest(
            missing_cells,
            z_sub,
            how="left",
        ).sort_values(["cell_id", "zone_id"]).drop_duplicates(subset=["cell_id"], keep="first")
        fallback_zone = dict(zip(nearest["cell_id"], nearest["zone_id"]))
        fallback_name = dict(zip(nearest["cell_id"], nearest["zone_name"]))
        merged.loc[missing_mask, "zone_id"] = merged.loc[missing_mask, "cell_id"].map(fallback_zone)
        merged.loc[missing_mask, "zone_name"] = merged.loc[missing_mask, "cell_id"].map(
            fallback_name
        )

    if merged["zone_id"].isna().any():
        raise ValueError("Unassigned zone_id remains in H3 cells")

    merged["parish_id"] = merged["zone_id"].map(ZONE_TO_PARISH)
    merged["parish_name"] = merged["parish_id"].map(lambda pid: PARISH_INFO[pid]["parish_name"])
    return merged


def allocate_buildings_to_h3(
    refined_buildings_gdf: gpd.GeoDataFrame,
    zones_23_gdf: gpd.GeoDataFrame,
    resolution: int,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Allocate building-level 2021 Census counts to H3 cells via normalized area overlay."""
    cand_grid = build_candidate_h3_grid(refined_buildings_gdf, resolution=resolution)

    b_sub = refined_buildings_gdf[["objectid", "total", "old", "geometry"]].copy()
    b_sub["bldg_area"] = b_sub.geometry.area

    inter = gpd.overlay(
        b_sub,
        cand_grid[["cell_id", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    inter["inter_area"] = inter.geometry.area
    inter = inter[inter["inter_area"] > 0.0].copy()

    matched_oids = set(inter["objectid"].unique())
    all_oids = set(b_sub["objectid"].unique())
    if matched_oids != all_oids:
        raise ValueError(
            f"H3 res-{resolution} candidate grid missed {len(all_oids - matched_oids)} buildings"
        )

    b_inter_sum = inter.groupby("objectid")["inter_area"].transform("sum")
    min_cov_ratio = float((b_inter_sum / inter["bldg_area"]).min())

    inter["weight"] = inter["inter_area"] / b_inter_sum
    inter["w_pop_total"] = inter["total"] * inter["weight"]
    inter["w_pop_65plus"] = inter["old"] * inter["weight"]

    cell_pops = (
        inter.groupby("cell_id", as_index=False)[["w_pop_total", "w_pop_65plus"]]
        .sum()
        .rename(columns={"w_pop_total": "pop_total", "w_pop_65plus": "pop_65plus"})
    )
    cell_pops = cell_pops[cell_pops["pop_total"] > 0.0].copy()

    cells = cand_grid.merge(cell_pops, on="cell_id", how="inner")
    cells = cells.sort_values("cell_id").reset_index(drop=True)

    centroids = cells.geometry.centroid
    cells["centroid_x"] = centroids.x.astype(float)
    cells["centroid_y"] = centroids.y.astype(float)
    cells["area_km2"] = (cells.geometry.area / 1e6).astype(float)
    cells["pop_total"] = cells["pop_total"].astype(float)
    cells["pop_65plus"] = cells["pop_65plus"].astype(float)
    cells["pop_density"] = (cells["pop_total"] / cells["area_km2"]).astype(float)
    cells["elderly_share"] = (cells["pop_65plus"] / cells["pop_total"]).clip(0.0, 1.0).astype(float)

    cells = _assign_zone_and_parish(cells, zones_23_gdf)

    tot_sum = float(cells["pop_total"].sum())
    old_sum = float(cells["pop_65plus"].sum())
    tot_rel_drift = abs(tot_sum - DSEC_TERRESTRIAL_POP_TOTAL) / DSEC_TERRESTRIAL_POP_TOTAL
    old_rel_drift = abs(old_sum - DSEC_TERRESTRIAL_POP_65PLUS) / DSEC_TERRESTRIAL_POP_65PLUS
    if tot_rel_drift >= 0.0005 or old_rel_drift >= 0.0005:
        raise ValueError(
            f"H3 res-{resolution} population drift exceeds 0.05%: "
            f"tot_drift={tot_rel_drift:.8%}, old_drift={old_rel_drift:.8%}"
        )

    audit = {
        "resolution": resolution,
        "candidate_cells_seeded": int(len(cand_grid)),
        "populated_cells_retained": int(len(cells)),
        "min_building_coverage_ratio": round(min_cov_ratio, 8),
        "pop_total_sum": round(tot_sum, 4),
        "pop_65plus_sum": round(old_sum, 4),
        "pop_total_rel_drift_pct": round(100.0 * tot_rel_drift, 8),
        "pop_65plus_rel_drift_pct": round(100.0 * old_rel_drift, 8),
    }
    return cells, audit


def _classify_osm_buildings_leak_free(
    gpkg_path: Path,
    buildings_pop_gdf: gpd.GeoDataFrame,
    coarse_threshold_m2: float = COARSE_GEOGROUP_AREA_M2,
) -> gpd.GeoDataFrame:
    """Attribute OSM buildings to residential/commercial weights without using any EV/car tags."""
    osm_bldgs = gpd.read_file(gpkg_path, layer="gis_osm_buildings_a_free")
    if osm_bldgs.crs is None:
        osm_bldgs = osm_bldgs.set_crs(WGS84_CRS)
    osm_bldgs = osm_bldgs.to_crs(ANALYSIS_CRS)
    osm_bldgs["geometry"] = osm_bldgs.geometry.apply(
        lambda g: make_valid(g) if not g.is_valid else g
    )
    osm_bldgs = osm_bldgs[osm_bldgs.geometry.area > 0.0].copy().reset_index(drop=True)
    osm_bldgs["bldg_idx"] = np.arange(len(osm_bldgs))
    osm_bldgs["bldg_area"] = osm_bldgs.geometry.area
    osm_bldgs["b_type"] = osm_bldgs["type"].fillna("").astype(str).str.strip().str.lower()

    # 1. Explicit OSM building type tags
    w_res = np.zeros(len(osm_bldgs), dtype=float)
    w_com = np.zeros(len(osm_bldgs), dtype=float)
    is_res_explicit = osm_bldgs["b_type"].isin(RES_BLDG_TYPES).to_numpy()
    is_com_explicit = osm_bldgs["b_type"].isin(COM_BLDG_TYPES).to_numpy()
    is_untagged = (osm_bldgs["b_type"] == "").to_numpy()

    w_res[is_res_explicit] = 1.0
    w_com[is_com_explicit] = 1.0

    # 2. For untagged OSM buildings (Trap 8: ~85% untagged), infer via DSEC compact buildings,
    #    landuse_a, and strictly non-EV commercial POIs
    untagged_gdf = osm_bldgs[is_untagged][["bldg_idx", "bldg_area", "geometry"]].copy()
    if not untagged_gdf.empty:
        # 2a. >= 15% overlap with compact DSEC populated residential buildings (<= 100,000 m²)
        dsec_compact = buildings_pop_gdf[
            buildings_pop_gdf.geometry.area <= coarse_threshold_m2
        ][["geometry"]].copy()
        ov_dsec = gpd.overlay(
            untagged_gdf, dsec_compact, how="intersection", keep_geom_type=False
        )
        ov_dsec["ov_area"] = ov_dsec.geometry.area
        dsec_area_by_idx = ov_dsec.groupby("bldg_idx")["ov_area"].sum()
        dsec_share = untagged_gdf["bldg_idx"].map(dsec_area_by_idx).fillna(0.0) / untagged_gdf[
            "bldg_area"
        ]
        r_dsec_ids = set(untagged_gdf.loc[dsec_share >= 0.15, "bldg_idx"])

        # 2b. Landuse polygon containment of representative points
        landuse_a = gpd.read_file(gpkg_path, layer="gis_osm_landuse_a_free").to_crs(ANALYSIS_CRS)
        rep_pts = untagged_gdf.copy()
        rep_pts["geometry"] = rep_pts.geometry.representative_point()

        lu_res = landuse_a[landuse_a["fclass"] == "residential"][["geometry"]]
        lu_com = landuse_a[landuse_a["fclass"].isin(["commercial", "retail"])][["geometry"]]

        r_lu_ids = set(
            gpd.sjoin(rep_pts, lu_res, how="inner", predicate="within")["bldg_idx"].unique()
        )
        c_lu_ids = set(
            gpd.sjoin(rep_pts, lu_com, how="inner", predicate="within")["bldg_idx"].unique()
        )

        # 2c. Non-EV commercial POIs inside untagged building footprints
        pois = gpd.read_file(gpkg_path, layer="gis_osm_pois_free").to_crs(ANALYSIS_CRS)
        pois_a = gpd.read_file(gpkg_path, layer="gis_osm_pois_a_free").to_crs(ANALYSIS_CRS)
        pois_a["geometry"] = pois_a.geometry.representative_point()
        com_pois = pd.concat(
            [
                pois[pois["fclass"].str.lower().isin(NON_EV_COM_POI_CLASSES)][["geometry"]],
                pois_a[pois_a["fclass"].str.lower().isin(NON_EV_COM_POI_CLASSES)][["geometry"]],
            ],
            ignore_index=True,
        )
        com_pois_gdf = gpd.GeoDataFrame(com_pois, geometry="geometry", crs=ANALYSIS_CRS)
        c_poi_ids = set(
            gpd.sjoin(untagged_gdf, com_pois_gdf, how="inner", predicate="contains")[
                "bldg_idx"
            ].unique()
        )

        r_all = r_dsec_ids | r_lu_ids
        c_all = c_lu_ids | c_poi_ids

        for idx in untagged_gdf["bldg_idx"]:
            has_r = idx in r_all
            has_c = idx in c_all
            if has_r and not has_c:
                w_res[idx] = 1.0
            elif has_c and not has_r:
                w_com[idx] = 1.0
            elif has_r and has_c:
                # Mixed-use residential tower over commercial podium
                w_res[idx] = 0.5
                w_com[idx] = 0.5

    osm_bldgs["w_res"] = w_res
    osm_bldgs["w_com"] = w_com
    return osm_bldgs[["bldg_idx", "w_res", "w_com", "geometry"]]


def attach_ev_targets(
    cells_gdf: gpd.GeoDataFrame,
    facilities_clean_gdf: gpd.GeoDataFrame,
    include_counts_and_capacity: bool = True,
) -> gpd.GeoDataFrame:
    """Attach EV charger target and audit columns to an H3 cell GeoDataFrame."""
    ev_gdf = facilities_clean_gdf[facilities_clean_gdf["facility_type"] == "ev_charger"][
        ["facility_id", "capacity", "geometry"]
    ].copy()

    joined = gpd.sjoin(
        ev_gdf,
        cells_gdf[["cell_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    # Deduplicate in case a station point sits on a shared hex boundary
    joined = joined.sort_values(["facility_id", "cell_id"]).drop_duplicates(
        subset=["facility_id"], keep="first"
    )

    agg = (
        joined.groupby("cell_id", as_index=False)
        .agg(
            ev_station_count=("facility_id", "count"),
            ev_charger_capacity=("capacity", lambda s: int(round(float(s.sum())))),
        )
    )

    out = cells_gdf.merge(agg, on="cell_id", how="left")
    out["ev_station_count"] = out["ev_station_count"].fillna(0).astype(int)
    out["ev_charger_capacity"] = out["ev_charger_capacity"].fillna(0).astype(int)
    out["has_ev_charger"] = (out["ev_station_count"] > 0).astype(int)

    if not include_counts_and_capacity:
        out = out.drop(columns=["ev_station_count", "ev_charger_capacity"])
    return out


def compute_h3_r9_features_and_targets(
    cells_r9_gdf: gpd.GeoDataFrame,
    G_walk: nx.MultiDiGraph,
    gpkg_path: Path,
    buildings_pop_gdf: gpd.GeoDataFrame,
    facilities_clean_gdf: gpd.GeoDataFrame,
    lrt_stations_gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Compute all 11 leak-free ML features and 3 EV target/audit columns on H3 res-9 cells."""
    out = cells_r9_gdf.copy()

    # 1. intersection_density (nodes with street_count >= 3 per km²)
    nodes_gdf = ox.graph_to_gdfs(G_walk, nodes=True, edges=False).to_crs(ANALYSIS_CRS)
    inter_nodes = nodes_gdf[nodes_gdf["street_count"].fillna(0).astype(int) >= 3][
        ["geometry"]
    ].copy()
    joined_nodes = gpd.sjoin(
        inter_nodes,
        out[["cell_id", "geometry"]],
        how="inner",
        predicate="within",
    )
    node_counts = joined_nodes.groupby("cell_id").size()
    out["intersection_density"] = (
        out["cell_id"].map(node_counts).fillna(0.0).astype(float) / out["area_km2"]
    )

    # 2. road_density_km_per_km2 (undirected walk edges clipped to cell, km / km²)
    G_undir = ox.convert.to_undirected(G_walk)
    edges_gdf = ox.graph_to_gdfs(G_undir, nodes=False, edges=True).to_crs(ANALYSIS_CRS)
    clipped_edges = gpd.overlay(
        edges_gdf[["geometry"]],
        out[["cell_id", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    clipped_edges["len_km"] = clipped_edges.geometry.length / 1000.0
    road_km_by_cell = clipped_edges.groupby("cell_id")["len_km"].sum()
    out["road_density_km_per_km2"] = (
        out["cell_id"].map(road_km_by_cell).fillna(0.0).astype(float) / out["area_km2"]
    )

    # 3. Building morphology & land-use shares (bldg_coverage_ratio, res_bldg_share, com_bldg_share)
    osm_bldgs = _classify_osm_buildings_leak_free(gpkg_path, buildings_pop_gdf)
    bldg_clips = gpd.overlay(
        osm_bldgs,
        out[["cell_id", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    bldg_clips["clip_area"] = bldg_clips.geometry.area
    bldg_clips = bldg_clips[bldg_clips["clip_area"] > 0.0].copy()
    bldg_clips["res_area"] = bldg_clips["clip_area"] * bldg_clips["w_res"]
    bldg_clips["com_area"] = bldg_clips["clip_area"] * bldg_clips["w_com"]

    # Union building footprints per cell to prevent podium+tower overlap double-counting
    union_area_by_cell = (
        bldg_clips.groupby("cell_id")["geometry"]
        .apply(lambda gs: float( gs.union_all().area))
        .to_dict()
    )
    area_sums = bldg_clips.groupby("cell_id")[["clip_area", "res_area", "com_area"]].sum()

    res_share_map = (area_sums["res_area"] / area_sums["clip_area"]).to_dict()
    com_share_map = (area_sums["com_area"] / area_sums["clip_area"]).to_dict()

    out["res_bldg_share"] = (
        out["cell_id"].map(res_share_map).fillna(0.0).clip(0.0, 1.0).astype(float)
    )
    out["com_bldg_share"] = (
        out["cell_id"].map(com_share_map).fillna(0.0).clip(0.0, 1.0).astype(float)
    )
    out["bldg_coverage_ratio"] = (
        (out["cell_id"].map(union_area_by_cell).fillna(0.0) / (out["area_km2"] * 1e6))
        .clip(0.0, 1.0)
        .astype(float)
    )

    # 4. Transit proximity (dist_nearest_lrt_m, dist_nearest_bus_m)
    cell_xy = np.column_stack([out["centroid_x"].to_numpy(), out["centroid_y"].to_numpy()])

    lrt_xy = np.column_stack(
        [lrt_stations_gdf.geometry.x.to_numpy(), lrt_stations_gdf.geometry.y.to_numpy()]
    )
    dist_lrt, _ = cKDTree(lrt_xy).query(cell_xy)
    out["dist_nearest_lrt_m"] = dist_lrt.astype(float)

    bus_gdf = facilities_clean_gdf[facilities_clean_gdf["facility_type"] == "bus_stop"]
    bus_xy = np.column_stack([bus_gdf.geometry.x.to_numpy(), bus_gdf.geometry.y.to_numpy()])
    dist_bus, _ = cKDTree(bus_xy).query(cell_xy)
    out["dist_nearest_bus_m"] = dist_bus.astype(float)

    # 5. Attach EV target & audit columns
    out = attach_ev_targets(out, facilities_clean_gdf, include_counts_and_capacity=True)

    # Verify zero NaNs / Infs across all 11 ML features
    for col in ML_FEATURES:
        vals = out[col].to_numpy(dtype=float)
        if np.isnan(vals).any() or np.isinf(vals).any():
            raise ValueError(f"NaN or Inf detected in ML feature '{col}'")

    if ((out["res_bldg_share"] + out["com_bldg_share"]) > 1.0 + 1e-9).any():
        raise ValueError("res_bldg_share + com_bldg_share exceeds 1.0")

    feature_audit = {
        "n_cells": int(len(out)),
        "n_pos_ev_cells": int(out["has_ev_charger"].sum()),
        "ev_cell_prevalence": round(float(out["has_ev_charger"].mean()), 4),
        "total_ev_stations_in_populated_cells": int(out["ev_station_count"].sum()),
        "total_ev_capacity_in_populated_cells": int(out["ev_charger_capacity"].sum()),
        "osm_buildings_total": int(len(osm_bldgs)),
        "osm_buildings_effective_res": round(float(osm_bldgs["w_res"].sum()), 1),
        "osm_buildings_effective_com": round(float(osm_bldgs["w_com"].sum()), 1),
    }
    return out, feature_audit


def write_feature_manifest(out_path: Path) -> dict[str, Any]:
    """Write canonical data/frozen/feature_manifest.json."""
    manifest = {
        "ml_features": list(ML_FEATURES),
        "excluded_target_and_audit": list(EXCLUDED_TARGET_AND_AUDIT),
        "analysis_prefix_excluded": list(ANALYSIS_PREFIX_EXCLUDED),
        "random_seed": RANDOM_SEED,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def _is_ev_poi(row: pd.Series) -> bool:
    """Two-stage POI-level EV/charging predicate (avoids parking fee charge=6MOP false positive)."""
    cat_val = str(row.get("cat_val") or "").strip().lower()
    amenity = str(row.get("amenity") or "").strip().lower()
    if cat_val == "charging_station" or amenity == "charging_station":
        return True

    tag_keys = row.get("tag_keys") or []
    for tk in tag_keys:
        tk_low = str(tk).lower()
        if tk_low.startswith("socket:") or "charging" in tk_low:
            return True

    text_blob = " ".join(
        [
            cat_val,
            str(row.get("name") or ""),
            str(row.get("name_en") or ""),
            str(row.get("brand") or ""),
            str(row.get("operator") or ""),
        ]
    )
    return bool(EV_REGEX_EXT.search(text_blob) or EV_REGEX_SPEC.search(text_blob))


def build_cell_poi_texts(
    cells_r9_gdf: gpd.GeoDataFrame,
    pbf_pois_gdf: gpd.GeoDataFrame,
    out_csv_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build EV-stripped OSM POI text per H3 res-9 cell for Phase B Route B."""
    joined = gpd.sjoin(
        pbf_pois_gdf,
        cells_r9_gdf[["cell_id", "zone_name", "parish_name", "geometry"]],
        how="inner",
        predicate="within",
    )
    joined = joined.sort_values(["cell_id", "osm_type", "osm_id"]).drop_duplicates(
        subset=["cell_id", "osm_type", "osm_id"], keep="first"
    )

    is_ev_mask = joined.apply(_is_ev_poi, axis=1) if not joined.empty else pd.Series([], dtype=bool)
    stripped_df = joined[is_ev_mask].copy() if not joined.empty else joined.iloc[0:0].copy()
    kept_df = joined[~is_ev_mask].copy() if not joined.empty else joined.iloc[0:0].copy()

    raw_counts = joined.groupby("cell_id").size().to_dict() if not joined.empty else {}
    filt_counts = kept_df.groupby("cell_id").size().to_dict() if not kept_df.empty else {}

    tokens_by_cell: dict[str, list[str]] = {}
    if not kept_df.empty:
        for cid, grp in kept_df.groupby("cell_id"):
            toks: set[str] = set()
            for _, r in grp.iterrows():
                cat_v = str(r.get("cat_val") or "").strip()
                nm = str(r.get("name") or "").strip()
                tok = f"{cat_v}: {nm}" if nm else cat_v
                # Stage 2 token-level sanitization
                tok = EV_REGEX_EXT.sub("", tok)
                tok = EV_REGEX_SPEC.sub("", tok)
                tok = " ".join(tok.split()).strip(" :;-|")
                if tok:
                    toks.add(tok)
            tokens_by_cell[str(cid)] = sorted(toks)

    rows: list[dict[str, Any]] = []
    for _, cell in cells_r9_gdf.sort_values("cell_id").iterrows():
        cid = str(cell["cell_id"])
        raw_c = int(raw_counts.get(cid, 0))
        filt_c = int(filt_counts.get(cid, 0))
        toks = tokens_by_cell.get(cid, [])

        district_header = f"District: {cell['zone_name']} ({cell['parish_name']})"
        poi_body = "; ".join(toks) if toks else "none"
        text = f"{district_header} | POIs: {poi_body}"
        text = EV_REGEX_EXT.sub("", text)
        text = EV_REGEX_SPEC.sub("", text)
        text = " ".join(text.split())

        if EV_REGEX_SPEC.search(text) or EV_REGEX_EXT.search(text):
            raise ValueError(f"EV token leakage detected in cell_poi_texts for cell {cid}: {text}")

        rows.append(
            {
                "cell_id": cid,
                "poi_count_raw": raw_c,
                "poi_count_filtered": filt_c,
                "poi_text": text,
            }
        )

    df = pd.DataFrame(rows)
    if (df["poi_count_filtered"] > df["poi_count_raw"]).any():
        raise ValueError("poi_count_filtered cannot exceed poi_count_raw")
    if int(df["poi_count_raw"].sum()) <= int(df["poi_count_filtered"].sum()):
        raise ValueError("Expected at least 1 EV POI stripped across populated H3 cells")

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)

    stripped_items = [
        {
            "cell_id": str(r["cell_id"]),
            "osm_ref": f"{r['osm_type']}/{r['osm_id']}",
            "cat_val": str(r["cat_val"]),
            "name": str(r["name"]),
            "brand": str(r["brand"]),
            "operator": str(r["operator"]),
        }
        for _, r in stripped_df.iterrows()
    ]
    audit = {
        "total_cells": int(len(df)),
        "cells_with_pois": int((df["poi_count_filtered"] > 0).sum()),
        "poi_count_raw_total": int(df["poi_count_raw"].sum()),
        "poi_count_filtered_total": int(df["poi_count_filtered"].sum()),
        "ev_pois_stripped_total": int(df["poi_count_raw"].sum() - df["poi_count_filtered"].sum()),
        "regex_spec_matches_remaining": 0,
        "regex_ext_matches_remaining": 0,
        "stripped_ev_pois": stripped_items,
    }
    return df, audit
