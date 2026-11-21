"""Comprehensive Milestone M1 unit and integration tests (A01–A06)."""

from __future__ import annotations

import json
import os
import re

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pytest
from shapely.geometry import Point

from macao_equity.config import (
    ANALYSIS_CRS,
    DATA_FROZEN_DIR,
    DATA_RAW_DIR,
    DSEC_TERRESTRIAL_POP_65PLUS,
    DSEC_TERRESTRIAL_POP_TOTAL,
    EV_REGEX_EXT,
    EV_REGEX_SPEC,
    EXCLUDED_TARGET_AND_AUDIT,
    IC_LIBRARY_WHITELIST,
    MACAO_UTM_X_BOUNDS,
    MACAO_UTM_Y_BOUNDS,
    ML_FEATURES,
    OUTPUTS_METRICS_DIR,
    PENINSULA_PARISH_IDS,
    RANDOM_SEED,
    SSM_HEALTH_CENTRE_WHITELIST,
    WGS84_CRS,
    make_transformer,
)
from macao_equity.ingest.dsec import esri_rings_to_valid_geom
from macao_equity.ingest.osm import compute_landmark_walk_check
from macao_equity.preprocess.clean import dedup_cross_source_25m
from macao_equity.preprocess.graph_and_folds import build_spatial_folds


# ---------------------------------------------------------------------------
# A01: Environment & Unit Tests for Core Geometry / Dedup Helpers
# ---------------------------------------------------------------------------
def test_a01_environment_and_mplbackend() -> None:
    """Verify headless matplotlib Agg backend and pandas 2.x x PySAL compatibility."""
    import esda  # noqa: F401
    import h3
    import inequality  # noqa: F401
    import libpysal  # noqa: F401
    import matplotlib
    import pypdf  # noqa: F401
    import spreg  # noqa: F401

    assert os.environ.get("MPLBACKEND", "").lower() == "agg"
    assert matplotlib.get_backend().lower() == "agg"
    assert pd.__version__.startswith("2.")
    assert h3.__version__.startswith("4.")
    assert ox.__version__.startswith("2.")


def test_unit_esri_rings_cw_ccw_and_sliver_repair() -> None:
    """Verify Esri JSON CW outer / CCW hole nesting, <1e-3 sliver drop, and bowtie fix."""
    # 100x100 CW outer ring (area = 10,000)
    cw_outer = [[0.0, 0.0], [0.0, 100.0], [100.0, 100.0], [100.0, 0.0], [0.0, 0.0]]
    # 20x20 CCW inner hole (area = 400) -> net area should be 9,600
    ccw_hole = [[20.0, 20.0], [40.0, 20.0], [40.0, 40.0], [20.0, 40.0], [20.0, 20.0]]
    # Degenerate sub-millimeter 4-pt sliver (area = 1e-8 < 1e-3)
    sliver = [[0.0, 0.0], [0.0, 1e-4], [1e-4, 1e-4], [0.0, 0.0]]

    geom = esri_rings_to_valid_geom([cw_outer, ccw_hole, sliver])
    assert geom.is_valid
    assert geom.geom_type == "Polygon"
    assert pytest.approx(geom.area, rel=1e-6) == 9600.0

    # Bowtie self-intersecting ring -> repaired to valid polygon/multipolygon
    bowtie = [[0.0, 0.0], [10.0, 10.0], [10.0, 0.0], [0.0, 10.0], [0.0, 0.0]]
    repaired = esri_rings_to_valid_geom([bowtie])
    assert repaired.is_valid
    assert repaired.area > 0.0


def test_unit_cross_source_only_25m_dedup() -> None:
    """Verify cross-source 25m dedup preserves within-source close points (e.g. 6.2m / 13.5m)."""
    primary = gpd.GeoDataFrame(
        {
            "id": ["p1", "p2"],
            "geometry": [Point(760000.0, 2450000.0), Point(760006.2, 2450000.0)],
        },
        geometry="geometry",
        crs=ANALYSIS_CRS,
    )
    secondary = gpd.GeoDataFrame(
        {
            "id": ["s_dup", "s_far"],
            "geometry": [Point(760010.0, 2450000.0), Point(760100.0, 2450000.0)],
        },
        geometry="geometry",
        crs=ANALYSIS_CRS,
    )
    combined, dropped = dedup_cross_source_25m(primary, secondary, radius_m=25.0)
    assert combined["id"].tolist() == ["p1", "p2", "s_far"]
    assert dropped["id"].tolist() == ["s_dup"]


# ---------------------------------------------------------------------------
# A02: Live Data Snapshots & Provenance Audit
# ---------------------------------------------------------------------------
def test_a02_raw_snapshots_and_provenance() -> None:
    """Verify raw snapshots in data/raw/ and mirrored data_provenance.json completeness."""
    expected_raw = [
        DATA_RAW_DIR / "dsec_zones_2021.json",
        DATA_RAW_DIR / "dsec_buildings_2021.json",
        DATA_RAW_DIR / "cem_ev_en.json",
        DATA_RAW_DIR / "cem_ev_tc.json",
        DATA_RAW_DIR / "macau-latest.osm.pbf",
        DATA_RAW_DIR / "macau-latest-free.gpkg.zip",
        DATA_RAW_DIR / "osm_walk_graph.graphml",
        DATA_RAW_DIR / "data_provenance.json",
        OUTPUTS_METRICS_DIR / "data_provenance.json",
    ]
    for path in expected_raw:
        assert path.exists(), f"Missing required raw/provenance file: {path}"
        assert path.stat().st_size > 0, f"Empty file: {path}"

    with (DATA_RAW_DIR / "data_provenance.json").open("r", encoding="utf-8") as f:
        prov_raw = json.load(f)
    with (OUTPUTS_METRICS_DIR / "data_provenance.json").open("r", encoding="utf-8") as f:
        prov_metrics = json.load(f)

    assert prov_raw == prov_metrics
    assert prov_raw["analysis_crs"] == ANALYSIS_CRS
    assert prov_raw["random_seed"] == RANDOM_SEED
    assert prov_raw["walking_speed_kmh"] == 4.5

    # Verify data.gov.mo Macao One Account login gate note
    gate = prov_raw["data_gov_mo_access_gate"]
    assert gate["status"] == "blocked_by_macao_one_account_login_gate"
    assert "e7b2e84d-3333-42f0-b676-64ce95306f0d" in gate["attempted_dataset_ids"]

    # Verify all 6 primary sources have SHA-256 hashes, timestamps, and URLs
    sources = {s["source_id"]: s for s in prov_raw["sources"]}
    for sid in (
        "dsec_zones_2021",
        "dsec_buildings_2021",
        "cem_ev_en",
        "cem_ev_tc",
        "geofabrik_macau_pbf",
        "geofabrik_macau_gpkg_zip",
    ):
        assert sid in sources
        assert len(sources[sid]["sha256"]) == 64
        assert sources[sid]["url"].startswith("https://")
        assert "+08:00" in sources[sid]["fetched_at_utc8"]

    # Verify CRS landmark ground-truth alignment (< 50 m)
    landmarks = {item["landmark"]: item for item in prov_raw["crs_landmark_verification"]}
    assert landmarks["逸景樓"]["centroid_error_m"] < 15.0
    assert landmarks["快達樓"]["centroid_error_m"] < 30.0
    assert landmarks["湖畔大廈"]["centroid_error_m"] < 50.0
    assert all(item["passed"] for item in prov_raw["crs_landmark_verification"])


# ---------------------------------------------------------------------------
# A03: Facilities Cleaning, Whitelists, Cross-Source Dedup & LRT
# ---------------------------------------------------------------------------
def test_a03_facilities_clean_and_lrt_stations() -> None:
    """Verify all 5 facility types in facilities_clean.gpkg and lrt_stations.gpkg."""
    fac_path = DATA_FROZEN_DIR / "facilities_clean.gpkg"
    lrt_path = DATA_FROZEN_DIR / "lrt_stations.gpkg"
    assert fac_path.exists()
    assert lrt_path.exists()

    fac = gpd.read_file(fac_path, layer="facilities")
    assert fac.crs is not None and fac.crs.to_epsg() == 32649
    assert fac.geometry.notna().all() and (~fac.geometry.is_empty).all()
    assert (fac.geometry.geom_type == "Point").all()

    req_cols = [
        "facility_id",
        "facility_type",
        "name",
        "source",
        "capacity",
        "capacity_unit",
        "capacity_imputed",
        "pilenum",
        "total_kw",
        "in_zona_a",
        "geometry",
    ]
    assert list(fac.columns) == req_cols
    assert fac["facility_id"].is_unique
    assert (fac["capacity"] > 0).all() and fac["capacity"].notna().all()
    assert set(fac["facility_type"].unique()) == {
        "park",
        "health_centre",
        "library",
        "bus_stop",
        "ev_charger",
    }

    # 1. Health Centres: strictly 11 official SSM terrestrial primary health centres/stations
    hc = fac[fac["facility_type"] == "health_centre"]
    assert len(hc) == 11
    bad_hc_re = re.compile(r"(?i)hospital|牙科|同善堂|工聯|臨床中心|大學醫療|新城A區")
    assert not hc["name"].str.contains(bad_hc_re).any()
    assert set(hc["name"]) == set(SSM_HEALTH_CENTRE_WHITELIST.values())

    # 2. Libraries: strictly 15 official IC public libraries with relocated 青洲坊圖書館
    lib = fac[fac["facility_type"] == "library"]
    assert len(lib) == 15
    bad_lib_re = re.compile(r"(?i)伍宜孫|科技大學|M\.U\.S\.T|劉楊梅少|藝術博物館|八角亭")
    assert not lib["name"].str.contains(bad_lib_re).any()
    assert set(lib["name"]) == set(IC_LIBRARY_WHITELIST.values())

    tr_4326_to_32649 = make_transformer(WGS84_CRS, ANALYSIS_CRS)
    expected_iv_x, expected_iv_y = tr_4326_to_32649.transform(113.543562, 22.212084)
    iv_lib = lib[lib["name"].str.contains("青洲坊圖書館")]
    assert len(iv_lib) == 1
    assert iv_lib.geometry.iloc[0].distance(Point(expected_iv_x, expected_iv_y)) < 30.0

    # 3. Parks: zero racetrack/stadium/FUFA/private contamination, hole-subtracted reservoir park
    parks = fac[fac["facility_type"] == "park"]
    assert 180 <= len(parks) <= 250
    bad_park_re = re.compile(
        r"氹仔馬場|東亞運動會體育館|奧林匹克體育中心|國際射擊中心|小型賽車場|FUFA|聖母岩"
    )
    assert not parks["name"].str.contains(bad_park_re).any()
    assert parks["capacity"].max() < 95_000.0
    reservoir = parks[parks["name"].str.contains("水塘公園")]
    assert len(reservoir) == 1
    assert 35_000.0 <= float(reservoir["capacity"].iloc[0]) <= 50_000.0
    assert int(parks["capacity_imputed"].sum()) == 13

    # 4. Bus Stops: exact osm_id dedup, zero casino shuttles, opposite-side street stops preserved
    bus = fac[fac["facility_type"] == "bus_stop"]
    assert 560 <= len(bus) <= 600
    bad_bus_re = re.compile(r"(?i)Duplicate of|free\s*shuttle|免費接駁|免費穿梭")
    assert not bus["name"].str.contains(bad_bus_re).any()
    # Verify opposite-side DSAT stops C664 and C663 (13.5 m apart) are BOTH preserved
    assert bus["name"].str.contains("竹灣泳池-1").any()
    assert bus["name"].str.contains("竹灣泳池-2").any()

    # 5. EV Chargers: 84 CEM + 1 public non-CEM OSM, exact Zona A = 4, 6.2m CEM pair kept
    ev = fac[fac["facility_type"] == "ev_charger"]
    assert len(ev) in (84, 85)
    cem_ev = ev[ev["source"] == "cem_api"]
    assert len(cem_ev) == 84
    assert pytest.approx(float(cem_ev["capacity"].sum()), abs=1e-6) == 1622.0
    assert pytest.approx(float(cem_ev["total_kw"].dropna().sum()), abs=1e-3) == 21722.0
    assert int(ev["in_zona_a"].sum()) == 4
    # Verify CEM stations 382819 & 387622 (6.2 m apart) are both preserved
    assert "cem_382819" in set(cem_ev["facility_id"])
    assert "cem_387622" in set(cem_ev["facility_id"])

    # 6. LRT Stations: 14 or 15 operational MLM stations, Lotus deduplicated, 0 Zhuhai CRH
    lrt = gpd.read_file(lrt_path, layer="lrt_stations")
    assert lrt.crs is not None and lrt.crs.to_epsg() == 32649
    assert len(lrt) in (14, 15)
    assert int(lrt["name"].str.contains("蓮花|Lotus").sum()) == 1
    assert "14330568544" not in set(lrt["osm_id"].astype(str))
    assert not lrt["name"].str.contains("珠海|湾仔北|长隆|東線").any()


# ---------------------------------------------------------------------------
# A04: Zones (23), Parishes (7), and H3 Res-9 / Res-8 Population Conservation
# ---------------------------------------------------------------------------
def test_a04_zones_parishes_and_h3_conservation() -> None:
    """Verify zones_23.gpkg, parishes_7.gpkg, cells_base_res9.gpkg, and cells_base_res8.gpkg."""
    zones = gpd.read_file(DATA_FROZEN_DIR / "zones_23.gpkg")
    parishes = gpd.read_file(DATA_FROZEN_DIR / "parishes_7.gpkg")
    cells_r9 = gpd.read_file(DATA_FROZEN_DIR / "cells_base_res9.gpkg")
    cells_r8 = gpd.read_file(DATA_FROZEN_DIR / "cells_base_res8.gpkg")

    for gdf in (zones, parishes, cells_r9, cells_r8):
        assert gdf.crs is not None and gdf.crs.to_epsg() == 32649
        assert gdf.geometry.notna().all() and (~gdf.geometry.is_empty).all()
        assert gdf.geometry.is_valid.all()

    # Zones (23 terrestrial zones Z01..Z23, excluding Z26)
    assert len(zones) == 23
    assert set(zones["zone_id"]) == {f"Z{i:02d}" for i in range(1, 24)}
    assert zones.loc[zones["zone_id"] == "Z06", "parish_id"].iloc[0] == "P01"
    assert pytest.approx(float(zones["pop_total"].sum()), abs=1e-2) == DSEC_TERRESTRIAL_POP_TOTAL
    assert pytest.approx(float(zones["pop_65plus"].sum()), abs=1e-2) == DSEC_TERRESTRIAL_POP_65PLUS

    # Parishes (7 official parishes P01..P07 dissolved from zones_23)
    assert len(parishes) == 7
    assert set(parishes["parish_id"]) == {f"P0{i}" for i in range(1, 8)}
    p01_area = float(parishes.loc[parishes["parish_id"] == "P01", "area_km2"].iloc[0])
    p02_area = float(parishes.loc[parishes["parish_id"] == "P02", "area_km2"].iloc[0])
    assert abs(p01_area - 3.20) < 0.15
    assert abs(p02_area - 1.14) < 0.15
    assert pytest.approx(float(parishes["pop_total"].sum()), abs=1e-2) == float(
        zones["pop_total"].sum()
    )

    # H3 Res-9 (~250-400 populated cells, exact population conservation < 0.05%)
    assert 250 <= len(cells_r9) <= 400
    assert cells_r9["cell_id"].is_unique
    assert (cells_r9["pop_total"] > 0).all()
    assert (
        abs(float(cells_r9["pop_total"].sum()) - DSEC_TERRESTRIAL_POP_TOTAL)
        / DSEC_TERRESTRIAL_POP_TOTAL
        < 1e-5
    )
    assert (
        abs(float(cells_r9["pop_65plus"].sum()) - DSEC_TERRESTRIAL_POP_65PLUS)
        / DSEC_TERRESTRIAL_POP_65PLUS
        < 1e-5
    )
    assert cells_r9[["zone_id", "zone_name", "parish_id", "parish_name"]].notna().all().all()
    assert (cells_r9["centroid_x"].between(*MACAO_UTM_X_BOUNDS)).all()
    assert (cells_r9["centroid_y"].between(*MACAO_UTM_Y_BOUNDS)).all()

    # H3 Res-8 (MAUP sensitivity grid, exact population conservation < 0.05%)
    assert 45 <= len(cells_r8) <= 90
    assert cells_r8["cell_id"].is_unique
    assert (cells_r8["pop_total"] > 0).all()
    assert (
        abs(float(cells_r8["pop_total"].sum()) - DSEC_TERRESTRIAL_POP_TOTAL)
        / DSEC_TERRESTRIAL_POP_TOTAL
        < 1e-5
    )
    assert (
        abs(float(cells_r8["pop_65plus"].sum()) - DSEC_TERRESTRIAL_POP_65PLUS)
        / DSEC_TERRESTRIAL_POP_65PLUS
        < 1e-5
    )


# ---------------------------------------------------------------------------
# A05: Walkable Network, 11 Leak-Free ML Features & EV-Stripped POI Text
# ---------------------------------------------------------------------------
def test_a05_walk_graph_features_and_poi_texts() -> None:
    """Verify osm_walk_graph.graphml, feature_manifest.json, 11 ML features, and cell_poi_texts."""
    G = ox.load_graphml(DATA_RAW_DIR / "osm_walk_graph.graphml")
    assert G.number_of_nodes() >= 10_000
    wcc_sizes = [len(c) for c in nx.weakly_connected_components(G)]
    assert min(wcc_sizes) >= 10
    ys = [float(G.nodes[n]["y"]) for n in G.nodes]
    assert max(ys) > 22.19 and min(ys) < 22.15
    assert all(int(G.nodes[n].get("street_count", 0)) >= 1 for n in G.nodes)
    assert all(float(d.get("length", 0.0)) > 0.0 for _, _, d in G.edges(data=True))

    landmark = compute_landmark_walk_check(G)
    assert landmark["symmetric"] is True
    assert 5.0 <= landmark["ruins_to_senado_min"] <= 8.0

    # Feature manifest & 11 leak-free ML features
    with (DATA_FROZEN_DIR / "feature_manifest.json").open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["ml_features"] == ML_FEATURES
    assert manifest["excluded_target_and_audit"] == EXCLUDED_TARGET_AND_AUDIT
    assert manifest["analysis_prefix_excluded"] == ["acc10_", "acc15_", "lisa_"]
    assert manifest["random_seed"] == RANDOM_SEED
    assert set(manifest["ml_features"]).isdisjoint(set(manifest["excluded_target_and_audit"]))
    leak_col_re = re.compile(r"(?i)(ev|charg|pile|acc10_|acc15_|lisa_)")
    assert not any(leak_col_re.search(c) for c in manifest["ml_features"])

    cells_r9 = gpd.read_file(DATA_FROZEN_DIR / "cells_base_res9.gpkg")
    for col in ML_FEATURES:
        assert col in cells_r9.columns
        arr = cells_r9[col].to_numpy(dtype=float)
        assert not np.isnan(arr).any() and not np.isinf(arr).any()

    for ratio_col in ("elderly_share", "res_bldg_share", "com_bldg_share", "bldg_coverage_ratio"):
        assert ((cells_r9[ratio_col] >= 0.0) & (cells_r9[ratio_col] <= 1.0)).all()
    assert ((cells_r9["res_bldg_share"] + cells_r9["com_bldg_share"]) <= 1.0 + 1e-9).all()
    assert (cells_r9["dist_nearest_lrt_m"] >= 0.0).all()
    assert (cells_r9["dist_nearest_bus_m"] >= 0.0).all()

    # EV-stripped POI texts
    poi_df = pd.read_csv(DATA_FROZEN_DIR / "cell_poi_texts.csv")
    assert list(poi_df.columns) == [
        "cell_id",
        "poi_count_raw",
        "poi_count_filtered",
        "poi_text",
    ]
    assert set(poi_df["cell_id"]) == set(cells_r9["cell_id"])
    assert len(poi_df) == len(cells_r9)
    assert poi_df["poi_text"].notna().all() and (poi_df["poi_text"].str.len() > 0).all()
    assert (poi_df["poi_count_filtered"] <= poi_df["poi_count_raw"]).all()
    assert int(poi_df["poi_count_raw"].sum()) > int(poi_df["poi_count_filtered"].sum())
    assert not poi_df["poi_text"].str.contains(EV_REGEX_SPEC).any()
    assert not poi_df["poi_text"].str.contains(EV_REGEX_EXT).any()


# ---------------------------------------------------------------------------
# A06: 10-Min Pedestrian Graph & Frozen 5 Spatial Blocks
# ---------------------------------------------------------------------------
def test_a06_graph_edges_and_spatial_folds(tmp_path) -> None:
    """Verify graph_edges.csv symmetry/bounds and spatial_folds.csv determinism & class balance."""
    cells_r9 = gpd.read_file(DATA_FROZEN_DIR / "cells_base_res9.gpkg")
    folds_df = pd.read_csv(DATA_FROZEN_DIR / "spatial_folds.csv")
    edges_df = pd.read_csv(DATA_FROZEN_DIR / "graph_edges.csv")
    with (DATA_FROZEN_DIR / "split_metadata.json").open("r", encoding="utf-8") as f:
        split_meta = json.load(f)

    # 1. Spatial folds 1-to-1 with cells_base_res9 & positive/negative class guard in every block
    assert list(folds_df.columns) == ["cell_id", "spatial_block"]
    assert len(folds_df) == len(cells_r9)
    merged_folds = cells_r9[["cell_id", "spatial_block", "has_ev_charger", "parish_id"]].merge(
        folds_df, on="cell_id", suffixes=("_gpkg", "_csv")
    )
    assert (merged_folds["spatial_block_gpkg"] == merged_folds["spatial_block_csv"]).all()
    assert set(folds_df["spatial_block"].unique()) == {0, 1, 2, 3, 4}

    for blk_id in range(5):
        sub = merged_folds[merged_folds["spatial_block_gpkg"] == blk_id]
        n_pos = int(sub["has_ev_charger"].sum())
        n_neg = int(len(sub) - n_pos)
        assert n_pos >= 3, f"Block {blk_id} has n_pos={n_pos} < 3"
        assert n_neg >= 1, f"Block {blk_id} has n_neg={n_neg} < 1"

    assert split_meta["seed"] == RANDOM_SEED
    assert len(split_meta["blocks"]) == 5

    # Verify deterministic reproduction of spatial_folds.csv
    tmp_folds_csv = tmp_path / "spatial_folds_recheck.csv"
    _, folds_recheck, _ = build_spatial_folds(cells_r9, tmp_folds_csv, seed=RANDOM_SEED)
    pd.testing.assert_frame_equal(folds_df, folds_recheck)

    # 2. 10-Min Pedestrian Graph invariants
    assert list(edges_df.columns) == [
        "src_cell_id",
        "dst_cell_id",
        "walk_time_min",
        "weight",
        "same_block",
    ]
    assert len(edges_df) > 0 and len(edges_df) % 2 == 0
    valid_ids = set(cells_r9["cell_id"])
    assert set(edges_df["src_cell_id"]).issubset(valid_ids)
    assert set(edges_df["dst_cell_id"]).issubset(valid_ids)
    assert (edges_df["src_cell_id"] != edges_df["dst_cell_id"]).all()
    assert ((edges_df["walk_time_min"] > 0.0) & (edges_df["walk_time_min"] <= 10.0)).all()
    assert np.allclose(
        edges_df["weight"].to_numpy(),
        1.0 / edges_df["walk_time_min"].to_numpy(),
        rtol=1e-6,
    )

    # Verify strict undirected symmetry: (u, v) iff (v, u) with identical walk_time_min & same_block
    fwd = edges_df.set_index(["src_cell_id", "dst_cell_id"])
    rev = edges_df.rename(
        columns={"src_cell_id": "dst_cell_id", "dst_cell_id": "src_cell_id"}
    ).set_index(["src_cell_id", "dst_cell_id"])
    assert set(fwd.index) == set(rev.index)
    rev_aligned = rev.loc[fwd.index]
    assert np.allclose(fwd["walk_time_min"], rev_aligned["walk_time_min"], atol=1e-9)
    assert (fwd["same_block"] == rev_aligned["same_block"]).all()

    # Verify zero cross-harbor 10-min pedestrian edges (Peninsula <-> Taipa-Cotai-Coloane)
    is_pen_map = dict(
        zip(cells_r9["cell_id"], cells_r9["parish_id"].isin(PENINSULA_PARISH_IDS))
    )
    src_pen = edges_df["src_cell_id"].map(is_pen_map)
    dst_pen = edges_df["dst_cell_id"].map(is_pen_map)
    assert (src_pen == dst_pen).all()
