#!/usr/bin/env python3
"""Milestone M1 CLI (A01–A06): Live data ingestion, CRS cleaning, H3 grids, walk graph & folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import osmnx as ox

# Ensure src/ is importable if script is invoked directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from macao_equity.config import (  # noqa: E402
    ANALYSIS_CRS,
    DATA_FROZEN_DIR,
    DATA_INTERIM_DIR,
    DATA_RAW_DIR,
    DSEC_BUILDINGS_URL,
    DSEC_ZONES_URL,
    OFFICIAL_REGISTRY_URLS,
    OUTPUTS_METRICS_DIR,
    RANDOM_SEED,
    WALK_SPEED_KMH,
    ensure_directories,
    sha256_file,
    utc8_now_iso,
)
from macao_equity.ingest.cem_ev import (  # noqa: E402
    build_cem_provenance_entries,
    dedup_ev_with_osm,
    download_cem_ev,
    parse_cem_ev,
)
from macao_equity.ingest.dsec import (  # noqa: E402
    aggregate_buildings_to_zones,
    build_dsec_provenance_entries,
    dissolve_parishes,
    download_dsec_layer,
    load_dsec_buildings,
    load_dsec_zones,
    verify_crs_landmarks,
)
from macao_equity.ingest.osm import (  # noqa: E402
    build_osm_provenance_entries,
    build_walk_graph_from_pbf,
    compute_landmark_walk_check,
    download_geofabrik_osm,
    extract_pbf_layers,
    probe_data_gov_mo_gate,
)
from macao_equity.preprocess.clean import (  # noqa: E402
    assemble_facilities_clean,
    build_bus_stops,
    build_health_centres,
    build_libraries,
    build_lrt_stations,
    build_parks,
)
from macao_equity.preprocess.graph_and_folds import (  # noqa: E402
    build_10min_pedestrian_graph,
    build_spatial_folds,
    write_split_metadata,
)
from macao_equity.preprocess.h3_grid import (  # noqa: E402
    allocate_buildings_to_h3,
    attach_ev_targets,
    build_cell_poi_texts,
    compute_h3_r9_features_and_targets,
    refine_coarse_dsec_buildings,
    write_feature_manifest,
)

CELLS_R9_COLUMNS = [
    "cell_id",
    "centroid_x",
    "centroid_y",
    "area_km2",
    "zone_id",
    "zone_name",
    "parish_id",
    "parish_name",
    "pop_total",
    "pop_65plus",
    "pop_density",
    "elderly_share",
    "intersection_density",
    "road_density_km_per_km2",
    "res_bldg_share",
    "com_bldg_share",
    "bldg_coverage_ratio",
    "dist_nearest_lrt_m",
    "dist_nearest_bus_m",
    "has_ev_charger",
    "ev_station_count",
    "ev_charger_capacity",
    "spatial_block",
    "geometry",
]

CELLS_R8_COLUMNS = [
    "cell_id",
    "area_km2",
    "zone_id",
    "parish_id",
    "parish_name",
    "pop_total",
    "pop_65plus",
    "pop_density",
    "elderly_share",
    "has_ev_charger",
    "geometry",
]


def run_m1_pipeline(skip_download: bool = False, force_download: bool = False) -> dict[str, Any]:
    """Execute Milestone M1 (A01–A06) end-to-end and write all frozen deliverables."""
    ensure_directories()

    # Raw snapshot file paths
    dsec_zones_raw_path = DATA_RAW_DIR / "dsec_zones_2021.json"
    dsec_bldgs_raw_path = DATA_RAW_DIR / "dsec_buildings_2021.json"
    cem_en_raw_path = DATA_RAW_DIR / "cem_ev_en.json"
    cem_tc_raw_path = DATA_RAW_DIR / "cem_ev_tc.json"
    osm_pbf_path = DATA_RAW_DIR / "macau-latest.osm.pbf"
    osm_gpkg_zip_path = DATA_RAW_DIR / "macau-latest-free.gpkg.zip"
    osm_gpkg_extract_path = DATA_INTERIM_DIR / "macau.gpkg"
    walk_graphml_path = DATA_RAW_DIR / "osm_walk_graph.graphml"

    # ------------------------------------------------------------------
    # Step 1: A02 Live Data Acquisition (with cache reuse / --skip-download)
    # ------------------------------------------------------------------
    print("[M1 1/7] Acquiring DSEC 2021 Census zones & building layers (native EPSG:8433)...")
    skip_zones = skip_download or (not force_download and dsec_zones_raw_path.exists())
    zones_payload, zones_fetched_at = download_dsec_layer(
        DSEC_ZONES_URL, dsec_zones_raw_path, skip_download=skip_zones
    )

    skip_bldgs = skip_download or (not force_download and dsec_bldgs_raw_path.exists())
    bldgs_payload, bldgs_fetched_at = download_dsec_layer(
        DSEC_BUILDINGS_URL, dsec_bldgs_raw_path, skip_download=skip_bldgs
    )

    print("[M1 2/7] Acquiring CEM official EV charging API snapshots (lang=en & lang=tc)...")
    skip_cem = skip_download or (
        not force_download and cem_en_raw_path.exists() and cem_tc_raw_path.exists()
    )
    cem_en_data, cem_tc_data, cem_fetched_at = download_cem_ev(
        cem_en_raw_path, cem_tc_raw_path, skip_download=skip_cem
    )

    print("[M1 3/7] Acquiring Geofabrik Macau OSM (.osm.pbf & .gpkg.zip)...")
    skip_osm = skip_download or (
        not force_download
        and osm_pbf_path.exists()
        and osm_pbf_path.stat().st_size > 100_000
        and osm_gpkg_zip_path.exists()
        and osm_gpkg_zip_path.stat().st_size > 100_000
    )
    osm_dl_meta = download_geofabrik_osm(
        osm_pbf_path,
        osm_gpkg_zip_path,
        osm_gpkg_extract_path,
        skip_download=skip_osm,
    )
    data_gov_gate_info = probe_data_gov_mo_gate(skip_download=skip_download)

    # ------------------------------------------------------------------
    # Step 2: A03 / A04 DSEC CRS Transformation, Zones, Parishes & Landmark Audit
    # ------------------------------------------------------------------
    print("[M1 4/7] Parsing DSEC EPSG:8433 -> EPSG:32649, zones_23, parishes_7 & CEM EV...")
    zones_raw_gdf, zones_audit = load_dsec_zones(zones_payload)
    buildings_all_gdf, buildings_pop_gdf, buildings_audit = load_dsec_buildings(bldgs_payload)
    cem_gdf, cem_audit = parse_cem_ev(cem_en_data, cem_tc_data)

    crs_landmark_results = verify_crs_landmarks(buildings_all_gdf, cem_gdf)

    zones_23_gdf = aggregate_buildings_to_zones(buildings_pop_gdf, zones_raw_gdf)
    parishes_7_gdf = dissolve_parishes(zones_23_gdf)

    zones_23_path = DATA_FROZEN_DIR / "zones_23.gpkg"
    parishes_7_path = DATA_FROZEN_DIR / "parishes_7.gpkg"
    zones_23_gdf.to_file(zones_23_path, layer="zones_23", driver="GPKG")
    parishes_7_gdf.to_file(parishes_7_path, layer="parishes_7", driver="GPKG")

    # ------------------------------------------------------------------
    # Step 3: A03 Facility Whitelisting & Cross-Source 25m Dedup
    # ------------------------------------------------------------------
    print("[M1 5/7] Extracting PBF boundary/tags & cleaning 5 facility layers + LRT stations...")
    pbf_layers = extract_pbf_layers(osm_pbf_path)
    macao_sar_32649 = pbf_layers["macao_sar_geom_32649"]

    health_gdf, health_audit = build_health_centres(osm_gpkg_extract_path)
    libraries_gdf, libraries_audit = build_libraries(osm_gpkg_extract_path)
    parks_gdf, parks_audit = build_parks(
        osm_gpkg_extract_path,
        pbf_layers["pbf_gardens_poly_gdf"],
        pbf_layers["pbf_gardens_pt_gdf"],
        pbf_layers["pbf_access_tags"],
        macao_sar_32649,
    )
    bus_gdf, bus_audit = build_bus_stops(
        osm_gpkg_extract_path,
        pbf_layers["pbf_bus_operators"],
        macao_sar_32649,
    )
    ev_clean_gdf, ev_audit = dedup_ev_with_osm(
        cem_gdf,
        pbf_layers["osm_ev_gdf"],
        cem_audit,
    )
    lrt_gdf, lrt_audit = build_lrt_stations(
        osm_gpkg_extract_path,
        pbf_layers["pbf_lrt_gdf"],
        macao_sar_32649,
    )

    facilities_clean_gdf = assemble_facilities_clean(
        parks_gdf=parks_gdf,
        health_gdf=health_gdf,
        libraries_gdf=libraries_gdf,
        bus_gdf=bus_gdf,
        ev_gdf=ev_clean_gdf,
    )

    facilities_path = DATA_FROZEN_DIR / "facilities_clean.gpkg"
    lrt_path = DATA_FROZEN_DIR / "lrt_stations.gpkg"
    facilities_clean_gdf.to_file(facilities_path, layer="facilities", driver="GPKG")
    lrt_gdf.to_file(lrt_path, layer="lrt_stations", driver="GPKG")

    # ------------------------------------------------------------------
    # Step 4: A05 Walkable Network & A04 H3 Res-9 / Res-8 Census Allocation
    # ------------------------------------------------------------------
    print("[M1 6/7] Building OSMnx walkable network & allocating building census to H3 r9/r8...")
    if skip_download and walk_graphml_path.exists():
        G_walk = ox.load_graphml(walk_graphml_path)
        landmark_walk = compute_landmark_walk_check(G_walk)
        walk_meta = {
            "source": (
                "Cached data/raw/osm_walk_graph.graphml (from Geofabrik macau-latest.osm.pbf)"
            ),
            "graphml_path": "data/raw/osm_walk_graph.graphml",
            "sha256": sha256_file(walk_graphml_path),
            "network_type": "walk",
            "bidirectional": True,
            "retain_all": True,
            "min_component_nodes_kept": 10,
            "pruned_nodes": G_walk.number_of_nodes(),
            "pruned_edges": G_walk.number_of_edges(),
            "walk_speed_kmh": WALK_SPEED_KMH,
            "landmark_sanity_check_min": landmark_walk,
        }
    else:
        G_walk, walk_meta = build_walk_graph_from_pbf(osm_pbf_path, walk_graphml_path)
    walk_meta["osm_snapshot_timestamp_utc"] = osm_dl_meta["pbf_snapshot_timestamp_utc"]

    refined_bldgs_gdf, refine_audit = refine_coarse_dsec_buildings(
        buildings_pop_gdf, osm_gpkg_extract_path
    )

    cells_r9_base, h3_r9_audit = allocate_buildings_to_h3(
        refined_bldgs_gdf, zones_23_gdf, resolution=9
    )
    cells_r8_base, h3_r8_audit = allocate_buildings_to_h3(
        refined_bldgs_gdf, zones_23_gdf, resolution=8
    )

    cells_r9_feat, r9_feat_audit = compute_h3_r9_features_and_targets(
        cells_r9_gdf=cells_r9_base,
        G_walk=G_walk,
        gpkg_path=osm_gpkg_extract_path,
        buildings_pop_gdf=buildings_pop_gdf,
        facilities_clean_gdf=facilities_clean_gdf,
        lrt_stations_gdf=lrt_gdf,
    )
    cells_r8_final = attach_ev_targets(
        cells_r8_base, facilities_clean_gdf, include_counts_and_capacity=False
    )

    feature_manifest_path = DATA_FROZEN_DIR / "feature_manifest.json"
    write_feature_manifest(feature_manifest_path)

    cell_poi_texts_path = DATA_FROZEN_DIR / "cell_poi_texts.csv"
    _, poi_audit = build_cell_poi_texts(
        cells_r9_gdf=cells_r9_feat,
        pbf_pois_gdf=pbf_layers["pbf_pois_gdf"],
        out_csv_path=cell_poi_texts_path,
    )

    # ------------------------------------------------------------------
    # Step 5: A06 Frozen 5 Spatial Blocks & 10-Min Pedestrian Graph
    # ------------------------------------------------------------------
    print("[M1 7/7] Generating frozen 5 spatial CV blocks, 10-min pedestrian graph & provenance...")
    spatial_folds_path = DATA_FROZEN_DIR / "spatial_folds.csv"
    cells_r9_final, _, blocks_meta = build_spatial_folds(
        cells_r9_gdf=cells_r9_feat,
        out_folds_csv=spatial_folds_path,
        seed=RANDOM_SEED,
    )

    graph_edges_path = DATA_FROZEN_DIR / "graph_edges.csv"
    _, graph_audit = build_10min_pedestrian_graph(
        cells_r9_gdf=cells_r9_final,
        G_walk=G_walk,
        out_edges_csv=graph_edges_path,
    )

    split_meta_path = DATA_FROZEN_DIR / "split_metadata.json"
    split_meta = write_split_metadata(
        out_json_path=split_meta_path,
        cells_r9_gdf=cells_r9_final,
        blocks_meta=blocks_meta,
        graph_audit=graph_audit,
        seed=RANDOM_SEED,
    )

    # Write final frozen H3 res-9 and res-8 GeoPackages
    cells_r9_out = cells_r9_final[CELLS_R9_COLUMNS].copy()
    cells_r8_out = cells_r8_final[CELLS_R8_COLUMNS].copy()

    cells_r9_path = DATA_FROZEN_DIR / "cells_base_res9.gpkg"
    cells_r8_path = DATA_FROZEN_DIR / "cells_base_res8.gpkg"
    cells_r9_out.to_file(cells_r9_path, layer="cells_base_res9", driver="GPKG")
    cells_r8_out.to_file(cells_r8_path, layer="cells_base_res8", driver="GPKG")

    # ------------------------------------------------------------------
    # Step 6: Build and Mirror data_provenance.json
    # ------------------------------------------------------------------
    sources_entries = (
        build_dsec_provenance_entries(
            zones_path=dsec_zones_raw_path,
            zones_fetched_at=zones_fetched_at,
            zones_audit=zones_audit,
            buildings_path=dsec_bldgs_raw_path,
            buildings_fetched_at=bldgs_fetched_at,
            buildings_audit=buildings_audit,
        )
        + build_cem_provenance_entries(
            en_path=cem_en_raw_path,
            tc_path=cem_tc_raw_path,
            fetched_at=cem_fetched_at,
            ev_audit=ev_audit,
        )
        + build_osm_provenance_entries(
            pbf_path=osm_pbf_path,
            gpkg_zip_path=osm_gpkg_zip_path,
            osm_dl_meta=osm_dl_meta,
        )
    )

    provenance = {
        "schema_version": "1.0.0",
        "project": "MacaoSpatialEquity — Plan 5 Phase A",
        "generated_at_utc8": utc8_now_iso(),
        "random_seed": RANDOM_SEED,
        "analysis_crs": ANALYSIS_CRS,
        "walking_speed_kmh": WALK_SPEED_KMH,
        "official_registry_urls": OFFICIAL_REGISTRY_URLS,
        "data_gov_mo_access_gate": data_gov_gate_info,
        "sources": sources_entries,
        "crs_landmark_verification": crs_landmark_results,
        "facility_capacity_and_dedup_audit": {
            "health_centre": health_audit,
            "library": libraries_audit,
            "park": parks_audit,
            "bus_stop": bus_audit,
            "ev_charger": ev_audit,
            "lrt_stations": lrt_audit,
        },
        "h3_allocation_and_feature_audit": {
            "coarse_geogrouping_refinement": refine_audit,
            "h3_res9": h3_r9_audit,
            "h3_res8": h3_r8_audit,
            "r9_features_and_targets": r9_feat_audit,
        },
        "osmnx_network_metadata": walk_meta,
        "pedestrian_graph_10min_audit": graph_audit,
        "spatial_folds_summary": split_meta,
        "poi_ev_anti_leakage_audit": poi_audit,
        "spatial_coverage_and_temporal_notes": {
            "census_vs_facilities_mismatch": (
                "DSEC population census is August 2021; "
                "CEM EV and Geofabrik OSM are November 2026."
            ),
            "um_hengqin_campus": (
                "0 buildings in DSEC c2021_Building_P5 intersect Z99 (UM Hengqin Campus, "
                "1.04 km²); per DSEC official footnote, UM student/resident population is "
                "statistically consolidated in Z20 (大學及北安灣區)."
            ),
            "zona_a_and_hzmb": (
                "4 CEM EV stations in Zona A (stationdistrict=='Zona A': 9480123, 9480425, "
                "9480527, 9488798) + 2 on HZMB Artificial Island (234, 239) lie 350–1,294 m "
                "east of 2021 terrestrial zones Z01–Z23 (where 2021 residential census "
                "population was 0)."
            ),
            "geogrouping_caveat": (
                "DSEC bundles multi-building blocks (e.g., OID 5322 Airport/Pac On/MUST, "
                "pop=4,212) into single geogrouping polygons; hybrid >100,000 m² OSM "
                "building-footprint refinement + per-building normalized areal weighting "
                "conserves exact 681,293 / 82,751 terrestrial totals while achieving 96.5% "
                "walk-graph snap < 300 m."
            ),
        },
    }

    prov_raw_path = DATA_RAW_DIR / "data_provenance.json"
    prov_metrics_path = OUTPUTS_METRICS_DIR / "data_provenance.json"
    for p in (prov_raw_path, prov_metrics_path):
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2, ensure_ascii=False)

    print(
        f"[M1 DONE] H3 r9={len(cells_r9_out)} cells (pop={cells_r9_out['pop_total'].sum():.1f}, "
        f"65+={cells_r9_out['pop_65plus'].sum():.1f}, EV+={cells_r9_out['has_ev_charger'].sum()}), "
        f"H3 r8={len(cells_r8_out)} cells, facilities={len(facilities_clean_gdf)}, "
        f"LRT={len(lrt_gdf)}, edges={graph_audit['total_directed_edges']}."
    )
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Milestone M1: Ingest, clean, allocate H3 grids, build walk graph & folds."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use cached snapshots in data/raw/ without making network requests.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download all raw files even if already cached in data/raw/.",
    )
    args = parser.parse_args(argv)
    run_m1_pipeline(skip_download=args.skip_download, force_download=args.force_download)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
