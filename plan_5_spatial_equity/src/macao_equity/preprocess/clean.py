"""Deterministic cleaning, official whitelisting, and cross-source dedup for Macao facilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.validation import make_valid

from macao_equity.config import (
    ANALYSIS_CRS,
    BUS_SHUTTLE_OPERATORS,
    BUS_SHUTTLE_REGEX,
    BUS_STOP_EXCLUDED_IDS,
    CROSS_SOURCE_DEDUP_M,
    IC_LIBRARY_RELOCATION_MAP,
    IC_LIBRARY_WHITELIST,
    LRT_EXCLUDED_DUPLICATE_IDS,
    LRT_STATION_META,
    PARK_EXCLUDED_ACCESS,
    PARK_EXCLUDED_NODE_IDS,
    PARK_EXTRA_RECREATION_WHITELIST,
    PARK_MEMBER_WAY_DUPLICATES,
    SSM_HEALTH_CENTRE_WHITELIST,
)

ZHUHAI_CRH_RE = re.compile(r"珠海|湾仔北|灣仔北|长隆|長隆|東線|东线")

CANONICAL_FACILITY_COLUMNS = [
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


def dedup_cross_source_25m(
    primary_gdf: gpd.GeoDataFrame,
    secondary_gdf: gpd.GeoDataFrame,
    radius_m: float = CROSS_SOURCE_DEDUP_M,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Cross-source-only spatial deduplication (EPSG:32649).

    NEVER deduplicates records within ``primary_gdf`` (preserves genuine nearby carparks or
    opposite-side street bus stops). Drops rows from ``secondary_gdf`` within ``radius_m``
    of any point in ``primary_gdf``.
    """
    if primary_gdf.empty:
        return secondary_gdf.copy().reset_index(drop=True), secondary_gdf.iloc[0:0].copy()
    if secondary_gdf.empty:
        return primary_gdf.copy().reset_index(drop=True), secondary_gdf.iloc[0:0].copy()

    p_xy = np.column_stack(
        [primary_gdf.geometry.x.to_numpy(), primary_gdf.geometry.y.to_numpy()]
    )
    s_xy = np.column_stack(
        [secondary_gdf.geometry.x.to_numpy(), secondary_gdf.geometry.y.to_numpy()]
    )
    dists, _ = cKDTree(p_xy).query(s_xy)
    drop_mask = dists <= radius_m

    kept_secondary = secondary_gdf[~drop_mask].copy()
    dropped_secondary = secondary_gdf[drop_mask].copy()
    combined = gpd.GeoDataFrame(
        pd.concat([primary_gdf, kept_secondary], ignore_index=True),
        geometry="geometry",
        crs=primary_gdf.crs,
    )
    return combined, dropped_secondary


def _read_gpkg_layer_utm(gpkg_path: Path, layer: str) -> gpd.GeoDataFrame:
    """Read a layer from Geofabrik macau.gpkg and project cleanly to EPSG:32649."""
    gdf = gpd.read_file(gpkg_path, layer=layer)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs(ANALYSIS_CRS)
    gdf["osm_id"] = gdf["osm_id"].astype(str).str.strip()
    return gdf


def build_health_centres(gpkg_path: Path) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Extract strictly the 11 official SSM terrestrial primary health centres/stations."""
    pois = _read_gpkg_layer_utm(gpkg_path, "gis_osm_pois_free")
    pois_a = _read_gpkg_layer_utm(gpkg_path, "gis_osm_pois_a_free")

    pois_clinic = pois[pois["fclass"] == "clinic"].copy()
    pois_a_clinic = pois_a[pois_a["fclass"] == "clinic"].copy()
    pois_a_clinic["geometry"] = pois_a_clinic.geometry.representative_point()

    raw_clinics = pd.concat([pois_clinic, pois_a_clinic], ignore_index=True)
    raw_count = int(len(raw_clinics))

    whitelist_ids = set(SSM_HEALTH_CENTRE_WHITELIST.keys())
    kept = raw_clinics[raw_clinics["osm_id"].isin(whitelist_ids)].drop_duplicates(
        subset=["osm_id"], keep="first"
    )
    kept = kept.sort_values("osm_id").reset_index(drop=True)

    missing = whitelist_ids - set(kept["osm_id"])
    if missing or len(kept) != 11:
        raise ValueError(
            f"Expected 11 official SSM health centres, got {len(kept)} (missing={sorted(missing)})"
        )

    rows = []
    for idx, row in kept.iterrows():
        oid = str(row["osm_id"])
        rows.append(
            {
                "facility_id": f"hc_{idx + 1:02d}",
                "osm_id": oid,
                "facility_type": "health_centre",
                "name": SSM_HEALTH_CENTRE_WHITELIST[oid],
                "source": "osm_geofabrik+gov_mo_ssm",
                "capacity": 1.0,
                "capacity_unit": "facility",
                "capacity_imputed": False,
                "pilenum": float("nan"),
                "total_kw": float("nan"),
                "in_zona_a": False,
                "geometry": row.geometry,
            }
        )

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=ANALYSIS_CRS)
    audit = {
        "raw_clinic_rows": raw_count,
        "final_count": int(len(gdf)),
        "capacity_completeness_pct": 100.0,
        "exclusions_documented": [
            (
                "Excluded 新城A區衛生站 (node/13304614301, fclass=doctors, "
                "2021 census pop=0 in Zona A)"
            ),
            "Excluded 橫琴澳門新街坊衛生站 (Mainland Hengqin, outside Macao SAR)",
            (
                "Excluded hospitals, specialized infectious-disease/TB centres, "
                "and private/union clinics"
            ),
        ],
    }
    return gdf, audit


def build_libraries(gpkg_path: Path) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Extract the 15 official IC general public lending libraries (applying IC Notice 548 fix)."""
    pois = _read_gpkg_layer_utm(gpkg_path, "gis_osm_pois_free")
    pois_a = _read_gpkg_layer_utm(gpkg_path, "gis_osm_pois_a_free")
    bldgs_a = _read_gpkg_layer_utm(gpkg_path, "gis_osm_buildings_a_free")

    pois_lib = pois[pois["fclass"] == "library"].copy()
    pois_a_lib = pois_a[pois_a["fclass"] == "library"].copy()
    pois_a_lib["geometry"] = pois_a_lib.geometry.representative_point()

    raw_libs = pd.concat([pois_lib, pois_a_lib], ignore_index=True)
    raw_count = int(len(raw_libs))

    whitelist_ids = set(IC_LIBRARY_WHITELIST.keys())
    kept = raw_libs[raw_libs["osm_id"].isin(whitelist_ids)].drop_duplicates(
        subset=["osm_id"], keep="first"
    )
    kept = kept.sort_values("osm_id").reset_index(drop=True)

    missing = whitelist_ids - set(kept["osm_id"])
    if missing or len(kept) != 15:
        raise ValueError(
            f"Expected 15 official IC public libraries, got {len(kept)} (missing={sorted(missing)})"
        )

    relocated_log: list[dict[str, Any]] = []
    rows = []
    for idx, row in kept.iterrows():
        oid = str(row["osm_id"])
        geom = row.geometry
        if oid in IC_LIBRARY_RELOCATION_MAP:
            target_bldg_id = IC_LIBRARY_RELOCATION_MAP[oid]
            bldg_match = bldgs_a[bldgs_a["osm_id"] == target_bldg_id]
            if bldg_match.empty:
                raise ValueError(
                    f"Relocation target building way/{target_bldg_id} missing from macau.gpkg"
                )
            old_pt = geom
            geom = bldg_match.geometry.iloc[0].representative_point()
            shift_m = float(old_pt.distance(geom))
            relocated_log.append(
                {
                    "osm_id": oid,
                    "new_building_osm_id": target_bldg_id,
                    "name": IC_LIBRARY_WHITELIST[oid],
                    "shift_m": round(shift_m, 2),
                    "basis": "IC Notice 548 (青洲坊圖書館 at 青洲坊大廈第二座)",
                }
            )

        rows.append(
            {
                "facility_id": f"lib_{idx + 1:02d}",
                "osm_id": oid,
                "facility_type": "library",
                "name": IC_LIBRARY_WHITELIST[oid],
                "source": "osm_geofabrik+ic_library_gov_mo",
                "capacity": 1.0,
                "capacity_unit": "facility",
                "capacity_imputed": False,
                "pilenum": float("nan"),
                "total_kw": float("nan"),
                "in_zona_a": False,
                "geometry": geom,
            }
        )

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=ANALYSIS_CRS)
    audit = {
        "raw_library_rows": raw_count,
        "final_count": int(len(gdf)),
        "capacity_completeness_pct": 100.0,
        "relocations_applied": relocated_log,
        "exclusions_documented": [
            "Excluded 5 non-IC libraries: UM Wu Yee Sun (241702236), MUST (192096008), "
            "St. Paul's School Lau Yeung Siu Mui (788174713), MAM Art Museum (3210223507), "
            "Octagonal Pavilion Chamber of Commerce (405411408)",
            "Excluded special rare-book reference library 議事亭藏書樓 (Senado Library) to keep "
            "strictly general neighbourhood lending branches (15 official IC branches)",
        ],
    }
    return gdf, audit


def build_parks(
    gpkg_path: Path,
    pbf_gardens_poly: gpd.GeoDataFrame,
    pbf_gardens_pt: gpd.GeoDataFrame,
    pbf_access_tags: dict[str, str],
    macao_sar_geom_32649: Any,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Build cleaned public parks/gardens/playgrounds/sitting-out areas with m² capacity."""
    landuse_a = _read_gpkg_layer_utm(gpkg_path, "gis_osm_landuse_a_free")
    pois_a = _read_gpkg_layer_utm(gpkg_path, "gis_osm_pois_a_free")
    pois = _read_gpkg_layer_utm(gpkg_path, "gis_osm_pois_free")

    # 1. Polygons from macau.gpkg (NEVER dump raw recreation_ground; preserves hole on rel/10266786)
    lu_park = landuse_a[landuse_a["fclass"] == "park"][["osm_id", "name", "geometry"]].copy()
    lu_park["source"] = "osm_geofabrik_landuse"

    lu_extra = landuse_a[landuse_a["osm_id"].isin(PARK_EXTRA_RECREATION_WHITELIST)][
        ["osm_id", "name", "geometry"]
    ].copy()
    lu_extra["source"] = "osm_geofabrik_recreation_whitelist"

    pa_play = pois_a[pois_a["fclass"] == "playground"][["osm_id", "name", "geometry"]].copy()
    pa_play["source"] = "osm_geofabrik_pois_a"

    polys = pd.concat([lu_park, lu_extra, pa_play], ignore_index=True)
    polys = polys.drop_duplicates(subset=["osm_id"], keep="first")
    polys = gpd.GeoDataFrame(polys, geometry="geometry", crs=ANALYSIS_CRS)
    polys["geometry"] = polys.geometry.apply(lambda g: make_valid(g) if not g.is_valid else g)
    polys = polys[polys.geometry.representative_point().within(macao_sar_geom_32649)].copy()

    # 2. Supplement leisure=garden polygons from PBF not already in macau.gpkg
    if not pbf_gardens_poly.empty:
        existing_ids = set(polys["osm_id"])
        gp_new = pbf_gardens_poly[~pbf_gardens_poly["osm_id"].isin(existing_ids)][
            ["osm_id", "name", "geometry"]
        ].copy()
        gp_new["source"] = "osm_pbf_garden"
        polys = gpd.GeoDataFrame(
            pd.concat([polys, gp_new], ignore_index=True),
            geometry="geometry",
            crs=ANALYSIS_CRS,
        )

    raw_poly_count = int(len(polys))

    # 3. Drop restricted access tags (e.g., way/1234891053 access=students;staff)
    polys["access"] = polys["osm_id"].map(lambda oid: pbf_access_tags.get(str(oid), "").lower())
    restricted_polys = polys[polys["access"].isin(PARK_EXCLUDED_ACCESS)]
    polys = polys[~polys["access"].isin(PARK_EXCLUDED_ACCESS)].copy()

    # 4. Drop relation member-way duplicates (rel/19216078 & rel/21464243 members)
    member_dup_mask = polys["osm_id"].isin(PARK_MEMBER_WAY_DUPLICATES)
    dropped_member_ways = int(member_dup_mask.sum())
    polys = polys[~member_dup_mask].copy()

    # 5. Drop nested child polygons whose representative_point lies inside a strictly larger polygon
    polys["area_m2"] = polys.geometry.area
    polys = polys[polys["area_m2"] > 0.0].sort_values(
        ["area_m2", "osm_id"], ascending=[False, True]
    ).reset_index(drop=True)

    kept_indices: list[int] = []
    kept_geoms: list[Any] = []
    kept_areas: list[float] = []
    nested_dropped = 0
    for idx, row in polys.iterrows():
        rp = row.geometry.representative_point()
        a_i = float(row["area_m2"])
        is_nested = any(
            (a_k > a_i) and g_k.contains(rp) for g_k, a_k in zip(kept_geoms, kept_areas)
        )
        if is_nested:
            nested_dropped += 1
        else:
            kept_indices.append(idx)
            kept_geoms.append(row.geometry)
            kept_areas.append(a_i)

    kept_polys = polys.loc[kept_indices].copy().reset_index(drop=True)

    # Verify 水塘公園 (rel/10266786) proper hole subtraction & zero racetrack leakage
    max_poly_area = float(kept_polys["area_m2"].max())
    if max_poly_area >= 95_000.0:
        raise ValueError(
            f"Park max polygon area {max_poly_area:.1f} m² >= 95,000 m² "
            "(possible racetrack or un-holed reservoir contamination)"
        )
    reservoir_row = kept_polys[kept_polys["osm_id"] == "10266786"]
    reservoir_area_m2 = (
        float(reservoir_row["area_m2"].iloc[0]) if not reservoir_row.empty else None
    )
    if reservoir_area_m2 is not None and not (30_000.0 < reservoir_area_m2 < 60_000.0):
        raise ValueError(
            f"rel/10266786 水塘公園 area {reservoir_area_m2:.1f} m² outside (30k, 60k) m²"
        )

    median_poly_area = float(kept_polys["area_m2"].median())

    # 6. Point-only outdoor parks (excluding FUFA indoor centre and points inside kept polygons)
    pt_gpkg = pois[pois["fclass"].isin(["park", "playground"])][
        ["osm_id", "name", "geometry"]
    ].copy()
    pt_gpkg["source"] = "osm_geofabrik_pois"
    pt_gpkg = pt_gpkg[pt_gpkg.geometry.within(macao_sar_geom_32649)].copy()

    if not pbf_gardens_pt.empty:
        gp_pt = pbf_gardens_pt[["osm_id", "name", "geometry"]].copy()
        gp_pt["source"] = "osm_pbf_garden_node"
        pts = gpd.GeoDataFrame(
            pd.concat([pt_gpkg, gp_pt], ignore_index=True),
            geometry="geometry",
            crs=ANALYSIS_CRS,
        )
    else:
        pts = pt_gpkg

    pts = pts.drop_duplicates(subset=["osm_id"], keep="first")
    pts["access"] = pts["osm_id"].map(lambda oid: pbf_access_tags.get(str(oid), "").lower())
    pts = pts[
        (~pts["osm_id"].isin(PARK_EXCLUDED_NODE_IDS))
        & (~pts["access"].isin(PARK_EXCLUDED_ACCESS))
    ].copy()

    kept_poly_union = kept_polys.geometry.union_all()
    inside_poly_mask = pts.geometry.within(kept_poly_union)
    pts_dropped_in_poly = int(inside_poly_mask.sum())
    kept_pts = pts[~inside_poly_mask].copy().reset_index(drop=True)

    # 7. Assemble final point-represented park facilities with area m² capacity
    rows: list[dict[str, Any]] = []
    for _, row in kept_polys.iterrows():
        name = str(row.get("name") or "").strip() or f"Public Park ({row['osm_id']})"
        rows.append(
            {
                "osm_id": str(row["osm_id"]),
                "facility_type": "park",
                "name": name,
                "source": str(row["source"]),
                "capacity": float(row["area_m2"]),
                "capacity_unit": "m2",
                "capacity_imputed": False,
                "pilenum": float("nan"),
                "total_kw": float("nan"),
                "in_zona_a": False,
                "geometry": row.geometry.representative_point(),
            }
        )

    for _, row in kept_pts.iterrows():
        name = str(row.get("name") or "").strip() or f"Public Park Point ({row['osm_id']})"
        rows.append(
            {
                "osm_id": str(row["osm_id"]),
                "facility_type": "park",
                "name": name,
                "source": str(row["source"]),
                "capacity": median_poly_area,
                "capacity_unit": "m2",
                "capacity_imputed": True,
                "pilenum": float("nan"),
                "total_kw": float("nan"),
                "in_zona_a": False,
                "geometry": row.geometry,
            }
        )

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=ANALYSIS_CRS)
    gdf = gdf.sort_values("osm_id").reset_index(drop=True)
    gdf.insert(0, "facility_id", [f"park_{i + 1:03d}" for i in range(len(gdf))])

    n_poly = int(len(kept_polys))
    n_pt = int(len(kept_pts))
    cap_completeness = round(100.0 * n_poly / len(gdf), 2) if len(gdf) else 0.0
    audit = {
        "raw_candidate_polygons": raw_poly_count,
        "dropped_restricted_access": int(len(restricted_polys)),
        "dropped_member_way_duplicates": dropped_member_ways,
        "dropped_nested_child_polygons": nested_dropped,
        "dropped_points_inside_polygons": pts_dropped_in_poly,
        "kept_polygon_parks": n_poly,
        "kept_point_only_parks": n_pt,
        "final_count": int(len(gdf)),
        "median_polygon_area_m2_used_for_imputation": round(median_poly_area, 2),
        "max_polygon_area_m2": round(max_poly_area, 2),
        "reservoir_park_rel_10266786_area_m2": (
            round(reservoir_area_m2, 2) if reservoir_area_m2 is not None else None
        ),
        "capacity_completeness_pct": cap_completeness,
    }
    return gdf, audit


def build_bus_stops(
    gpkg_path: Path,
    pbf_bus_operators: dict[str, str],
    macao_sar_geom_32649: Any,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Build cleaned public bus stops (osm_id dedup first, zero within-OSM spatial merging)."""
    transport = _read_gpkg_layer_utm(gpkg_path, "gis_osm_transport_free")
    bus_raw = transport[transport["fclass"].isin(["bus_stop", "bus_station"])].copy()
    bus_raw = bus_raw[bus_raw.geometry.within(macao_sar_geom_32649)].copy()
    raw_count = int(len(bus_raw))

    # Step 1: Deduplicate strictly by osm_id FIRST (eliminates 62 dual-tagged 0.0 m duplicates)
    bus_raw["_prio"] = bus_raw["fclass"].map({"bus_stop": 0, "bus_station": 1}).fillna(2)
    bus_dedup = (
        bus_raw.sort_values(["osm_id", "_prio"])
        .drop_duplicates(subset=["osm_id"], keep="first")
        .drop(columns=["_prio"])
        .reset_index(drop=True)
    )
    exact_id_dups_removed = raw_count - int(len(bus_dedup))

    # Step 2: Drop explicit duplicate node and private casino free-shuttle stops
    explicit_dup_mask = bus_dedup["osm_id"].isin(BUS_STOP_EXCLUDED_IDS)
    bus_dedup["operator"] = bus_dedup["osm_id"].map(lambda oid: pbf_bus_operators.get(str(oid), ""))
    shuttle_mask = bus_dedup["name"].fillna("").str.contains(BUS_SHUTTLE_REGEX) | bus_dedup[
        "operator"
    ].isin(BUS_SHUTTLE_OPERATORS)

    kept = bus_dedup[(~explicit_dup_mask) & (~shuttle_mask)].copy()
    kept = kept.sort_values("osm_id").reset_index(drop=True)

    # Step 3: Cross-source-only 25 m dedup (empty secondary since data.gov.mo DSAT is login-gated)
    empty_secondary = kept.iloc[0:0].copy()
    kept, _ = dedup_cross_source_25m(kept, empty_secondary, radius_m=CROSS_SOURCE_DEDUP_M)

    rows = []
    for idx, row in kept.iterrows():
        oid = str(row["osm_id"])
        name = str(row.get("name") or "").strip() or f"Bus Stop ({oid})"
        rows.append(
            {
                "facility_id": f"bus_{idx + 1:03d}",
                "osm_id": oid,
                "facility_type": "bus_stop",
                "name": name,
                "source": "osm_geofabrik",
                "capacity": 1.0,
                "capacity_unit": "stop",
                "capacity_imputed": False,
                "pilenum": float("nan"),
                "total_kw": float("nan"),
                "in_zona_a": False,
                "geometry": row.geometry,
            }
        )

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=ANALYSIS_CRS)
    audit = {
        "raw_transport_bus_rows": raw_count,
        "exact_same_osm_id_dups_removed": exact_id_dups_removed,
        "explicit_duplicate_nodes_removed": int(explicit_dup_mask.sum()),
        "private_casino_shuttle_stops_removed": int(shuttle_mask.sum()),
        "within_osm_spatial_merges": 0,
        "final_count": int(len(gdf)),
        "capacity_completeness_pct": 100.0,
    }
    return gdf, audit


def build_lrt_stations(
    gpkg_path: Path,
    pbf_lrt_gdf: gpd.GeoDataFrame,
    macao_sar_geom_32649: Any,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Build operational MLM LRT stations clipped to Macao SAR, deduplicating Lotus platform."""
    transport = _read_gpkg_layer_utm(gpkg_path, "gis_osm_transport_free")
    gpkg_rail = transport[transport["fclass"] == "railway_station"][
        ["osm_id", "name", "geometry"]
    ].copy()
    gpkg_rail = gpkg_rail[gpkg_rail.geometry.within(macao_sar_geom_32649)].copy()

    frames = []
    if not pbf_lrt_gdf.empty:
        p_sub = pbf_lrt_gdf[["osm_id", "name", "name_en", "geometry"]].copy()
        frames.append(p_sub)
    if not gpkg_rail.empty:
        g_sub = gpkg_rail.copy()
        g_sub["name_en"] = ""
        frames.append(g_sub)

    if not frames:
        raise ValueError("No LRT stations found in PBF or macau.gpkg")

    merged = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), geometry="geometry", crs=ANALYSIS_CRS
    )
    merged["osm_id"] = merged["osm_id"].astype(str).str.strip()
    merged = merged.drop_duplicates(subset=["osm_id"], keep="first")

    # Drop Mainland Zhuhai CRH / construction stations and explicit Lotus duplicate (79.4 m)
    zhuhai_mask = merged["name"].fillna("").str.contains(ZHUHAI_CRH_RE)
    lotus_dup_mask = merged["osm_id"].isin(LRT_EXCLUDED_DUPLICATE_IDS)
    kept = merged[(~zhuhai_mask) & (~lotus_dup_mask)].copy()

    # Keep only verified operational MLM stations (guards against stray non-MLM railway tags)
    known_mask = kept["osm_id"].isin(LRT_STATION_META.keys())
    kept = kept[known_mask].copy()

    # Order canonically by LRT_STATION_META insertion order
    order_map = {oid: i for i, oid in enumerate(LRT_STATION_META.keys())}
    kept["_ord"] = kept["osm_id"].map(order_map)
    kept = kept.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)

    rows = []
    for idx, row in kept.iterrows():
        oid = str(row["osm_id"])
        name_zh, name_en, line, in_main = LRT_STATION_META[oid]
        rows.append(
            {
                "station_id": f"LRT{idx + 1:02d}",
                "osm_id": oid,
                "name": f"{name_zh} ({name_en})",
                "name_en": name_en,
                "line": line,
                "in_main_islands": bool(in_main),
                "geometry": row.geometry,
            }
        )

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=ANALYSIS_CRS)
    if len(gdf) < 14:
        raise ValueError(f"Expected >= 14 operational MLM LRT stations, got {len(gdf)}")

    audit = {
        "final_count": int(len(gdf)),
        "dropped_lotus_duplicate_ids": sorted(
            merged.loc[lotus_dup_mask, "osm_id"].astype(str).tolist()
        ),
        "dropped_zhuhai_or_construction": int(zhuhai_mask.sum()),
        "station_ids": gdf["station_id"].tolist(),
        "osm_ids": gdf["osm_id"].tolist(),
    }
    return gdf, audit


def assemble_facilities_clean(
    parks_gdf: gpd.GeoDataFrame,
    health_gdf: gpd.GeoDataFrame,
    libraries_gdf: gpd.GeoDataFrame,
    bus_gdf: gpd.GeoDataFrame,
    ev_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Assemble all 5 cleaned facility layers into canonical facilities_clean.gpkg schema."""
    parts = [
        parks_gdf[CANONICAL_FACILITY_COLUMNS],
        health_gdf[CANONICAL_FACILITY_COLUMNS],
        libraries_gdf[CANONICAL_FACILITY_COLUMNS],
        bus_gdf[CANONICAL_FACILITY_COLUMNS],
        ev_gdf[CANONICAL_FACILITY_COLUMNS],
    ]
    combined = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True),
        geometry="geometry",
        crs=ANALYSIS_CRS,
    )
    combined["capacity"] = combined["capacity"].astype(float)
    combined["capacity_imputed"] = combined["capacity_imputed"].astype(bool)
    combined["pilenum"] = combined["pilenum"].astype(float)
    combined["total_kw"] = combined["total_kw"].astype(float)
    combined["in_zona_a"] = combined["in_zona_a"].astype(bool)

    if not combined["facility_id"].is_unique:
        raise ValueError("Duplicate facility_id detected in facilities_clean")
    if (combined["capacity"] <= 0).any() or combined["capacity"].isna().any():
        raise ValueError("All facilities must have finite capacity > 0")
    if not (combined.geometry.geom_type == "Point").all():
        raise ValueError("All facilities_clean geometries must be Point in EPSG:32649")
    return combined
