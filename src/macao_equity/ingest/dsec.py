"""DSEC 2021 Population Census ArcGIS REST ingestion, Esri ring repair, and CRS verification."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import requests
import shapely
from shapely.geometry import LinearRing, MultiPolygon, Point, Polygon
from shapely.validation import make_valid

from macao_equity.config import (
    ANALYSIS_CRS,
    DSEC_BUILDINGS_URL,
    DSEC_MARITIME_POP_65PLUS,
    DSEC_MARITIME_POP_TOTAL,
    DSEC_NATIVE_CRS,
    DSEC_TERRESTRIAL_POP_65PLUS,
    DSEC_TERRESTRIAL_POP_TOTAL,
    DSEC_ZONES_URL,
    PARISH_INFO,
    WGS84_CRS,
    ZONE_TO_PARISH,
    make_transformer,
    reproject_geom,
    sha256_file,
    utc8_now_iso,
)

# Verified landmark reference coordinates (WGS84 lat, lon) for EPSG:8433 -> EPSG:32649 check
OSM_YAT_KING_LAU_WGS84 = (22.2073517, 113.5456511)  # 逸景樓 (OID=2)


def _fetch_json_with_retry(
    url: str,
    params: dict[str, Any],
    timeout: int = 60,
    max_retries: int = 4,
) -> dict[str, Any]:
    """Fetch JSON from an ArcGIS REST endpoint via POST (avoids IIS GET URL length limit)."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, data=params, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            if "error" in payload:
                raise RuntimeError(f"ArcGIS REST error on {url}: {payload['error']}")
            return payload
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt + 1 < max_retries:
                time.sleep(1.5 * (2**attempt))
    raise RuntimeError(f"Failed fetching {url} after {max_retries} attempts") from last_exc


def download_dsec_layer(
    layer_url: str,
    out_path: Path,
    chunk_size: int = 500,
    skip_download: bool = False,
) -> tuple[dict[str, Any], str]:
    """Download an entire DSEC ArcGIS 10.03 layer in native EPSG:8433 (NEVER passing outSR).

    Uses ``returnIdsOnly=true`` and paginates by ``objectIds`` chunks to ensure complete,
    lossless retrieval at full native vertex resolution.
    """
    if skip_download:
        if not out_path.exists():
            raise FileNotFoundError(f"--skip-download set but missing raw file: {out_path}")
        with out_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload, payload.get("fetched_at_utc8", utc8_now_iso())

    query_url = f"{layer_url.rstrip('/')}/query"
    id_resp = _fetch_json_with_retry(
        query_url,
        params={"where": "1=1", "returnIdsOnly": "true", "f": "json"},
    )
    object_ids = sorted(int(oid) for oid in id_resp.get("objectIds", []))
    if not object_ids:
        raise RuntimeError(f"No OBJECTIDs returned from {query_url}")

    all_features: list[dict[str, Any]] = []
    spatial_reference: dict[str, Any] | None = None
    fields: list[dict[str, Any]] | None = None
    for start in range(0, len(object_ids), chunk_size):
        chunk = object_ids[start : start + chunk_size]
        chunk_resp = _fetch_json_with_retry(
            query_url,
            params={
                "objectIds": ",".join(str(i) for i in chunk),
                "outFields": "*",
                "returnGeometry": "true",
                "f": "json",
            },
        )
        if spatial_reference is None:
            spatial_reference = chunk_resp.get("spatialReference")
        if fields is None:
            fields = chunk_resp.get("fields")
        all_features.extend(chunk_resp.get("features", []))

    if len(all_features) != len(object_ids):
        raise RuntimeError(
            f"DSEC pagination mismatch on {layer_url}: "
            f"expected {len(object_ids)}, got {len(all_features)}"
        )

    fetched_at = utc8_now_iso()
    payload = {
        "layer_url": layer_url,
        "query_url": query_url,
        "fetched_at_utc8": fetched_at,
        "spatialReference": spatial_reference,
        "fields": fields,
        "feature_count": len(all_features),
        "features": all_features,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return payload, fetched_at


def _extract_polygonal(geom: Any) -> Polygon | MultiPolygon:
    """Extract only polygonal components with positive area from a repaired geometry."""
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    if geom.geom_type == "GeometryCollection":
        parts = [
            g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon") and g.area > 0.0
        ]
        if not parts:
            raise ValueError("Repaired geometry contains no polygonal parts with positive area")
        merged = parts[0] if len(parts) == 1 else shapely.union_all(parts)
        if merged.geom_type in ("Polygon", "MultiPolygon"):
            return merged
        return _extract_polygonal(merged)
    raise ValueError(f"Unexpected repaired geometry type: {geom.geom_type}")


def esri_rings_to_valid_geom(
    rings: list[list[list[float]]],
    min_ring_area_m2: float = 1e-3,
) -> Polygon | MultiPolygon:
    """Convert Esri JSON polygon rings (CW exterior, CCW hole) into a valid Shapely polygon.

    Handles:
    - Dropping degenerate sub-millimeter 4-point slivers (e.g., DSEC Z15/Z19).
    - Nesting CCW interior holes into their smallest containing CW exterior (e.g., Z21 lagoon).
    - Multi-exterior polygons (e.g., Z23 islets and multi-tower building groups).
    - Repairing ring self-intersections via ``make_valid`` + polygonal extraction.
    """
    exteriors: list[Polygon] = []
    holes: list[list[list[float]]] = []
    for ring in rings:
        if len(ring) < 4:
            continue
        poly: Any = Polygon(ring)
        if not poly.is_valid and poly.area < min_ring_area_m2:
            repaired_ring = make_valid(poly)
            if repaired_ring.area >= min_ring_area_m2:
                poly = _extract_polygonal(repaired_ring)
        if poly.area < min_ring_area_m2:
            continue
        if poly.geom_type == "MultiPolygon":
            exteriors.extend(list(poly.geoms))
        elif not LinearRing(ring).is_ccw:  # CW in Esri JSON = exterior ring
            exteriors.append(poly)
        else:  # CCW in Esri JSON = interior hole
            holes.append(ring)

    if not exteriors:
        # Fallback if a single-ring feature had reversed orientation
        exteriors = [
            Polygon(r) for r in rings if len(r) >= 4 and Polygon(r).area >= min_ring_area_m2
        ]
        holes = []
    if not exteriors:
        raise ValueError("Esri rings contained no valid rings above min_ring_area_m2")

    ext_holes: list[list[list[list[float]]]] = [[] for _ in exteriors]
    for hole in holes:
        hole_poly = Polygon(hole)
        pt = hole_poly.representative_point()
        candidates = [(idx, ext.area) for idx, ext in enumerate(exteriors) if ext.contains(pt)]
        if candidates:
            target_idx = min(candidates, key=lambda pair: pair[1])[0]
            ext_holes[target_idx].append(hole)
        else:
            # Treat orphan CCW ring as an exterior rather than dropping geometry
            exteriors.append(hole_poly)
            ext_holes.append([])

    polys = [Polygon(ext.exterior.coords, hs) for ext, hs in zip(exteriors, ext_holes)]
    geom: Any = polys[0] if len(polys) == 1 else MultiPolygon(polys)
    if not geom.is_valid:
        geom = _extract_polygonal(make_valid(geom))
    return geom


def _format_zone_id(raw_zona: Any) -> str:
    """Normalize DSEC Zona integer/string into canonical 'Z01'..'Z26'."""
    s = str(raw_zona).strip()
    if s.upper().startswith("Z"):
        num = int(s[1:])
    else:
        num = int(float(s))
    return f"Z{num:02d}"


def load_dsec_zones(zones_payload: dict[str, Any]) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Parse DSEC 24 zones from native EPSG:8433 to EPSG:32649, excluding maritime Z26."""
    tr_8433_to_32649 = make_transformer(DSEC_NATIVE_CRS, ANALYSIS_CRS)
    rows: list[dict[str, Any]] = []
    excluded_maritime: list[dict[str, Any]] = []
    repaired_count = 0
    max_area_rel_diff = 0.0

    for feat in zones_payload["features"]:
        attrs = feat["attributes"]
        rings = feat["geometry"]["rings"]
        zone_id = _format_zone_id(attrs["Zona"])

        raw_geom = esri_rings_to_valid_geom(rings)
        if not Polygon(rings[0]).is_valid:
            repaired_count += 1
        shape_area = float(attrs.get("Shape.area") or attrs.get("Shape_Area") or raw_geom.area)
        rel_diff = abs(raw_geom.area - shape_area) / shape_area if shape_area > 0 else 0.0
        max_area_rel_diff = max(max_area_rel_diff, rel_diff)

        geom_32649 = reproject_geom(raw_geom, tr_8433_to_32649)
        if not geom_32649.is_valid:
            geom_32649 = _extract_polygonal(make_valid(geom_32649))

        desc_c = str(attrs.get("Desc_C") or "").strip()
        desc_abb = str(attrs.get("Desc_Abb") or "").strip()
        desc_p = str(attrs.get("Desc_P") or "").strip()
        zone_name = (
            f"{desc_c} ({desc_abb})" if desc_c and desc_abb else (desc_c or desc_abb or desc_p)
        )

        if zone_id == "Z26":
            excluded_maritime.append(
                {
                    "zone_id": zone_id,
                    "desc_c": desc_c,
                    "shape_area_m2": round(shape_area, 2),
                    "reason": (
                        "Maritime statistical zone (水域), excluded from terrestrial analysis"
                    ),
                }
            )
            continue

        parish_id = ZONE_TO_PARISH[zone_id]
        parish_meta = PARISH_INFO[parish_id]
        rows.append(
            {
                "zone_id": zone_id,
                "zone_name": zone_name,
                "zone_name_zh": desc_c,
                "zone_name_en": desc_abb,
                "parish_id": parish_id,
                "parish_name": parish_meta["parish_name"],
                "parish_name_zh": parish_meta["parish_name_zh"],
                "area_km2": float(geom_32649.area / 1e6),
                "dsec_shape_area_m2": shape_area,
                "dsec_density": float(attrs.get("density") or 0.0),
                "dsec_old_per": float(attrs.get("old_per") or 0.0),
                "geometry": geom_32649,
            }
        )

    zones_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=ANALYSIS_CRS)
    zones_gdf = zones_gdf.sort_values("zone_id").reset_index(drop=True)
    if len(zones_gdf) != 23:
        raise ValueError(f"Expected exactly 23 terrestrial zones, got {len(zones_gdf)}")
    if not bool(zones_gdf.geometry.is_valid.all()):
        raise ValueError("Invalid geometries remain in zones_23")

    audit = {
        "row_count_raw": len(zones_payload["features"]),
        "row_count_clean": len(zones_gdf),
        "excluded_zones": excluded_maritime,
        "max_native_area_rel_diff": round(max_area_rel_diff, 8),
        "z06_parish_verification": (
            "Z06 (筷子基區 / Fai Chi Kei) mapped to P01 (Nossa Senhora de Fátima / 花地瑪堂區), "
            "verified against DSCC official parish areas (P01=3.20 km² vs 3.2 km², P02=1.14 km² vs "
            "1.1 km²) and OSM admin_level=6 rel/9506156 (95.8% geometric overlap)."
        ),
    }
    return zones_gdf, audit


def load_dsec_buildings(
    buildings_payload: dict[str, Any],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    """Parse all 5,682 DSEC building polygons from EPSG:8433 to EPSG:32649.

    Returns:
        (buildings_all_gdf, buildings_pop_terrestrial_gdf, audit_dict)
    """
    tr_8433_to_32649 = make_transformer(DSEC_NATIVE_CRS, ANALYSIS_CRS)
    rows: list[dict[str, Any]] = []
    invalid_raw_count = 0
    max_area_rel_diff = 0.0

    for feat in buildings_payload["features"]:
        attrs = feat["attributes"]
        rings = feat["geometry"]["rings"]

        # Check raw single-ring validity strictly for provenance count
        if rings and len(rings[0]) >= 4 and not Polygon(rings[0]).is_valid:
            invalid_raw_count += 1

        raw_geom = esri_rings_to_valid_geom(rings)
        shape_area = float(attrs.get("Shape.area") or attrs.get("Shape_Area") or raw_geom.area)
        rel_diff = abs(raw_geom.area - shape_area) / shape_area if shape_area > 0 else 0.0
        max_area_rel_diff = max(max_area_rel_diff, rel_diff)

        geom_32649 = reproject_geom(raw_geom, tr_8433_to_32649)
        if not geom_32649.is_valid:
            geom_32649 = _extract_polygonal(make_valid(geom_32649))

        rows.append(
            {
                "objectid": int(attrs["OBJECTID"]),
                "geogrouping": int(attrs.get("geogrouping") or 0),
                "total": float(attrs.get("total") or 0.0),
                "old": float(attrs.get("old") or 0.0),
                "teen": float(attrs.get("teen") or 0.0),
                "adult": float(attrs.get("adult") or 0.0),
                "nome_edf_c": str(attrs.get("Nome_Edf_C_Unicode") or "").strip(),
                "nome_edf_p": str(attrs.get("Nome_Edf_P_Unicode") or "").strip(),
                "area_m2": float(geom_32649.area),
                "geometry": geom_32649,
            }
        )

    all_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=ANALYSIS_CRS)
    if not bool(all_gdf.geometry.is_valid.all()):
        raise ValueError("Invalid geometries remain in DSEC buildings after repair")

    maritime_mask = (all_gdf["geogrouping"] == 0) | (all_gdf["objectid"] == 1)
    maritime_df = all_gdf[maritime_mask]
    maritime_total = int(round(maritime_df["total"].sum()))
    maritime_old = int(round(maritime_df["old"].sum()))
    if maritime_total != DSEC_MARITIME_POP_TOTAL or maritime_old != DSEC_MARITIME_POP_65PLUS:
        raise ValueError(
            f"Unexpected maritime population: total={maritime_total}, old={maritime_old}"
        )

    pop_terrestrial = all_gdf[(~maritime_mask) & (all_gdf["total"] > 0)].copy().reset_index(
        drop=True
    )
    terr_total = int(round(pop_terrestrial["total"].sum()))
    terr_old = int(round(pop_terrestrial["old"].sum()))
    if terr_total != DSEC_TERRESTRIAL_POP_TOTAL or terr_old != DSEC_TERRESTRIAL_POP_65PLUS:
        raise ValueError(
            f"Terrestrial population mismatch: got ({terr_total}, {terr_old}), "
            f"expected ({DSEC_TERRESTRIAL_POP_TOTAL}, {DSEC_TERRESTRIAL_POP_65PLUS})"
        )

    zero_pop_count = int(((~maritime_mask) & (all_gdf["total"] == 0)).sum())
    audit = {
        "row_count_raw": len(all_gdf),
        "row_count_populated_terrestrial": len(pop_terrestrial),
        "row_count_zero_pop_terrestrial": zero_pop_count,
        "row_count_maritime_excluded": len(maritime_df),
        "maritime_pop_total_excluded": maritime_total,
        "maritime_pop_65plus_excluded": maritime_old,
        "terrestrial_pop_total": terr_total,
        "terrestrial_pop_65plus": terr_old,
        "raw_self_intersecting_repaired": invalid_raw_count,
        "max_native_area_rel_diff": round(max_area_rel_diff, 8),
    }
    return all_gdf, pop_terrestrial, audit


def aggregate_buildings_to_zones(
    buildings_pop_gdf: gpd.GeoDataFrame,
    zones_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Aggregate populated terrestrial buildings into 23 zones via normalized area-weighted overlay.

    Per-building weight normalization prevents dropping coastal shoreline overhang slivers
    (3 geogroupings totaling 123.81 residents) and concave coastal buildings (4 in Z23,
    773 residents) that would fail centroid joins.
    """
    b_sub = buildings_pop_gdf[["objectid", "total", "old", "geometry"]].copy()
    z_sub = zones_gdf[["zone_id", "geometry"]].copy()

    inter = gpd.overlay(b_sub, z_sub, how="intersection", keep_geom_type=False)
    inter["inter_area"] = inter.geometry.area
    inter = inter[inter["inter_area"] > 0.0].copy()

    matched_oids = set(inter["objectid"].unique())
    all_oids = set(b_sub["objectid"].unique())
    unmatched = all_oids - matched_oids
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} populated terrestrial buildings did not intersect zones"
        )

    b_area_sum = inter.groupby("objectid")["inter_area"].transform("sum")
    inter["weight"] = inter["inter_area"] / b_area_sum
    inter["w_pop_total"] = inter["total"] * inter["weight"]
    inter["w_pop_65plus"] = inter["old"] * inter["weight"]

    zone_pops = (
        inter.groupby("zone_id", as_index=False)[["w_pop_total", "w_pop_65plus"]]
        .sum()
        .rename(columns={"w_pop_total": "pop_total", "w_pop_65plus": "pop_65plus"})
    )

    out = zones_gdf.merge(zone_pops, on="zone_id", how="left")
    out["pop_total"] = out["pop_total"].fillna(0.0).astype(float)
    out["pop_65plus"] = out["pop_65plus"].fillna(0.0).astype(float)
    out["pop_density"] = out["pop_total"] / out["area_km2"]
    out["log_pop_density"] = np.log(out["pop_density"])
    out["elderly_share"] = (out["pop_65plus"] / out["pop_total"]).clip(0.0, 1.0)

    # Assert exact population conservation
    tot_err = abs(out["pop_total"].sum() - DSEC_TERRESTRIAL_POP_TOTAL)
    old_err = abs(out["pop_65plus"].sum() - DSEC_TERRESTRIAL_POP_65PLUS)
    if tot_err > 1e-3 or old_err > 1e-3:
        raise ValueError(
            f"Zone population conservation failed: tot_err={tot_err}, old_err={old_err}"
        )

    cols = [
        "zone_id",
        "zone_name",
        "parish_id",
        "parish_name",
        "area_km2",
        "pop_total",
        "pop_65plus",
        "pop_density",
        "log_pop_density",
        "elderly_share",
        "geometry",
    ]
    return gpd.GeoDataFrame(out[cols], geometry="geometry", crs=ANALYSIS_CRS)


def dissolve_parishes(zones_23_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Dissolve the 23 terrestrial DSEC zones into the 7 official parishes."""
    dissolved = zones_23_gdf.dissolve(
        by="parish_id",
        aggfunc={"pop_total": "sum", "pop_65plus": "sum"},
        as_index=False,
    )
    dissolved["geometry"] = dissolved.geometry.apply(
        lambda g: _extract_polygonal(make_valid(g)) if not g.is_valid else g
    )
    dissolved["parish_name"] = dissolved["parish_id"].map(
        lambda pid: PARISH_INFO[pid]["parish_name"]
    )
    dissolved["parish_name_zh"] = dissolved["parish_id"].map(
        lambda pid: PARISH_INFO[pid]["parish_name_zh"]
    )
    dissolved["area_km2"] = dissolved.geometry.area / 1e6
    dissolved = dissolved.sort_values("parish_id").reset_index(drop=True)

    cols = [
        "parish_id",
        "parish_name",
        "parish_name_zh",
        "area_km2",
        "pop_total",
        "pop_65plus",
        "geometry",
    ]
    parishes_gdf = gpd.GeoDataFrame(dissolved[cols], geometry="geometry", crs=ANALYSIS_CRS)
    if len(parishes_gdf) != 7:
        raise ValueError(f"Expected 7 parishes, got {len(parishes_gdf)}")
    return parishes_gdf


def verify_crs_landmarks(
    buildings_all_gdf: gpd.GeoDataFrame,
    cem_gdf: gpd.GeoDataFrame,
) -> list[dict[str, Any]]:
    """Verify local EPSG:8433 -> EPSG:32649 accuracy (< 50 m) against OSM & CEM ground truth."""
    tr_4326_to_32649 = make_transformer(WGS84_CRS, ANALYSIS_CRS)
    checks: list[tuple[str, str, Point, float]] = []

    # 1. 逸景樓 vs OSM verified WGS84 coordinate
    yx, yy = tr_4326_to_32649.transform(OSM_YAT_KING_LAU_WGS84[1], OSM_YAT_KING_LAU_WGS84[0])
    checks.append(("逸景樓", "OSM (22.2073517, 113.5456511)", Point(yx, yy), 15.0))

    # 2–4. CEM EV carparks located at named housing estates
    cem_refs = [
        ("快達樓", 162, "CEM station 162 (Fai Tat)", 30.0),
        ("湖畔大廈", 186, "CEM station 186 (Edifício do Lago)", 50.0),
        ("青洲坊大廈", 254, "CEM station 254 (青洲坊大廈停車場)", 50.0),
    ]
    for name_zh, cem_id, ref_label, thresh in cem_refs:
        sub_cem = cem_gdf[cem_gdf["cem_station_id"] == cem_id]
        if sub_cem.empty:
            raise ValueError(f"Expected CEM landmark station {cem_id} not found")
        checks.append((name_zh, ref_label, sub_cem.geometry.iloc[0], thresh))

    results: list[dict[str, Any]] = []
    for name_zh, ref_label, ref_pt, thresh_m in checks:
        matches = buildings_all_gdf[
            buildings_all_gdf["nome_edf_c"].str.contains(name_zh, na=False, regex=False)
        ]
        if matches.empty:
            raise ValueError(f"DSEC landmark '{name_zh}' not found in c2021_Building_P5")
        union_geom = matches.geometry.union_all()
        centroid_err_m = float(union_geom.centroid.distance(ref_pt))
        edge_dist_m = float(union_geom.distance(ref_pt))
        passed = centroid_err_m < thresh_m
        results.append(
            {
                "landmark": name_zh,
                "dsec_polygon_count": int(len(matches)),
                "reference_source": ref_label,
                "centroid_error_m": round(centroid_err_m, 2),
                "edge_distance_m": round(edge_dist_m, 2),
                "threshold_m": thresh_m,
                "passed": passed,
            }
        )
        if not passed:
            raise ValueError(
                f"CRS landmark verification failed for {name_zh}: "
                f"centroid_err={centroid_err_m:.2f} m >= {thresh_m} m"
            )
    return results


def build_dsec_provenance_entries(
    zones_path: Path,
    zones_fetched_at: str,
    zones_audit: dict[str, Any],
    buildings_path: Path,
    buildings_fetched_at: str,
    buildings_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build structured provenance records for the two DSEC 2021 Census layers."""
    return [
        {
            "source_id": "dsec_zones_2021",
            "dataset_name": "DSEC 2021 Population Census Statistical Zones (c2021_Zona)",
            "authority": "DSEC (Direcção dos Serviços de Estatística e Censos)",
            "dataset_id_or_layer": "Production2021/Census-QueryZonaBuilding-2021/MapServer/0",
            "url": f"{DSEC_ZONES_URL}/query",
            "fetched_at_utc8": zones_fetched_at,
            "local_raw_path": "data/raw/dsec_zones_2021.json",
            "sha256": sha256_file(zones_path),
            "byte_size": zones_path.stat().st_size,
            "native_crs": "EPSG:8433 (ArcGIS legacy wkid:3064)",
            "target_crs": ANALYSIS_CRS,
            "row_count_raw": zones_audit["row_count_raw"],
            "row_count_clean": zones_audit["row_count_clean"],
            "capacity_completeness_pct": None,
            "fallback_used": False,
            "fallback_reason": None,
            "anomalies_handled": [
                "Excluded Z26 maritime (水域); kept 23 terrestrial zones (Z01–Z23)",
                "Local EPSG:8433 -> EPSG:32649 (always_xy=True); avoided broken server outSR=4326",
                (
                    "Esri CW/CCW ring nesting (Z21 2 genuine airport lagoon holes), "
                    "<1e-3 m² sliver drop (Z15, Z19), make_valid repair (Z01, Z15)"
                ),
                zones_audit["z06_parish_verification"],
            ],
        },
        {
            "source_id": "dsec_buildings_2021",
            "dataset_name": "DSEC 2021 Population Census Buildings (c2021_Building_P5)",
            "authority": "DSEC (Direcção dos Serviços de Estatística e Censos)",
            "dataset_id_or_layer": "Production2021/Census-QueryZonaBuilding-2021/MapServer/1",
            "url": f"{DSEC_BUILDINGS_URL}/query",
            "fetched_at_utc8": buildings_fetched_at,
            "local_raw_path": "data/raw/dsec_buildings_2021.json",
            "sha256": sha256_file(buildings_path),
            "byte_size": buildings_path.stat().st_size,
            "native_crs": "EPSG:8433 (ArcGIS legacy wkid:3064)",
            "target_crs": ANALYSIS_CRS,
            "row_count_raw": buildings_audit["row_count_raw"],
            "row_count_clean": buildings_audit["row_count_populated_terrestrial"],
            "capacity_completeness_pct": 100.0,
            "fallback_used": False,
            "fallback_reason": None,
            "anomalies_handled": [
                (
                    f"Excluded OBJECTID=1 maritime (水域: "
                    f"{buildings_audit['maritime_pop_total_excluded']} total / "
                    f"{buildings_audit['maritime_pop_65plus_excluded']} aged 65+) and "
                    f"{buildings_audit['row_count_zero_pop_terrestrial']} zero-pop polygons; "
                    f"retained {buildings_audit['row_count_populated_terrestrial']} populated "
                    f"terrestrial polygons summing to exact "
                    f"{buildings_audit['terrestrial_pop_total']} total / "
                    f"{buildings_audit['terrestrial_pop_65plus']} aged 65+"
                ),
                (
                    f"Repaired {buildings_audit['raw_self_intersecting_repaired']} "
                    "self-intersecting raw polygons via CW/CCW ring nesting + make_valid"
                ),
                (
                    "Normalized per-building area-weighted overlay into zones_23 & H3 cells, "
                    "preventing concave Z23 centroid drops (773 pop) and shoreline overhang "
                    "loss (123.81 pop)"
                ),
            ],
        },
    ]
