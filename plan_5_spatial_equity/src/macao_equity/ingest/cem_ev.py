"""CEM official EV charging API ingestion, coordinate swap repair, and cross-source OSM dedup."""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree
from shapely.geometry import Point

from macao_equity.config import (
    ANALYSIS_CRS,
    CEM_EV_EN_URL,
    CEM_EV_TC_URL,
    CROSS_SOURCE_DEDUP_M,
    MACAO_LAT_BOUNDS,
    MACAO_LON_BOUNDS,
    WGS84_CRS,
    make_transformer,
    sha256_file,
    utc8_now_iso,
)

CEM_OPERATOR_RE = re.compile(r"(?i)cem|澳電")
BUS_FLEET_RE = re.compile(r"(?i)\bbus\b|巴士")
TESLA_BRAND_RE = re.compile(r"(?i)tesla|特斯拉")


def _get_json_retry(url: str, timeout: int = 45, max_retries: int = 4) -> dict[str, Any]:
    """Fetch JSON from an HTTP endpoint with retries."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt + 1 < max_retries:
                time.sleep(1.5 * (2**attempt))
    raise RuntimeError(f"Failed fetching {url} after {max_retries} attempts") from last_exc


def download_cem_ev(
    en_path: Path,
    tc_path: Path,
    skip_download: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Download CEM official EV API in BOTH lang=en and lang=tc (NEVER lang=zh)."""
    if skip_download:
        for p in (en_path, tc_path):
            if not p.exists():
                raise FileNotFoundError(f"--skip-download set but missing raw file: {p}")
        with en_path.open("r", encoding="utf-8") as f:
            en_data = json.load(f)
        with tc_path.open("r", encoding="utf-8") as f:
            tc_data = json.load(f)
        fetched_at = en_data.get("_fetched_at_utc8", utc8_now_iso())
        return en_data, tc_data, fetched_at

    en_data = _get_json_retry(CEM_EV_EN_URL)
    tc_data = _get_json_retry(CEM_EV_TC_URL)
    fetched_at = utc8_now_iso()
    en_data["_fetched_at_utc8"] = fetched_at
    tc_data["_fetched_at_utc8"] = fetched_at

    en_path.parent.mkdir(parents=True, exist_ok=True)
    with en_path.open("w", encoding="utf-8") as f:
        json.dump(en_data, f, ensure_ascii=False)
    with tc_path.open("w", encoding="utf-8") as f:
        json.dump(tc_data, f, ensure_ascii=False)
    return en_data, tc_data, fetched_at


def parse_cem_ev(
    en_data: dict[str, Any],
    tc_data: dict[str, Any],
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Parse CEM EV stations, fixing swapped lat/lon and deduplicating multi-connector heads."""
    tr_4326_to_32649 = make_transformer(WGS84_CRS, ANALYSIS_CRS)
    tc_by_id = {int(s["stationid"]): s for s in tc_data.get("list", [])}

    rows: list[dict[str, Any]] = []
    fallback_imputations: list[dict[str, Any]] = []
    zona_a_ids: list[int] = []
    hzmb_ids: list[int] = []

    for s in en_data.get("list", []):
        sid = int(s["stationid"])
        # CEM API swaps stationlongitude (=LATITUDE) and stationlatitude (=LONGITUDE) on all rows
        lat = float(s["stationlongitude"])
        lon = float(s["stationlatitude"])
        if not (
            MACAO_LAT_BOUNDS[0] <= lat <= MACAO_LAT_BOUNDS[1]
            and MACAO_LON_BOUNDS[0] <= lon <= MACAO_LON_BOUNDS[1]
        ):
            raise ValueError(
                f"CEM station {sid} out of Macao bounds after lat/lon swap: ({lat}, {lon})"
            )

        x, y = tr_4326_to_32649.transform(lon, lat)
        recs = s.get("pilerecords") or []
        pilenum = float(s["pilenum"])

        if recs:
            # Group by unique physical chargerid to avoid +59.7% multi-connector double counting
            by_cid: dict[str, list[float]] = defaultdict(list)
            for r in recs:
                cid = str(r["chargerid"]).strip()
                pw = float(r.get("poweroutput") or 0.0)
                by_cid[cid].append(pw)
            capacity = float(len(by_cid))
            capacity_imputed = False
            total_kw = float(sum(max(pws) for pws in by_cid.values()))
        else:
            capacity = pilenum
            capacity_imputed = True
            total_kw = float("nan")
            fallback_imputations.append(
                {
                    "facility_id": f"cem_{sid}",
                    "field": "capacity",
                    "value": capacity,
                    "basis": "pilenum fallback (pilerecords empty)",
                }
            )

        district_en = str(s.get("stationdistrict") or "").strip()
        in_zona_a = district_en == "Zona A"
        in_hzmb = district_en == "HZM Island macau area"
        if in_zona_a:
            zona_a_ids.append(sid)
        if in_hzmb:
            hzmb_ids.append(sid)

        name_en = str(s.get("stationname") or "").strip()
        name_tc = str((tc_by_id.get(sid) or {}).get("stationname") or "").strip()
        if name_tc and name_en and name_tc != name_en:
            full_name = f"{name_tc} ({name_en})"
        else:
            full_name = name_tc or name_en or f"CEM Station {sid}"

        rows.append(
            {
                "facility_id": f"cem_{sid}",
                "cem_station_id": sid,
                "facility_type": "ev_charger",
                "name": full_name,
                "name_en": name_en,
                "name_tc": name_tc,
                "source": "cem_api",
                "capacity": capacity,
                "capacity_unit": "charger",
                "capacity_imputed": capacity_imputed,
                "pilenum": pilenum,
                "total_kw": total_kw,
                "in_zona_a": in_zona_a,
                "in_hzmb_island": in_hzmb,
                "station_district": district_en,
                "geometry": Point(x, y),
            }
        )

    cem_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=ANALYSIS_CRS)
    cem_gdf = cem_gdf.sort_values("cem_station_id").reset_index(drop=True)
    if len(cem_gdf) == 0:
        raise ValueError("CEM EV API returned zero stations")

    audit = {
        "cem_raw": int(len(cem_gdf)),
        "cem_total_capacity": float(cem_gdf["capacity"].sum()),
        "cem_total_pilenum": float(cem_gdf["pilenum"].sum()),
        "cem_total_kw_non_null": float(cem_gdf["total_kw"].dropna().sum()),
        "zona_a_station_ids": sorted(zona_a_ids),
        "hzmb_island_station_ids": sorted(hzmb_ids),
        "fallback_imputations": fallback_imputations,
    }
    return cem_gdf, audit


def _parse_numeric_capacity(raw_cap: Any) -> float | None:
    """Parse an OSM capacity tag into a positive float if cleanly numeric."""
    if raw_cap is None:
        return None
    s = str(raw_cap).strip()
    m = re.search(r"\d+(?:\.\d+)?", s)
    if not m:
        return None
    val = float(m.group(0))
    return val if val > 0 else None


def dedup_ev_with_osm(
    cem_gdf: gpd.GeoDataFrame,
    osm_ev_gdf: gpd.GeoDataFrame,
    cem_audit: dict[str, Any],
    dedup_radius_m: float = CROSS_SOURCE_DEDUP_M,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Cross-source deduplicate OSM amenity=charging_station against authoritative CEM stations.

    NEVER collapses within-CEM nearby carparks (e.g., 382819 & 387622 at 6.2 m).
    Logs every OSM EV element's distance to nearest CEM and exact keep/drop decision.
    """
    cem_coords = np.column_stack([cem_gdf.geometry.x.to_numpy(), cem_gdf.geometry.y.to_numpy()])
    cem_tree = cKDTree(cem_coords)

    kept_osm_rows: list[dict[str, Any]] = []
    osm_item_decisions: list[dict[str, Any]] = []
    counts = {
        "osm_raw": int(len(osm_ev_gdf)),
        "osm_dropped_25m": 0,
        "osm_dropped_cem_operator_dup": 0,
        "osm_dropped_fleet_or_stub": 0,
        "osm_dropped_proprietary_tesla": 0,
        "osm_retained_public": 0,
    }

    for _, row in osm_ev_gdf.sort_values(["osm_type", "osm_id"]).iterrows():
        pt = row.geometry
        dist_m, nearest_idx = cem_tree.query([pt.x, pt.y])
        dist_m = float(dist_m)
        nearest_cem_id = int(cem_gdf.iloc[int(nearest_idx)]["cem_station_id"])

        osm_ref = f"{row['osm_type']}/{row['osm_id']}"
        name = str(row.get("name") or "").strip()
        operator = str(row.get("operator") or "").strip()
        brand = str(row.get("brand") or "").strip()
        access = str(row.get("access") or "").strip()
        raw_cap = row.get("capacity")
        parsed_cap = _parse_numeric_capacity(raw_cap)
        combined_text = " ".join([name, operator, brand])

        if dist_m <= dedup_radius_m:
            decision = "drop_cross_source_25m"
            reason = f"Within {dedup_radius_m} m of CEM station {nearest_cem_id} ({dist_m:.1f} m)"
            counts["osm_dropped_25m"] += 1
        elif CEM_OPERATOR_RE.search(operator):
            decision = "drop_cem_operator_dup"
            reason = (
                f"CEM-operated station offset {dist_m:.1f} m from CEM {nearest_cem_id}; "
                "CEM API is authoritative"
            )
            counts["osm_dropped_cem_operator_dup"] += 1
        elif BUS_FLEET_RE.search(combined_text):
            decision = "drop_fleet_bus_depot"
            reason = "Electric bus fleet depot (non-public passenger EV charging)"
            counts["osm_dropped_fleet_or_stub"] += 1
        elif TESLA_BRAND_RE.search(combined_text):
            decision = "drop_proprietary_tesla"
            reason = "Proprietary single-brand Tesla Supercharger/Destination hotel/resort charger"
            counts["osm_dropped_proprietary_tesla"] += 1
        elif not name and not operator and parsed_cap is None:
            decision = "drop_unnamed_stub"
            reason = "Unnamed unverified OSM charging_station stub without operator or capacity"
            counts["osm_dropped_fleet_or_stub"] += 1
        else:
            decision = "retain_public_non_cem"
            cap_val = parsed_cap if parsed_cap is not None else 1.0
            cap_imp = parsed_cap is None
            reason = (
                f"Verified public non-CEM charging station ({dist_m:.1f} m from nearest CEM "
                f"{nearest_cem_id}, capacity={cap_val})"
            )
            counts["osm_retained_public"] += 1
            kept_osm_rows.append(
                {
                    "facility_id": f"osm_ev_{row['osm_id']}",
                    "facility_type": "ev_charger",
                    "name": name or f"OSM Public EV Station {row['osm_id']}",
                    "source": "osm_public_non_cem",
                    "capacity": float(cap_val),
                    "capacity_unit": "charger",
                    "capacity_imputed": bool(cap_imp),
                    "pilenum": float("nan"),
                    "total_kw": float("nan"),
                    "in_zona_a": False,
                    "geometry": pt,
                }
            )

        osm_item_decisions.append(
            {
                "osm_ref": osm_ref,
                "name": name,
                "operator": operator,
                "brand": brand,
                "access": access,
                "raw_capacity": None if raw_cap is None else str(raw_cap),
                "nearest_cem_station_id": nearest_cem_id,
                "nearest_cem_dist_m": round(dist_m, 2),
                "decision": decision,
                "reason": reason,
            }
        )

    canonical_cols = [
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
    base_cem = cem_gdf[canonical_cols].copy()
    if kept_osm_rows:
        osm_keep_gdf = gpd.GeoDataFrame(kept_osm_rows, geometry="geometry", crs=ANALYSIS_CRS)
        combined = gpd.GeoDataFrame(
            pd.concat([base_cem, osm_keep_gdf], ignore_index=True),
            geometry="geometry",
            crs=ANALYSIS_CRS,
        )
    else:
        combined = base_cem.reset_index(drop=True)

    non_imputed = int((~combined["capacity_imputed"]).sum())
    cap_completeness_pct = round(100.0 * non_imputed / len(combined), 2)

    ev_audit = {
        **cem_audit,
        **counts,
        "final_count": int(len(combined)),
        "final_total_capacity": float(combined["capacity"].sum()),
        "capacity_completeness_pct": cap_completeness_pct,
        "osm_item_decisions": osm_item_decisions,
    }
    return combined, ev_audit


def build_cem_provenance_entries(
    en_path: Path,
    tc_path: Path,
    fetched_at: str,
    ev_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build provenance records for CEM official EV API snapshots (lang=en & lang=tc)."""
    return [
        {
            "source_id": "cem_ev_en",
            "dataset_name": "CEM Public EV Charging Stations API (English/Portuguese)",
            "authority": "CEM (Companhia de Electricidade de Macau)",
            "dataset_id_or_layer": "Ajax/GetMapData?lang=en",
            "url": CEM_EV_EN_URL,
            "fetched_at_utc8": fetched_at,
            "local_raw_path": "data/raw/cem_ev_en.json",
            "sha256": sha256_file(en_path),
            "byte_size": en_path.stat().st_size,
            "native_crs": "EPSG:4326 (swapped stationlongitude=lat, stationlatitude=lon)",
            "target_crs": ANALYSIS_CRS,
            "row_count_raw": ev_audit["cem_raw"],
            "row_count_clean": ev_audit["final_count"],
            "capacity_completeness_pct": ev_audit["capacity_completeness_pct"],
            "fallback_used": len(ev_audit["fallback_imputations"]) > 0,
            "fallback_reason": (
                "Station 9488798 has empty pilerecords; imputed capacity=12.0 from pilenum"
                if ev_audit["fallback_imputations"]
                else None
            ),
            "anomalies_handled": [
                "Swapped stationlongitude -> lat and stationlatitude -> lon on all 84/84 rows",
                (
                    "Computed capacity=nunique(chargerid) & "
                    "total_kw=sum(max(poweroutput) per chargerid)"
                ),
                "Flagged exact 4 Zona A stations via stationdistrict=='Zona A' (plus 2 HZMB noted)",
                (
                    "No within-CEM dedup (preserves 382819 & 387622 6.2 m apart); "
                    "cross-source 25 m dedup vs OSM"
                ),
            ],
        },
        {
            "source_id": "cem_ev_tc",
            "dataset_name": "CEM Public EV Charging Stations API (Traditional Chinese)",
            "authority": "CEM (Companhia de Electricidade de Macau)",
            "dataset_id_or_layer": "Ajax/GetMapData?lang=tc",
            "url": CEM_EV_TC_URL,
            "fetched_at_utc8": fetched_at,
            "local_raw_path": "data/raw/cem_ev_tc.json",
            "sha256": sha256_file(tc_path),
            "byte_size": tc_path.stat().st_size,
            "native_crs": "EPSG:4326 (swapped stationlongitude=lat, stationlatitude=lon)",
            "target_crs": ANALYSIS_CRS,
            "row_count_raw": ev_audit["cem_raw"],
            "row_count_clean": ev_audit["cem_raw"],
            "capacity_completeness_pct": ev_audit["capacity_completeness_pct"],
            "fallback_used": False,
            "fallback_reason": None,
            "anomalies_handled": [
                "Used lang=tc instead of lang=zh (lang=zh returns blank station names/addresses)"
            ],
        },
    ]
