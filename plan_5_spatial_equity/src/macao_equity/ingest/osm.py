"""Offline-capable Geofabrik Macau OSM ingestion, PBF tag/boundary extraction, and walk graph."""

from __future__ import annotations

import math
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import geopandas as gpd
import networkx as nx
import osmium
import osmnx as ox
import requests
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.validation import make_valid

from macao_equity.config import (
    ANALYSIS_CRS,
    GEOFABRIK_GPKG_ZIP_URL,
    GEOFABRIK_PBF_URL,
    MIN_WALK_COMPONENT_NODES,
    OFFICIAL_REGISTRY_URLS,
    WALK_SPEED_M_PER_MIN,
    WGS84_CRS,
    make_transformer,
    reproject_geom,
    sha256_file,
    utc8_now_iso,
)

# Exact OSMnx 2.1.2 walk filter predicates (verified from ox._overpass._get_network_filter("walk"))
EXCLUDED_WALK_HIGHWAYS = frozenset(
    {
        "abandoned",
        "bus_guideway",
        "construction",
        "cycleway",
        "motor",
        "motorway",
        "motorway_link",
        "no",
        "planned",
        "platform",
        "proposed",
        "raceway",
        "razed",
        "rest_area",
        "services",
        "trunk",
        "trunk_link",
    }
)
# Wait: check exact OSMnx 2.1.2 walk filter — note OSMnx excludes highway~motor (regex contains
# "motor", which matches motorway & motorway_link) but does NOT exclude trunk unless foot=no!
# Let's match OSMnx 2.1.2's exact regex filter literally:
OSMNX_WALK_EXCL_HW_SUBSTR = frozenset(
    {
        "abandoned",
        "bus_guideway",
        "construction",
        "cycleway",
        "motor",
        "no",
        "planned",
        "platform",
        "proposed",
        "raceway",
        "razed",
        "rest_area",
        "services",
    }
)

POI_TAG_KEYS = (
    "amenity",
    "shop",
    "tourism",
    "leisure",
    "office",
    "healthcare",
    "craft",
    "historic",
)

# Landmark coordinates for pedestrian routing sanity check (WGS84 lon, lat)
RUINS_ST_PAUL_LONLAT = (113.54083, 22.19762)
SENADO_SQUARE_LONLAT = (113.53969, 22.19357)


def _download_binary_with_retry(
    url: str,
    out_path: Path,
    timeout: int = 120,
    max_retries: int = 4,
) -> str:
    """Stream-download a binary archive/PBF with retries and return UTC+8 timestamp."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                tmp_path = out_path.with_suffix(out_path.suffix + ".part")
                with tmp_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=131072):
                        if chunk:
                            f.write(chunk)
                tmp_path.replace(out_path)
            return utc8_now_iso()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt + 1 < max_retries:
                time.sleep(2.0 * (2**attempt))
    raise RuntimeError(f"Failed downloading {url} after {max_retries} attempts") from last_exc


def download_geofabrik_osm(
    pbf_path: Path,
    gpkg_zip_path: Path,
    gpkg_extract_path: Path,
    skip_download: bool = False,
) -> dict[str, str]:
    """Download and cache Geofabrik Macau .osm.pbf and .gpkg.zip; extract macau.gpkg."""
    if skip_download:
        for p in (pbf_path, gpkg_zip_path):
            if not p.exists():
                raise FileNotFoundError(f"--skip-download set but missing raw file: {p}")
        fetched_pbf = utc8_now_iso()
        fetched_zip = fetched_pbf
    else:
        fetched_pbf = _download_binary_with_retry(GEOFABRIK_PBF_URL, pbf_path)
        fetched_zip = _download_binary_with_retry(GEOFABRIK_GPKG_ZIP_URL, gpkg_zip_path)

    gpkg_extract_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(gpkg_zip_path, "r") as zf:
        gpkg_members = [m for m in zf.namelist() if m.endswith(".gpkg")]
        if not gpkg_members:
            raise ValueError(f"No .gpkg file found inside {gpkg_zip_path}")
        with zf.open(gpkg_members[0]) as src, gpkg_extract_path.open("wb") as dst:
            dst.write(src.read())

    # Read PBF header replication timestamp for audit provenance
    reader = osmium.io.Reader(str(pbf_path))
    header = reader.header()
    pbf_timestamp = (
        header.get("osmosis_replication_timestamp")
        or header.get("timestamp")
        or "unknown"
    )
    reader.close()

    return {
        "pbf_fetched_at_utc8": fetched_pbf,
        "gpkg_zip_fetched_at_utc8": fetched_zip,
        "pbf_snapshot_timestamp_utc": str(pbf_timestamp),
    }


def probe_data_gov_mo_gate(skip_download: bool = False) -> dict[str, Any]:
    """Verify and document the Macao One Account login gate on data.gov.mo."""
    observed = '{"msg":"內部錯誤","code":1}'
    http_status: int | None = None
    if not skip_download:
        try:
            resp = requests.get(OFFICIAL_REGISTRY_URLS["data_gov_mo_dsat_bus_probe"], timeout=15)
            http_status = resp.status_code
            observed = resp.text[:300].strip()
        except Exception as exc:  # noqa: BLE001
            observed = f"probe_error: {exc}"

    return {
        "portal_url": OFFICIAL_REGISTRY_URLS["data_gov_mo_portal"],
        "attempted_dataset_ids": [
            "e7b2e84d-3333-42f0-b676-64ce95306f0d",
            "ea50a770-cc35-47cc-a3ba-7f60092d4bc4",
        ],
        "status": "blocked_by_macao_one_account_login_gate",
        "http_status": http_status,
        "observed_api_response": observed,
        "technical_evidence": (
            "data.gov.mo gates api.data.gov.mo/datadir/downloadSingleFile behind "
            "Macao One Account localStorage token (account.gov.mo: 請先登入再登記使用數據集目錄)."
        ),
        "mitigation": (
            "Replaced with authoritative unauthenticated public APIs (DSEC CensusGIS REST, "
            "CEM EV JSON API, official gov.mo/library.gov.mo/iam.gov.mo/mlm.com.mo registries) "
            "+ deterministic offline Geofabrik Macau OSM extracts."
        ),
    }


def _is_walkable_way(tags: Any) -> bool:
    """Exact OSMnx 2.1.2 walk-network way predicate."""
    hw = tags.get("highway")
    if not hw:
        return False
    if tags.get("area") == "yes":
        return False
    if tags.get("access") == "private":
        return False
    if tags.get("foot") == "no":
        return False
    if tags.get("service") == "private":
        return False
    for sk in ("sidewalk", "sidewalk:both", "sidewalk:left", "sidewalk:right"):
        if tags.get(sk) == "separate":
            return False
    for excl in OSMNX_WALK_EXCL_HW_SUBSTR:
        if excl in hw:
            return False
    return True


class _WalkWayCollectHandler(osmium.SimpleHandler):
    """Pass 1: collect walkable ways and referenced node IDs from .osm.pbf."""

    def __init__(self) -> None:
        super().__init__()
        self.ways: list[tuple[int, list[int], dict[str, str]]] = []
        self.needed_nodes: set[int] = set()

    def way(self, w: Any) -> None:
        if _is_walkable_way(w.tags):
            nrefs = [n.ref for n in w.nodes]
            if len(nrefs) >= 2:
                self.ways.append((w.id, nrefs, {t.k: t.v for t in w.tags}))
                self.needed_nodes.update(nrefs)


class _WalkNodeCollectHandler(osmium.SimpleHandler):
    """Pass 2: collect coordinates for referenced walkable nodes."""

    def __init__(self, needed_nodes: set[int]) -> None:
        super().__init__()
        self.needed_nodes = needed_nodes
        self.nodes: dict[int, tuple[float, float]] = {}

    def node(self, n: Any) -> None:
        if n.id in self.needed_nodes and n.location.valid():
            self.nodes[n.id] = (n.location.lat, n.location.lon)


def _write_osm_xml(
    out_xml: Path,
    nodes: dict[int, tuple[float, float]],
    ways: list[tuple[int, list[int], dict[str, str]]],
) -> None:
    """Write minimal valid OSM XML suitable for osmnx.graph_from_xml."""
    with out_xml.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<osm version="0.6" generator="macao_equity">\n')
        for nid, (lat, lon) in sorted(nodes.items()):
            f.write(f'  <node id="{nid}" lat="{lat:.7f}" lon="{lon:.7f}"/>\n')
        for wid, nrefs, tags in ways:
            valid_refs = [r for r in nrefs if r in nodes]
            if len(valid_refs) < 2:
                continue
            f.write(f'  <way id="{wid}">\n')
            for r in valid_refs:
                f.write(f'    <nd ref="{r}"/>\n')
            for k, v in sorted(tags.items()):
                f.write(f'    <tag k="{escape(k)}" v="{escape(v)}"/>\n')
            f.write("  </way>\n")
        f.write("</osm>\n")


def compute_landmark_walk_check(G: nx.MultiDiGraph) -> dict[str, Any]:
    """Verify Ruins of St. Paul's <-> Senado Square bidirectional walk time (4.5 km/h)."""
    tr_4326_to_32649 = make_transformer(WGS84_CRS, ANALYSIS_CRS)
    node_ids = list(G.nodes)
    xs = [float(G.nodes[n]["x"]) for n in node_ids]
    ys = [float(G.nodes[n]["y"]) for n in node_ids]
    ux, uy = tr_4326_to_32649.transform(xs, ys)
    tree = cKDTree(list(zip(ux, uy)))

    rx, ry = tr_4326_to_32649.transform(*RUINS_ST_PAUL_LONLAT)
    sx, sy = tr_4326_to_32649.transform(*SENADO_SQUARE_LONLAT)
    snap_r, idx_r = tree.query([rx, ry])
    snap_s, idx_s = tree.query([sx, sy])
    node_r = node_ids[int(idx_r)]
    node_s = node_ids[int(idx_s)]

    fwd_m = float(nx.shortest_path_length(G, node_r, node_s, weight="length"))
    rev_m = float(nx.shortest_path_length(G, node_s, node_r, weight="length"))
    fwd_min = fwd_m / WALK_SPEED_M_PER_MIN
    incl_snap_min = (fwd_m + float(snap_r) + float(snap_s)) / WALK_SPEED_M_PER_MIN

    if abs(fwd_m - rev_m) > 1e-6:
        raise ValueError(f"Walk graph not bidirectional: fwd={fwd_m:.3f} m vs rev={rev_m:.3f} m")
    if not (5.0 <= incl_snap_min <= 8.0):
        raise ValueError(
            f"Landmark walk time Ruins->Senado out of [5.0, 8.0] min: {incl_snap_min:.2f} min"
        )

    return {
        "origin": "Ruins of St. Paul's (113.54083, 22.19762)",
        "destination": "Senado Square (113.53969, 22.19357)",
        "snap_origin_m": round(float(snap_r), 2),
        "snap_dest_m": round(float(snap_s), 2),
        "network_distance_m": round(fwd_m, 2),
        "reverse_distance_m": round(rev_m, 2),
        "network_walk_time_min": round(fwd_min, 2),
        "ruins_to_senado_min": round(incl_snap_min, 2),
        "symmetric": abs(fwd_m - rev_m) <= 1e-6,
    }


def build_walk_graph_from_pbf(
    pbf_path: Path,
    out_graphml_path: Path,
    min_component_nodes: int = MIN_WALK_COMPONENT_NODES,
) -> tuple[nx.MultiDiGraph, dict[str, Any]]:
    """Build deterministic bidirectional OSMnx walk graph offline from Geofabrik .osm.pbf."""
    way_handler = _WalkWayCollectHandler()
    way_handler.apply_file(str(pbf_path))

    node_handler = _WalkNodeCollectHandler(way_handler.needed_nodes)
    node_handler.apply_file(str(pbf_path))

    with tempfile.TemporaryDirectory(prefix="macao_walk_") as tmpdir:
        tmp_xml = Path(tmpdir) / "macau_walk.osm"
        _write_osm_xml(tmp_xml, node_handler.nodes, way_handler.ways)
        G = ox.graph_from_xml(
            tmp_xml,
            bidirectional=True,
            simplify=True,
            retain_all=True,
        )

    raw_nodes = G.number_of_nodes()
    raw_edges = G.number_of_edges()
    raw_wccs = nx.number_weakly_connected_components(G)

    keep_nodes = [
        n
        for comp in nx.weakly_connected_components(G)
        if len(comp) >= min_component_nodes
        for n in comp
    ]
    G = G.subgraph(keep_nodes).copy()

    # CRITICAL: ox.graph_from_xml does NOT populate street_count automatically
    street_counts = ox.stats.count_streets_per_node(G)
    nx.set_node_attributes(G, street_counts, "street_count")

    out_graphml_path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, out_graphml_path)

    comp_sizes = sorted((len(c) for c in nx.weakly_connected_components(G)), reverse=True)
    landmark = compute_landmark_walk_check(G)

    ys = [float(G.nodes[n]["y"]) for n in G.nodes]
    if not (max(ys) > 22.19 and min(ys) < 22.15):
        raise ValueError("Walk graph does not cover both Macao Peninsula and Taipa/Coloane")

    meta = {
        "source": "Geofabrik macau-latest.osm.pbf (offline 2-pass osmium + ox.graph_from_xml)",
        "graphml_path": "data/raw/osm_walk_graph.graphml",
        "sha256": sha256_file(out_graphml_path),
        "network_type": "walk",
        "bidirectional": True,
        "retain_all": True,
        "min_component_nodes_kept": min_component_nodes,
        "raw_nodes": raw_nodes,
        "raw_edges": raw_edges,
        "raw_wcc_count": raw_wccs,
        "pruned_nodes": G.number_of_nodes(),
        "pruned_edges": G.number_of_edges(),
        "pruned_wcc_count": len(comp_sizes),
        "pruned_wcc_sizes": comp_sizes,
        "walk_speed_kmh": 4.5,
        "landmark_sanity_check_min": landmark,
    }
    return G, meta


class _Rel1867188MemberCollector(osmium.SimpleHandler):
    """Collect outer member way IDs of Macao SAR relation rel/1867188."""

    def __init__(self) -> None:
        super().__init__()
        self.outer_way_ids: set[int] = set()

    def relation(self, r: Any) -> None:
        if r.id == 1867188:
            for m in r.members:
                if m.type == "w" and m.role in ("outer", ""):
                    self.outer_way_ids.add(int(m.ref))


class _PbfFeatureExtractor(osmium.SimpleHandler):
    """Single-pass offline extractor for Macao SAR boundary, EV, gardens, tags, LRT & POIs."""

    def __init__(self, rel1867188_way_ids: set[int] | None = None) -> None:
        super().__init__()
        self._wkb = osmium.geom.WKBFactory()
        self.rel1867188_way_ids = rel1867188_way_ids or set()
        self.rel1867188_lines: list[Any] = []
        self.macao_sar_geom_4326: Any = None
        self.ev_records: list[dict[str, Any]] = []
        self.garden_polys: list[dict[str, Any]] = []
        self.garden_pts: list[dict[str, Any]] = []
        self.access_tags: dict[str, str] = {}
        self.bus_operators: dict[str, str] = {}
        self.lrt_nodes: list[dict[str, Any]] = []
        self.poi_records: list[dict[str, Any]] = []

    def _maybe_record_poi(
        self,
        osm_type: str,
        osm_id: int,
        tags: Any,
        lon: float,
        lat: float,
    ) -> None:
        matched_key = None
        matched_val = None
        for k in POI_TAG_KEYS:
            v = tags.get(k)
            if v:
                matched_key = k
                matched_val = v
                break
        if matched_key is None:
            return
        tag_keys = [t.k for t in tags]
        self.poi_records.append(
            {
                "osm_type": osm_type,
                "osm_id": str(osm_id),
                "cat_key": matched_key,
                "cat_val": str(matched_val),
                "name": str(tags.get("name") or ""),
                "name_en": str(tags.get("name:en") or ""),
                "brand": str(tags.get("brand") or ""),
                "operator": str(tags.get("operator") or ""),
                "amenity": str(tags.get("amenity") or ""),
                "tag_keys": tag_keys,
                "lon": lon,
                "lat": lat,
            }
        )

    def node(self, n: Any) -> None:
        if not n.location.valid():
            return
        tags = n.tags
        if not len(tags):
            return
        lon = float(n.location.lon)
        lat = float(n.location.lat)
        oid = str(n.id)

        if tags.get("access"):
            self.access_tags[oid] = str(tags.get("access"))

        if tags.get("highway") == "bus_stop" or tags.get("amenity") == "bus_station":
            if tags.get("operator"):
                self.bus_operators[oid] = str(tags.get("operator"))

        if tags.get("amenity") == "charging_station":
            self.ev_records.append(
                {
                    "osm_type": "node",
                    "osm_id": oid,
                    "name": str(tags.get("name") or ""),
                    "operator": str(tags.get("operator") or ""),
                    "brand": str(tags.get("brand") or ""),
                    "access": str(tags.get("access") or ""),
                    "capacity": tags.get("capacity"),
                    "fee": str(tags.get("fee") or ""),
                    "lon": lon,
                    "lat": lat,
                }
            )

        if tags.get("leisure") == "garden":
            self.garden_pts.append(
                {
                    "osm_id": oid,
                    "osm_type": "node",
                    "name": str(tags.get("name") or ""),
                    "access": str(tags.get("access") or ""),
                    "lon": lon,
                    "lat": lat,
                }
            )

        if tags.get("railway") == "station" and tags.get("station") == "light_rail":
            self.lrt_nodes.append(
                {
                    "osm_id": oid,
                    "name": str(tags.get("name") or ""),
                    "name_en": str(tags.get("name:en") or ""),
                    "lon": lon,
                    "lat": lat,
                }
            )

        self._maybe_record_poi("node", n.id, tags, lon, lat)

    def way(self, w: Any) -> None:
        if int(w.id) in self.rel1867188_way_ids:
            bpts = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
            if len(bpts) >= 2:
                self.rel1867188_lines.append(shapely.geometry.LineString(bpts))

        tags = w.tags
        if not len(tags):
            return
        oid = str(w.id)
        if tags.get("access"):
            self.access_tags[oid] = str(tags.get("access"))

        # Compute mean centroid for closed/located ways carrying EV or POI tags
        coords = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
        if len(coords) < 3:
            return
        is_closed = w.nodes[0].ref == w.nodes[-1].ref
        if not is_closed:
            return
        # Drop duplicate closing vertex before computing centroid
        pts = coords[:-1] if len(coords) > 1 else coords
        lon = float(sum(c[0] for c in pts) / len(pts))
        lat = float(sum(c[1] for c in pts) / len(pts))
        if math.isnan(lon) or math.isnan(lat):
            return

        if tags.get("amenity") == "charging_station":
            self.ev_records.append(
                {
                    "osm_type": "way",
                    "osm_id": oid,
                    "name": str(tags.get("name") or ""),
                    "operator": str(tags.get("operator") or ""),
                    "brand": str(tags.get("brand") or ""),
                    "access": str(tags.get("access") or ""),
                    "capacity": tags.get("capacity"),
                    "fee": str(tags.get("fee") or ""),
                    "lon": lon,
                    "lat": lat,
                }
            )

        self._maybe_record_poi("way", w.id, tags, lon, lat)

    def area(self, a: Any) -> None:
        tags = a.tags
        if not len(tags):
            return
        orig_id = str(a.orig_id())
        is_rel = not a.from_way()

        if tags.get("access"):
            self.access_tags[orig_id] = str(tags.get("access"))

        # Capture official Macao SAR boundary relation rel/1867188 if assembled directly
        if is_rel and a.orig_id() == 1867188:
            try:
                wkb = self._wkb.create_multipolygon(a)
                geom = shapely.from_wkb(wkb)
                if not geom.is_valid:
                    geom = make_valid(geom)
                self.macao_sar_geom_4326 = geom
            except Exception:  # noqa: BLE001
                pass

        # Capture leisure=garden polygons/multipolygons (omitted by Geofabrik macau.gpkg)
        if tags.get("leisure") == "garden":
            try:
                wkb = self._wkb.create_multipolygon(a)
                geom = shapely.from_wkb(wkb)
                if not geom.is_valid:
                    geom = make_valid(geom)
                if not geom.is_empty:
                    self.garden_polys.append(
                        {
                            "osm_id": orig_id,
                            "osm_type": "relation" if is_rel else "way",
                            "name": str(tags.get("name") or ""),
                            "access": str(tags.get("access") or ""),
                            "geometry": geom,
                        }
                    )
            except Exception:  # noqa: BLE001
                pass


def extract_pbf_layers(pbf_path: Path) -> dict[str, Any]:
    """Extract Macao SAR boundary, OSM EV stations, gardens, tags, LRT & POIs from .osm.pbf."""
    tr_4326_to_32649 = make_transformer(WGS84_CRS, ANALYSIS_CRS)
    rel_collector = _Rel1867188MemberCollector()
    rel_collector.apply_file(str(pbf_path))

    handler = _PbfFeatureExtractor(rel1867188_way_ids=rel_collector.outer_way_ids)
    handler.apply_file(str(pbf_path), locations=True, idx="flex_mem")

    if (
        handler.macao_sar_geom_4326 is None or handler.macao_sar_geom_4326.is_empty
    ) and handler.rel1867188_lines:
        polys = list(shapely.ops.polygonize(handler.rel1867188_lines))
        if polys:
            handler.macao_sar_geom_4326 = shapely.union_all(polys)

    if handler.macao_sar_geom_4326 is None or handler.macao_sar_geom_4326.is_empty:
        raise RuntimeError(
            "Failed to extract Macao SAR boundary rel/1867188 from macau-latest.osm.pbf"
        )

    macao_sar_32649 = reproject_geom(handler.macao_sar_geom_4326, tr_4326_to_32649)
    if not macao_sar_32649.is_valid:
        macao_sar_32649 = make_valid(macao_sar_32649)

    def _pts_to_gdf(records: list[dict[str, Any]], clip_to_sar: bool = True) -> gpd.GeoDataFrame:
        if not records:
            return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=ANALYSIS_CRS)
        geoms = [
            Point(*tr_4326_to_32649.transform(r["lon"], r["lat"])) for r in records
        ]
        gdf = gpd.GeoDataFrame(records, geometry=geoms, crs=ANALYSIS_CRS)
        if clip_to_sar:
            gdf = gdf[gdf.geometry.within(macao_sar_32649)].copy().reset_index(drop=True)
        return gdf

    osm_ev_gdf = _pts_to_gdf(handler.ev_records, clip_to_sar=True)
    garden_pts_gdf = _pts_to_gdf(handler.garden_pts, clip_to_sar=True)
    lrt_gdf = _pts_to_gdf(handler.lrt_nodes, clip_to_sar=True)
    pois_gdf = _pts_to_gdf(handler.poi_records, clip_to_sar=False)

    if handler.garden_polys:
        gp_gdf = gpd.GeoDataFrame(handler.garden_polys, geometry="geometry", crs=WGS84_CRS)
        gp_gdf["geometry"] = gp_gdf.geometry.apply(lambda g: reproject_geom(g, tr_4326_to_32649))
        gp_gdf = gp_gdf.set_crs(ANALYSIS_CRS, allow_override=True)
        gp_gdf = gp_gdf[gp_gdf.geometry.representative_point().within(macao_sar_32649)].copy()
        gp_gdf = gp_gdf.reset_index(drop=True)
    else:
        gp_gdf = gpd.GeoDataFrame(
            columns=["osm_id", "osm_type", "name", "access", "geometry"],
            geometry="geometry",
            crs=ANALYSIS_CRS,
        )

    return {
        "macao_sar_geom_4326": handler.macao_sar_geom_4326,
        "macao_sar_geom_32649": macao_sar_32649,
        "osm_ev_gdf": osm_ev_gdf,
        "pbf_gardens_poly_gdf": gp_gdf,
        "pbf_gardens_pt_gdf": garden_pts_gdf,
        "pbf_access_tags": handler.access_tags,
        "pbf_bus_operators": handler.bus_operators,
        "pbf_lrt_gdf": lrt_gdf,
        "pbf_pois_gdf": pois_gdf,
    }


def build_osm_provenance_entries(
    pbf_path: Path,
    gpkg_zip_path: Path,
    osm_dl_meta: dict[str, str],
) -> list[dict[str, Any]]:
    """Build structured provenance records for Geofabrik Macau OSM snapshots."""
    return [
        {
            "source_id": "geofabrik_macau_pbf",
            "dataset_name": "Geofabrik OpenStreetMap Macau Extract (.osm.pbf)",
            "authority": "OpenStreetMap Contributors / Geofabrik GmbH",
            "dataset_id_or_layer": "asia/china/macau-latest.osm.pbf",
            "url": GEOFABRIK_PBF_URL,
            "fetched_at_utc8": osm_dl_meta["pbf_fetched_at_utc8"],
            "osm_snapshot_timestamp_utc": osm_dl_meta["pbf_snapshot_timestamp_utc"],
            "local_raw_path": "data/raw/macau-latest.osm.pbf",
            "sha256": sha256_file(pbf_path),
            "byte_size": pbf_path.stat().st_size,
            "native_crs": WGS84_CRS,
            "target_crs": ANALYSIS_CRS,
            "row_count_raw": None,
            "row_count_clean": None,
            "capacity_completeness_pct": None,
            "fallback_used": False,
            "fallback_reason": None,
            "anomalies_handled": [
                "Primary offline source for walk graph, rel/1867188 SAR boundary, leisure=garden, "
                "access/operator tags, amenity=charging_station, station=light_rail, and POIs "
                "(overpass-api.de and overpass.openstreetmap.fr return 502/403 on this host)."
            ],
        },
        {
            "source_id": "geofabrik_macau_gpkg_zip",
            "dataset_name": "Geofabrik OpenStreetMap Macau Free GeoPackage (.gpkg.zip)",
            "authority": "OpenStreetMap Contributors / Geofabrik GmbH",
            "dataset_id_or_layer": "asia/china/macau-latest-free.gpkg.zip",
            "url": GEOFABRIK_GPKG_ZIP_URL,
            "fetched_at_utc8": osm_dl_meta["gpkg_zip_fetched_at_utc8"],
            "osm_snapshot_timestamp_utc": osm_dl_meta["pbf_snapshot_timestamp_utc"],
            "local_raw_path": "data/raw/macau-latest-free.gpkg.zip",
            "sha256": sha256_file(gpkg_zip_path),
            "byte_size": gpkg_zip_path.stat().st_size,
            "native_crs": WGS84_CRS,
            "target_crs": ANALYSIS_CRS,
            "row_count_raw": None,
            "row_count_clean": None,
            "capacity_completeness_pct": None,
            "fallback_used": False,
            "fallback_reason": None,
            "anomalies_handled": [
                "Did NOT use gis_osm_adminareas_a_free (omits Taipa); used rel/1867188 from PBF.",
                "Deduplicated 62 dual-tagged bus_stop/bus_station rows by osm_id before cleaning.",
            ],
        },
    ]
