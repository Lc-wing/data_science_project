"""Pure-Python 3.11 stdlib + Poppler artifact readers and OGC/PNG/PDF writers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import shutil
import sqlite3
import struct
import subprocess
import tomllib
import unicodedata
import xml.etree.ElementTree as ET
import zlib
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from e2e_lib.contracts import (
    ISLANDS_MAX_LAT,
    ISLANDS_MAX_N,
    PENINSULA_MIN_LAT,
    PENINSULA_MIN_N,
)

# Full OGC WKT definitions for pure-Python GPKG writer
WKT_EPSG_32649 = (
    'PROJCS["WGS 84 / UTM zone 49N",'
    'GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
    'AUTHORITY["EPSG","6326"]],'
    'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
    'AUTHORITY["EPSG","4326"]],'
    'PROJECTION["Transverse_Mercator"],'
    'PARAMETER["latitude_of_origin",0],'
    'PARAMETER["central_meridian",111],'
    'PARAMETER["scale_factor",0.9996],'
    'PARAMETER["false_easting",500000],'
    'PARAMETER["false_northing",0],'
    'UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
    'AXIS["Easting",EAST],AXIS["Northing",NORTH],'
    'AUTHORITY["EPSG","32649"]]'
)
WKT_EPSG_4326 = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
    'AUTHORITY["EPSG","6326"]],'
    'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
    'AUTHORITY["EPSG","4326"]]'
)

_TRAILING_WKT_EPSG_RE = re.compile(r'(?:AUTHORITY\["EPSG","(\d+)"\]|ID\["EPSG",(\d+)\])\s*\]\s*$')
_GPB_ENV_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
_WKB_TYPE_NAMES = {
    1: "Point",
    2: "LineString",
    3: "Polygon",
    4: "MultiPoint",
    5: "MultiLineString",
    6: "MultiPolygon",
    7: "GeometryCollection",
}

_BACKMATTER_HEADING_RE = re.compile(
    r"(?i)^(?:\d+\.?\s*)?(?:"
    r"references|bibliography|"
    r"(?:separate\s+)?(?:team\s+|author\s+|member\s+)?contribution\s+statement|"
    r"team\s+contributions?|author\s+contributions?|"
    r"(?:generative\s+)?ai\s+(?:usage|use)\s+declaration|"
    r"declaration\s+of\s+(?:generative\s+)?ai"
    r")\b.*$"
)
_MAIN_START_RE = re.compile(
    r"(?i)^(?:1\.?\s+)?(?:introduction|problem\s+definition|background\s+and\s+motivation)\b"
)


# ---------------------------------------------------------------------------
# Basic strict readers & helpers
# ---------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    """Compute lowercase hexadecimal SHA-256 digest of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _reject_nonfinite_json(token: str) -> Any:
    raise ValueError(f"Non-standard JSON constant not permitted: {token!r}")


def read_json_strict(path: Path) -> Any:
    """Read UTF-8 JSON strictly rejecting bare NaN / Infinity tokens."""
    text = path.read_text(encoding="utf-8")
    return json.loads(text, parse_constant=_reject_nonfinite_json)


def read_toml_strict(path: Path) -> dict[str, Any]:
    """Read TOML using Python 3.11 stdlib tomllib."""
    return tomllib.loads(path.read_text(encoding="utf-8"))


def read_csv_strict(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read UTF-8 CSV asserting unique non-empty headers and consistent column counts."""
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV or missing header row: {path}")
        headers = [h.strip() for h in reader.fieldnames]
        if any(not h for h in headers):
            raise ValueError(f"Blank header column in CSV: {path}")
        if len(set(headers)) != len(headers):
            raise ValueError(f"Duplicate header columns in CSV {path}: {headers}")
        rows: list[dict[str, str]] = []
        for row_idx, raw_row in enumerate(reader, start=2):
            if None in raw_row or any(v is None for v in raw_row.values()):
                raise ValueError(f"Column count mismatch at row {row_idx} in CSV {path}")
            rows.append({k.strip(): v for k, v in raw_row.items()})
    return headers, rows


def as_bool(val: Any) -> bool:
    """Normalize Python/SQLite/CSV boolean representations cleanly."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        if val in (0, 1):
            return bool(val)
        raise ValueError(f"Numeric value is not boolean 0/1: {val!r}")
    s = str(val).strip().lower()
    if s in {"true", "1", "t", "yes"}:
        return True
    if s in {"false", "0", "f", "no"}:
        return False
    raise ValueError(f"Cannot coerce to bool: {val!r}")


def as_float(val: Any) -> float:
    """Parse value to finite float, raising ValueError on NaN/Inf/empty."""
    if val is None or (isinstance(val, str) and not val.strip()):
        raise ValueError("Missing numeric value")
    f = float(val)
    if not math.isfinite(f):
        raise ValueError(f"Non-finite float value: {val!r}")
    return f


def normalize_pdf_text(raw: str) -> str:
    """Apply NFKC, de-hyphenation, Unicode minus normalization, and whitespace collapse."""
    s = unicodedata.normalize("NFKC", raw)
    s = re.sub(r"-\s*\n\s*", "", s)
    s = s.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def wgs84_to_utm49n(lon_deg: float, lat_deg: float) -> tuple[float, float]:
    """Pure-Python WGS84 geographic (lon, lat) to UTM Zone 49N (EPSG:32649) meters."""
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    ep2 = e2 / (1.0 - e2)
    k0 = 0.9996
    lon0 = math.radians(111.0)
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    tan_lat = math.tan(lat)

    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    t = tan_lat * tan_lat
    c = ep2 * cos_lat * cos_lat
    a_term = cos_lat * (lon - lon0)

    e4 = e2 * e2
    e6 = e4 * e2
    m = a * (
        (1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * lat
        - (3.0 * e2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * math.sin(2.0 * lat)
        + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * math.sin(4.0 * lat)
        - (35.0 * e6 / 3072.0) * math.sin(6.0 * lat)
    )
    easting = 500000.0 + k0 * n * (
        a_term
        + (1.0 - t + c) * (a_term**3) / 6.0
        + (5.0 - 18.0 * t + t * t + 72.0 * c - 58.0 * ep2) * (a_term**5) / 120.0
    )
    northing = k0 * (
        m
        + n
        * tan_lat
        * (
            (a_term**2) / 2.0
            + (5.0 - t + 9.0 * c + 4.0 * c * c) * (a_term**4) / 24.0
            + (61.0 - 58.0 * t + t * t + 600.0 * c - 330.0 * ep2) * (a_term**6) / 720.0
        )
    )
    return easting, northing


# ---------------------------------------------------------------------------
# Pure-Python GeoPackage (.gpkg) Reader & OGC 1.3 Writer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpkgGeom:
    """Parsed GeoPackage geometry summary."""

    geom_type: str
    srs_id: int
    is_empty: bool
    is_valid: bool
    bbox: tuple[float, float, float, float] | None
    centroid: tuple[float, float] | None
    area: float
    point_xy: tuple[float, float] | None
    polygons: tuple[
        tuple[
            tuple[float, float, float, float],
            tuple[tuple[float, float], ...],
            tuple[tuple[tuple[float, float], ...], ...],
        ],
        ...,
    ]


@dataclass
class GpkgLayer:
    """Parsed GeoPackage feature layer (read strictly via SQLite ?mode=ro)."""

    path: Path
    table_name: str
    geom_column: str
    srs_id: int
    epsg: int | None
    geometry_type_name: str
    columns: list[str]
    rows: list[dict[str, Any]]
    geoms: list[GpkgGeom]
    layer_names: list[str]


def _ring_signed_area_and_centroid(
    ring: Sequence[tuple[float, float]],
) -> tuple[float, float, float]:
    """Return (signed_area, cx, cy) using exact shoelace polygon formulas."""
    n = len(ring)
    if n < 4:
        return 0.0, 0.0, 0.0
    twice_area = 0.0
    cx_acc = 0.0
    cy_acc = 0.0
    for i in range(n - 1):
        x0, y0 = ring[i]
        x1, y1 = ring[i + 1]
        cross = x0 * y1 - x1 * y0
        twice_area += cross
        cx_acc += (x0 + x1) * cross
        cy_acc += (y0 + y1) * cross
    signed_area = 0.5 * twice_area
    if abs(signed_area) <= 1e-18:
        xs = [p[0] for p in ring[:-1]]
        ys = [p[1] for p in ring[:-1]]
        return 0.0, sum(xs) / len(xs), sum(ys) / len(ys)
    return signed_area, cx_acc / (6.0 * signed_area), cy_acc / (6.0 * signed_area)


def _points_bbox(pts: Sequence[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _merge_bboxes(
    bboxes: Sequence[tuple[float, float, float, float] | None],
) -> tuple[float, float, float, float] | None:
    valid = [b for b in bboxes if b is not None]
    if not valid:
        return None
    return (
        min(b[0] for b in valid),
        min(b[1] for b in valid),
        max(b[2] for b in valid),
        max(b[3] for b in valid),
    )


def _parse_wkb_at(blob: bytes, offset: int) -> tuple[dict[str, Any], int]:
    """Parse ISO WKB or EWKB geometry starting at offset."""
    if offset + 5 > len(blob):
        raise ValueError("Truncated WKB geometry header")
    bo_flag = blob[offset]
    endian = "<" if bo_flag == 1 else ">"
    raw_type = struct.unpack_from(f"{endian}I", blob, offset + 1)[0]
    pos = offset + 5

    has_z = bool(raw_type & 0x80000000)
    has_m = bool(raw_type & 0x40000000)
    has_srid = bool(raw_type & 0x20000000)
    base_raw = raw_type & 0x1FFFFFFF
    if has_srid:
        pos += 4

    iso_dim = base_raw // 1000
    base_type = base_raw % 1000
    if iso_dim == 1:
        has_z = True
    elif iso_dim == 2:
        has_m = True
    elif iso_dim == 3:
        has_z = True
        has_m = True
    dims = 2 + int(has_z) + int(has_m)
    coord_fmt = f"{endian}{'d' * dims}"
    coord_size = 8 * dims

    def _read_coords(count: int) -> tuple[list[tuple[float, float]], bool]:
        nonlocal pos
        if pos + count * coord_size > len(blob):
            raise ValueError("Truncated WKB coordinate sequence")
        pts: list[tuple[float, float]] = []
        all_finite = True
        for _ in range(count):
            vals = struct.unpack_from(coord_fmt, blob, pos)
            pos += coord_size
            if not all(math.isfinite(v) for v in vals):
                all_finite = False
            pts.append((vals[0], vals[1]))
        return pts, all_finite

    type_name = _WKB_TYPE_NAMES.get(base_type, f"Unknown({base_type})")

    if base_type == 1:  # Point
        pts, finite = _read_coords(1)
        x, y = pts[0]
        is_empty = math.isnan(x) or math.isnan(y)
        return {
            "geom_type": type_name,
            "is_empty": is_empty,
            "is_valid": finite and not is_empty,
            "bbox": None if is_empty else (x, y, x, y),
            "centroid": None if is_empty else (x, y),
            "area": 0.0,
            "point_xy": None if is_empty else (x, y),
            "polygons": [],
        }, pos

    if base_type == 2:  # LineString
        n_pts = struct.unpack_from(f"{endian}I", blob, pos)[0]
        pos += 4
        pts, finite = _read_coords(n_pts)
        is_empty = n_pts == 0
        centroid = (
            (sum(p[0] for p in pts) / n_pts, sum(p[1] for p in pts) / n_pts)
            if not is_empty
            else None
        )
        return {
            "geom_type": type_name,
            "is_empty": is_empty,
            "is_valid": finite and n_pts >= 2,
            "bbox": None if is_empty else _points_bbox(pts),
            "centroid": centroid,
            "area": 0.0,
            "point_xy": None,
            "polygons": [],
        }, pos

    if base_type == 3:  # Polygon
        n_rings = struct.unpack_from(f"{endian}I", blob, pos)[0]
        pos += 4
        if n_rings == 0:
            return {
                "geom_type": type_name,
                "is_empty": True,
                "is_valid": False,
                "bbox": None,
                "centroid": None,
                "area": 0.0,
                "point_xy": None,
                "polygons": [],
            }, pos
        rings: list[tuple[tuple[float, float], ...]] = []
        valid = True
        for _ in range(n_rings):
            n_pts = struct.unpack_from(f"{endian}I", blob, pos)[0]
            pos += 4
            pts, finite = _read_coords(n_pts)
            if not finite or n_pts < 4:
                valid = False
            elif math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) > 1e-6:
                valid = False
            rings.append(tuple(pts))
        outer = rings[0]
        holes = tuple(rings[1:])
        outer_sa, ocx, ocy = _ring_signed_area_and_centroid(outer)
        outer_area = abs(outer_sa)
        net_area = outer_area
        cx_num = outer_area * ocx
        cy_num = outer_area * ocy
        for h in holes:
            h_sa, hcx, hcy = _ring_signed_area_and_centroid(h)
            h_area = abs(h_sa)
            net_area -= h_area
            cx_num -= h_area * hcx
            cy_num -= h_area * hcy
        if outer_area <= 0.0 or net_area <= 0.0:
            valid = False
        bbox = _points_bbox(outer)
        centroid = (cx_num / net_area, cy_num / net_area) if net_area > 0.0 else (ocx, ocy)
        return {
            "geom_type": type_name,
            "is_empty": False,
            "is_valid": valid,
            "bbox": bbox,
            "centroid": centroid,
            "area": max(net_area, 0.0),
            "point_xy": None,
            "polygons": [(bbox, outer, holes)],
        }, pos

    if base_type in (4, 5, 6, 7):  # Multi* or GeometryCollection
        n_geoms = struct.unpack_from(f"{endian}I", blob, pos)[0]
        pos += 4
        parts: list[dict[str, Any]] = []
        for _ in range(n_geoms):
            part, pos = _parse_wkb_at(blob, pos)
            parts.append(part)
        if not parts or all(p["is_empty"] for p in parts):
            return {
                "geom_type": type_name,
                "is_empty": True,
                "is_valid": False,
                "bbox": None,
                "centroid": None,
                "area": 0.0,
                "point_xy": None,
                "polygons": [],
            }, pos
        total_area = sum(p["area"] for p in parts)
        if total_area > 0.0:
            cx = sum(p["centroid"][0] * p["area"] for p in parts if p["centroid"]) / total_area
            cy = sum(p["centroid"][1] * p["area"] for p in parts if p["centroid"]) / total_area
            centroid = (cx, cy)
        else:
            c_pts = [p["centroid"] for p in parts if p["centroid"] is not None]
            centroid = (
                (sum(c[0] for c in c_pts) / len(c_pts), sum(c[1] for c in c_pts) / len(c_pts))
                if c_pts
                else None
            )
        all_polys = [poly for p in parts for poly in p["polygons"]]
        point_xy = parts[0]["point_xy"] if base_type == 4 and len(parts) == 1 else None
        return {
            "geom_type": type_name,
            "is_empty": False,
            "is_valid": all(p["is_valid"] for p in parts),
            "bbox": _merge_bboxes([p["bbox"] for p in parts]),
            "centroid": centroid,
            "area": total_area,
            "point_xy": point_xy,
            "polygons": all_polys,
        }, pos

    raise ValueError(f"Unsupported WKB geometry type: {base_type}")


def _optional_shapely_valid(wkb_bytes: bytes) -> bool | None:
    """Optionally cross-check GEOS topology validity when shapely is importable."""
    try:
        import shapely  # type: ignore[import-not-found]

        geom = shapely.from_wkb(wkb_bytes)
        return bool(shapely.is_valid(geom)) and not bool(shapely.is_empty(geom))
    except Exception:
        return None


def _parse_gpb(blob: bytes | None) -> GpkgGeom:
    """Parse GeoPackage Binary (GPB) blob containing header + ISO/EWKB payload."""
    if blob is None or len(blob) < 8:
        return GpkgGeom("Unknown", -1, True, False, None, None, 0.0, None, ())
    if blob[0:2] != b"GP":
        raise ValueError(f"Invalid GeoPackage Binary magic: {blob[0:2]!r}")
    version = blob[2]
    if version != 0:
        raise ValueError(f"Unsupported GPB version: {version}")
    flags = blob[3]
    header_endian = "<" if (flags & 0x01) else ">"
    env_code = (flags >> 1) & 0x07
    if env_code not in _GPB_ENV_BYTES:
        raise ValueError(f"Invalid GPB envelope indicator: {env_code}")
    is_empty_flag = bool((flags >> 4) & 0x01)
    srs_id = struct.unpack_from(f"{header_endian}i", blob, 4)[0]
    wkb_offset = 8 + _GPB_ENV_BYTES[env_code]
    if is_empty_flag or wkb_offset >= len(blob):
        return GpkgGeom("Empty", srs_id, True, False, None, None, 0.0, None, ())
    parsed, _ = _parse_wkb_at(blob, wkb_offset)
    is_valid = bool(parsed["is_valid"])
    shapely_check = _optional_shapely_valid(blob[wkb_offset:])
    if shapely_check is not None:
        is_valid = is_valid and shapely_check
    return GpkgGeom(
        geom_type=parsed["geom_type"],
        srs_id=srs_id,
        is_empty=bool(parsed["is_empty"]),
        is_valid=is_valid,
        bbox=parsed["bbox"],
        centroid=parsed["centroid"],
        area=float(parsed["area"]),
        point_xy=parsed["point_xy"],
        polygons=tuple(parsed["polygons"]),
    )


def _resolve_epsg(
    organization: str | None,
    org_coordsys_id: int | None,
    definition: str | None,
    srs_id: int,
) -> int | None:
    if organization and organization.strip().upper() == "EPSG" and org_coordsys_id is not None:
        return int(org_coordsys_id)
    if definition:
        match = _TRAILING_WKT_EPSG_RE.search(definition.strip())
        if match:
            code = match.group(1) or match.group(2)
            if code:
                return int(code)
    if srs_id > 0:
        return int(srs_id)
    return None


def read_gpkg(
    path: Path,
    layer: str | None = None,
    allow_multi_layer: bool = False,
) -> GpkgLayer:
    """Read a GeoPackage strictly via SQLite read-only URI (?mode=ro)."""
    if not path.is_file():
        raise FileNotFoundError(f"GeoPackage not found: {path}")
    uri = f"file:{quote(str(path.resolve()))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cur = conn.cursor()
        meta_rows = cur.execute(
            """
            SELECT c.table_name, c.srs_id, g.column_name, g.geometry_type_name,
                   g.srs_id AS geom_srs_id, s.organization,
                   s.organization_coordsys_id, s.definition
            FROM gpkg_contents c
            JOIN gpkg_geometry_columns g ON c.table_name = g.table_name
            LEFT JOIN gpkg_spatial_ref_sys s ON c.srs_id = s.srs_id
            WHERE c.data_type = 'features'
            ORDER BY c.table_name
            """
        ).fetchall()
        if not meta_rows:
            raise ValueError(f"No feature layers found in GeoPackage: {path}")

        available_layers = [str(r[0]) for r in meta_rows]
        if layer is not None:
            selected = [r for r in meta_rows if r[0] == layer]
            if not selected:
                raise ValueError(
                    f"Requested layer {layer!r} not in {path}; available: {available_layers}"
                )
        elif len(meta_rows) == 1:
            selected = meta_rows
        elif allow_multi_layer:
            selected = meta_rows
        else:
            raise ValueError(
                f"Ambiguous multi-layer GeoPackage {path} with layers {available_layers}"
            )

        all_rows: list[dict[str, Any]] = []
        all_geoms: list[GpkgGeom] = []
        merged_cols: list[str] = []
        epsg_values: set[int | None] = set()
        first_meta = selected[0]

        for (
            table_name,
            contents_srs_id,
            geom_col,
            geom_type_name,
            geom_srs_id,
            org,
            org_id,
            wkt_def,
        ) in selected:
            if contents_srs_id != geom_srs_id:
                raise ValueError(
                    f"SRS mismatch in {path} layer {table_name}: "
                    f"gpkg_contents.srs_id={contents_srs_id} vs "
                    f"gpkg_geometry_columns.srs_id={geom_srs_id}"
                )
            epsg = _resolve_epsg(org, org_id, wkt_def, int(contents_srs_id))
            epsg_values.add(epsg)

            quoted_tbl = '"' + str(table_name).replace('"', '""') + '"'
            cur.execute(f"SELECT * FROM {quoted_tbl}")
            col_names = [desc[0] for desc in cur.description]
            if geom_col not in col_names:
                raise ValueError(f"Geometry column {geom_col!r} missing in {table_name}")
            geom_idx = col_names.index(geom_col)
            attr_cols = [c for c in col_names if c != geom_col and c.lower() != "fid"]
            for c in attr_cols:
                if c not in merged_cols:
                    merged_cols.append(c)
            if len(selected) > 1 and "facility_type" not in attr_cols:
                if "facility_type" not in merged_cols:
                    merged_cols.append("facility_type")

            for db_row in cur.fetchall():
                geom = _parse_gpb(db_row[geom_idx])
                row_dict = {
                    col_names[i]: db_row[i]
                    for i in range(len(col_names))
                    if i != geom_idx and col_names[i].lower() != "fid"
                }
                if len(selected) > 1 and "facility_type" not in row_dict:
                    row_dict["facility_type"] = str(table_name)
                all_rows.append(row_dict)
                all_geoms.append(geom)

        if len(epsg_values) != 1:
            raise ValueError(f"Inconsistent EPSG across layers in {path}: {epsg_values}")

        return GpkgLayer(
            path=path,
            table_name=str(first_meta[0]),
            geom_column=str(first_meta[2]),
            srs_id=int(first_meta[1]),
            epsg=next(iter(epsg_values)),
            geometry_type_name=str(first_meta[3]),
            columns=merged_cols,
            rows=all_rows,
            geoms=all_geoms,
            layer_names=available_layers,
        )
    finally:
        conn.close()


def _encode_wkb_point(x: float, y: float) -> bytes:
    return struct.pack("<BIdd", 1, 1, float(x), float(y))


def _encode_wkb_polygon(rings: Sequence[Sequence[tuple[float, float]]]) -> bytes:
    parts = [struct.pack("<BII", 1, 3, len(rings))]
    for ring in rings:
        pts = list(ring)
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        parts.append(struct.pack("<I", len(pts)))
        for x, y in pts:
            parts.append(struct.pack("<dd", float(x), float(y)))
    return b"".join(parts)


def _encode_wkb_multipolygon(
    polygons: Sequence[Sequence[Sequence[tuple[float, float]]]],
) -> bytes:
    parts = [struct.pack("<BII", 1, 6, len(polygons))]
    for poly in polygons:
        parts.append(_encode_wkb_polygon(poly))
    return b"".join(parts)


def _encode_gpb(
    geom_spec: tuple[str, Any], srs_id: int
) -> tuple[bytes, tuple[float, float, float, float]]:
    kind, data = geom_spec
    kind_u = kind.upper()
    if kind_u == "POINT":
        x, y = float(data[0]), float(data[1])
        wkb = _encode_wkb_point(x, y)
        bbox = (x, y, x, y)
    elif kind_u == "POLYGON":
        wkb = _encode_wkb_polygon(data)
        outer = data[0]
        bbox = _points_bbox(outer)
    elif kind_u == "MULTIPOLYGON":
        wkb = _encode_wkb_multipolygon(data)
        all_pts = [pt for poly in data for pt in poly[0]]
        bbox = _points_bbox(all_pts)
    else:
        raise ValueError(f"Unsupported geom_spec kind for write_gpkg: {kind!r}")
    # flags = 0x03 -> little-endian (bit 0 = 1) + XY envelope (env_code = 1 -> bits 1..3 = 001)
    header = struct.pack(
        "<2sBBidddd",
        b"GP",
        0,
        0x03,
        int(srs_id),
        bbox[0],
        bbox[2],
        bbox[1],
        bbox[3],
    )
    return header + wkb, bbox


def write_gpkg(
    path: Path,
    table_name: str,
    columns: Sequence[tuple[str, str]],
    rows: Sequence[dict[str, Any]],
    geoms: Sequence[tuple[str, Any]],
    geom_type: str = "POLYGON",
    srs_id: int = 32649,
) -> Path:
    """Write a standards-compliant OGC 1.3 GeoPackage in pure Python stdlib."""
    if len(rows) != len(geoms):
        raise ValueError("rows and geoms must have identical length")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    conn = sqlite3.connect(str(path))
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA application_id = 0x47504B47")
        cur.execute("PRAGMA user_version = 10300")
        cur.executescript(
            """
            CREATE TABLE gpkg_spatial_ref_sys (
                srs_name TEXT NOT NULL,
                srs_id INTEGER NOT NULL PRIMARY KEY,
                organization TEXT NOT NULL,
                organization_coordsys_id INTEGER NOT NULL,
                definition TEXT NOT NULL,
                description TEXT
            );
            CREATE TABLE gpkg_contents (
                table_name TEXT NOT NULL PRIMARY KEY,
                data_type TEXT NOT NULL,
                identifier TEXT UNIQUE,
                description TEXT DEFAULT '',
                last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                min_x DOUBLE,
                min_y DOUBLE,
                max_x DOUBLE,
                max_y DOUBLE,
                srs_id INTEGER,
                CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id)
                    REFERENCES gpkg_spatial_ref_sys(srs_id)
            );
            CREATE TABLE gpkg_geometry_columns (
                table_name TEXT NOT NULL,
                column_name TEXT NOT NULL,
                geometry_type_name TEXT NOT NULL,
                srs_id INTEGER NOT NULL,
                z TINYINT NOT NULL,
                m TINYINT NOT NULL,
                CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name),
                CONSTRAINT uk_gc_table_name UNIQUE (table_name),
                CONSTRAINT fk_gc_tn FOREIGN KEY (table_name)
                    REFERENCES gpkg_contents(table_name),
                CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id)
                    REFERENCES gpkg_spatial_ref_sys(srs_id)
            );
            """
        )
        cur.executemany(
            "INSERT INTO gpkg_spatial_ref_sys VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("Undefined cartesian SRS", -1, "NONE", -1, "undefined", None),
                ("Undefined geographic SRS", 0, "NONE", 0, "undefined", None),
                ("WGS 84 geodetic", 4326, "EPSG", 4326, WKT_EPSG_4326, "WGS 84"),
                (
                    "WGS 84 / UTM zone 49N",
                    32649,
                    "EPSG",
                    32649,
                    WKT_EPSG_32649,
                    "UTM 49N",
                ),
            ],
        )

        col_defs = ['"fid" INTEGER PRIMARY KEY AUTOINCREMENT', '"geom" BLOB']
        for col_name, col_type in columns:
            col_defs.append(f'"{col_name}" {col_type}')
        cur.execute(f'CREATE TABLE "{table_name}" ({", ".join(col_defs)})')

        encoded_blobs: list[bytes] = []
        bboxes: list[tuple[float, float, float, float]] = []
        for gspec in geoms:
            blob, bbox = _encode_gpb(gspec, srs_id)
            encoded_blobs.append(blob)
            bboxes.append(bbox)

        overall_bbox = _merge_bboxes(bboxes) or (0.0, 0.0, 0.0, 0.0)
        cur.execute(
            """
            INSERT INTO gpkg_contents
                (table_name, data_type, identifier, description, last_change,
                 min_x, min_y, max_x, max_y, srs_id)
            VALUES (?, 'features', ?, '', '2026-11-20T00:00:00.000Z', ?, ?, ?, ?, ?)
            """,
            (
                table_name,
                table_name,
                overall_bbox[0],
                overall_bbox[1],
                overall_bbox[2],
                overall_bbox[3],
                int(srs_id),
            ),
        )
        cur.execute(
            """
            INSERT INTO gpkg_geometry_columns
                (table_name, column_name, geometry_type_name, srs_id, z, m)
            VALUES (?, 'geom', ?, ?, 0, 0)
            """,
            (table_name, geom_type.upper(), int(srs_id)),
        )

        insert_cols = ["geom", *(c for c, _ in columns)]
        placeholders = ", ".join("?" for _ in insert_cols)
        col_sql = ", ".join(f'"{c}"' for c in insert_cols)
        batch = [
            (
                encoded_blobs[i],
                *(
                    int(rows[i][c]) if isinstance(rows[i].get(c), bool) else rows[i].get(c)
                    for c, _ in columns
                ),
            )
            for i in range(len(rows))
        ]
        cur.executemany(f'INSERT INTO "{table_name}" ({col_sql}) VALUES ({placeholders})', batch)
        conn.commit()
    finally:
        conn.close()
    return path


# ---------------------------------------------------------------------------
# Streaming GraphML Reader
# ---------------------------------------------------------------------------


@dataclass
class GraphMLSummary:
    """Summary of OSMnx pedestrian GraphML network."""

    path: Path
    node_count: int
    edge_count: int
    crs_guess: str
    component_sizes: list[int]
    peninsula_nodes: int
    island_nodes: int
    min_edge_length: float
    max_edge_length: float
    invalid_edge_lengths: int
    nodes_utm49n: list[tuple[float, float]]


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def read_graphml_summary(path: Path) -> GraphMLSummary:
    """Stream-parse GraphML with ElementTree.iterparse + iterative BFS components."""
    if not path.is_file():
        raise FileNotFoundError(f"GraphML not found: {path}")

    node_key_map: dict[str, str] = {}
    edge_key_map: dict[str, str] = {}
    nodes_raw: dict[str, tuple[float, float]] = {}
    adj: dict[str, list[str]] = {}
    edge_count = 0
    invalid_lengths = 0
    min_len = float("inf")
    max_len = 0.0

    for _event, elem in ET.iterparse(str(path), events=("end",)):
        tag = _strip_ns(elem.tag)
        if tag == "key":
            kid = elem.attrib.get("id", "")
            attr_name = elem.attrib.get("attr.name", "")
            for_what = elem.attrib.get("for", "")
            if for_what == "node":
                node_key_map[kid] = attr_name
            elif for_what == "edge":
                edge_key_map[kid] = attr_name
            elem.clear()
        elif tag == "node":
            nid = elem.attrib.get("id", "")
            x_val: float | None = None
            y_val: float | None = None
            for child in elem:
                if _strip_ns(child.tag) == "data":
                    name = node_key_map.get(child.attrib.get("key", ""), "")
                    if name == "x" and child.text:
                        x_val = float(child.text.strip())
                    elif name == "y" and child.text:
                        y_val = float(child.text.strip())
            if x_val is not None and y_val is not None:
                nodes_raw[nid] = (x_val, y_val)
            adj.setdefault(nid, [])
            elem.clear()
        elif tag == "edge":
            src = elem.attrib.get("source", "")
            dst = elem.attrib.get("target", "")
            edge_count += 1
            adj.setdefault(src, []).append(dst)
            adj.setdefault(dst, []).append(src)
            length_val: float | None = None
            for child in elem:
                if _strip_ns(child.tag) == "data":
                    name = edge_key_map.get(child.attrib.get("key", ""), "")
                    if name == "length" and child.text:
                        try:
                            length_val = float(child.text.strip())
                        except ValueError:
                            length_val = float("nan")
            if length_val is None or not math.isfinite(length_val) or length_val <= 0.0:
                invalid_lengths += 1
            else:
                min_len = min(min_len, length_val)
                max_len = max(max_len, length_val)
            elem.clear()

    # Iterative BFS connected components
    visited: set[str] = set()
    comp_sizes: list[int] = []
    for nid in adj:
        if nid in visited:
            continue
        size = 0
        queue: deque[str] = deque([nid])
        visited.add(nid)
        while queue:
            curr = queue.popleft()
            size += 1
            for nxt in adj.get(curr, ()):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        comp_sizes.append(size)
    comp_sizes.sort(reverse=True)

    # Auto-detect CRS (EPSG:4326 vs EPSG:32649) and count landmass nodes
    coords = list(nodes_raw.values())
    if coords and all(100.0 < x < 130.0 and 15.0 < y < 30.0 for x, y in coords[:200]):
        crs_guess = "EPSG:4326"
        pen_nodes = sum(1 for _, y in coords if y >= PENINSULA_MIN_LAT)
        isl_nodes = sum(1 for _, y in coords if y <= ISLANDS_MAX_LAT)
        nodes_utm = [wgs84_to_utm49n(x, y) for x, y in coords]
    else:
        crs_guess = "EPSG:32649"
        pen_nodes = sum(1 for _, y in coords if y >= PENINSULA_MIN_N)
        isl_nodes = sum(1 for _, y in coords if y <= ISLANDS_MAX_N)
        nodes_utm = coords

    return GraphMLSummary(
        path=path,
        node_count=len(nodes_raw),
        edge_count=edge_count,
        crs_guess=crs_guess,
        component_sizes=comp_sizes,
        peninsula_nodes=pen_nodes,
        island_nodes=isl_nodes,
        min_edge_length=min_len if math.isfinite(min_len) else 0.0,
        max_edge_length=max_len,
        invalid_edge_lengths=invalid_lengths,
        nodes_utm49n=nodes_utm,
    )


def nearest_node_snap_fraction(
    cell_centroids: Sequence[tuple[float, float]],
    graph_nodes_utm: Sequence[tuple[float, float]],
    max_dist_m: float = 300.0,
) -> tuple[float, float]:
    """Return (fraction_within_max_dist, max_observed_dist_m) using spatial grid hashing."""
    if not cell_centroids or not graph_nodes_utm:
        return 0.0, float("inf")
    grid_size = max_dist_m
    buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for nx, ny in graph_nodes_utm:
        key = (int(math.floor(nx / grid_size)), int(math.floor(ny / grid_size)))
        buckets.setdefault(key, []).append((nx, ny))

    within = 0
    max_obs = 0.0
    for cx, cy in cell_centroids:
        gx = int(math.floor(cx / grid_size))
        gy = int(math.floor(cy / grid_size))
        best = float("inf")
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for nx, ny in buckets.get((gx + dx, gy + dy), ()):
                    d = math.hypot(cx - nx, cy - ny)
                    if d < best:
                        best = d
        if not math.isfinite(best):
            best = min(math.hypot(cx - nx, cy - ny) for nx, ny in graph_nodes_utm)
        if best <= max_dist_m:
            within += 1
        max_obs = max(max_obs, best)
    return within / len(cell_centroids), max_obs


# ---------------------------------------------------------------------------
# Pure-Python PNG Inspector & 300-DPI Writer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PngInfo:
    """Validated PNG metadata and decoded pixel variance summary."""

    path: Path
    size_bytes: int
    width: int
    height: int
    bit_depth: int
    color_type: int
    dpi_x: float | None
    dpi_y: float | None
    pixel_std: float
    crc_valid: bool


def _paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _sample_png_pixel_std(
    raw_scanlines: bytes,
    width: int,
    height: int,
    bit_depth: int,
    color_type: int,
) -> float:
    """Unfilter PNG scanlines (8-bit RGB/RGBA/Gray) and compute RGB byte standard deviation."""
    channels_map = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    channels = channels_map.get(color_type, 3)
    if bit_depth != 8:
        sample = raw_scanlines[:: max(1, len(raw_scanlines) // 8192)]
        if not sample:
            return 0.0
        mean = sum(sample) / len(sample)
        return math.sqrt(sum((b - mean) ** 2 for b in sample) / len(sample))

    bpp = channels
    stride = width * bpp
    expected = height * (stride + 1)
    if len(raw_scanlines) < expected:
        raise ValueError(
            f"Decompressed IDAT shorter than expected: {len(raw_scanlines)} < {expected}"
        )

    prev_row = bytearray(stride)
    row_step = max(1, height // 64)
    col_step = max(1, width // 64)
    samples: list[int] = []

    offset = 0
    for y in range(height):
        flt = raw_scanlines[offset]
        offset += 1
        cur_row = bytearray(raw_scanlines[offset : offset + stride])
        offset += stride

        if flt == 1:  # Sub
            for i in range(bpp, stride):
                cur_row[i] = (cur_row[i] + cur_row[i - bpp]) & 0xFF
        elif flt == 2:  # Up
            for i in range(stride):
                cur_row[i] = (cur_row[i] + prev_row[i]) & 0xFF
        elif flt == 3:  # Average
            for i in range(stride):
                left = cur_row[i - bpp] if i >= bpp else 0
                cur_row[i] = (cur_row[i] + ((left + prev_row[i]) >> 1)) & 0xFF
        elif flt == 4:  # Paeth
            for i in range(stride):
                a = cur_row[i - bpp] if i >= bpp else 0
                b = prev_row[i]
                c = prev_row[i - bpp] if i >= bpp else 0
                cur_row[i] = (cur_row[i] + _paeth_predictor(a, b, c)) & 0xFF

        if y % row_step == 0:
            rgb_channels = min(3, channels)
            for x in range(0, width, col_step):
                base = x * bpp
                for ch in range(rgb_channels):
                    samples.append(cur_row[base + ch])
        prev_row = cur_row

    if not samples:
        return 0.0
    mean = sum(samples) / len(samples)
    return math.sqrt(sum((v - mean) ** 2 for v in samples) / len(samples))


def inspect_png(path: Path) -> PngInfo:
    """Validate PNG signature, chunk CRC32s, IHDR, pHYs DPI, and IDAT pixel variance."""
    if not path.is_file():
        raise FileNotFoundError(f"PNG file not found: {path}")
    data = path.read_bytes()
    if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Invalid PNG signature: {path}")

    pos = 8
    width = height = bit_depth = color_type = 0
    dpi_x: float | None = None
    dpi_y: float | None = None
    idat_parts: list[bytes] = []
    seen_ihdr = False
    seen_iend = False
    crc_valid = True

    while pos + 12 <= len(data):
        length = struct.unpack_from(">I", data, pos)[0]
        ctype = data[pos + 4 : pos + 8]
        cdata_start = pos + 8
        cdata_end = cdata_start + length
        if cdata_end + 4 > len(data):
            raise ValueError(f"Truncated PNG chunk {ctype!r} in {path}")
        cdata = data[cdata_start:cdata_end]
        stored_crc = struct.unpack_from(">I", data, cdata_end)[0]
        calc_crc = zlib.crc32(ctype + cdata) & 0xFFFFFFFF
        if calc_crc != stored_crc:
            crc_valid = False
        pos = cdata_end + 4

        if ctype == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack_from(">IIBB", cdata, 0)
            seen_ihdr = True
        elif ctype == b"pHYs" and length == 9:
            ppm_x, ppm_y, unit = struct.unpack_from(">IIB", cdata, 0)
            if unit == 1:
                dpi_x = ppm_x * 0.0254
                dpi_y = ppm_y * 0.0254
        elif ctype == b"IDAT":
            idat_parts.append(cdata)
        elif ctype == b"IEND":
            seen_iend = True
            break

    if not (seen_ihdr and seen_iend and idat_parts):
        raise ValueError(f"Incomplete PNG structure in {path}")
    if not crc_valid:
        raise ValueError(f"Corrupt PNG CRC32 checksum in {path}")

    raw_scanlines = zlib.decompress(b"".join(idat_parts))
    pixel_std = _sample_png_pixel_std(raw_scanlines, width, height, bit_depth, color_type)

    # Optional PIL secondary check if Pillow is importable
    try:
        from PIL import Image  # type: ignore[import-not-found]

        with Image.open(path) as img:
            img.verify()
    except ImportError:
        pass

    return PngInfo(
        path=path,
        size_bytes=len(data),
        width=width,
        height=height,
        bit_depth=bit_depth,
        color_type=color_type,
        dpi_x=dpi_x,
        dpi_y=dpi_y,
        pixel_std=pixel_std,
        crc_valid=crc_valid,
    )


def _png_chunk(ctype: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(ctype + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", crc)


def write_png(
    path: Path,
    width: int,
    height: int,
    dpi: int = 300,
    seed: int = 42,
) -> Path:
    """Write a fast, valid, multi-color, non-blank PNG (~50-90 KB) with exact pHYs DPI."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    stride = width * 3
    band_len = max(1, width // 6)
    palette_rows: list[bytes] = []
    for idx in range(12):
        c1 = bytes(((idx * 41 + 20) % 256, (idx * 67 + 40) % 256, (idx * 23 + 110) % 256))
        c2 = bytes(((240 - idx * 29) % 256, (idx * 53 + 30) % 256, (200 - idx * 17) % 256))
        c3 = bytes(((idx * 19 + 80) % 256, (220 - idx * 31) % 256, (idx * 47 + 50) % 256))
        pattern = c1 * band_len + c2 * band_len + c3 * band_len
        row_bytes = (pattern * ((width // (3 * band_len)) + 2))[:stride]
        palette_rows.append(b"\x00" + row_bytes)

    raw_rows = [palette_rows[(y // 24) % len(palette_rows)] for y in range(height)]
    # Add 8 high-detail textured legend/inset rows so compressed size is ~50-90 KB (>20 KB)
    for y in range(min(8, height)):
        raw_rows[y] = b"\x00" + rng.randbytes(stride)
    idat_compressed = zlib.compress(b"".join(raw_rows), level=1)

    ppm = int(round(dpi / 0.0254))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    phys = struct.pack(">IIB", ppm, ppm, 1)

    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"pHYs", phys)
        + _png_chunk(b"IDAT", idat_compressed)
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png_bytes)
    return path


# ---------------------------------------------------------------------------
# Poppler-Primary PDF Inspector & Pure-Python Multi-Page A4 PDF Writer
# ---------------------------------------------------------------------------


@dataclass
class PdfInfo:
    """Comprehensive Poppler-verified PDF layout, pagination, and segmented text info."""

    path: Path
    total_pages: int
    encrypted: bool
    page_sizes_pt: list[tuple[float, float]]
    pages_raw_text: list[str]
    cover_pages: list[int]
    main_start_page: int
    main_end_page: int
    main_fractional_pages: float
    single_column_ratio: float
    median_line_pitch_pt: float
    embedded_image_count: int
    fonts: list[str]
    cover_text: str
    main_text: str
    references_text: str
    contribution_text: str
    ai_declaration_text: str
    backmatter_text: str
    full_text: str


def _run_cmd(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}"
        )
    return proc.stdout


def _group_words_into_lines(
    words: Sequence[tuple[float, float, float, float, str]],
) -> list[tuple[float, float, float, float, str]]:
    """Group (xMin, yMin, xMax, yMax, text) boxes into horizontal visual lines."""
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (round(w[1] / 3.0), w[0]))
    lines: list[list[tuple[float, float, float, float, str]]] = []
    for w in ordered:
        if not lines or abs(w[1] - lines[-1][0][1]) > 4.5:
            lines.append([w])
        else:
            lines[-1].append(w)
    result: list[tuple[float, float, float, float, str]] = []
    for grp in lines:
        grp_sorted = sorted(grp, key=lambda w: w[0])
        xmin = min(w[0] for w in grp_sorted)
        ymin = min(w[1] for w in grp_sorted)
        xmax = max(w[2] for w in grp_sorted)
        ymax = max(w[3] for w in grp_sorted)
        text = " ".join(w[4] for w in grp_sorted).strip()
        result.append((xmin, ymin, xmax, ymax, text))
    result.sort(key=lambda ln: ln[1])
    return result


def _parse_bbox_xhtml(
    xhtml: str,
) -> list[tuple[float, float, list[tuple[float, float, float, float, str]]]]:
    """Parse pdftotext -bbox XHTML into per-page (width, height, words)."""
    root = ET.fromstring(xhtml)
    pages: list[tuple[float, float, list[tuple[float, float, float, float, str]]]] = []
    for elem in root.iter():
        if _strip_ns(elem.tag) != "page":
            continue
        pw = float(elem.attrib.get("width", "595.28"))
        ph = float(elem.attrib.get("height", "841.89"))
        words: list[tuple[float, float, float, float, str]] = []
        for child in elem.iter():
            if _strip_ns(child.tag) == "word" and child.text and child.text.strip():
                words.append(
                    (
                        float(child.attrib["xMin"]),
                        float(child.attrib["yMin"]),
                        float(child.attrib["xMax"]),
                        float(child.attrib["yMax"]),
                        child.text.strip(),
                    )
                )
        pages.append((pw, ph, words))
    return pages


def _split_backmatter_sections(back_text: str) -> tuple[str, str, str]:
    """Split normalized backmatter into (references, contribution, ai_declaration)."""
    ref_m = re.search(r"(?im)^(?:\d+\.?\s*)?(?:references|bibliography)\b", back_text)
    contrib_m = re.search(
        r"(?im)^(?:\d+\.?\s*)?(?:(?:separate\s+)?(?:team\s+|author\s+|member\s+)?"
        r"contribution\s+statement|team\s+contributions?|author\s+contributions?)\b",
        back_text,
    )
    ai_m = re.search(
        r"(?im)^(?:\d+\.?\s*)?(?:(?:generative\s+)?ai\s+(?:usage|use)\s+declaration|"
        r"declaration\s+of\s+(?:generative\s+)?ai)\b",
        back_text,
    )
    markers = sorted(
        [
            (m.start(), name)
            for name, m in (("ref", ref_m), ("contrib", contrib_m), ("ai", ai_m))
            if m
        ],
        key=lambda item: item[0],
    )
    slices: dict[str, str] = {"ref": "", "contrib": "", "ai": ""}
    for idx, (start, name) in enumerate(markers):
        end = markers[idx + 1][0] if idx + 1 < len(markers) else len(back_text)
        slices[name] = back_text[start:end].strip()
    return slices["ref"], slices["contrib"], slices["ai"]


def inspect_pdf(path: Path) -> PdfInfo:
    """Inspect a PDF using Poppler CLI utilities (pdfinfo, pdftotext -bbox, pdffonts, pdfimages)."""
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if shutil.which("pdfinfo") is None or shutil.which("pdftotext") is None:
        raise RuntimeError("Poppler utilities (pdfinfo / pdftotext) are required on PATH")

    info_out = _run_cmd(["pdfinfo", str(path)])
    pages_m = re.search(r"^Pages:\s*(\d+)", info_out, re.MULTILINE)
    if not pages_m:
        raise ValueError(f"Could not parse page count from pdfinfo on {path}")
    total_pages = int(pages_m.group(1))
    enc_m = re.search(r"^Encrypted:\s*(\S+)", info_out, re.MULTILINE)
    encrypted = bool(enc_m and enc_m.group(1).lower() != "no")

    sizes_out = _run_cmd(["pdfinfo", "-f", "1", "-l", str(total_pages), str(path)])
    page_sizes: list[tuple[float, float]] = []
    for match in re.finditer(
        r"^Page\s+\d+\s+size:\s*([0-9.]+)\s+x\s+([0-9.]+)\s+pts",
        sizes_out,
        re.MULTILINE,
    ):
        page_sizes.append((float(match.group(1)), float(match.group(2))))
    if len(page_sizes) != total_pages:
        single_m = re.search(
            r"^Page size:\s*([0-9.]+)\s+x\s+([0-9.]+)\s+pts",
            info_out,
            re.MULTILINE,
        )
        if single_m:
            page_sizes = [(float(single_m.group(1)), float(single_m.group(2)))] * total_pages

    raw_text = _run_cmd(["pdftotext", str(path), "-"])
    raw_pages = raw_text.split("\x0c")
    # Strip ONLY the single trailing form-feed sentinel appended by pdftotext
    if len(raw_pages) == total_pages + 1 and raw_pages[-1] == "":
        raw_pages.pop()

    bbox_xhtml = _run_cmd(["pdftotext", "-bbox", str(path), "-"])
    bbox_pages = _parse_bbox_xhtml(bbox_xhtml)

    # Fonts & embedded images
    fonts: list[str] = []
    if shutil.which("pdffonts") is not None:
        fonts_out = _run_cmd(["pdffonts", str(path)])
        for ln in fonts_out.splitlines()[2:]:
            parts = ln.split()
            if parts:
                fonts.append(parts[0])

    embedded_images = 0
    if shutil.which("pdfimages") is not None:
        imgs_out = _run_cmd(["pdfimages", "-list", str(path)])
        for ln in imgs_out.splitlines()[2:]:
            if ln.strip():
                embedded_images += 1

    # Segment Cover Page -> Main Text -> Excluded Back Matter using bbox lines
    page_lines = [_group_words_into_lines(words) for _, _, words in bbox_pages]

    cover_pages: list[int] = []
    main_start = 1
    if total_pages >= 2:
        p1_norm = normalize_pdf_text(raw_pages[0]) if raw_pages else ""
        p1_has_main_start = any(_MAIN_START_RE.match(ln[4]) for ln in page_lines[0])
        p1_looks_cover = not p1_has_main_start and bool(
            re.search(
                r"(?i)(cs482|student\s+id|group\s+|cover\s+page|phase\s+a)",
                p1_norm,
            )
        )
        if p1_looks_cover:
            cover_pages = [1]
            main_start = 2

    # Find first back-matter heading at or after main_start
    back_page_idx: int | None = None
    back_heading_y: float | None = None
    for p_num in range(main_start, total_pages + 1):
        lines = page_lines[p_num - 1] if p_num - 1 < len(page_lines) else []
        for _xmin, ymin, _xmax, _ymax, text in lines:
            if len(text) <= 80 and _BACKMATTER_HEADING_RE.match(text.strip()):
                back_page_idx = p_num
                back_heading_y = ymin
                break
        if back_page_idx is not None:
            break

    # Helper to exclude bottom page-number footers
    def _body_lines(
        p_num: int, max_y_cutoff: float | None = None
    ) -> list[tuple[float, float, float, float, str]]:
        if p_num - 1 >= len(bbox_pages):
            return []
        _pw, ph, _words = bbox_pages[p_num - 1]
        lines = page_lines[p_num - 1]
        out = []
        for ln in lines:
            xmin, ymin, xmax, ymax, txt = ln
            if max_y_cutoff is not None and ymin >= max_y_cutoff - 2.0:
                continue
            if ymin > ph - 65.0 and re.fullmatch(
                r"(?:page\s*)?\d+(?:\s*/\s*\d+)?", txt.strip(), re.I
            ):
                continue
            out.append(ln)
        return out

    # Determine exact main_end and fractional fill
    if back_page_idx is None:
        main_end = total_pages
        last_page_cutoff = None
    else:
        above = _body_lines(back_page_idx, max_y_cutoff=back_heading_y)
        if len(above) >= 2 and back_heading_y is not None and back_heading_y > 120.0:
            main_end = back_page_idx
            last_page_cutoff = back_heading_y
        else:
            main_end = max(main_start, back_page_idx - 1)
            last_page_cutoff = None

    # Compute typical vertical body span from full main-text pages
    full_pages = list(range(main_start, main_end))
    top_ys: list[float] = []
    bot_ys: list[float] = []
    for p_num in full_pages:
        bl = _body_lines(p_num)
        if bl:
            top_ys.append(min(ln[1] for ln in bl))
            bot_ys.append(max(ln[3] for ln in bl))

    body_top = min(top_ys) if top_ys else 72.0
    body_bottom = max(bot_ys) if bot_ys else 760.0
    body_height = max(body_bottom - body_top, 400.0)

    last_lines = _body_lines(main_end, max_y_cutoff=last_page_cutoff)
    if last_lines:
        last_ymax = max(ln[3] for ln in last_lines)
        last_fill = max(0.05, min(1.0, (last_ymax - body_top) / body_height))
    else:
        last_fill = 0.0
    main_fractional = max(0.0, float(main_end - main_start) + last_fill)

    # Single-column check & median body line pitch across main-text pages
    wide_lines = 0
    eligible_lines = 0
    pitches: list[float] = []
    for p_num in range(main_start, main_end + 1):
        pw = bbox_pages[p_num - 1][0] if p_num - 1 < len(bbox_pages) else 595.28
        cutoff = last_page_cutoff if p_num == main_end else None
        bl = _body_lines(p_num, max_y_cutoff=cutoff)
        for ln in bl:
            xmin, _ymin, xmax, _ymax, txt = ln
            if len(txt) >= 35:
                eligible_lines += 1
                if xmin < 0.44 * pw and xmax > 0.56 * pw:
                    wide_lines += 1
        for i in range(len(bl) - 1):
            dy = bl[i + 1][1] - bl[i][1]
            if 10.0 <= dy <= 30.0:
                pitches.append(dy)

    single_col_ratio = (wide_lines / eligible_lines) if eligible_lines > 0 else 1.0
    pitches.sort()
    median_pitch = pitches[len(pitches) // 2] if pitches else 18.0

    # Assemble segmented normalized texts
    cover_text = normalize_pdf_text(
        "\n".join(raw_pages[p - 1] for p in cover_pages if p <= len(raw_pages))
    )
    main_chunks: list[str] = []
    back_chunks: list[str] = []
    for p_num in range(main_start, total_pages + 1):
        page_txt = raw_pages[p_num - 1] if p_num - 1 < len(raw_pages) else ""
        if back_page_idx is None or p_num < back_page_idx:
            main_chunks.append(page_txt)
        elif p_num == back_page_idx and last_page_cutoff is not None:
            above_txt = "\n".join(ln[4] for ln in _body_lines(p_num, max_y_cutoff=last_page_cutoff))
            below_txt = "\n".join(
                ln[4] for ln in page_lines[p_num - 1] if ln[1] >= last_page_cutoff - 2.0
            )
            main_chunks.append(above_txt)
            back_chunks.append(below_txt)
        else:
            back_chunks.append(page_txt)

    main_text = normalize_pdf_text("\n".join(main_chunks))
    backmatter_text = normalize_pdf_text("\n".join(back_chunks))
    ref_text, contrib_text, ai_text = _split_backmatter_sections(backmatter_text)
    full_text = normalize_pdf_text("\n".join(raw_pages))

    return PdfInfo(
        path=path,
        total_pages=total_pages,
        encrypted=encrypted,
        page_sizes_pt=page_sizes,
        pages_raw_text=raw_pages,
        cover_pages=cover_pages,
        main_start_page=main_start,
        main_end_page=main_end,
        main_fractional_pages=main_fractional,
        single_column_ratio=single_col_ratio,
        median_line_pitch_pt=median_pitch,
        embedded_image_count=embedded_images,
        fonts=fonts,
        cover_text=cover_text,
        main_text=main_text,
        references_text=ref_text,
        contribution_text=contrib_text,
        ai_declaration_text=ai_text,
        backmatter_text=backmatter_text,
        full_text=full_text,
    )


def _pdf_escape(text: str) -> str:
    safe = text.encode("latin-1", errors="replace").decode("latin-1")
    return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(
    path: Path,
    pages_lines: Sequence[Sequence[str]],
    page_width: float = 595.28,
    page_height: float = 841.89,
    line_pitch: float = 18.0,
    font_size: float = 11.0,
    embed_image_on_page: int | None = None,
) -> Path:
    """Write a valid multi-page PDF 1.4 with /Helvetica text & optional RGB Image XObject."""
    path.parent.mkdir(parents=True, exist_ok=True)
    objects: list[bytes] = []

    def _add_obj(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    catalog_id = _add_obj(b"")  # placeholder 1
    pages_id = _add_obj(b"")  # placeholder 2
    font_id = _add_obj(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    )

    image_id: int | None = None
    if embed_image_on_page is not None:
        img_w, img_h = 32, 16
        rgb_buf = bytearray()
        for y in range(img_h):
            for x in range(img_w):
                rgb_buf.extend(((x * 8) & 0xFF, (y * 16) & 0xFF, ((x + y) * 5) & 0xFF))
        img_data = bytes(rgb_buf)
        image_id = _add_obj(
            f"<< /Type /XObject /Subtype /Image /Width {img_w} /Height {img_h} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Length {len(img_data)} >>\n"
            f"stream\n".encode("ascii")
            + img_data
            + b"\nendstream"
        )

    page_obj_ids: list[int] = []
    for p_idx, lines in enumerate(pages_lines, start=1):
        cmds: list[str] = []
        y = page_height - 72.0
        for line in lines:
            cmds.append(f"BT /F1 {font_size:.1f} Tf 72.0 {y:.2f} Td ({_pdf_escape(line)}) Tj ET")
            y -= line_pitch
        if image_id is not None and embed_image_on_page == p_idx:
            # Draw embedded RGB image XObject cleanly below the text block
            img_y = max(90.0, y - 95.0)
            cmds.append(f"q 320 0 0 80 120 {img_y:.2f} cm /Im1 Do Q")
        stream_bytes = ("\n".join(cmds) + "\n").encode("latin-1", errors="replace")
        content_id = _add_obj(
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
            + stream_bytes
            + b"endstream"
        )
        xobj_part = (
            f" /XObject << /Im1 {image_id} 0 R >>"
            if (image_id and embed_image_on_page == p_idx)
            else ""
        )
        page_id = _add_obj(
            f"<< /Type /Page /Parent {pages_id} 0 R "
            f"/MediaBox [0 0 {page_width:.2f} {page_height:.2f}] "
            f"/Resources << /Font << /F1 {font_id} 0 R >>{xobj_part} >> "
            f"/Contents {content_id} 0 R >>".encode("ascii")
        )
        page_obj_ids.append(page_id)

    kids = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    objects[catalog_id - 1] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("ascii")
    objects[pages_id - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_obj_ids)} >>".encode(
        "ascii"
    )

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = [0]
    for obj_num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{obj_num} 0 obj\n".encode("ascii"))
        out.extend(body)
        out.extend(b"\nendobj\n")

    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    out.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(bytes(out))
    return path
