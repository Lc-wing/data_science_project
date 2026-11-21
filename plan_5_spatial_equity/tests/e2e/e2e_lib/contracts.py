"""Authoritative opaque-box contracts, schemas, bounds, and regexes from PROJECT.md."""

from __future__ import annotations

import re
from pathlib import Path

# tests/e2e/e2e_lib/contracts.py -> Plan 5 project root (plan_5_spatial_equity/)
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]

# Geofabrik extract cached by M1 at data/interim/macau.gpkg. It stays CRS84
# (gpkg srs_id 100000), not an EPSG:32649 analysis product.
RAW_SOURCE_GPKG_NAMES: frozenset[str] = frozenset({"macau.gpkg"})

# Core geospatial & reproducibility constants
TARGET_EPSG = 32649
GLOBAL_SEED = 42
WALK_SPEED_KMH = 4.5
WALK_SPEED_M_PER_MIN = 75.0
MAX_GRAPH_WALK_MIN = 10.0

# Macao SAR projected EPSG:32649 (WGS84 / UTM Zone 49N) envelopes
MACAO_POLY_E_MIN = 756_000.0
MACAO_POLY_E_MAX = 772_000.0
MACAO_POLY_N_MIN = 2_444_000.0
MACAO_POLY_N_MAX = 2_463_000.0

MACAO_PT_E_MIN = 758_000.0
MACAO_PT_E_MAX = 770_500.0
MACAO_PT_N_MIN = 2_445_500.0
MACAO_PT_N_MAX = 2_461_500.0

# Peninsula vs. Taipa-Cotai-Coloane harbour divide
PENINSULA_MIN_N = 2_454_500.0
ISLANDS_MAX_N = 2_454_000.0
PENINSULA_MIN_LAT = 22.175
ISLANDS_MAX_LAT = 22.170
ZONA_A_MIN_E = 764_000.0

# Terrestrial 2021 DSEC census totals (excluding maritime Z26: 777 / 61)
TERRESTRIAL_POP_TOTAL = 681_293.0
TERRESTRIAL_POP_65PLUS = 82_751.0
POP_TOL_ZONES_PARISHES = 0.0005  # +/- 0.05%
POP_TOL_H3_TOTAL = 0.0005  # +/- 0.05%
POP_TOL_H3_65PLUS = 0.0010  # +/- 0.10%

# Canonical enumerations
FACILITY_TYPES: tuple[str, ...] = (
    "park",
    "health_centre",
    "library",
    "bus_stop",
    "ev_charger",
)
TRADITIONAL_FACILITY_TYPES: tuple[str, ...] = (
    "park",
    "health_centre",
    "library",
    "bus_stop",
)
UNIT_CAPACITY_FACILITIES: frozenset[str] = frozenset({"health_centre", "library", "bus_stop"})


def facility_source_family(source: str) -> str:
    """Provider family used by the cross-source 25 m dedup contract.

    M1 writes OSM layer names (`osm_geofabrik_landuse`, `osm_pbf_garden`) into
    `source`. Those are one OSM extract, not independent registries. Cross-source
    dedup compares families (`cem` vs `osm`), and does not collapse nearby
    features that share a family (opposite bus poles, adjacent OSM parks).
    """
    token = source.strip().lower().split("+", 1)[0].split("_", 1)[0]
    return token or source.strip().lower()
LISA_CLUSTERS: frozenset[str] = frozenset({"HH", "LL", "LH", "HL", "NS"})
H3_MODEL_TYPES: frozenset[str] = frozenset({"OLS_HC3", "SEM_ML", "SLM_ML"})
ML_MODEL_NAMES: tuple[str, ...] = ("naive_prior", "logistic_regression", "random_forest")
ML_METRICS: tuple[str, ...] = ("roc_auc", "pr_auc", "brier", "top20_hit_rate")
ML_AGGREGATE_KEYS: tuple[str, ...] = tuple(
    f"{prefix}_{metric}" for prefix in ("mean", "std", "oof") for metric in ML_METRICS
)
PENINSULA_PARISH_IDS: frozenset[str] = frozenset({"P01", "P02", "P03", "P04", "P05"})
ISLAND_PARISH_IDS: frozenset[str] = frozenset({"P06", "P07"})

# Facility count & capacity bounds
FACILITY_COUNT_BOUNDS: dict[str, tuple[int, int]] = {
    "park": (50, 600),
    "health_centre": (8, 15),
    "library": (10, 25),
    "bus_stop": (250, 1000),
    "ev_charger": (60, 120),
}
EV_CAPACITY_MIN = 1200.0
EV_CAPACITY_MAX = 1800.0
LRT_COUNT_BOUNDS = (10, 20)
H3_R9_COUNT_BOUNDS = (180, 500)
H3_R8_COUNT_MIN = 15

# Required columns per artifact contract (PROJECT.md § Interface Contracts)
ML_FEATURES_REQUIRED: tuple[str, ...] = (
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
)
TARGET_AND_AUDIT_COLS: tuple[str, ...] = (
    "has_ev_charger",
    "ev_station_count",
    "ev_charger_capacity",
)
CELLS_BASE_R9_IDENTITY_COLS: tuple[str, ...] = (
    "cell_id",
    "centroid_x",
    "centroid_y",
    "area_km2",
    "zone_id",
    "zone_name",
    "parish_id",
    "parish_name",
)
CELLS_BASE_R9_REQUIRED_COLS: tuple[str, ...] = (
    *CELLS_BASE_R9_IDENTITY_COLS,
    *ML_FEATURES_REQUIRED,
    *TARGET_AND_AUDIT_COLS,
    "spatial_block",
)
CELLS_BASE_R8_REQUIRED_COLS: tuple[str, ...] = (
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
)
ZONES_23_REQUIRED_COLS: tuple[str, ...] = (
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
)
PARISHES_7_REQUIRED_COLS: tuple[str, ...] = (
    "parish_id",
    "parish_name",
    "parish_name_zh",
    "area_km2",
    "pop_total",
    "pop_65plus",
)
FACILITIES_REQUIRED_COLS: tuple[str, ...] = (
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
)
GRAPH_EDGES_REQUIRED_COLS: tuple[str, ...] = (
    "src_cell_id",
    "dst_cell_id",
    "walk_time_min",
    "weight",
    "same_block",
)
SPATIAL_FOLDS_REQUIRED_COLS: tuple[str, ...] = ("cell_id", "spatial_block")
POI_TEXTS_REQUIRED_COLS: tuple[str, ...] = (
    "cell_id",
    "poi_count_raw",
    "poi_count_filtered",
    "poi_text",
)

# 2SFCA columns (14 total: 12 non-negative individual/unweighted + 2 composite)
ACC_INDIVIDUAL_TYPES: tuple[str, ...] = (*FACILITY_TYPES, "ev_unweighted")
ACC_10_INDIVIDUAL_COLS: tuple[str, ...] = tuple(f"acc10_{t}" for t in ACC_INDIVIDUAL_TYPES)
ACC_15_INDIVIDUAL_COLS: tuple[str, ...] = tuple(f"acc15_{t}" for t in ACC_INDIVIDUAL_TYPES)
ACC_INDIVIDUAL_COLS: tuple[str, ...] = (*ACC_10_INDIVIDUAL_COLS, *ACC_15_INDIVIDUAL_COLS)
ACC_COMPOSITE_COLS: tuple[str, ...] = ("acc10_composite", "acc15_composite")
ALL_ACC_COLS: tuple[str, ...] = (
    *(f"acc10_{t}" for t in (*FACILITY_TYPES, "composite", "ev_unweighted")),
    *(f"acc15_{t}" for t in (*FACILITY_TYPES, "composite", "ev_unweighted")),
)

# Analytical table schemas
H1_REQUIRED_COLS: tuple[str, ...] = (
    "resolution",
    "threshold_min",
    "facility_type",
    "gini",
    "gini_ci_low",
    "gini_ci_high",
    "theil_total",
    "theil_within",
    "theil_between",
    "theil_total_ci_low",
    "theil_total_ci_high",
    "theil_between_share",
    "moran_i",
    "moran_p",
    "n_ll_fdr",
    "n_hh_fdr",
    "mean_acc",
    "median_acc",
    "zero_acc_pct",
)
H2_REQUIRED_COLS: tuple[str, ...] = (
    "threshold_min",
    "comparison",
    "gini_ev",
    "gini_other",
    "gini_diff",
    "ci_low",
    "ci_high",
    "p_boot_raw",
    "p_holm",
    "n_boot",
    "seed",
)
H3_REQUIRED_COLS: tuple[str, ...] = (
    "threshold_min",
    "outcome",
    "weights_spec",
    "model_type",
    "term",
    "coef",
    "se",
    "ci_low",
    "ci_high",
    "p_value",
    "r2_or_pseudo_r2",
    "aic",
    "resid_moran_i",
    "resid_moran_p",
    "n_obs",
)

# Required repository deliverables (relative paths)
FROZEN_CANONICAL_FILES: tuple[str, ...] = (
    "data/frozen/cells.gpkg",
    "data/frozen/cells_res8.gpkg",
    "data/frozen/zones_23.gpkg",
    "data/frozen/parishes_7.gpkg",
    "data/frozen/facilities_clean.gpkg",
    "data/frozen/lrt_stations.gpkg",
    "data/frozen/graph_edges.csv",
    "data/frozen/spatial_folds.csv",
    "data/frozen/split_metadata.json",
    "data/frozen/cell_poi_texts.csv",
    "data/frozen/feature_manifest.json",
    "data/frozen/FREEZE_MANIFEST.json",
    "data/frozen/data_dictionary.md",
)
FROZEN_CORE_HASHED_FILES: tuple[str, ...] = tuple(
    p for p in FROZEN_CANONICAL_FILES if not p.endswith("FREEZE_MANIFEST.json")
)
REQUIRED_FIGURES: tuple[str, ...] = (
    "outputs/figures/fig1_accessibility_lisa.png",
    "outputs/figures/eda_facilities_pop.png",
    "outputs/figures/eda_accessibility_lorenz.png",
    "outputs/figures/maup_r8_vs_r9.png",
)
TABLE1_FILES: tuple[str, ...] = (
    "outputs/tables/table1_summary.csv",
    "outputs/tables/table1_summary.md",
    "outputs/tables/table1_summary.tex",
)
REPORT_TEX_REL = "report/phase_a_draft.tex"
REPORT_BIB_REL = "report/references.bib"
REPORT_PDF_REL = "report/CS482_GroupX_MacaoSpatialEquity_PhaseA_Draft.pdf"
REPORT_AUDIT_REL = "report/go_nogo_audit.md"

REQUIRED_PYPROJECT_PACKAGES: tuple[str, ...] = (
    "geopandas",
    "shapely",
    "pyproj",
    "h3",
    "osmnx",
    "networkx",
    "libpysal",
    "esda",
    "spreg",
    "statsmodels",
    "scikit-learn",
    "matplotlib",
    "pytest",
)

# PDF A4 & budget tolerances
A4_WIDTH_PT = 595.28
A4_HEIGHT_PT = 841.89
A4_TOL_PT = 3.5
PDF_MAIN_PAGES_MIN = 2.95
PDF_MAIN_PAGES_MAX = 3.85
PNG_MIN_DPI = 299.5
PNG_MIN_BYTES = 20_000
PNG_MIN_PIXEL_STD = 5.0

# Authoritative validation regexes
H3_R9_RE = re.compile(r"^89[0-9a-f]{13}$")
H3_R8_RE = re.compile(r"^88[0-9a-f]{13}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EV_TEXT_LEAK_RE = re.compile(r"(?i)\b(ev|evse|supercharger|tesla)\b|charg|充電|充电|電動車|电动车")
EV_TEXT_LEAK_MULTILINGUAL_RE = re.compile(
    r"(?i)\b(ev|evse|supercharger|tesla)\b|charg|充電|充电|電動車|电动车|"
    r"carregador|ve[ií]culo\s+el[eé]trico"
)
EV_COL_LEAK_RE = re.compile(
    r"(?i)(^|_)(ev|evse|charger|charging|tesla|pile|pilenum|total_kw|"
    r"acc10|acc15|lisa|target|spatial_block|fold)(_|$)|2sfca|charg"
)
MARITIME_ZONE_RE = re.compile(r"(?i)(^z?26$|水域|maritime)")
TAIPA_PARISH_RE = re.compile(r"(?i)(嘉模|carmo|taipa)")
NON_PUBLIC_LIBRARY_RE = re.compile(
    r"(?i)\b(must|university\s+of\s+macau|um\s+library|art\s+museum|school\s+library)\b"
)
ANALYTICAL_PLACEHOLDER_RE = re.compile(
    r"\b(?:TBD_NUM|TODO|FIXME|LOREM\s+IPSUM|NaN|Infinity)\b|\[\?\]|\?\?"
)
TABLE_PLACEHOLDER_RE = re.compile(r"(?i)\b(?:nan|none|inf|todo|tbd|tbd_num|fixme|xxx)\b")
FORBIDDEN_HYPE_PHRASES: tuple[str, ...] = (
    "target 35/35",
    "huge originality advantage",
    "instructors love",
    "cotai price cliff",
    "guaranteed a+",
    "guaranteed 100%",
)
SECRET_TOKEN_RE = re.compile(
    r"(?i)(ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|"
    r"authorization\s*:\s*bearer\s+[A-Za-z0-9._-]{15,}|"
    r"api[_-]?key[\"']?\s*[:=]\s*[\"'][A-Za-z0-9._-]{16,}[\"'])"
)


PROVENANCE_TIMESTAMP_KEYS: tuple[str, ...] = (
    "fetched_at_utc8",
    "extracted_at",
    "downloaded_at",
    "timestamp",
    "date",
)
PROVENANCE_LOCAL_PATH_KEYS: tuple[str, ...] = (
    "local_raw_path",
    "local_path",
    "file",
    "path",
)
PROVENANCE_COUNT_KEYS: tuple[str, ...] = (
    "record_count",
    "row_count_raw",
    "row_count_clean",
    "count",
    "byte_size",
    "size_bytes",
)


def _first_present(entry: dict[object, object], keys: tuple[str, ...]) -> object | None:
    for key in keys:
        val = entry.get(key)
        if val is None or val == "":
            continue
        return val
    return None


def provenance_timestamp(entry: dict[object, object]) -> str:
    """ISO timestamp from an M1 (`fetched_at_utc8`) or synthetic (`extracted_at`) entry."""
    val = _first_present(entry, PROVENANCE_TIMESTAMP_KEYS)
    return "" if val is None else str(val)


def provenance_local_rel(entry: dict[object, object]) -> str:
    """Relative snapshot path. M1 records `local_raw_path`."""
    val = _first_present(entry, PROVENANCE_LOCAL_PATH_KEYS)
    return "" if val is None else str(val)


def provenance_count(entry: dict[object, object]) -> float | None:
    """Positive size/count. Skips JSON nulls such as Geofabrik `row_count_raw`."""
    val = _first_present(entry, PROVENANCE_COUNT_KEYS)
    if val is None:
        return None
    return float(val)


def resolve_m1_cells_r9_path(project_root: Path) -> Path:
    """H3 r9 analysis cells: M4 `cells.gpkg` when present, else M1 `cells_base_res9.gpkg`."""
    canonical = project_root / "data/frozen/cells.gpkg"
    if canonical.is_file():
        return canonical
    return project_root / "data/frozen/cells_base_res9.gpkg"


def resolve_m1_cells_r8_path(project_root: Path) -> Path:
    """H3 r8 analysis cells: M4 `cells_res8.gpkg` when present, else M1 `cells_base_res8.gpkg`."""
    canonical = project_root / "data/frozen/cells_res8.gpkg"
    if canonical.is_file():
        return canonical
    return project_root / "data/frozen/cells_base_res8.gpkg"


def resolve_provenance_paths(project_root: Path) -> list[Path]:
    """Return existing canonical data_provenance.json path(s) in preference order."""
    candidates = [
        project_root / "outputs" / "metrics" / "data_provenance.json",
        project_root / "data" / "raw" / "data_provenance.json",
    ]
    return [p for p in candidates if p.is_file()]


def resolve_zones_with_2sfca_path(project_root: Path) -> Path:
    """Prefer data/interim/zones_23_with_2sfca.gpkg; fallback to data/frozen/zones_23.gpkg."""
    interim = project_root / "data" / "interim" / "zones_23_with_2sfca.gpkg"
    if interim.is_file():
        return interim
    return project_root / "data" / "frozen" / "zones_23.gpkg"


def normalize_resolution_label(val: object) -> str:
    """Normalize resolution strings such as 'res9', 'r9', 'H3_r9', '9' -> 'res9'/'res8'."""
    s = str(val).strip().lower()
    if s in {"res9", "r9", "9", "h3_r9", "h3_res9", "res_9"}:
        return "res9"
    if s in {"res8", "r8", "8", "h3_r8", "h3_res8", "res_8"}:
        return "res8"
    return s


def normalize_weights_label(val: object) -> str:
    """Classify spatial weights spec label into 'knn4' or 'bridge_queen' (or lowercase raw)."""
    s = str(val).strip().lower()
    if "knn" in s or "k4" in s or "k=4" in s or "k_4" in s or "4nn" in s:
        return "knn4"
    if "queen" in s or "bridge" in s:
        return "bridge_queen"
    return s
