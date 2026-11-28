"""Canonical configuration, constants, verified whitelists, and helpers for MacaoSpatialEquity."""

from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from pyproj import Transformer

# Enforce headless plotting before any matplotlib import
os.environ.setdefault("MPLBACKEND", "Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DATA_FROZEN_DIR = PROJECT_ROOT / "data" / "frozen"
OUTPUTS_METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

# ---------------------------------------------------------------------------
# CRS, Seeds & Spatial Constants
# ---------------------------------------------------------------------------
ANALYSIS_CRS = "EPSG:32649"
WGS84_CRS = "EPSG:4326"
DSEC_NATIVE_CRS = "EPSG:8433"

RANDOM_SEED = 42
WALK_SPEED_KMH = 4.5
WALK_SPEED_M_PER_MIN = WALK_SPEED_KMH * 1000.0 / 60.0  # 75.0 m/min
CUTOFF_10_MIN = 10.0
CUTOFF_15_MIN = 15.0
CUTOFF_10_M = CUTOFF_10_MIN * WALK_SPEED_M_PER_MIN  # 750.0 m

CROSS_SOURCE_DEDUP_M = 25.0
COARSE_GEOGROUP_AREA_M2 = 100_000.0
MIN_WALK_COMPONENT_NODES = 10

MACAO_LAT_BOUNDS = (22.10, 22.22)
MACAO_LON_BOUNDS = (113.52, 113.60)
MACAO_UTM_X_BOUNDS = (755_000.0, 772_000.0)
MACAO_UTM_Y_BOUNDS = (2_445_000.0, 2_462_000.0)

# Exact 2021 DSEC terrestrial population totals (excluding Z26 maritime: 777 total / 61 aged 65+)
DSEC_TERRESTRIAL_POP_TOTAL = 681_293
DSEC_TERRESTRIAL_POP_65PLUS = 82_751
DSEC_MARITIME_POP_TOTAL = 777
DSEC_MARITIME_POP_65PLUS = 61

# ---------------------------------------------------------------------------
# Live Data Endpoints (Unauthenticated Public APIs & Snapshots)
# ---------------------------------------------------------------------------
DSEC_ZONES_URL = (
    "https://www.dsec.gov.mo/CensusGIS/rest/services/Production2021/"
    "Census-QueryZonaBuilding-2021/MapServer/0"
)
DSEC_BUILDINGS_URL = (
    "https://www.dsec.gov.mo/CensusGIS/rest/services/Production2021/"
    "Census-QueryZonaBuilding-2021/MapServer/1"
)
CEM_EV_EN_URL = "https://ev.cem-macau.com/Ajax/GetMapData?lang=en"
CEM_EV_TC_URL = "https://ev.cem-macau.com/Ajax/GetMapData?lang=tc"
GEOFABRIK_PBF_URL = "https://download.geofabrik.de/asia/china/macau-latest.osm.pbf"
GEOFABRIK_GPKG_ZIP_URL = "https://download.geofabrik.de/asia/china/macau-latest-free.gpkg.zip"

OFFICIAL_REGISTRY_URLS: dict[str, str] = {
    "ssm_health_centres_zh": "https://www.gov.mo/zh-hant/services/ps-1547/ps-1547a/",
    "ssm_health_centres_en": "https://www.gov.mo/en/services/ps-1547/ps-1547a/",
    "ic_libraries_zh": "https://www.library.gov.mo/zh-hant/branch-libraries/public-library/macao",
    "ic_libraries_en": "https://www.library.gov.mo/en/branch-libraries/public-library/macao",
    "ic_notice_548_ilha_verde_relocation": "https://www.library.gov.mo/en/aboutus/notice/548",
    "iam_parks_and_gardens": "https://www.iam.gov.mo/c/garden/itemList",
    "mlm_lrt_routes": "https://www.mlm.com.mo/en/route.html",
    "data_gov_mo_portal": "https://data.gov.mo",
    "data_gov_mo_dsat_bus_probe": (
        "https://api.data.gov.mo/datadir/downloadSingleFile"
        "?id=e7b2e84d-3333-42f0-b676-64ce95306f0d"
    ),
}

# ---------------------------------------------------------------------------
# Verified 7 Official Parishes Mapping (Z01–Z23 -> P01–P07)
# Empirically triple-verified against DSCC land areas & OSM admin_level=6:
#   Z06 (筷子基區 / Fai Chi Kei) belongs to P01 (Fátima), 95.8% inside rel/9506156.
# ---------------------------------------------------------------------------
PARISH_INFO: dict[str, dict[str, str]] = {
    "P01": {"parish_name": "Nossa Senhora de Fátima", "parish_name_zh": "花地瑪堂區"},
    "P02": {"parish_name": "Santo António", "parish_name_zh": "花王堂區"},
    "P03": {"parish_name": "São Lázaro", "parish_name_zh": "望德堂區"},
    "P04": {"parish_name": "Sé", "parish_name_zh": "大堂區"},
    "P05": {"parish_name": "São Lourenço", "parish_name_zh": "風順堂區"},
    "P06": {"parish_name": "Nossa Senhora do Carmo", "parish_name_zh": "嘉模堂區"},
    "P07": {"parish_name": "São Francisco Xavier", "parish_name_zh": "聖方濟各堂區"},
}

ZONE_TO_PARISH: dict[str, str] = {
    "Z01": "P01",
    "Z02": "P01",
    "Z03": "P01",
    "Z04": "P01",
    "Z05": "P01",
    "Z06": "P01",
    "Z07": "P02",
    "Z08": "P02",
    "Z09": "P02",
    "Z10": "P02",
    "Z11": "P03",
    "Z12": "P03",
    "Z13": "P04",
    "Z14": "P04",
    "Z15": "P04",
    "Z16": "P05",
    "Z17": "P05",
    "Z18": "P06",
    "Z19": "P06",
    "Z20": "P06",
    "Z21": "P06",
    "Z22": "P06",
    "Z23": "P07",
}

PENINSULA_PARISH_IDS = frozenset({"P01", "P02", "P03", "P04", "P05"})
ISLANDS_PARISH_IDS = frozenset({"P06", "P07"})

# ---------------------------------------------------------------------------
# Verified Official Facility Whitelists & Exclusions
# ---------------------------------------------------------------------------
SSM_HEALTH_CENTRE_WHITELIST: dict[str, str] = {
    "328757103": "黑沙環衛生中心 Hac Sa Wan Health Centre",
    "206824015": "筷子基衛生中心 Fai Chi Kei Health Centre",
    "5469633936": "青洲衛生中心 Ilha Verde Health Centre",
    "205784460": "塔石衛生中心 Tap Seac Health Centre",
    "398850722": "海傍衛生中心 Hoi Pong Health Centre",
    "6164047586": "下環衛生中心 Praia do Manduco Health Centre",
    "7686442906": "海洋花園衛生中心 Ocean Gardens Health Centre",
    "3530373653": "湖畔嘉模衛生中心 Nossa Senhora do Carmo – Lago Health Centre",
    "2950694038": "石排灣衛生中心 Seac Pai Van Health Centre",
    "1067575197": "路環衛生站 Coloane Health Station",
    "3294161007": "氹仔老人保健站 Taipa Elderly Health Station",
}

IC_LIBRARY_WHITELIST: dict[str, str] = {
    "172301930": "澳門中央圖書館 Macao Central Library",
    "402151732": "何賢公園圖書館 Library in Ho Yin Garden",
    "279957436": "何東圖書館 Sir Robert Ho Tung Library",
    "2992379552": "下環圖書館 S. Lourenço Library",
    "399238639": "沙梨頭圖書館 Patane Library",
    "421036564": "紅街市圖書館 Red Market Library",
    "406123254": "白鴿巢公園黃營均圖書館 Wong Ieng Kuan Library in Luís de Camões Garden",
    "4074554769": "青洲坊圖書館 Bairro da Ilha Verde Library",
    "3377691823": "望廈圖書館 Mong Ha Library",
    "402296695": (
        "紀念孫中山公園黃營均圖書館 Wong Ieng Kuan Library in Dr. Sun Yat Sen Municipal Park"
    ),
    "350566600": "黑沙環公園黃營均圖書館 Wong Ieng Kuan Library in Areia Preta Urban Park",
    "328757096": (
        "黑沙環公園黃營均兒童圖書館 Wong Ieng Kuan Children's Library in Areia Preta Urban Park"
    ),
    "405678242": "氹仔圖書館 Taipa Library",
    "229184907": "路環圖書館 Coloane Library",
    "7545205378": "石排灣圖書館 Seac Pai Van Library",
}

# IC Notice 548: old 青洲圖書館 (node/4074554769) relocated to 青洲坊大廈第二座 (way/668949326)
IC_LIBRARY_RELOCATION_MAP: dict[str, str] = {
    "4074554769": "668949326",
}

PARK_EXTRA_RECREATION_WHITELIST = frozenset({"578956858"})  # 青洲河邊馬路休憩區
PARK_MEMBER_WAY_DUPLICATES = frozenset(
    {"206109394", "206109395", "206109396", "406362576", "405411409"}
)
PARK_EXCLUDED_NODE_IDS = frozenset({"8720439953"})  # FUFA indoor play centre
PARK_EXCLUDED_ACCESS = frozenset({"private", "no", "customers", "students;staff", "permit"})

BUS_STOP_EXCLUDED_IDS = frozenset({"13264387080"})  # "Duplicate of ..."
BUS_SHUTTLE_REGEX = re.compile(
    r"(?i)free\s*shuttle|shuttle\s*(?:bus|to)|免費接駁|免費穿梭|casino\s*shuttle"
)
BUS_SHUTTLE_OPERATORS = frozenset({"澳娛綜合 SJM Resorts"})

LRT_EXCLUDED_DUPLICATE_IDS = frozenset({"14330568544"})  # Duplicate Lotus Hengqin platform (79.4 m)

# Official MLM LRT line metadata keyed by OSM node_id
LRT_STATION_META: dict[str, tuple[str, str, str, bool]] = {
    "11401270262": ("媽閣", "Barra", "Taipa", True),
    "7051569813": ("海洋", "Ocean", "Taipa", True),
    "7051569804": ("馬會", "Jockey Club", "Taipa", True),
    "7051569801": ("運動場", "Stadium", "Taipa", True),
    "7051540683": ("排角", "Pai Kok", "Taipa", True),
    "7051569798": ("路氹西", "Cotai West", "Taipa", True),
    "7051569795": ("蓮花", "Lotus", "Taipa;Hengqin", True),
    "10132281747": ("協和醫院", "Union Hospital", "Taipa;Seac Pai Van", True),
    "7051569792": ("東亞運", "East Asian Games", "Taipa", True),
    "7051569789": ("路氹東", "Cotai East", "Taipa", True),
    "7051569786": ("科大", "MUST", "Taipa", True),
    "7051569807": ("機場", "Airport", "Taipa", True),
    "7051569810": ("氹仔碼頭", "Taipa Ferry Terminal", "Taipa", True),
    "12304462028": ("石排灣", "Seac Pai Van", "Seac Pai Van", True),
    "12391564890": ("橫琴", "Hengqin", "Hengqin", False),
}

# ---------------------------------------------------------------------------
# Feature Manifest & Anti-Leakage Regexes
# ---------------------------------------------------------------------------
ML_FEATURES: list[str] = [
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
]

EXCLUDED_TARGET_AND_AUDIT: list[str] = [
    "has_ev_charger",
    "ev_station_count",
    "ev_charger_capacity",
]

ANALYSIS_PREFIX_EXCLUDED: list[str] = ["acc10_", "acc15_", "lisa_"]

EV_REGEX_SPEC = re.compile(r"(?i)\b(?:ev|evse|supercharger|tesla)\b|charg|充電|充电|電動車|电动车")
EV_REGEX_EXT = re.compile(
    r"(?i)\b(?:ev|evse|supercharger|tesla|byd|nio|xpeng)\b|"
    r"charg|充電|充电|超充|電動車|电动车|電動汽車|电动汽车|停車充電|停车充电|新能源|"
    r"特斯拉|蔚來|蔚来|比亞迪|比亚迪|carreg"
)

# ---------------------------------------------------------------------------
# Shared Helper Utilities
# ---------------------------------------------------------------------------
UTC8 = timezone(timedelta(hours=8))


def utc8_now_iso() -> str:
    """Return current timestamp formatted in ISO-8601 with UTC+8 offset."""
    return datetime.now(UTC8).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file on disk."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def make_transformer(src_crs: str, dst_crs: str) -> Transformer:
    """Create a strict always_xy pyproj Transformer."""
    return Transformer.from_crs(src_crs, dst_crs, always_xy=True)


def reproject_geom(geom: Any, transformer: Transformer) -> Any:
    """Reproject a Shapely geometry cleanly without Shapely 2.2 deprecation warnings."""
    if geom is None or geom.is_empty:
        return geom

    def _transform_coords(xy: np.ndarray) -> np.ndarray:
        x, y = transformer.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([x, y])

    return shapely.transform(geom, _transform_coords)


def ensure_directories() -> None:
    """Create canonical data and output directories if absent."""
    for d in (DATA_RAW_DIR, DATA_INTERIM_DIR, DATA_FROZEN_DIR, OUTPUTS_METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)
