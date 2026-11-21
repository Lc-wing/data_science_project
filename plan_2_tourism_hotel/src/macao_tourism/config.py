"""Paths, source registry, and fixed analysis constants for Plan 2 (MacaoTourismHotel)."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DATA_FROZEN_DIR = PROJECT_ROOT / "data" / "frozen"
OUTPUTS_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUTS_METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
OUTPUTS_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

RAW_DSEC_TSDB_DIR = DATA_RAW_DIR / "dsec_tsdb"
RAW_DSEC_RELEASE_DIR = DATA_RAW_DIR / "dsec_releases"
RAW_CN_HOLIDAY_DIR = DATA_RAW_DIR / "cn_holiday_notices"
RAW_SMG_DIR = DATA_RAW_DIR / "smg_tropical_cyclone"
# The SMG case-report PDFs are about 160 MB in total. They are cached here, outside git;
# the repository keeps their URL, size, SHA-256, and the extracted signal-table text.
SMG_PDF_CACHE_DIR = PROJECT_ROOT / ".cache" / "smg_tc_pdf"
PROVENANCE_PATH = DATA_RAW_DIR / "data_provenance.json"
FREEZE_MANIFEST_PATH = DATA_FROZEN_DIR / "FREEZE_MANIFEST.json"

RANDOM_SEED = 42
USER_AGENT = "Mozilla/5.0 (CS482 coursework; Plan 2 data download)"

# ---------------------------------------------------------------------------
# Sample window (plan section 2.2). Raw downloads start earlier so that 12-month
# lags exist for the first panel month.
# ---------------------------------------------------------------------------
PANEL_START = "2010-01"
PANEL_END = "2026-09"
DOWNLOAD_FROM_YEAR = 2008
DOWNLOAD_TO_YEAR = 2026

TIERS: tuple[str, ...] = ("5star", "4star", "3star", "2star")

# Period flags (plan section 2.2). January 2020 is a transition month: it belongs to
# neither the pre-pandemic nor the pandemic period and is left out of the main analysis.
PERIODS: dict[str, tuple[str, str]] = {
    "pre_pandemic": ("2010-01", "2019-12"),
    "transition_2020_01": ("2020-01", "2020-01"),
    "pandemic": ("2020-02", "2022-12"),
    "post_pandemic": ("2023-01", PANEL_END),
}
MAIN_SAMPLE_PERIODS: tuple[str, ...] = ("pre_pandemic", "post_pandemic")

# Rolling-origin backtest (plan section 2.3). An origin is the last observed month;
# the target is the month after it.
TEST_ORIGIN_FIRST = "2024-07"
TEST_ORIGIN_LAST = "2026-08"
# Orders and hyperparameters are chosen on these targets only, all before the first
# test origin.
VALIDATION_TARGET_WINDOWS: tuple[tuple[str, str], ...] = (
    ("2018-01", "2019-12"),
    ("2023-07", "2024-06"),
)
FORECAST_LAGS: tuple[int, ...] = (1, 2, 3, 12)


def period_label(month: str) -> str:
    for label, (start, end) in PERIODS.items():
        if start <= month <= end:
            return label
    raise ValueError(f"{month} is outside the panel window")


# ---------------------------------------------------------------------------
# DSEC (Statistics and Census Service) endpoints
# ---------------------------------------------------------------------------
DSEC_BASE = "https://www.dsec.gov.mo"
DSEC_VISITOR_PAGE = f"{DSEC_BASE}/en-US/Statistic?id=402"
DSEC_HOTEL_PAGE = f"{DSEC_BASE}/en-US/Statistic?id=403"
DSEC_TSDB_PAGE = f"{DSEC_BASE}/ts/#!/step1/en-US"
DSEC_TSDB_CHILDREN_URL = f"{DSEC_BASE}/TimeSeriesApi/App/Indicatorv3"
DSEC_TSDB_VALUES_URL = f"{DSEC_BASE}/TimeSeriesApi/App/IndicatorValue/LatestSameEndPeriodv3"
DSEC_RELEASES_INDEX_URL = f"{DSEC_BASE}/StatisticsJSON/AllReleases_en-US_TourismAndServices.json"
DSEC_RELEASE_SECTIONS: dict[str, str] = {
    "visitor_arrivals": "VisitorArrivals",
    "hotel_occupancy": "PackageToursAndHotelOccupancyRate",
}


@dataclass(frozen=True)
class Indicator:
    """One monthly series in the DSEC Time Series Database."""

    key: str
    indicator_id: int
    parent_id: int
    label_en: str
    unit_en: str
    group: str
    dsec_page: str


def _hotel(
    key: str, indicator_id: int, parent_id: int, label: str, unit: str, group: str
) -> Indicator:
    return Indicator(key, indicator_id, parent_id, label, unit, group, DSEC_HOTEL_PAGE)


def _visitor(
    key: str, indicator_id: int, parent_id: int, label: str, unit: str = "No."
) -> Indicator:
    group = "visitor_stay" if unit == "Day" else "visitor_arrivals"
    return Indicator(key, indicator_id, parent_id, label, unit, group, DSEC_VISITOR_PAGE)


def _tiered(
    prefix: str, ids: tuple[int, int, int, int], parent: int, unit: str, group: str
) -> list[Indicator]:
    stars = ("5", "4", "3", "2")
    return [
        _hotel(f"{prefix}_{star}star", indicator_id, parent, f"{star}-star hotels", unit, group)
        for star, indicator_id in zip(stars, ids, strict=True)
    ]


_BY_DOCUMENT = "By Issuing Place of Travel Document"

INDICATORS: tuple[Indicator, ...] = (
    *_tiered("occupancy", (13011, 13012, 13013, 13014), 13016, "%", "hotel_occupancy"),
    _hotel("occupancy_hotels", 13016, 13005, "Hotels", "%", "hotel_occupancy"),
    _hotel("occupancy_guesthouses", 13015, 13005, "Guest-houses", "%", "hotel_occupancy"),
    _hotel(
        "occupancy_economical",
        67592,
        13005,
        "Economical accommodation establishments",
        "%",
        "hotel_occupancy",
    ),
    *_tiered("rooms", (25008, 25009, 25010, 25011), 25007, "No.", "hotel_supply"),
    _hotel("rooms_hotels", 25007, 13004, "Hotels", "No.", "hotel_supply"),
    *_tiered("establishments", (25002, 25003, 25004, 25005), 25001, "No.", "hotel_supply"),
    _hotel("establishments_hotels", 25001, 13003, "Hotels", "No.", "hotel_supply"),
    *_tiered("guests", (25093, 25122, 25151, 25180), 25063, "No.", "hotel_guests"),
    _hotel("guests_hotels", 25064, 25063, "Total", "No.", "hotel_guests"),
    _hotel("package_tour_visitors", 26001, 13009, "Grand total", "No.", "package_tours"),
    _visitor("visitors_total", 14040, 60397, _BY_DOCUMENT),
    _visitor("visitors_same_day", 14272, 60395, _BY_DOCUMENT),
    _visitor("visitors_overnight", 57251, 60396, _BY_DOCUMENT),
    _visitor("visitors_mainland", 14042, 14040, "Chinese mainland"),
    _visitor("visitors_hong_kong", 14043, 14040, "Hong Kong SAR"),
    _visitor("visitors_taiwan", 14044, 14040, "Taiwan region"),
    _visitor("visitors_by_sea", 14024, 14023, "By sea"),
    _visitor("visitors_by_land", 14028, 14023, "By land"),
    _visitor("visitors_by_air", 14032, 14023, "By air"),
    _visitor(
        "stay_all_visitors", 14302, 14301, "Visitors by Issuing Place of Travel Document", "Day"
    ),
    _visitor(
        "stay_overnight_visitors",
        14330,
        14301,
        "Overnight Visitors by Issuing Place of Travel Document",
        "Day",
    ),
)

INDICATOR_BY_KEY: dict[str, Indicator] = {ind.key: ind for ind in INDICATORS}

# ---------------------------------------------------------------------------
# State Council holiday notices (gov.cn). One notice per calendar year, plus the
# three later notices that changed a published schedule.
# ---------------------------------------------------------------------------
CN_HOLIDAY_NOTICES: dict[int, str] = {
    2010: "https://www.gov.cn/zwgk/2009-12/08/content_1482691.htm",
    2011: "https://www.gov.cn/zwgk/2010-12/10/content_1762643.htm",
    2012: "https://www.gov.cn/zwgk/2011-12/06/content_2012097.htm",
    2013: "https://www.gov.cn/zwgk/2012-12/10/content_2286598.htm",
    2014: "https://www.gov.cn/zwgk/2013-12/11/content_2546204.htm",
    2015: "https://www.gov.cn/zhengce/content/2014-12/16/content_9302.htm",
    2016: "https://www.gov.cn/zhengce/content/2015-12/10/content_10394.htm",
    2017: "https://www.gov.cn/zhengce/content/2016-12/01/content_5141603.htm",
    2018: "https://www.gov.cn/zhengce/content/2017-11/30/content_5243579.htm",
    2019: "https://www.gov.cn/zhengce/content/2018-12/06/content_5346276.htm",
    2020: "https://www.gov.cn/zhengce/content/2019-11/21/content_5454164.htm",
    2021: "https://www.gov.cn/zhengce/content/2020-11/25/content_5564127.htm",
    2022: "https://www.gov.cn/zhengce/content/2021-10/25/content_5644835.htm",
    2023: "https://www.gov.cn/zhengce/content/2022-12/08/content_5730844.htm",
    2024: "https://www.gov.cn/zhengce/content/202310/content_6911527.htm",
    2025: "https://www.gov.cn/zhengce/content/202411/content_6986382.htm",
    2026: "https://www.gov.cn/zhengce/content/202511/content_7047090.htm",
}
CN_HOLIDAY_AMENDMENTS: dict[str, str] = {
    "2015_victory_day": "https://www.gov.cn/zhengce/content/2015-05/13/content_9742.htm",
    "2019_labour_day": "https://www.gov.cn/zhengce/content/2019-03/22/content_5375877.htm",
    "2020_spring_festival_extension": (
        "https://www.gov.cn/zhengce/content/2020-01/27/content_5472352.htm"
    ),
}
GOLDEN_WEEK_HOLIDAYS: frozenset[str] = frozenset({"spring_festival", "national_day"})

# ---------------------------------------------------------------------------
# SMG (Macao Meteorological and Geophysical Bureau) tropical cyclone sources
# ---------------------------------------------------------------------------
SMG_REPORT_LIST_URL = "https://new-api.smg.gov.mo/reportlist?folder=/pdf/tc/{year}"
SMG_REPORT_PDF_URL = "https://cms.smg.gov.mo/uploads/sync/pdf/tc/{year}/{name}"
SMG_REPORT_NAMES_URL = "https://cms.smg.gov.mo/uploads/sync/pdf/tc/typhoon_report_mapping.json"
SMG_ANNUAL_SIGNAL_STATS_URL = "https://cms.smg.gov.mo/en/api/sitecontent/259"
SMG_MONTHLY_SUMMARY_LIST_URL = (
    "https://new-api.smg.gov.mo/reportlist?folder=/pdf/CLI/CLI_Month_report"
)
SMG_MONTHLY_SUMMARY_PDF_URL = "https://cms.smg.gov.mo/uploads/sync/pdf/CLI/CLI_Month_report/{name}"
SMG_REPORT_PAGE = "https://www.smg.gov.mo/en/subpage/355/report/typhoon-yearly-report"
SMG_ANNUAL_STATS_PAGE = "https://www.smg.gov.mo/en/subpage/355/page/259"
SMG_FIRST_YEAR = 2010
SMG_LAST_YEAR = 2026
# A "typhoon day" is a calendar day on which signal No. 8 or higher was in force.
TYPHOON_DAY_MIN_SIGNAL = 8

UTC8 = timezone(timedelta(hours=8))


def utc8_now_iso() -> str:
    return datetime.now(UTC8).isoformat(timespec="seconds")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
