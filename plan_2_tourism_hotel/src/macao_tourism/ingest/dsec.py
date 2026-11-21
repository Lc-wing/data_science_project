"""DSEC monthly series: Time Series Database downloads and the latest Excel releases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd

from macao_tourism.config import (
    DOWNLOAD_FROM_YEAR,
    DOWNLOAD_TO_YEAR,
    DSEC_BASE,
    DSEC_RELEASE_SECTIONS,
    DSEC_RELEASES_INDEX_URL,
    DSEC_TSDB_CHILDREN_URL,
    DSEC_TSDB_VALUES_URL,
    INDICATORS,
    RAW_DSEC_RELEASE_DIR,
    RAW_DSEC_TSDB_DIR,
    Indicator,
)
from macao_tourism.ingest.http import Provenance, fetch_bytes, save_snapshot

ZERO_REMARK = "Absolute value equals zero"
MISSING_REMARK = "No figure provided"
REVISED_REMARK = "Revised figures"
RELEASES_INDEX_NAME = "AllReleases_en-US_TourismAndServices.json"


def _loads(payload: bytes) -> Any:
    return json.loads(payload.decode("utf-8-sig"))


def children_path(parent_id: int, raw_dir: Path = RAW_DSEC_TSDB_DIR) -> Path:
    return raw_dir / f"children_{parent_id}.json"


def values_path(group: str, raw_dir: Path = RAW_DSEC_TSDB_DIR) -> Path:
    return raw_dir / f"values_monthly_{group}.json"


def values_request_body(indicator_ids: list[int]) -> dict[str, Any]:
    return {
        "indicator_ids": indicator_ids,
        "language": "en-US",
        "types": ["VAL"] * len(indicator_ids),
        "dataPeriods": ["MONTHLY"],
        "fromYear": DOWNLOAD_FROM_YEAR,
        "toYear": DOWNLOAD_TO_YEAR,
    }


def download_tsdb(provenance: Provenance, raw_dir: Path = RAW_DSEC_TSDB_DIR) -> None:
    """Save the indicator metadata and the monthly values of every registered series."""
    for parent_id in sorted({ind.parent_id for ind in INDICATORS}):
        url = f"{DSEC_TSDB_CHILDREN_URL}/{parent_id}"
        path = children_path(parent_id, raw_dir)
        save_snapshot(path, fetch_bytes(url))
        provenance.record(
            path,
            url=url,
            source="DSEC Time Series Database",
            description=f"Metadata of the indicators under node {parent_id}",
        )
    for group in sorted({ind.group for ind in INDICATORS}):
        ids = [ind.indicator_id for ind in INDICATORS if ind.group == group]
        body = values_request_body(ids)
        path = values_path(group, raw_dir)
        save_snapshot(path, fetch_bytes(DSEC_TSDB_VALUES_URL, json_body=body, timeout=240.0))
        provenance.record(
            path,
            url=DSEC_TSDB_VALUES_URL,
            source="DSEC Time Series Database",
            description=f"Monthly values, group {group}",
            request_body=body,
        )


def load_indicator_meta(raw_dir: Path = RAW_DSEC_TSDB_DIR) -> dict[int, dict[str, Any]]:
    """Metadata rows keyed by indicator id, read from the saved child listings."""
    meta: dict[int, dict[str, Any]] = {}
    for parent_id in sorted({ind.parent_id for ind in INDICATORS}):
        payload = _loads(children_path(parent_id, raw_dir).read_bytes())
        for row in payload["Value"]:
            meta[int(row["IndicatorID"])] = row
    return meta


def verify_registry(meta: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Check each registered id against DSEC's own label, unit, parent, and frequency.

    Raises ``ValueError`` on any disagreement, so a wrong id cannot enter the panel.
    """
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for ind in INDICATORS:
        row = meta.get(ind.indicator_id)
        if row is None:
            problems.append(f"{ind.key}: id {ind.indicator_id} not under node {ind.parent_id}")
            continue
        label = str(row["DescriptionEngl"]).strip()
        unit = str(row["UnitLabelEngl"]).strip()
        parent = int(row["Parent"])
        monthly = str(row["MonthlyAvailable"]) == "True"
        if label != ind.label_en:
            problems.append(f"{ind.key}: label {label!r} != {ind.label_en!r}")
        if unit != ind.unit_en:
            problems.append(f"{ind.key}: unit {unit!r} != {ind.unit_en!r}")
        if parent != ind.parent_id:
            problems.append(f"{ind.key}: parent {parent} != {ind.parent_id}")
        if not monthly:
            problems.append(f"{ind.key}: DSEC does not publish this series monthly")
        rows.append(
            {
                "key": ind.key,
                "indicator_id": ind.indicator_id,
                "parent_id": parent,
                "label_en": label,
                "label_zh": str(row["Description"]).strip(),
                "path_en": str(row["IndicatorPath_enUS"]).strip(),
                "unit_en": unit,
                "dsec_min_year": row["minYear"],
                "dsec_max_year": row["maxYear"],
            }
        )
    if problems:
        raise ValueError("DSEC indicator registry mismatch: " + "; ".join(problems))
    return rows


def parse_values_payload(payload: dict[str, Any], indicators: list[Indicator]) -> pd.DataFrame:
    """Flatten one values response into rows of (key, period, value, remarks)."""
    if payload.get("Status") != "OK":
        raise ValueError(f"DSEC values response status is {payload.get('Status')!r}")
    by_id = {ind.indicator_id: ind for ind in indicators}
    seen: set[int] = set()
    rows: list[dict[str, Any]] = []
    for block in payload["Value"]:
        indicator_id = int(block["indicatorId"])
        if indicator_id not in by_id:
            raise ValueError(f"Unexpected indicator id {indicator_id} in DSEC response")
        seen.add(indicator_id)
        ind = by_id[indicator_id]
        for obs in block["dsecIndicatorData"]:
            month = obs.get("PeriodID")
            if month is None or not 1 <= int(month) <= 12:
                raise ValueError(f"{ind.key}: non-monthly row {obs.get('ReferencePeriod')!r}")
            remarks = sorted(
                str(r["DescriptionEngl"]).strip() for r in (obs.get("AssociatedRemark") or []) if r
            )
            value = obs.get("IndicatorValue")
            is_zero = ZERO_REMARK in remarks
            if value is None and is_zero:
                value = 0.0
            last_update = str(obs.get("LastUpdateDate") or "")[:10]
            rows.append(
                {
                    "key": ind.key,
                    "indicator_id": indicator_id,
                    "period": f"{int(obs['Year']):04d}-{int(month):02d}",
                    "value": None if value is None else float(value),
                    "zero_by_remark": bool(is_zero and obs.get("IndicatorValue") is None),
                    "revised": REVISED_REMARK in remarks,
                    "remarks": " | ".join(remarks),
                    "dsec_last_update": "" if last_update.startswith("0001") else last_update,
                }
            )
    missing = sorted(set(by_id) - seen)
    if missing:
        raise ValueError(f"DSEC response is missing indicator ids {missing}")
    frame = pd.DataFrame(rows)
    if frame.duplicated(["key", "period"]).any():
        raise ValueError("Duplicate (key, period) rows in DSEC response")
    return frame


def load_tsdb_long(raw_dir: Path = RAW_DSEC_TSDB_DIR) -> pd.DataFrame:
    """All registered monthly series in long form, sorted by key and period."""
    frames = []
    for group in sorted({ind.group for ind in INDICATORS}):
        members = [ind for ind in INDICATORS if ind.group == group]
        payload = _loads(values_path(group, raw_dir).read_bytes())
        frames.append(parse_values_payload(payload, members))
    frame = pd.concat(frames, ignore_index=True)
    return frame.sort_values(["key", "period"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Latest monthly Excel releases (plan 2.1: keep the raw .xlsx and the download date)
# ---------------------------------------------------------------------------
def latest_monthly_xlsx(index: dict[str, Any], section: str) -> dict[str, Any]:
    """Newest monthly release of ``section`` that carries an .xlsx statistical table."""
    releases = index[section]["Containers"][section]["Publication"]["Monthly"]
    candidates = []
    for release in releases:
        for attachment in release.get("File") or []:
            if attachment.get("Type") == ".xlsx":
                candidates.append(
                    {
                        "year": int(release["Year"]),
                        "month": int(release["Period"]),
                        "release_id": int(release["ReleaseID"]),
                        "url": DSEC_BASE + attachment["Link"],
                        "file_name": Path(attachment["Link"]).stem + ".xlsx",
                    }
                )
    if not candidates:
        raise ValueError(f"No monthly .xlsx release listed for {section}")
    return max(candidates, key=lambda item: (item["year"], item["month"]))


def download_releases(
    provenance: Provenance, raw_dir: Path = RAW_DSEC_RELEASE_DIR
) -> dict[str, dict[str, Any]]:
    index_bytes = fetch_bytes(DSEC_RELEASES_INDEX_URL, timeout=120.0)
    index_path = raw_dir / RELEASES_INDEX_NAME
    save_snapshot(index_path, index_bytes)
    provenance.record(
        index_path,
        url=DSEC_RELEASES_INDEX_URL,
        source="DSEC",
        description="Index of all tourism releases (publication list behind id=402 and id=403)",
    )
    index = _loads(index_bytes)
    chosen: dict[str, dict[str, Any]] = {}
    for name, section in DSEC_RELEASE_SECTIONS.items():
        release = latest_monthly_xlsx(index, section)
        path = raw_dir / release["file_name"]
        save_snapshot(path, fetch_bytes(release["url"], timeout=240.0))
        provenance.record(
            path,
            url=release["url"],
            source="DSEC",
            description=(
                f"Statistical tables, {section}, {release['year']}-{release['month']:02d} "
                f"(release {release['release_id']})"
            ),
        )
        chosen[name] = release
    return chosen


def selected_releases(raw_dir: Path = RAW_DSEC_RELEASE_DIR) -> dict[str, dict[str, Any]]:
    index = _loads((raw_dir / RELEASES_INDEX_NAME).read_bytes())
    return {
        name: latest_monthly_xlsx(index, section) for name, section in DSEC_RELEASE_SECTIONS.items()
    }


def _numeric_cells(row: tuple[Any, ...]) -> list[float]:
    return [float(v) for v in row if isinstance(v, int | float) and not isinstance(v, bool)]


def _find_row(sheet: Any, english_label: str) -> list[float]:
    for row in sheet.iter_rows(values_only=True):
        texts = [str(v).strip() for v in row if isinstance(v, str)]
        if english_label in texts:
            return _numeric_cells(row)
    raise ValueError(f"Row {english_label!r} not found in sheet {sheet.title!r}")


def release_crosscheck_values(raw_dir: Path = RAW_DSEC_RELEASE_DIR) -> pd.DataFrame:
    """Values printed in the two latest Excel releases, for the release month.

    In every table used here the first numeric cell is the same month one year
    earlier and the second is the release month. Rooms and guests are printed in
    thousands with one decimal.
    """
    releases = selected_releases(raw_dir)
    rows: list[dict[str, Any]] = []

    hotel = releases["hotel_occupancy"]
    book = openpyxl.load_workbook(raw_dir / hotel["file_name"], read_only=True, data_only=True)
    period = f"{hotel['year']:04d}-{hotel['month']:02d}"
    for star in ("5", "4", "3", "2"):
        supply = _find_row(book["5"], f"{star}-star")
        demand = _find_row(book["6"], f"{star}-star")
        rows += [
            _check_row(f"establishments_{star}star", period, supply[1], 1.0, hotel),
            _check_row(f"rooms_{star}star", period, supply[4], 1000.0, hotel),
            _check_row(f"occupancy_{star}star", period, demand[1], 1.0, hotel),
            _check_row(f"guests_{star}star", period, demand[4], 1000.0, hotel),
        ]
    book.close()

    visitor = releases["visitor_arrivals"]
    book = openpyxl.load_workbook(raw_dir / visitor["file_name"], read_only=True, data_only=True)
    period = f"{visitor['year']:04d}-{visitor['month']:02d}"
    for key, sheet, label in (
        ("visitors_total", "1", "Total"),
        ("visitors_mainland", "1", "Chinese mainland"),
        ("visitors_hong_kong", "1", "Hong Kong Special Administrative Region"),
        ("visitors_taiwan", "1", "Taiwan region"),
        ("visitors_same_day", "5", "Total"),
        ("visitors_overnight", "9", "Total"),
    ):
        rows.append(_check_row(key, period, _find_row(book[sheet], label)[1], 1.0, visitor))
    book.close()
    return pd.DataFrame(rows)


def release_table_notes(raw_dir: Path = RAW_DSEC_RELEASE_DIR) -> list[str]:
    """English footnotes under the two star-class tables of the latest hotel release."""
    hotel = selected_releases(raw_dir)["hotel_occupancy"]
    book = openpyxl.load_workbook(raw_dir / hotel["file_name"], read_only=True, data_only=True)
    notes: set[str] = set()
    for sheet in ("5", "6"):
        for row in book[sheet].iter_rows(values_only=True):
            notes.update(
                " ".join(v.split()) for v in row if isinstance(v, str) and v.startswith("Note")
            )
    book.close()
    return sorted(notes)


def _check_row(
    key: str, period: str, printed: float, scale: float, release: dict[str, Any]
) -> dict[str, Any]:
    return {
        "key": key,
        "period": period,
        "xlsx_value": printed,
        "xlsx_scale": scale,
        "xlsx_file": release["file_name"],
    }
