"""SMG tropical cyclone warning signals, read from the bureau's own case reports.

Each tropical cyclone that led to a signal in Macao has a report PDF with a table of
the time every signal was issued and the time all signals were cancelled. The PDFs are
cached outside git; the repository keeps their URL, size, SHA-256, and extracted text.
Two independent SMG publications check the result: the annual signal table
(1968 to the last full year) and the monthly climate summaries of the current year.
"""

from __future__ import annotations

import html
import io
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from macao_tourism.config import (
    PANEL_END,
    RAW_SMG_DIR,
    SMG_ANNUAL_SIGNAL_STATS_URL,
    SMG_FIRST_YEAR,
    SMG_LAST_YEAR,
    SMG_MONTHLY_SUMMARY_LIST_URL,
    SMG_MONTHLY_SUMMARY_PDF_URL,
    SMG_PDF_CACHE_DIR,
    SMG_REPORT_LIST_URL,
    SMG_REPORT_NAMES_URL,
    SMG_REPORT_PDF_URL,
    sha256_bytes,
)
from macao_tourism.ingest.http import Provenance, fetch_bytes, save_snapshot

SIGNAL_LEVELS: dict[str, int] = {"一": 1, "三": 3, "八": 8, "九": 9, "十": 10}
_SIGNAL_RE = re.compile(r"([一三八九十])號(?:東北|東南|西北|西南)?風球")
_SIGNAL_WORD_RE = re.compile(r"風球|熱帶氣旋.*信號")
_CANCEL_RE = re.compile(r"除下所有風球|取消所有熱帶氣旋(?:警告)?信號")
_EVENT_RE = re.compile(
    r"(\d{1,2})\s*月\s*(\d{1,2})\s*日\s+(\d{1,2})\s*[:：]\s*(\d{2})\s+(\S.*?)\s*$"
)
_REPORT_NAME_RE = re.compile(r"^c_(\d{2})(.+)\.pdf$")
_SUMMARY_NAME_RE = re.compile(r"^c_resumo_(\d{4})(\d{2})\.pdf$")
_ANNUAL_LEVELS = (1, 3, 8, 9, 10)


@dataclass(frozen=True)
class SignalEvent:
    when: datetime
    level: int
    text: str


def report_list_path(year: int, raw_dir: Path = RAW_SMG_DIR) -> Path:
    return raw_dir / f"report_list_{year}.json"


def report_text_path(year: int, name: str, raw_dir: Path = RAW_SMG_DIR) -> Path:
    return raw_dir / "report_text" / str(year) / (Path(name).stem + ".txt")


def summary_text_path(name: str, raw_dir: Path = RAW_SMG_DIR) -> Path:
    return raw_dir / "monthly_summary_text" / (Path(name).stem + ".txt")


def chinese_reports(listing: list[str]) -> list[str]:
    return sorted(name for name in listing if _REPORT_NAME_RE.match(name))


def pdf_to_text(payload: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(payload))
    return "\n".join((page.extract_text() or "") for page in reader.pages) + "\n"


def _fetch_pdf_text(
    url: str, cache_path: Path, text_path: Path, timeout: float = 300.0
) -> dict[str, Any]:
    cached = cache_path.is_file()
    if cached:
        payload = cache_path.read_bytes()
    else:
        payload = fetch_bytes(url, timeout=timeout)
        save_snapshot(cache_path, payload)
    if not payload.startswith(b"%PDF"):
        cache_path.unlink(missing_ok=True)
        raise ValueError(f"{url} did not return a PDF")
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(pdf_to_text(payload), encoding="utf-8")
    return {"cached": cached, "pdf_sha256": sha256_bytes(payload), "pdf_bytes": len(payload)}


def download_smg(
    provenance: Provenance,
    raw_dir: Path = RAW_SMG_DIR,
    cache_dir: Path = SMG_PDF_CACHE_DIR,
    workers: int = 6,
) -> None:
    """Save report lists, name mapping, annual table, and the text of every report."""
    for path, url, what in (
        (raw_dir / "typhoon_report_mapping.json", SMG_REPORT_NAMES_URL, "Report code to name"),
        (
            raw_dir / "annual_signal_stats.json",
            SMG_ANNUAL_SIGNAL_STATS_URL,
            "Annual frequency of each tropical cyclone signal since 1968",
        ),
        (
            raw_dir / "monthly_summary_list.json",
            SMG_MONTHLY_SUMMARY_LIST_URL,
            "List of monthly climate summary PDFs",
        ),
    ):
        save_snapshot(path, fetch_bytes(url, timeout=120.0))
        provenance.record(path, url=url, source="SMG", description=what)

    jobs: list[tuple[str, Path, Path, str]] = []
    for year in range(SMG_FIRST_YEAR, SMG_LAST_YEAR + 1):
        url = SMG_REPORT_LIST_URL.format(year=year)
        path = report_list_path(year, raw_dir)
        save_snapshot(path, fetch_bytes(url, timeout=120.0))
        provenance.record(
            path, url=url, source="SMG", description=f"List of tropical cyclone reports, {year}"
        )
        for name in chinese_reports(json.loads(path.read_text(encoding="utf-8"))):
            jobs.append(
                (
                    SMG_REPORT_PDF_URL.format(year=year, name=name),
                    cache_dir / str(year) / name,
                    report_text_path(year, name, raw_dir),
                    f"Tropical cyclone report {name} ({year}), text extracted with pypdf",
                )
            )
    for name in current_year_summaries(raw_dir):
        jobs.append(
            (
                SMG_MONTHLY_SUMMARY_PDF_URL.format(name=name),
                cache_dir / "monthly_summary" / name,
                summary_text_path(name, raw_dir),
                f"Monthly climate summary {name}, text extracted with pypdf",
            )
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda job: _fetch_pdf_text(job[0], job[1], job[2]), jobs))
    for (url, _cache, text_path, what), info in zip(jobs, results, strict=True):
        pdf = {"pdf_sha256": info["pdf_sha256"], "pdf_bytes": info["pdf_bytes"]}
        earlier = provenance.downloaded_at(text_path, **pdf) if info["cached"] else None
        provenance.record(
            text_path, url=url, source="SMG", description=what, extra=pdf, downloaded_at=earlier
        )


def current_year_summaries(raw_dir: Path = RAW_SMG_DIR) -> list[str]:
    """Monthly summaries of the years the annual table does not cover yet."""
    listing = json.loads((raw_dir / "monthly_summary_list.json").read_text(encoding="utf-8"))
    last_annual_year = max(load_annual_stats(raw_dir))
    names = []
    for name in sorted(listing):
        match = _SUMMARY_NAME_RE.match(name)
        if match and int(match.group(1)) > last_annual_year:
            if f"{match.group(1)}-{match.group(2)}" <= PANEL_END:
                names.append(name)
    return names


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_signal_lines(text: str, year: int) -> list[SignalEvent]:
    """Every 'M月D日 HH:MM <signal or cancellation>' line, deduplicated, in time order."""
    events: set[SignalEvent] = set()
    # Some PDFs encode 北 and other characters as CJK compatibility ideographs.
    for line in unicodedata.normalize("NFKC", text).splitlines():
        match = _EVENT_RE.search(line)
        if not match:
            continue
        tail = match.group(5)
        signal = _SIGNAL_RE.search(tail)
        cancel = _CANCEL_RE.search(tail)
        if not signal and not cancel:
            if _SIGNAL_WORD_RE.search(tail):
                raise ValueError(f"Unreadable tropical cyclone signal record: {line.strip()!r}")
            continue
        month, day, hour, minute = (int(match.group(k)) for k in range(1, 5))
        base = datetime(year, month, day) + timedelta(hours=hour, minutes=minute)
        level = SIGNAL_LEVELS[signal.group(1)] if signal else 0
        events.add(SignalEvent(base, level, (signal or cancel).group(0)))
    return sorted(events, key=lambda event: (event.when, event.level == 0))


def validate_episodes(events: list[SignalEvent], label: str) -> None:
    """A report must read: signal(s) ..., cancellation, with strictly increasing times."""
    if not events:
        raise ValueError(f"{label}: no signal record found")
    if events[0].level == 0 or events[-1].level != 0:
        raise ValueError(f"{label}: record does not start with a signal and end with a cancel")
    for earlier, later in zip(events, events[1:], strict=False):
        if later.when <= earlier.when:
            raise ValueError(f"{label}: two records share the time {later.when}")
        if earlier.level == 0 and later.level == 0:
            raise ValueError(f"{label}: two cancellations in a row")


def load_report_names(raw_dir: Path = RAW_SMG_DIR) -> dict[str, dict[str, str]]:
    return json.loads((raw_dir / "typhoon_report_mapping.json").read_text(encoding="utf-8"))


def load_signal_events(raw_dir: Path = RAW_SMG_DIR) -> pd.DataFrame:
    names = load_report_names(raw_dir)
    rows: list[dict[str, Any]] = []
    for year in range(SMG_FIRST_YEAR, SMG_LAST_YEAR + 1):
        listing = json.loads(report_list_path(year, raw_dir).read_text(encoding="utf-8"))
        for name in chinese_reports(listing):
            text = report_text_path(year, name, raw_dir).read_text(encoding="utf-8")
            events = parse_signal_lines(text, year)
            validate_episodes(events, f"{year}/{name}")
            code = name[len("c") : -len(".pdf")]
            label = names.get(code, {})
            for event in events:
                rows.append(
                    {
                        "year": year,
                        "report": name,
                        "name_en": label.get("disp_name_en", ""),
                        "name_zh": label.get("disp_name_zh", ""),
                        "issued_at": event.when.isoformat(timespec="minutes"),
                        "signal": event.level,
                        "signal_text": event.text,
                    }
                )
    frame = pd.DataFrame(rows).sort_values(["issued_at", "report"]).reset_index(drop=True)
    _check_no_overlap(frame)
    return frame


def _check_no_overlap(events: pd.DataFrame) -> None:
    """Signals of two reports must not be in force at the same time."""
    spans = (
        events.groupby("report")["issued_at"].agg(["min", "max"]).sort_values("min").reset_index()
    )
    for k in range(1, len(spans)):
        if spans.loc[k, "min"] < spans.loc[k - 1, "max"]:
            raise ValueError(
                f"Signal periods overlap: {spans.loc[k - 1, 'report']} and {spans.loc[k, 'report']}"
            )


def signal_intervals(events: pd.DataFrame) -> pd.DataFrame:
    """One row per stretch of time during which one signal was in force."""
    rows: list[dict[str, Any]] = []
    for report, group in events.groupby("report", sort=False):
        records = group.sort_values("issued_at").to_dict("records")
        for current, following in zip(records, records[1:], strict=False):
            if current["signal"] == 0:
                continue
            rows.append(
                {
                    "report": report,
                    "start": datetime.fromisoformat(current["issued_at"]),
                    "end": datetime.fromisoformat(following["issued_at"]),
                    "signal": int(current["signal"]),
                }
            )
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)


def daily_signals(intervals: pd.DataFrame) -> pd.DataFrame:
    """Calendar days with a signal in force: highest signal and hours at No. 8 or above."""
    by_day: dict[date, dict[str, Any]] = {}
    for row in intervals.to_dict("records"):
        cursor = row["start"]
        while cursor < row["end"]:
            midnight = datetime.combine(cursor.date() + timedelta(days=1), datetime.min.time())
            stop = min(midnight, row["end"])
            entry = by_day.setdefault(
                cursor.date(), {"max_signal": 0, "hours_signal8plus": 0.0, "reports": set()}
            )
            entry["max_signal"] = max(entry["max_signal"], row["signal"])
            if row["signal"] >= 8:
                entry["hours_signal8plus"] += (stop - cursor).total_seconds() / 3600.0
            entry["reports"].add(row["report"])
            cursor = stop
    return pd.DataFrame(
        [
            {
                "date": day.isoformat(),
                "max_signal": entry["max_signal"],
                "hours_signal8plus": round(entry["hours_signal8plus"], 4),
                "reports": "+".join(sorted(entry["reports"])),
            }
            for day, entry in sorted(by_day.items())
        ]
    )


def monthly_signal_days(daily: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    months = [str(m) for m in pd.period_range(start, end, freq="M")]
    month_of = daily["date"].str.slice(0, 7)
    rows = []
    for month in months:
        days = daily[month_of == month]
        rows.append(
            {
                "period": month,
                "typhoon_days_signal8plus": int((days["max_signal"] >= 8).sum()),
                "signal3plus_days": int((days["max_signal"] >= 3).sum()),
                "signal1plus_days": int((days["max_signal"] >= 1).sum()),
                "hours_signal8plus": round(float(days["hours_signal8plus"].sum()), 4),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-checks against other SMG publications
# ---------------------------------------------------------------------------
def load_annual_stats(raw_dir: Path = RAW_SMG_DIR) -> dict[int, dict[str, Any]]:
    """SMG's own yearly count of tropical cyclones that reached each signal."""
    payload = json.loads((raw_dir / "annual_signal_stats.json").read_text(encoding="utf-8"))
    content = payload[0]["translations"]["en"]["content"]
    stats: dict[int, dict[str, Any]] = {}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", content, flags=re.S):
        cells = [
            html.unescape(re.sub(r"<[^>]+>", "", cell)).strip()
            for cell in re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.S)
        ]
        if len(cells) != 11 or not re.fullmatch(r"\d{4}", cells[0]):
            continue
        counts = {}
        for level, cell in zip(_ANNUAL_LEVELS, cells[1:6], strict=True):
            number = re.match(r"\d+", cell)
            counts[level] = int(number.group(0)) if number else 0
        stats[int(cells[0])] = {"counts": counts, "first": cells[9], "last": cells[10]}
    if not stats:
        raise ValueError("SMG annual signal table could not be read")
    return stats


def annual_crosscheck(events: pd.DataFrame, raw_dir: Path = RAW_SMG_DIR) -> pd.DataFrame:
    """Per year: reports that reached each signal, next to SMG's annual table."""
    stats = load_annual_stats(raw_dir)
    signals = events[events["signal"] > 0]
    rows = []
    for year in range(SMG_FIRST_YEAR, SMG_LAST_YEAR + 1):
        of_year = signals[signals["year"] == year]
        row: dict[str, Any] = {"year": year, "in_annual_table": year in stats}
        for level in _ANNUAL_LEVELS:
            parsed = int(of_year.loc[of_year["signal"] == level, "report"].nunique())
            row[f"parsed_signal{level}"] = parsed
            row[f"smg_signal{level}"] = stats[year]["counts"][level] if year in stats else None
        # SMG prints the day the first and the last cyclone of the year got their first signal.
        starts = of_year.groupby("report")["issued_at"].min()
        row["parsed_first"] = starts.min()[5:10].replace("-", "/")
        row["parsed_last"] = starts.max()[5:10].replace("-", "/")
        row["smg_first"] = stats[year]["first"] if year in stats else None
        row["smg_last"] = stats[year]["last"] if year in stats else None
        row["match"] = (
            all(row[f"parsed_signal{lv}"] == row[f"smg_signal{lv}"] for lv in _ANNUAL_LEVELS)
            and row["parsed_first"] == row["smg_first"]
            and row["parsed_last"] == row["smg_last"]
            if year in stats
            else None
        )
        rows.append(row)
    return pd.DataFrame(rows)


def monthly_summary_crosscheck(events: pd.DataFrame, raw_dir: Path = RAW_SMG_DIR) -> pd.DataFrame:
    """Per month not covered by the annual table: summary records vs report records."""
    rows = []
    for name in current_year_summaries(raw_dir):
        match = _SUMMARY_NAME_RE.match(name)
        year, month = int(match.group(1)), int(match.group(2))
        text = summary_text_path(name, raw_dir).read_text(encoding="utf-8")
        summary = {
            (event.when.isoformat(timespec="minutes"), event.level)
            for event in parse_signal_lines(text, year)
        }
        prefix = f"{year:04d}-{month:02d}"
        in_month = events[events["issued_at"].str.startswith(prefix)]
        report = set(zip(in_month["issued_at"], in_month["signal"].astype(int), strict=True))
        # A summary also prints the earlier records of a cyclone that began the month before.
        summary_in_month = {item for item in summary if item[0].startswith(prefix)}
        compact = re.sub(r"\s+", "", text)
        # A month without a cyclone has no tropical cyclone section at all.
        mentions_cyclone = "熱帶氣旋" in compact
        rows.append(
            {
                "period": prefix,
                "summary_file": name,
                "is_summary_of_month": f"{year}年{month}月氣象觀測總結" in compact,
                "summary_mentions_cyclone": mentions_cyclone,
                "summary_records": len(summary_in_month),
                "report_records": len(report),
                "match": summary_in_month == report and mentions_cyclone == bool(report),
            }
        )
    return pd.DataFrame(rows)
