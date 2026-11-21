"""Mainland public-holiday periods parsed from the State Council notices on gov.cn.

Rule for one holiday period (applied to every notice the same way): the dates a notice
declares as days off (放假, 补休, 调休放假), joined with the Saturdays and Sundays
that touch them and that the notice does not declare as working days (上班). Where a
notice states the length ("共N天"), the parsed period must have exactly that length.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from macao_tourism.config import (
    CN_HOLIDAY_AMENDMENTS,
    CN_HOLIDAY_NOTICES,
    GOLDEN_WEEK_HOLIDAYS,
    RAW_CN_HOLIDAY_DIR,
)
from macao_tourism.ingest.http import Provenance, fetch_bytes, save_snapshot

HOLIDAY_NAMES: dict[str, str] = {
    "元旦": "new_year",
    "春节": "spring_festival",
    "清明节": "qingming",
    "劳动节": "labour_day",
    "端午节": "dragon_boat",
    "中秋节": "mid_autumn",
    "国庆节": "national_day",
}
_NAME_ALT = "|".join(HOLIDAY_NAMES)
_ITEM_RE = re.compile(
    rf"[一二三四五六七八九]、((?:{_NAME_ALT})(?:、(?:{_NAME_ALT}))*)：(.*?)"
    rf"(?=[一二三四五六七八九]、(?:{_NAME_ALT})|$)"
)
_DATE_RE = re.compile(r"(?:(\d{4})年)?(?:(\d{1,2})月)?(\d{1,2})日")
_COUNT_RE = re.compile(r"(?:共|放假)(\d+)天")
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)


@dataclass
class HolidayPeriod:
    notice_year: int
    holidays: tuple[str, ...]
    start: date
    end: date
    declared_days: int | None
    source: str
    amended_by: str = ""

    @property
    def n_days(self) -> int:
        return (self.end - self.start).days + 1

    @property
    def golden_week(self) -> bool:
        return bool(GOLDEN_WEEK_HOLIDAYS.intersection(self.holidays))

    def days(self) -> list[date]:
        return [self.start + timedelta(days=k) for k in range(self.n_days)]


@dataclass
class ParsedItem:
    holidays: tuple[str, ...]
    off_days: set[date] = field(default_factory=set)
    work_days: set[date] = field(default_factory=set)
    declared_days: int | None = None


def notice_path(year: int, raw_dir: Path = RAW_CN_HOLIDAY_DIR) -> Path:
    return raw_dir / f"notice_{year}.html"


def amendment_path(name: str, raw_dir: Path = RAW_CN_HOLIDAY_DIR) -> Path:
    return raw_dir / f"amendment_{name}.html"


def download_notices(provenance: Provenance, raw_dir: Path = RAW_CN_HOLIDAY_DIR) -> None:
    for year, url in CN_HOLIDAY_NOTICES.items():
        path = notice_path(year, raw_dir)
        save_snapshot(path, fetch_bytes(url, timeout=60.0))
        provenance.record(
            path,
            url=url,
            source="State Council of the PRC (gov.cn)",
            description=f"Notice on the {year} public-holiday arrangement",
        )
    for name, url in CN_HOLIDAY_AMENDMENTS.items():
        path = amendment_path(name, raw_dir)
        save_snapshot(path, fetch_bytes(url, timeout=60.0))
        provenance.record(
            path,
            url=url,
            source="State Council of the PRC (gov.cn)",
            description=f"Later notice changing a published schedule: {name}",
        )


def html_to_text(raw: bytes) -> str:
    """Strip markup and all whitespace; the notices are continuous Chinese prose."""
    text = raw.decode("utf-8", errors="replace")
    text = _TAG_RE.sub("", _SCRIPT_RE.sub("", text))
    return re.sub(r"\s+", "", html.unescape(text))


def notice_body(text: str) -> str:
    start = text.find("通知如下")
    if start < 0:
        raise ValueError("Notice body marker '通知如下' not found")
    end = text.find("节假日期间", start)
    return text[start : end if end > 0 else len(text)]


def _dates_in(clause: str, default_year: int, month: int | None) -> tuple[list[date], int | None]:
    """Dates named in a clause. A bare '21日' reuses the month of the date before it."""
    found: list[date] = []
    year = default_year
    for match in _DATE_RE.finditer(clause):
        if match.group(1):
            year = int(match.group(1))
        if match.group(2):
            month = int(match.group(2))
        if month is None:
            raise ValueError(f"Date without a month in {clause!r}")
        found.append(date(year, month, int(match.group(3))))
    return found, month


def parse_item_text(holidays: tuple[str, ...], text: str, year: int) -> ParsedItem:
    item = ParsedItem(holidays=holidays)
    text = re.sub(r"（[^）]*）", "", text)
    for sentence in text.split("。"):
        month: int | None = None
        for clause in sentence.split("，"):
            count = _COUNT_RE.search(clause)
            if count:
                item.declared_days = int(count.group(1))
            dates, month = _dates_in(clause, year, month)
            if not dates:
                continue
            if "上班" in clause:
                item.work_days.update(dates)
            elif "至" in clause and "放假" in clause:
                if len(dates) != 2 or dates[1] < dates[0]:
                    raise ValueError(f"Cannot read a date range from {clause!r}")
                span = (dates[1] - dates[0]).days
                item.off_days.update(dates[0] + timedelta(days=k) for k in range(span + 1))
            elif any(word in clause for word in ("放假", "补休", "调休")):
                item.off_days.update(dates)
    if not item.off_days:
        raise ValueError(f"No day off found for {holidays} in {text!r}")
    return item


def parse_notice_items(text: str, year: int) -> list[ParsedItem]:
    items = []
    for match in _ITEM_RE.finditer(notice_body(text)):
        names = tuple(HOLIDAY_NAMES[name] for name in match.group(1).split("、"))
        items.append(parse_item_text(names, match.group(2), year))
    found = {name for item in items for name in item.holidays}
    if found != set(HOLIDAY_NAMES.values()):
        missing = sorted(set(HOLIDAY_NAMES.values()) - found)
        raise ValueError(f"{year} notice: holidays not found: {missing}")
    return items


def period_from_item(
    item: ParsedItem, all_work_days: set[date], notice_year: int, source: str
) -> HolidayPeriod:
    """Join the declared days off with the adjoining free weekend days."""

    def is_free(day: date) -> bool:
        return day in item.off_days or (day.weekday() >= 5 and day not in all_work_days)

    start, end = min(item.off_days), max(item.off_days)
    cursor = start
    while cursor <= end:
        if not is_free(cursor):
            raise ValueError(f"{item.holidays} {notice_year}: working day {cursor} inside period")
        cursor += timedelta(days=1)
    while is_free(start - timedelta(days=1)):
        start -= timedelta(days=1)
    while is_free(end + timedelta(days=1)):
        end += timedelta(days=1)
    period = HolidayPeriod(notice_year, item.holidays, start, end, item.declared_days, source)
    if item.declared_days is not None and period.n_days != item.declared_days:
        raise ValueError(
            f"{item.holidays} {notice_year}: parsed {period.n_days} days "
            f"({start}..{end}) but the notice states {item.declared_days}"
        )
    return period


def _periods_of(items: list[ParsedItem], year: int, source: str) -> list[HolidayPeriod]:
    work_days = {day for item in items for day in item.work_days}
    return [period_from_item(item, work_days, year, source) for item in items]


def _apply_2015_victory_day(text: str, periods: list[HolidayPeriod]) -> None:
    match = re.search(r"(\d{1,2})月(\d{1,2})日至(\d{1,2})日调休放假，共(\d+)天", text)
    if not match or "抗日战争" not in text:
        raise ValueError("2015 Victory Day notice: holiday range not found")
    month, first, last, declared = (int(g) for g in match.groups())
    period = HolidayPeriod(
        2015,
        ("victory_day_70th",),
        date(2015, month, first),
        date(2015, month, last),
        declared,
        CN_HOLIDAY_AMENDMENTS["2015_victory_day"],
        amended_by="2015_victory_day",
    )
    if period.n_days != declared:
        raise ValueError("2015 Victory Day notice: stated length does not match the range")
    periods.append(period)


def _apply_2019_labour_day(text: str, periods: list[HolidayPeriod]) -> None:
    match = re.search(r"通知如下。一、(.*?)二、", text)
    if not match or "调整2019年劳动节" not in text:
        raise ValueError("2019 Labour Day notice: item not found")
    item = parse_item_text(("labour_day",), match.group(1), 2019)
    new = period_from_item(
        item, set(item.work_days), 2019, CN_HOLIDAY_AMENDMENTS["2019_labour_day"]
    )
    new.amended_by = "2019_labour_day"
    for index, period in enumerate(periods):
        if period.notice_year == 2019 and period.holidays == ("labour_day",):
            periods[index] = new
            return
    raise ValueError("2019 Labour Day period to replace not found")


def _apply_2020_extension(text: str, periods: list[HolidayPeriod]) -> None:
    match = re.search(r"延长2020年春节假期至(\d{1,2})月(\d{1,2})日", text)
    if not match:
        raise ValueError("2020 Spring Festival extension notice: new end date not found")
    new_end = date(2020, int(match.group(1)), int(match.group(2)))
    for period in periods:
        if period.notice_year == 2020 and period.holidays == ("spring_festival",):
            if new_end <= period.end:
                raise ValueError("2020 extension does not lengthen the Spring Festival period")
            period.end = new_end
            period.declared_days = None
            period.amended_by = "2020_spring_festival_extension"
            return
    raise ValueError("2020 Spring Festival period to extend not found")


def _annual_periods(raw_dir: Path) -> list[HolidayPeriod]:
    """Periods as first scheduled by the annual notices, before any later notice."""
    periods: list[HolidayPeriod] = []
    for year, url in CN_HOLIDAY_NOTICES.items():
        text = html_to_text(notice_path(year, raw_dir).read_bytes())
        periods.extend(_periods_of(parse_notice_items(text, year), year, url))
    return periods


def load_holiday_periods(raw_dir: Path = RAW_CN_HOLIDAY_DIR) -> list[HolidayPeriod]:
    periods = _annual_periods(raw_dir)
    appliers = {
        "2015_victory_day": _apply_2015_victory_day,
        "2019_labour_day": _apply_2019_labour_day,
        "2020_spring_festival_extension": _apply_2020_extension,
    }
    for name, apply in appliers.items():
        apply(html_to_text(amendment_path(name, raw_dir).read_bytes()), periods)
    periods.sort(key=lambda period: period.start)
    for earlier, later in zip(periods, periods[1:], strict=False):
        if later.start <= earlier.end:
            raise ValueError(f"Holiday periods overlap: {earlier} and {later}")
    return periods


_PUBLISHED_RE = re.compile(r"发布日期：(\d{4})年(\d{1,2})月(\d{1,2})日")
_URL_DATE_RE = re.compile(r"/(\d{4})-(\d{2})/(\d{2})/")


def publication_date(raw: bytes, url: str) -> date:
    """Day gov.cn published a notice: the page's 发布日期, else the date in its URL."""
    match = _PUBLISHED_RE.search(html_to_text(raw)) or _URL_DATE_RE.search(url)
    if not match:
        raise ValueError(f"No publication date found for {url}")
    return date(*(int(group) for group in match.groups()))


def _annual_publication_dates(raw_dir: Path) -> dict[int, date]:
    return {
        year: publication_date(notice_path(year, raw_dir).read_bytes(), url)
        for year, url in CN_HOLIDAY_NOTICES.items()
    }


@dataclass(frozen=True)
class ScheduleVersion:
    """One published schedule of a holiday period."""

    period: HolidayPeriod
    published: date
    superseded: bool

    @property
    def key(self) -> tuple[int, tuple[str, ...]]:
        return (self.period.notice_year, self.period.holidays)


def load_schedule_versions(raw_dir: Path = RAW_CN_HOLIDAY_DIR) -> list[ScheduleVersion]:
    """Every final period with its publication day, plus the schedules later notices replaced.

    A period changed by a later notice has two versions: the one in the annual notice
    and the final one, dated by the later notice.
    """
    annual = _annual_publication_dates(raw_dir)
    amended = {
        name: publication_date(amendment_path(name, raw_dir).read_bytes(), url)
        for name, url in CN_HOLIDAY_AMENDMENTS.items()
    }
    final = load_holiday_periods(raw_dir)
    versions = [
        ScheduleVersion(
            period,
            amended[period.amended_by] if period.amended_by else annual[period.notice_year],
            superseded=False,
        )
        for period in final
    ]
    changed = {version.key for version in versions if version.period.amended_by}
    versions += [
        ScheduleVersion(period, annual[period.notice_year], superseded=True)
        for period in _annual_periods(raw_dir)
        if (period.notice_year, period.holidays) in changed
    ]
    return sorted(versions, key=lambda version: (version.period.start, version.published))


def monthly_holiday_days_at_origin(
    versions: list[ScheduleVersion], start: str, end: str
) -> pd.DataFrame:
    """Holiday days of each month as scheduled at the end of the month before it.

    This is what a forecaster standing at the origin knows about the target month. A
    schedule published later counts for nothing; a schedule replaced later still counts
    in its earlier form.
    """
    rows = []
    for month in pd.period_range(start, end, freq="M"):
        cutoff = date(month.year, month.month, 1) - timedelta(days=1)
        latest: dict[tuple[int, tuple[str, ...]], ScheduleVersion] = {}
        for version in versions:
            if version.published > cutoff:
                continue
            known = latest.get(version.key)
            if known is None or version.published > known.published:
                latest[version.key] = version
        total = golden = 0
        for version in latest.values():
            inside = sum(
                1
                for day in version.period.days()
                if (day.year, day.month) == (month.year, month.month)
            )
            total += inside
            golden += inside * int(version.period.golden_week)
        rows.append(
            {
                "period": str(month),
                "cn_holiday_days_at_origin": total,
                "cn_golden_week_days_at_origin": golden,
            }
        )
    return pd.DataFrame(rows)


def versions_frame(versions: list[ScheduleVersion]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "notice_year": v.period.notice_year,
                "holiday": "+".join(v.period.holidays),
                "start": v.period.start.isoformat(),
                "end": v.period.end.isoformat(),
                "n_days": v.period.n_days,
                "golden_week": v.period.golden_week,
                "published": v.published.isoformat(),
                "superseded": v.superseded,
                "amended_by": v.period.amended_by,
                "source_url": v.period.source,
            }
            for v in versions
        ]
    )


def periods_frame(periods: list[HolidayPeriod]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "notice_year": p.notice_year,
                "holiday": "+".join(p.holidays),
                "start": p.start.isoformat(),
                "end": p.end.isoformat(),
                "n_days": p.n_days,
                "declared_days": p.declared_days,
                "golden_week": p.golden_week,
                "amended_by": p.amended_by,
                "source_url": p.source,
            }
            for p in periods
        ]
    )


def monthly_holiday_days(periods: list[HolidayPeriod], start: str, end: str) -> pd.DataFrame:
    """Days of each month that fall in a holiday period, total and golden-week only."""
    months = pd.period_range(start, end, freq="M")
    total = dict.fromkeys(months.astype(str), 0)
    golden = dict.fromkeys(months.astype(str), 0)
    for period in periods:
        for day in period.days():
            key = f"{day.year:04d}-{day.month:02d}"
            if key in total:
                total[key] += 1
                golden[key] += int(period.golden_week)
    return pd.DataFrame(
        {
            "period": list(total),
            "cn_holiday_days": list(total.values()),
            "cn_golden_week_days": list(golden.values()),
        }
    )
