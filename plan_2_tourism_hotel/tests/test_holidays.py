from __future__ import annotations

from datetime import date

import pytest

from macao_tourism.config import PANEL_END, PANEL_START
from macao_tourism.ingest import holidays as h


def _period(names: tuple[str, ...], text: str, year: int, work: set[date] | None = None):
    item = h.parse_item_text(names, text, year)
    return h.period_from_item(item, (work or set()) | item.work_days, year, "test")


def test_range_with_stated_length_and_working_weekend() -> None:
    period = _period(
        ("spring_festival",),
        "2月13日至19日放假调休，共7天。2月20日（星期六）、21日（星期日）上班。",
        2010,
    )
    assert (period.start, period.end) == (date(2010, 2, 13), date(2010, 2, 19))
    assert period.n_days == period.declared_days == 7
    assert period.golden_week


def test_bare_day_reuses_month_of_previous_date() -> None:
    item = h.parse_item_text(
        ("spring_festival",), "2月13日至19日放假调休，共7天。2月20日、21日上班。", 2010
    )
    assert item.work_days == {date(2010, 2, 20), date(2010, 2, 21)}


def test_single_day_joined_with_weekend() -> None:
    period = _period(("dragon_boat",), "6月2日放假，与周末连休。", 2014)
    assert (period.start, period.end) == (date(2014, 5, 31), date(2014, 6, 2))
    assert period.declared_days is None


def test_single_day_with_compensation_day() -> None:
    period = _period(("qingming",), "4月5日放假，4月7日（星期一）补休。", 2014)
    assert (period.start, period.end) == (date(2014, 4, 5), date(2014, 4, 7))


def test_midweek_single_day_is_not_extended() -> None:
    period = _period(("new_year",), "1月1日（周三）放假1天，不调休。", 2025)
    assert period.n_days == period.declared_days == 1


def test_range_across_new_year() -> None:
    period = _period(
        ("new_year",),
        "2018年12月30日至2019年1月1日放假调休，共3天。2018年12月29日（星期六）上班。",
        2019,
    )
    assert (period.start, period.end) == (date(2018, 12, 30), date(2019, 1, 1))


def test_range_across_months_with_lunar_notes() -> None:
    period = _period(
        ("spring_festival",),
        "1月28日（农历除夕、周二）至2月4日（农历正月初七、周二）放假调休，共8天。"
        "1月26日（周日）、2月8日（周六）上班。",
        2025,
    )
    assert (period.start, period.end) == (date(2025, 1, 28), date(2025, 2, 4))


def test_stated_length_mismatch_is_an_error() -> None:
    with pytest.raises(ValueError, match="notice states 4"):
        _period(
            ("dragon_boat",),
            "6月10日至12日放假调休，共4天。6月8日（星期六）、6月9日（星期日）上班。",
            2013,
        )


def test_item_without_day_off_is_an_error() -> None:
    with pytest.raises(ValueError, match="No day off"):
        h.parse_item_text(("labour_day",), "5月4日（星期日）上班。", 2014)


def test_notice_missing_a_holiday_is_an_error() -> None:
    text = "通知如下。一、元旦：1月1日放假1天。节假日期间"
    with pytest.raises(ValueError, match="holidays not found"):
        h.parse_notice_items(text, 2014)


@pytest.fixture(scope="module")
def periods() -> list[h.HolidayPeriod]:
    return h.load_holiday_periods()


def _find(periods: list[h.HolidayPeriod], year: int, holiday: str) -> h.HolidayPeriod:
    (match,) = [p for p in periods if p.notice_year == year and holiday in p.holidays]
    return match


def test_all_notices_parse(periods: list[h.HolidayPeriod]) -> None:
    assert len(periods) == 115
    assert {p.notice_year for p in periods} == set(range(2010, 2027))
    stated = [p for p in periods if p.declared_days is not None]
    assert len(stated) == 94
    assert all(p.n_days == p.declared_days for p in stated)


def test_amending_notices_are_applied(periods: list[h.HolidayPeriod]) -> None:
    spring_2020 = _find(periods, 2020, "spring_festival")
    assert (spring_2020.start, spring_2020.end) == (date(2020, 1, 24), date(2020, 2, 2))
    labour_2019 = _find(periods, 2019, "labour_day")
    assert (labour_2019.start, labour_2019.end) == (date(2019, 5, 1), date(2019, 5, 4))
    victory = _find(periods, 2015, "victory_day_70th")
    assert (victory.start, victory.end) == (date(2015, 9, 3), date(2015, 9, 5))


def test_known_periods(periods: list[h.HolidayPeriod]) -> None:
    national_2023 = _find(periods, 2023, "national_day")
    assert national_2023.holidays == ("mid_autumn", "national_day")
    assert (national_2023.start, national_2023.end) == (date(2023, 9, 29), date(2023, 10, 6))
    spring_2026 = _find(periods, 2026, "spring_festival")
    assert spring_2026.n_days == 9


def test_monthly_holiday_days(periods: list[h.HolidayPeriod]) -> None:
    monthly = h.monthly_holiday_days(periods, PANEL_START, PANEL_END).set_index("period")
    assert len(monthly) == 201
    assert monthly.loc["2023-09", "cn_holiday_days"] == 2
    assert monthly.loc["2023-10", "cn_holiday_days"] == 6
    assert monthly.loc["2023-10", "cn_golden_week_days"] == 6
    assert monthly.loc["2020-01", "cn_holiday_days"] == 1 + 8
    assert monthly.loc["2020-02", "cn_golden_week_days"] == 2
    assert monthly.loc["2015-09", "cn_holiday_days"] == 3 + 2
    assert monthly.loc["2015-09", "cn_golden_week_days"] == 0
    assert monthly.loc["2018-12", "cn_holiday_days"] == 2
    assert (monthly["cn_golden_week_days"] <= monthly["cn_holiday_days"]).all()
    assert monthly.loc["2012-01", "cn_holiday_days"] == 3 + 7
    assert monthly["cn_holiday_days"].max() == 10
