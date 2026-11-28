from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from macao_tourism.config import PANEL_END, PANEL_START
from macao_tourism.ingest import smg

OLD_FORMAT = """信號懸掛記錄：
日期 時間 懸掛信號
7 月 21 日 18:00 一號風球
7 月 23 日 06:30 三號風球
7 月 23 日 19:00 八號東北風球
7 月 24 日 02:15 九號風球
7 月 24 日 05:00 八號東南風球
7 月 24 日 09:30 三號風球
7 月 24 日 16:20 除下所有風球
"""

NEW_FORMAT = """熱帶氣旋信號發出記錄
日期 發出時間 發出之熱帶氣旋信號
8 月 30 日 16:30 一號風球
9 月 1 日 14:00 八號西北風球
9 月 2 日 01:00 十號風球
9 月 2 日 15:00 三號風球
9 月 3 日 00:00 取消所有熱帶氣旋警告信號
9 月 2 日 13:30 取消所有風暴潮警告信號
8 月 30 日 16:30 一號風球
"""


def _frame(text: str, year: int, report: str = "c_test.pdf") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "year": year,
                "report": report,
                "issued_at": e.when.isoformat(timespec="minutes"),
                "signal": e.level,
            }
            for e in smg.parse_signal_lines(text, year)
        ]
    )


def test_old_format_levels_and_times() -> None:
    events = smg.parse_signal_lines(OLD_FORMAT, 2012)
    assert [e.level for e in events] == [1, 3, 8, 9, 8, 3, 0]
    assert events[3].when == datetime(2012, 7, 24, 2, 15)
    smg.validate_episodes(events, "old")


def test_new_format_skips_storm_surge_and_duplicates() -> None:
    events = smg.parse_signal_lines(NEW_FORMAT, 2023)
    assert [e.level for e in events] == [1, 8, 10, 3, 0]
    assert events[-1].when == datetime(2023, 9, 3, 0, 0)
    smg.validate_episodes(events, "new")


def test_zero_padded_dates_and_summary_layout() -> None:
    text = (
        "08 月 23 日 11:30 十號風球\n"
        "沙德爾 8月31日 18:30 一號風球\n"
        "9月3日 21:00 取消所有熱帶氣旋信號"
    )
    events = smg.parse_signal_lines(text, 2026)
    assert [(e.when.month, e.when.day, e.level) for e in events] == [
        (8, 23, 10),
        (8, 31, 1),
        (9, 3, 0),
    ]


def test_compatibility_ideographs_are_read() -> None:
    line = "10 月 15 日 10:00 八號東\uf963風球"
    (event,) = smg.parse_signal_lines(line, 2017)
    assert event.level == 8


def test_unreadable_signal_line_is_an_error() -> None:
    with pytest.raises(ValueError, match="Unreadable"):
        smg.parse_signal_lines("10 月 15 日 10:00 七號風球", 2017)


def test_validate_rejects_incomplete_records() -> None:
    with pytest.raises(ValueError, match="no signal record"):
        smg.validate_episodes([], "empty")
    without_cancel = smg.parse_signal_lines("7 月 21 日 18:00 一號風球", 2012)
    with pytest.raises(ValueError, match="end with a cancel"):
        smg.validate_episodes(without_cancel, "open")


def test_typhoon_day_counts_only_days_with_signal_in_force() -> None:
    daily = smg.daily_signals(smg.signal_intervals(_frame(NEW_FORMAT, 2023))).set_index("date")
    assert list(daily.index) == ["2023-08-30", "2023-08-31", "2023-09-01", "2023-09-02"]
    assert daily.loc["2023-09-01", "max_signal"] == 8
    assert daily.loc["2023-09-01", "hours_signal8plus"] == 10.0
    assert daily.loc["2023-09-02", "max_signal"] == 10
    assert daily.loc["2023-09-02", "hours_signal8plus"] == 15.0
    assert "2023-09-03" not in daily.index
    monthly = smg.monthly_signal_days(daily.reset_index(), "2023-08", "2023-10")
    assert monthly["typhoon_days_signal8plus"].tolist() == [0, 2, 0]
    assert monthly["signal1plus_days"].tolist() == [2, 2, 0]


def test_overlapping_reports_are_rejected() -> None:
    first = _frame(OLD_FORMAT, 2012, "c_a.pdf")
    second = _frame(OLD_FORMAT, 2012, "c_b.pdf")
    with pytest.raises(ValueError, match="overlap"):
        smg._check_no_overlap(pd.concat([first, second], ignore_index=True))


@pytest.fixture(scope="module")
def events() -> pd.DataFrame:
    return smg.load_signal_events()


def test_reports_cover_every_listed_cyclone(events: pd.DataFrame) -> None:
    assert events["report"].nunique() == 104
    assert events["year"].min() == 2010
    assert events["year"].max() == 2026
    assert set(events["signal"].unique()) == {0, 1, 3, 8, 9, 10}


def test_annual_table_agrees_with_reports(events: pd.DataFrame) -> None:
    check = smg.annual_crosscheck(events)
    covered = check[check["in_annual_table"]]
    assert covered["year"].tolist() == list(range(2010, 2026))
    assert covered["match"].all()
    assert check.loc[~check["in_annual_table"], "year"].tolist() == [2026]


def test_monthly_summaries_agree_for_the_current_year(events: pd.DataFrame) -> None:
    check = smg.monthly_summary_crosscheck(events)
    assert check["period"].tolist() == [f"2026-{m:02d}" for m in range(1, 10)]
    assert check["match"].all()
    assert check["is_summary_of_month"].all()
    assert check["summary_mentions_cyclone"].tolist() == [False] * 6 + [True] * 3


def test_known_typhoons(events: pd.DataFrame) -> None:
    daily = smg.daily_signals(smg.signal_intervals(events)).set_index("date")
    assert daily.loc["2017-08-23", "max_signal"] == 10
    assert daily.loc["2018-09-16", "max_signal"] == 10
    assert daily.loc["2023-09-02", "max_signal"] == 10
    monthly = smg.monthly_signal_days(daily.reset_index(), PANEL_START, PANEL_END)
    assert len(monthly) == 201
    by_month = monthly.set_index("period")["typhoon_days_signal8plus"]
    assert by_month["2017-08"] == 2
    assert by_month["2018-09"] == 2
    assert (monthly["typhoon_days_signal8plus"] <= monthly["signal3plus_days"]).all()
    assert (monthly["signal3plus_days"] <= monthly["signal1plus_days"]).all()
