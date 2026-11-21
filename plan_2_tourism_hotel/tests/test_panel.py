from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from macao_tourism import panel as pn
from macao_tourism.config import PANEL_END, PANEL_START, TIERS
from macao_tourism.ingest import dsec
from macao_tourism.ingest import holidays as h


@pytest.fixture(scope="module")
def months() -> pd.DataFrame:
    return pn.build_months()


@pytest.fixture(scope="module")
def panel(months: pd.DataFrame) -> pd.DataFrame:
    return pn.build_panel(months)


def test_logit_round_trip_and_bounds() -> None:
    values = pd.Series([5.0, 50.0, 97.2], index=["a", "b", "c"])
    logits = pn.logit_pct(values)
    assert logits.index.tolist() == ["a", "b", "c"]
    assert logits["b"] == pytest.approx(0.0)
    np.testing.assert_allclose(pn.inv_logit_pct(logits), values)
    assert pn.inv_logit_pct(0.0) == pytest.approx(50.0)
    for bad in (0.0, 100.0, 120.0):
        with pytest.raises(ValueError, match="logit is undefined"):
            pn.logit_pct(bad)


def test_month_table_covers_the_window_once(months: pd.DataFrame) -> None:
    assert list(months.columns) == list(pn.MONTH_COLUMNS)
    assert len(months) == 201
    assert (months["period"].iloc[0], months["period"].iloc[-1]) == (PANEL_START, PANEL_END)
    assert months["period"].is_unique
    assert not months.isna().any().any()


def test_period_flags_follow_the_plan(months: pd.DataFrame) -> None:
    by_month = months.set_index("period")
    assert months["period_label"].value_counts().to_dict() == {
        "pre_pandemic": 120,
        "post_pandemic": 45,
        "pandemic": 35,
        "transition_2020_01": 1,
    }
    assert by_month.loc["2019-12", ["post", "in_main_sample"]].tolist() == [0, 1]
    assert by_month.loc["2020-01", ["post", "in_main_sample"]].tolist() == [0, 0]
    assert by_month.loc["2022-12", ["post", "in_main_sample"]].tolist() == [0, 0]
    assert by_month.loc["2023-01", ["post", "in_main_sample"]].tolist() == [1, 1]
    assert months["in_main_sample"].sum() == 165
    assert months["post"].sum() == 45


def test_shares_are_computed_from_the_dsec_counts(months: pd.DataFrame) -> None:
    first = months.set_index("period").loc["2010-01"]
    assert (first["visitors_total"], first["visitors_same_day"]) == (2046556, 1040977)
    assert first["same_day_share_pct"] == pytest.approx(100 * 1040977 / 2046556)
    assert first["international_share_pct"] == pytest.approx(
        100 * (1 - (1131660 + 544603 + 115673) / 2046556)
    )
    assert first["package_tour_share_pct"] == pytest.approx(100 * 458907 / 2046556)
    assert first["log_visitors_overnight"] == pytest.approx(np.log(1005579))
    total = months["visitors_same_day"] + months["visitors_overnight"]
    assert (total == months["visitors_total"]).all()
    for column in ("same_day_share_pct", "international_share_pct", "package_tour_share_pct"):
        assert months[column].between(0, 100).all()


def test_calendar_and_typhoon_columns(months: pd.DataFrame) -> None:
    by_month = months.set_index("period")
    assert by_month.loc["2023-10", ["cn_holiday_days", "cn_golden_week_days"]].tolist() == [6, 6]
    assert by_month.loc["2017-08", "typhoon_days"] == 2
    assert months["typhoon_days"].sum() == 44
    assert months["cn_holiday_days"].sum() == 492
    assert by_month.loc["2024-02", "days_in_month"] == 29


def test_holiday_days_at_origin_differ_only_where_a_notice_came_late(
    months: pd.DataFrame,
) -> None:
    differs = months[months["cn_holiday_days_at_origin"] != months["cn_holiday_days"]]
    columns = ["cn_holiday_days", "cn_holiday_days_at_origin"]
    assert differs.set_index("period")[columns].to_dict("index") == {
        "2016-12": {"cn_holiday_days": 1, "cn_holiday_days_at_origin": 0},
        "2018-12": {"cn_holiday_days": 2, "cn_holiday_days_at_origin": 0},
        "2020-01": {"cn_holiday_days": 9, "cn_holiday_days_at_origin": 8},
        "2022-12": {"cn_holiday_days": 1, "cn_holiday_days_at_origin": 0},
    }
    by_month = months.set_index("period")
    assert by_month.loc["2020-01", "cn_golden_week_days_at_origin"] == 7
    assert by_month.loc["2020-02", "cn_holiday_days_at_origin"] == 2
    assert by_month.loc["2019-05", "cn_holiday_days_at_origin"] == 4
    assert by_month.loc["2015-09", "cn_holiday_days_at_origin"] == 5
    after_2023 = months[months["period"] >= "2023-01"]
    assert (after_2023["cn_holiday_days_at_origin"] == after_2023["cn_holiday_days"]).all()


def _version(start: date, end: date, published: date, superseded: bool) -> h.ScheduleVersion:
    period = h.HolidayPeriod(2030, ("spring_festival",), start, end, None, "test")
    return h.ScheduleVersion(period, published, superseded)


def test_schedule_at_origin_uses_only_notices_published_before_the_month() -> None:
    first = _version(date(2030, 2, 2), date(2030, 2, 8), date(2029, 11, 20), superseded=True)
    longer = _version(date(2030, 2, 2), date(2030, 2, 10), date(2030, 2, 1), superseded=False)
    counts = h.monthly_holiday_days_at_origin([first, longer], "2030-02", "2030-02")
    assert counts["cn_holiday_days_at_origin"].tolist() == [7]
    assert counts["cn_golden_week_days_at_origin"].tolist() == [7]

    in_time = _version(date(2030, 2, 2), date(2030, 2, 10), date(2030, 1, 31), superseded=False)
    counts = h.monthly_holiday_days_at_origin([first, in_time], "2030-02", "2030-02")
    assert counts["cn_holiday_days_at_origin"].tolist() == [9]

    late = _version(date(2030, 2, 2), date(2030, 2, 8), date(2030, 2, 1), superseded=False)
    counts = h.monthly_holiday_days_at_origin([late], "2030-02", "2030-03")
    assert counts["cn_holiday_days_at_origin"].tolist() == [0, 0]


def test_schedule_versions_keep_the_two_replaced_schedules() -> None:
    versions = h.load_schedule_versions()
    assert sum(not v.superseded for v in versions) == 115
    replaced = [v for v in versions if v.superseded]
    assert [(v.period.start, v.period.end, v.published) for v in replaced] == [
        (date(2019, 5, 1), date(2019, 5, 1), date(2018, 12, 6)),
        (date(2020, 1, 24), date(2020, 1, 30), date(2019, 11, 21)),
    ]
    assert all(v.published < v.period.start for v in versions if not v.period.amended_by)


def test_panel_is_balanced_and_matches_the_raw_series(
    months: pd.DataFrame, panel: pd.DataFrame
) -> None:
    assert list(panel.columns) == list(pn.PANEL_COLUMNS)
    assert len(panel) == 804
    assert not panel.duplicated(["tier", "period"]).any()
    assert panel.groupby("period")["tier"].apply(list).map(lambda t: t == list(TIERS)).all()
    assert not panel.isna().any().any()

    long = dsec.load_tsdb_long().set_index(["key", "period"])["value"]
    for tier in TIERS:
        rows = panel[panel["tier"] == tier].set_index("period")
        in_window = rows.index
        assert (rows["occupancy_pct"] == long[f"occupancy_{tier}"].loc[in_window]).all()
        assert (rows["rooms"] == long[f"rooms_{tier}"].loc[in_window]).all()
        assert (rows["guests"] == long[f"guests_{tier}"].loc[in_window]).all()
    latest = panel[panel["period"] == "2026-09"].set_index("tier")["occupancy_pct"]
    assert latest.to_dict() == {"5star": 90.1, "4star": 87.0, "3star": 83.7, "2star": 85.4}
    np.testing.assert_allclose(panel["log_rooms"], np.log(panel["rooms"]), atol=1e-9)


def test_month_variables_are_identical_across_star_classes(
    months: pd.DataFrame, panel: pd.DataFrame
) -> None:
    month_columns = [c for c in pn.MONTH_COLUMNS if c != "period"]
    for tier in TIERS:
        rows = panel.loc[panel["tier"] == tier, ["period", *month_columns]].reset_index(drop=True)
        pd.testing.assert_frame_equal(rows, months[["period", *month_columns]])


def test_validate_panel_stops_on_a_broken_panel(months: pd.DataFrame, panel: pd.DataFrame) -> None:
    pn.validate_panel(months, panel)
    with pytest.raises(ValueError, match="not balanced"):
        pn.validate_panel(months, panel.iloc[:-1])
    with_gap = panel.copy()
    with_gap.loc[3, "occupancy_pct"] = np.nan
    with pytest.raises(ValueError, match="Missing values in panel"):
        pn.validate_panel(months, with_gap)
    twice = pd.concat([panel.iloc[:-1], panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        pn.validate_panel(months, twice)
    full = panel.copy()
    full.loc[0, "occupancy_pct"] = 100.0
    with pytest.raises(ValueError, match="strictly between 0 and 100"):
        pn.validate_panel(months, full)
    wrong_total = months.copy()
    wrong_total.loc[0, "visitors_total"] += 1
    with pytest.raises(ValueError, match="does not equal total"):
        pn.validate_panel(wrong_total, panel)


def test_variable_summary_is_not_split_by_period_or_class(
    months: pd.DataFrame, panel: pd.DataFrame
) -> None:
    summary = pn.variable_summary(months, panel)
    assert summary["variable"].is_unique
    assert set(summary["level"]) == {"tier_month", "month"}
    assert (summary["missing"] == 0).all()
    assert summary.set_index("variable").loc["occupancy_pct", "n"] == 804
