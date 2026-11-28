from __future__ import annotations

import copy

import pandas as pd
import pytest

from macao_tourism import audit
from macao_tourism.config import INDICATOR_BY_KEY, INDICATORS, PANEL_END, PANEL_START, TIERS
from macao_tourism.ingest import dsec

OCC5 = INDICATOR_BY_KEY["occupancy_5star"]
TOURS = INDICATOR_BY_KEY["package_tour_visitors"]


def _obs(year: int, month: int, value: float | None, remark: str | None = None) -> dict:
    return {
        "Year": year,
        "PeriodID": month,
        "ReferencePeriod": f"{month}/{year}",
        "IndicatorValue": value,
        "LastUpdateDate": "2026-10-29T10:00:00",
        "AssociatedRemark": [{"DescriptionEngl": remark}] if remark else [],
    }


def _payload(blocks: dict[int, list[dict]]) -> dict:
    return {
        "Status": "OK",
        "Value": [{"indicatorId": key, "dsecIndicatorData": rows} for key, rows in blocks.items()],
    }


def test_values_use_year_and_month_number() -> None:
    payload = _payload({OCC5.indicator_id: [_obs(2023, 5, 81.2), _obs(2023, 9, 85.0)]})
    frame = dsec.parse_values_payload(payload, [OCC5])
    assert frame["period"].tolist() == ["2023-05", "2023-09"]
    assert frame["value"].tolist() == [81.2, 85.0]
    assert frame["key"].unique().tolist() == ["occupancy_5star"]


def test_zero_remark_becomes_zero_and_missing_stays_missing() -> None:
    payload = _payload(
        {
            TOURS.indicator_id: [
                _obs(2020, 4, None, dsec.ZERO_REMARK),
                _obs(2020, 5, None, dsec.MISSING_REMARK),
                _obs(2026, 8, 120.0, dsec.REVISED_REMARK),
            ]
        }
    )
    frame = dsec.parse_values_payload(payload, [TOURS]).set_index("period")
    assert frame.loc["2020-04", "value"] == 0.0
    assert frame.loc["2020-04", "zero_by_remark"]
    assert pd.isna(frame.loc["2020-05", "value"])
    assert not frame.loc["2020-05", "zero_by_remark"]
    assert frame.loc["2026-08", "revised"]


def test_payload_errors() -> None:
    good = _payload({OCC5.indicator_id: [_obs(2023, 5, 81.2)]})
    failed = copy.deepcopy(good)
    failed["Status"] = "ERROR"
    with pytest.raises(ValueError, match="status"):
        dsec.parse_values_payload(failed, [OCC5])
    with pytest.raises(ValueError, match="missing indicator ids"):
        dsec.parse_values_payload(good, [OCC5, TOURS])
    with pytest.raises(ValueError, match="Unexpected indicator id"):
        dsec.parse_values_payload(good, [TOURS])
    quarterly = _payload({OCC5.indicator_id: [_obs(2023, 13, 81.2)]})
    with pytest.raises(ValueError, match="non-monthly"):
        dsec.parse_values_payload(quarterly, [OCC5])
    twice = _payload({OCC5.indicator_id: [_obs(2023, 5, 81.2), _obs(2023, 5, 81.3)]})
    with pytest.raises(ValueError, match="Duplicate"):
        dsec.parse_values_payload(twice, [OCC5])


def test_registry_check_rejects_a_wrong_label() -> None:
    meta = copy.deepcopy(dsec.load_indicator_meta())
    assert len(dsec.verify_registry(meta)) == len(INDICATORS)
    meta[OCC5.indicator_id]["DescriptionEngl"] = "4-star hotels"
    with pytest.raises(ValueError, match="occupancy_5star: label"):
        dsec.verify_registry(meta)
    del meta[OCC5.indicator_id]
    with pytest.raises(ValueError, match="not under node"):
        dsec.verify_registry(meta)


def test_latest_monthly_xlsx_picks_newest_release_with_a_table() -> None:
    def release(year: int, month: int, kind: str) -> dict:
        return {
            "Year": str(year),
            "Period": month,
            "ReleaseID": year * 100 + month,
            "File": [{"Type": kind, "Link": f"/getAttachment/x/T_{year}_M{month:02d}.aspx"}],
        }

    monthly = [release(2026, 8, ".xlsx"), release(2026, 10, ".pdf"), release(2026, 9, ".xlsx")]
    index = {"S": {"Containers": {"S": {"Publication": {"Monthly": monthly}}}}}
    chosen = dsec.latest_monthly_xlsx(index, "S")
    assert (chosen["year"], chosen["month"]) == (2026, 9)
    assert chosen["file_name"] == "T_2026_M09.xlsx"
    assert chosen["url"].startswith("https://www.dsec.gov.mo/getAttachment/")


@pytest.fixture(scope="module")
def long() -> pd.DataFrame:
    return dsec.load_tsdb_long()


def test_tier_series_are_complete_in_the_panel_window(long: pd.DataFrame) -> None:
    months = audit.panel_months()
    assert (months[0], months[-1], len(months)) == (PANEL_START, PANEL_END, 201)
    for measure in audit.TIER_MEASURES:
        for tier in TIERS:
            series = long[(long["key"] == f"{measure}_{tier}") & long["period"].isin(months)]
            assert series["value"].notna().sum() == 201, f"{measure}_{tier}"


def test_values_agree_with_the_printed_releases(long: pd.DataFrame) -> None:
    check = audit.release_crosscheck(long, dsec.release_crosscheck_values())
    assert len(check) == 22
    assert check["match"].all()
    exact = check[check["xlsx_scale"] == 1.0]
    assert (exact["abs_diff"] == 0).all()
    occupancy = check[check["key"].str.startswith("occupancy_")].set_index("key")["tsdb_value"]
    assert occupancy.to_dict() == {
        "occupancy_5star": 90.1,
        "occupancy_4star": 87.0,
        "occupancy_3star": 83.7,
        "occupancy_2star": 85.4,
    }


def test_accounting_identities_hold(long: pd.DataFrame) -> None:
    checks = audit.identity_checks(long)
    for name in (
        "same_day_plus_overnight_equals_total",
        "sea_plus_land_plus_air_equals_total",
        "tier_rooms_sum_equals_hotel_rooms",
        "tier_establishments_sum_equals_hotels",
        "tier_guests_sum_equals_hotel_guests",
    ):
        assert checks[name]["months_compared"] == 201, name
        assert checks[name]["months_exact"] == 201, name
    assert checks["tier_occupancy_within_0_100"]["ok"]
    assert checks["international_share_within_0_1"]["ok"]


def test_package_tour_zero_months_are_kept_as_zero(long: pd.DataFrame) -> None:
    tours = long[long["key"] == "package_tour_visitors"].set_index("period")
    zero_months = tours.index[tours["zero_by_remark"]].tolist()
    assert zero_months == ["2020-04", "2020-05", "2020-06", "2020-07"]
    assert (tours.loc[zero_months, "value"] == 0).all()
