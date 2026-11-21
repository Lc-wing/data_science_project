from __future__ import annotations

import json

import pandas as pd
import pytest

from macao_tourism import audit
from macao_tourism.config import (
    DATA_INTERIM_DIR,
    DATA_RAW_DIR,
    OUTPUTS_METRICS_DIR,
    OUTPUTS_TABLES_DIR,
    PROJECT_ROOT,
    PROVENANCE_PATH,
    sha256_file,
)
from macao_tourism.ingest import dsec, holidays, smg


def test_period_counts_follow_the_plan() -> None:
    counts = audit.period_counts()
    assert counts == {
        "pre_pandemic": 120,
        "transition_2020_01": 1,
        "pandemic": 35,
        "post_pandemic": 45,
        "all_months": 201,
    }
    assert audit.period_label("2020-01") == "transition_2020_01"
    assert audit.period_label("2023-01") == "post_pandemic"
    with pytest.raises(ValueError):
        audit.period_label("2009-12")


def test_release_crosscheck_tolerance_for_rounded_thousands() -> None:
    long = pd.DataFrame(
        {
            "key": ["rooms_5star", "rooms_4star", "occupancy_5star"],
            "period": ["2026-09"] * 3,
            "value": [26578.0, 9351.0, 90.2],
        }
    )
    printed = pd.DataFrame(
        {
            "key": ["rooms_5star", "rooms_4star", "occupancy_5star"],
            "period": ["2026-09"] * 3,
            "xlsx_value": [26.6, 9.2, 90.1],
            "xlsx_scale": [1000.0, 1000.0, 1.0],
            "xlsx_file": ["f.xlsx"] * 3,
        }
    )
    check = audit.release_crosscheck(long, printed).set_index("key")
    assert check.loc["rooms_5star", "match"]
    assert not check.loc["rooms_4star", "match"]
    assert not check.loc["occupancy_5star", "match"]


def test_offsetting_moves_need_a_gain_and_a_loss() -> None:
    changes = pd.DataFrame(
        {
            "tier": ["5star", "4star", "3star"],
            "period": ["2021-01", "2021-01", "2016-09"],
            "establishments_before": [33, 15, 13],
            "establishments_after": [32, 16, 15],
            "rooms_before": [21684, 6174, 2528],
            "rooms_after": [21351, 6747, 5477],
        }
    )
    moves = audit.offsetting_reclassifications(changes)
    assert moves["period"].tolist() == ["2021-01"]


def test_supply_jumps_are_reported_not_removed() -> None:
    long = dsec.load_tsdb_long()
    jumps = audit.supply_jumps(long)
    largest = jumps.loc[jumps["rooms_change_pct"].abs().idxmax()]
    assert (largest["tier"], largest["period"]) == ("3star", "2016-09")
    assert (largest["rooms_before"], largest["rooms_after"]) == (2528, 5477)
    assert set(jumps["period_label"]) <= set(audit.PERIODS)


def test_committed_tables_match_a_fresh_parse_of_the_raw_files() -> None:
    """The interim CSVs must be reproducible from data/raw alone."""
    long = pd.read_csv(DATA_INTERIM_DIR / "dsec_monthly_long.csv", keep_default_na=False)
    fresh = dsec.load_tsdb_long()
    assert len(long) == len(fresh)
    assert long["key"].tolist() == fresh["key"].tolist()
    assert long["period"].tolist() == fresh["period"].tolist()
    committed_values = pd.to_numeric(long["value"].replace("", None))
    pd.testing.assert_series_equal(committed_values, fresh["value"], check_names=False)

    periods = holidays.load_holiday_periods()
    monthly = pd.read_csv(DATA_INTERIM_DIR / "cn_holiday_days_monthly.csv")
    pd.testing.assert_frame_equal(
        monthly, holidays.monthly_holiday_days(periods, "2010-01", "2026-09")
    )

    events = smg.load_signal_events()
    daily = smg.daily_signals(smg.signal_intervals(events))
    signal_months = pd.read_csv(DATA_INTERIM_DIR / "smg_signal_days_monthly.csv")
    pd.testing.assert_frame_equal(
        signal_months, smg.monthly_signal_days(daily, "2010-01", "2026-09")
    )


def test_audit_headline_numbers() -> None:
    result = json.loads((OUTPUTS_METRICS_DIR / "day1_data_audit.json").read_text(encoding="utf-8"))
    assert result["dsec"]["tier_month_observations"] == 804
    assert result["dsec"]["tier_series_complete_in_panel_window"] == 16
    assert result["dsec"]["release_crosscheck"]["values_matching"] == 22
    assert result["dsec"]["tier_classification"]["all_four_tiers_every_month"] is True
    assert result["cn_holidays"]["periods"] == 115
    assert result["cn_holidays"]["periods_with_stated_length"] == 94
    assert result["cn_holidays"]["periods_matching_stated_length"] == 94
    signals = result["smg_typhoon_signals"]
    assert signals["reports_parsed"] == 104
    assert signals["annual_table_years_matching"] == signals["annual_table_years_checked"] == 16
    assert signals["monthly_summaries_matching"] == signals["monthly_summaries_checked"] == 9
    assert signals["typhoon_days_in_panel_window"] == 44
    assert sum(signals["typhoon_days_by_year"].values()) == 44
    assert signals["typhoon_days_by_year"]["2026"] == 0
    assert (OUTPUTS_TABLES_DIR / "day1_data_audit.md").is_file()


def test_every_raw_file_has_provenance_with_a_matching_hash() -> None:
    entries = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))["files"]
    on_disk = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in DATA_RAW_DIR.rglob("*")
        if path.is_file() and path != PROVENANCE_PATH
    }
    assert on_disk == set(entries)
    for rel, entry in entries.items():
        assert entry["url"].startswith("https://"), rel
        assert entry["downloaded_at_utc8"], rel
        assert entry["sha256"] == sha256_file(PROJECT_ROOT / rel), rel
        assert entry["bytes"] == (PROJECT_ROOT / rel).stat().st_size, rel
    hosts = {entry["url"].split("/")[2] for entry in entries.values()}
    assert hosts == {"www.dsec.gov.mo", "www.gov.cn", "cms.smg.gov.mo", "new-api.smg.gov.mo"}
    pdf_backed = [e for e in entries.values() if "pdf_sha256" in e]
    assert len(pdf_backed) == 104 + 9
    assert all(len(e["pdf_sha256"]) == 64 and e["pdf_bytes"] > 0 for e in pdf_backed)
