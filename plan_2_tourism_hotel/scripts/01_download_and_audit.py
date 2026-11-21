"""Phase A step 1: download the raw sources and write the Day-1 data audit.

uv run python scripts/01_download_and_audit.py                  # download, then audit
uv run python scripts/01_download_and_audit.py --skip-download  # audit the saved snapshots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from macao_tourism import audit
from macao_tourism.config import (
    CN_HOLIDAY_AMENDMENTS,
    CN_HOLIDAY_NOTICES,
    DATA_INTERIM_DIR,
    DSEC_HOTEL_PAGE,
    DSEC_TSDB_PAGE,
    DSEC_VISITOR_PAGE,
    OUTPUTS_METRICS_DIR,
    OUTPUTS_TABLES_DIR,
    PANEL_END,
    PANEL_START,
    PROJECT_ROOT,
    PROVENANCE_PATH,
    SMG_ANNUAL_STATS_PAGE,
    SMG_REPORT_PAGE,
    TIERS,
    TYPHOON_DAY_MIN_SIGNAL,
)
from macao_tourism.ingest import dsec, holidays, smg
from macao_tourism.ingest.http import Provenance


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _md_table(frame: pd.DataFrame) -> str:
    def cell(value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    header = "| " + " | ".join(frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def download_all() -> None:
    provenance = Provenance(PROVENANCE_PATH)
    dsec.download_tsdb(provenance)
    dsec.download_releases(provenance)
    holidays.download_notices(provenance)
    smg.download_smg(provenance)
    provenance.save()


def build_audit() -> dict[str, Any]:
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))["files"]
    downloaded = sorted(entry["downloaded_at_utc8"] for entry in provenance.values())

    # DSEC -----------------------------------------------------------------
    registry = dsec.verify_registry(dsec.load_indicator_meta())
    long = dsec.load_tsdb_long()
    coverage = audit.coverage_table(long, registry)
    identities = audit.identity_checks(long)
    crosscheck = audit.release_crosscheck(long, dsec.release_crosscheck_values())
    continuity, changes = audit.tier_continuity(long)
    reclass = audit.offsetting_reclassifications(changes)
    jumps = audit.supply_jumps(long)
    table_notes = dsec.release_table_notes()
    remarks = audit.remark_table(long)

    # Holidays -------------------------------------------------------------
    periods = holidays.load_holiday_periods()
    holiday_periods = holidays.periods_frame(periods)
    holiday_months = holidays.monthly_holiday_days(periods, PANEL_START, PANEL_END)

    # Typhoon signals ------------------------------------------------------
    events = smg.load_signal_events()
    daily = smg.daily_signals(smg.signal_intervals(events))
    signal_months = smg.monthly_signal_days(daily, PANEL_START, PANEL_END)
    annual = smg.annual_crosscheck(events)
    summaries = smg.monthly_summary_crosscheck(events)

    _write_csv(pd.DataFrame(registry), DATA_INTERIM_DIR / "dsec_indicator_registry.csv")
    _write_csv(long, DATA_INTERIM_DIR / "dsec_monthly_long.csv")
    _write_csv(holiday_periods, DATA_INTERIM_DIR / "cn_holiday_periods.csv")
    _write_csv(holiday_months, DATA_INTERIM_DIR / "cn_holiday_days_monthly.csv")
    _write_csv(events, DATA_INTERIM_DIR / "smg_signal_events.csv")
    _write_csv(daily, DATA_INTERIM_DIR / "smg_signal_days.csv")
    _write_csv(signal_months, DATA_INTERIM_DIR / "smg_signal_days_monthly.csv")
    _write_csv(coverage, OUTPUTS_TABLES_DIR / "day1_series_coverage.csv")
    _write_csv(crosscheck, OUTPUTS_TABLES_DIR / "day1_release_crosscheck.csv")
    _write_csv(continuity, OUTPUTS_TABLES_DIR / "day1_tier_continuity.csv")
    _write_csv(changes, OUTPUTS_TABLES_DIR / "day1_tier_establishment_changes.csv")
    _write_csv(jumps, OUTPUTS_TABLES_DIR / "day1_tier_room_jumps.csv")
    _write_csv(remarks, OUTPUTS_TABLES_DIR / "day1_dsec_remarks.csv")
    _write_csv(annual, OUTPUTS_TABLES_DIR / "day1_smg_annual_crosscheck.csv")
    _write_csv(summaries, OUTPUTS_TABLES_DIR / "day1_smg_monthly_summary_crosscheck.csv")

    tier_keys = [f"{m}_{t}" for m in audit.TIER_MEASURES for t in TIERS]
    tier_cov = coverage[coverage["key"].isin(tier_keys)]
    core_keys = [
        *tier_keys,
        "visitors_total",
        "visitors_same_day",
        "visitors_overnight",
        "visitors_mainland",
        "visitors_hong_kong",
        "visitors_taiwan",
        "package_tour_visitors",
    ]
    core_cov = coverage[coverage["key"].isin(core_keys)]
    annual_checked = annual[annual["in_annual_table"]]
    declared = holiday_periods.dropna(subset=["declared_days"])
    typhoon_days = daily[daily["max_signal"] >= TYPHOON_DAY_MIN_SIGNAL]
    in_window = typhoon_days[
        (typhoon_days["date"].str.slice(0, 7) >= PANEL_START)
        & (typhoon_days["date"].str.slice(0, 7) <= PANEL_END)
    ]

    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "step": "Phase A step 1: data download and Day-1 audit",
        "downloaded_between_utc8": [downloaded[0], downloaded[-1]],
        "raw_files_recorded": len(provenance),
        "panel_window": {
            "start": PANEL_START,
            "end": PANEL_END,
            "tiers": list(TIERS),
            "months_by_period": audit.period_counts(),
        },
        "dsec": {
            "pages": [DSEC_VISITOR_PAGE, DSEC_HOTEL_PAGE, DSEC_TSDB_PAGE],
            "series_downloaded": int(len(coverage)),
            "registry_verified_against_dsec_metadata": True,
            "tier_series": int(len(tier_cov)),
            "tier_series_complete_in_panel_window": int(
                (tier_cov["panel_months_missing"] == 0).sum()
            ),
            "tier_month_observations": int(
                tier_cov.loc[
                    tier_cov["key"].str.startswith("occupancy_"), "panel_months_with_value"
                ].sum()
            ),
            "core_series_with_gaps": {
                row.key: {"missing": int(row.panel_months_missing), "span": row.missing_span}
                for row in core_cov.itertuples()
                if row.panel_months_missing
            },
            "earliest_month_by_series": dict(
                zip(coverage["key"], coverage["first_month"], strict=True)
            ),
            "latest_month_by_series": dict(
                zip(coverage["key"], coverage["last_month"], strict=True)
            ),
            "identity_checks": identities,
            "release_crosscheck": {
                "values_compared": int(len(crosscheck)),
                "values_matching": int(crosscheck["match"].sum()),
                "files": sorted(crosscheck["xlsx_file"].unique()),
                "periods": sorted(crosscheck["period"].unique()),
            },
            "tier_classification": {
                "all_four_tiers_every_month": bool(
                    (
                        continuity[[f"{m}_months" for m in audit.TIER_MEASURES]]
                        == len(audit.panel_months())
                    )
                    .all()
                    .all()
                ),
                "months_with_establishment_count_change": int(len(changes)),
                "months_with_offsetting_moves_between_tiers": int(len(reclass)),
                "offsetting_moves": reclass.to_dict("records"),
                "tier_months_with_room_change_of_10pct_or_more": {
                    label: int((jumps["period_label"] == label).sum()) for label in audit.PERIODS
                },
                "largest_room_change": jumps.loc[
                    jumps["rooms_change_pct"].abs().idxmax()
                ].to_dict(),
                "dsec_table_notes": table_notes,
                "non_star_class": {
                    "guest_houses_last_month": _last(coverage, "occupancy_guesthouses"),
                    "economical_accommodation_first_month": _first(
                        coverage, "occupancy_economical"
                    ),
                },
                "distinct_remarks_on_tier_series": sorted(
                    remarks.loc[remarks["key"].isin(tier_keys), "remark"].unique()
                ),
            },
        },
        "cn_holidays": {
            "notices": len(CN_HOLIDAY_NOTICES),
            "amending_notices": len(CN_HOLIDAY_AMENDMENTS),
            "years": [min(CN_HOLIDAY_NOTICES), max(CN_HOLIDAY_NOTICES)],
            "periods": int(len(holiday_periods)),
            "periods_with_stated_length": int(len(declared)),
            "periods_matching_stated_length": int(
                (declared["n_days"] == declared["declared_days"]).sum()
            ),
            "holiday_days_in_panel_window": int(holiday_months["cn_holiday_days"].sum()),
            "golden_week_days_in_panel_window": int(holiday_months["cn_golden_week_days"].sum()),
            "months_with_holiday_days": int((holiday_months["cn_holiday_days"] > 0).sum()),
        },
        "smg_typhoon_signals": {
            "pages": [SMG_REPORT_PAGE, SMG_ANNUAL_STATS_PAGE],
            "reports_parsed": int(events["report"].nunique()),
            "signal_records": int(len(events)),
            "first_record": events["issued_at"].min(),
            "last_record": events["issued_at"].max(),
            "typhoon_day_definition": (
                f"calendar day (Macao time) on which signal No. {TYPHOON_DAY_MIN_SIGNAL} "
                "or higher was in force for any length of time"
            ),
            "typhoon_days_in_panel_window": int(len(in_window)),
            "months_with_typhoon_days": int((signal_months["typhoon_days_signal8plus"] > 0).sum()),
            "typhoon_days_by_year": {
                str(year): int((in_window["date"].str.slice(0, 4) == str(year)).sum())
                for year in range(int(PANEL_START[:4]), int(PANEL_END[:4]) + 1)
            },
            "annual_table_years_checked": int(len(annual_checked)),
            "annual_table_years_matching": int(annual_checked["match"].sum()),
            "annual_table_mismatches": annual_checked.loc[
                ~annual_checked["match"].astype(bool)
            ].to_dict("records"),
            "years_not_in_annual_table": annual.loc[~annual["in_annual_table"], "year"].tolist(),
            "monthly_summaries_checked": int(len(summaries)),
            "monthly_summaries_matching": int(
                (summaries["match"] & summaries["is_summary_of_month"]).sum()
            ),
        },
    }
    OUTPUTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_METRICS_DIR / "day1_data_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUTS_TABLES_DIR / "day1_data_audit.md").write_text(
        render_markdown(
            result, coverage, crosscheck, continuity, reclass, jumps, remarks, annual, summaries
        ),
        encoding="utf-8",
    )
    return result


def _first(coverage: pd.DataFrame, key: str) -> str:
    return str(coverage.loc[coverage["key"] == key, "first_month"].iloc[0])


def _last(coverage: pd.DataFrame, key: str) -> str:
    return str(coverage.loc[coverage["key"] == key, "last_month"].iloc[0])


def render_markdown(
    result: dict[str, Any],
    coverage: pd.DataFrame,
    crosscheck: pd.DataFrame,
    continuity: pd.DataFrame,
    reclass: pd.DataFrame,
    jumps: pd.DataFrame,
    remarks: pd.DataFrame,
    annual: pd.DataFrame,
    summaries: pd.DataFrame,
) -> str:
    d, h, s = result["dsec"], result["cn_holidays"], result["smg_typhoon_signals"]
    tc = d["tier_classification"]
    window = result["panel_window"]
    counts = window["months_by_period"]
    ident = d["identity_checks"]
    lines = [
        "# Plan 2 Day-1 data audit",
        "",
        "Written by `scripts/01_download_and_audit.py` from the raw snapshots in `data/raw/`. "
        f"Downloads ran between {result['downloaded_between_utc8'][0]} and "
        f"{result['downloaded_between_utc8'][1]} (UTC+8); `data/raw/data_provenance.json` "
        f"lists the URL, time, size, and SHA-256 of all {result['raw_files_recorded']} files.",
        "",
        "## Panel window",
        "",
        f"- {window['start']} to {window['end']}: {counts['all_months']} months × "
        f"{len(window['tiers'])} star classes = "
        f"{counts['all_months'] * len(window['tiers'])} tier-months.",
        f"- Pre-pandemic {counts['pre_pandemic']} months, 2020-01 transition "
        f"{counts['transition_2020_01']}, pandemic {counts['pandemic']}, "
        f"post-pandemic {counts['post_pandemic']}.",
        "",
        "## DSEC series (plan items 1 and 2)",
        "",
        f"- {d['series_downloaded']} monthly series from the DSEC Time Series Database, the "
        "database behind the `id=402` and `id=403` statistics pages. Every indicator id was "
        "checked against DSEC's own label, unit, and parent node before use.",
        f"- {d['tier_series_complete_in_panel_window']} of {d['tier_series']} star-class series "
        "(occupancy, guest rooms, operating establishments, guests × 4 classes) have a value "
        f"in every month of the window: {d['tier_month_observations']} tier-month occupancy "
        "observations.",
        "- Core series with gaps in the window: "
        + (
            "; ".join(
                f"`{key}` {gap['missing']} months ({gap['span']})"
                for key, gap in d["core_series_with_gaps"].items()
            )
            or "none"
        )
        + ".",
        "- Same-day + overnight = total visitors in "
        f"{ident['same_day_plus_overnight_equals_total']['months_exact']} of "
        f"{ident['same_day_plus_overnight_equals_total']['months_compared']} months; "
        "sea + land + air = total in "
        f"{ident['sea_plus_land_plus_air_equals_total']['months_exact']} of "
        f"{ident['sea_plus_land_plus_air_equals_total']['months_compared']}.",
        "- Four-class sum against the DSEC hotel total: rooms exact in "
        f"{ident['tier_rooms_sum_equals_hotel_rooms']['months_exact']} of "
        f"{ident['tier_rooms_sum_equals_hotel_rooms']['months_compared']} months, "
        "establishments in "
        f"{ident['tier_establishments_sum_equals_hotels']['months_exact']} of "
        f"{ident['tier_establishments_sum_equals_hotels']['months_compared']}, guests in "
        f"{ident['tier_guests_sum_equals_hotel_guests']['months_exact']} of "
        f"{ident['tier_guests_sum_equals_hotel_guests']['months_compared']}.",
        f"- Latest Excel releases ({', '.join(d['release_crosscheck']['files'])}): "
        f"{d['release_crosscheck']['values_matching']} of "
        f"{d['release_crosscheck']['values_compared']} printed values equal the database values.",
        "",
        "### Coverage by series",
        "",
        _md_table(
            coverage[
                [
                    "key",
                    "indicator_id",
                    "unit",
                    "first_month",
                    "last_month",
                    "panel_months_with_value",
                    "panel_months_missing",
                    "missing_span",
                    "zero_by_remark",
                ]
            ]
        ),
        "",
        "### Star-class continuity (plan item 2)",
        "",
        _md_table(
            continuity[
                [
                    "tier",
                    "occupancy_months",
                    "rooms_months",
                    "establishments_pre_pandemic",
                    "establishments_pandemic",
                    "establishments_post_pandemic",
                    "months_establishment_count_changed",
                    "largest_monthly_room_change_pct",
                    "month_of_largest_room_change",
                    "occupancy_min",
                    "occupancy_max",
                ]
            ]
        ),
        "",
        f"- The four star classes are published under the same labels in all "
        f"{counts['all_months']} months: "
        f"{'yes' if tc['all_four_tiers_every_month'] else 'NO'}.",
        "- The class below 2-star changed its label: `Guest-houses` ends "
        f"{tc['non_star_class']['guest_houses_last_month']}, "
        "`Economical accommodation establishments` starts "
        f"{tc['non_star_class']['economical_accommodation_first_month']}. "
        "It is outside the four classes analysed.",
        f"- The number of operating establishments in a class changed in "
        f"{tc['months_with_establishment_count_change']} tier-months "
        "(`day1_tier_establishment_changes.csv`). In "
        f"{tc['months_with_offsetting_moves_between_tiers']} months one "
        "class lost establishments while another gained them; the database does not say "
        "whether these are re-ratings or unrelated openings and closures.",
        "- Guest rooms of a class moved by 10% or more from one month to the next in "
        + ", ".join(
            f"{count} tier-months ({label})"
            for label, count in tc["tier_months_with_room_change_of_10pct_or_more"].items()
        )
        + f". The largest is {tc['largest_room_change']['tier']} in "
        f"{tc['largest_room_change']['period']}: {tc['largest_room_change']['rooms_before']} to "
        f"{tc['largest_room_change']['rooms_after']} rooms "
        f"({tc['largest_room_change']['rooms_change_pct']:+.1f}%).",
        "- DSEC note under the star-class tables: "
        + " ".join(f'"{note}"' for note in tc["dsec_table_notes"]),
        "- Footnotes DSEC attaches to star-class values in the database: "
        + ("; ".join(tc["distinct_remarks_on_tier_series"]) or "none")
        + ".",
        "- Reading: the class labels are continuous, the membership of each class is not "
        "fixed. Guest rooms per class are in the data so that later steps can control for "
        "supply; no class is merged, dropped, or re-labelled.",
        "",
    ]
    if len(reclass):
        lines += ["Months with offsetting moves between classes:", "", _md_table(reclass), ""]
    lines += [
        "Months with a room change of 10% or more:",
        "",
        _md_table(jumps.drop(columns=["period_label"])),
        "",
    ]
    if len(remarks):
        lines += ["### DSEC footnotes in the window", "", _md_table(remarks), ""]
    lines += [
        "### Latest release against the database",
        "",
        _md_table(crosscheck[["key", "period", "xlsx_value", "xlsx_scale", "tsdb_value", "match"]]),
        "",
        "## Mainland public holidays (plan item 4)",
        "",
        f"- {h['notices']} State Council notices ({h['years'][0]}–{h['years'][1]}) and "
        f"{h['amending_notices']} later notices that changed a schedule (September 2015, "
        "May 2019, Spring Festival 2020).",
        f"- {h['periods']} holiday periods. {h['periods_matching_stated_length']} of the "
        f"{h['periods_with_stated_length']} periods whose notice states a length parse to "
        "exactly that length.",
        "- Period rule: the days a notice declares off, joined with adjoining Saturdays and "
        "Sundays that the notice does not declare as working days.",
        f"- {h['holiday_days_in_panel_window']} holiday days in the window "
        f"({h['golden_week_days_in_panel_window']} in Spring Festival or National Day "
        f"periods), in {h['months_with_holiday_days']} months.",
        "",
        "## SMG tropical cyclone signals (plan item 4)",
        "",
        f"- {s['reports_parsed']} SMG tropical cyclone reports, {s['signal_records']} signal "
        f"records from {s['first_record']} to {s['last_record']}.",
        f"- Typhoon day: {s['typhoon_day_definition']}.",
        f"- {s['typhoon_days_in_panel_window']} typhoon days in the window, in "
        f"{s['months_with_typhoon_days']} months. By year: "
        + ", ".join(f"{year}: {count}" for year, count in s["typhoon_days_by_year"].items())
        + ".",
        f"- SMG's annual signal table agrees with the parsed reports in "
        f"{s['annual_table_years_matching']} of {s['annual_table_years_checked']} years "
        "(number of cyclones reaching signals 1, 3, 8, 9, and 10; dates on which the first "
        "and the last cyclone of the year got their first signal).",
        "- Not yet in the annual table: "
        + ", ".join(str(year) for year in s["years_not_in_annual_table"])
        + f". For its months in the window, {s['monthly_summaries_matching']} of "
        f"{s['monthly_summaries_checked']} SMG monthly climate summaries list the same "
        "signal records as the reports, and the summaries of months without a report have "
        "no tropical cyclone section.",
        "",
        "### Annual table against parsed reports",
        "",
        _md_table(annual),
        "",
        "### Monthly climate summaries against parsed reports",
        "",
        _md_table(summaries),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-download", action="store_true", help="audit the snapshots already in data/raw"
    )
    args = parser.parse_args()
    if not args.skip_download:
        download_all()
    result = build_audit()
    summary = {
        "tier_month_observations": result["dsec"]["tier_month_observations"],
        "release_values_matching": result["dsec"]["release_crosscheck"]["values_matching"],
        "holiday_periods": result["cn_holidays"]["periods"],
        "typhoon_days": result["smg_typhoon_signals"]["typhoon_days_in_panel_window"],
        "annual_table_years_matching": result["smg_typhoon_signals"]["annual_table_years_matching"],
    }
    print(json.dumps(summary, indent=2))
    print(f"Audit written under {OUTPUTS_TABLES_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
