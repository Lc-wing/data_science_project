"""Strict 16:00 America/New_York cutoff (13:00 ET on NYSE early-close days), dedup & <=5/day cap."""

from __future__ import annotations

import bisect
import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from stock_volatility import config

# All 15 NYSE scheduled 13:00 America/New_York early-close trading days in 2019-2025
# (day before Independence Day when weekday, Black Friday, and Christmas Eve when weekday).
NYSE_EARLY_CLOSE_DATES: frozenset[str] = frozenset(
    {
        "2019-07-03",
        "2019-11-29",
        "2019-12-24",
        "2020-11-27",
        "2020-12-24",
        "2021-11-26",
        "2022-11-25",
        "2023-07-03",
        "2023-11-24",
        "2024-07-03",
        "2024-11-29",
        "2024-12-24",
        "2025-07-03",
        "2025-11-28",
        "2025-12-24",
    }
)
EARLY_CLOSE_HOUR: int = 13

EVENTS_ALIGNED_COLUMNS: tuple[str, ...] = (
    "ticker",
    "cik",
    "accession_number",
    "form",
    "filing_date",
    "report_date",
    "acceptance_datetime_utc",
    "acceptance_datetime_et",
    "et_date",
    "et_tz_abbr",
    "aligned_date",
    "is_early_close_day",
    "shifted_due_to_early_close",
    "shift_reason",
    "items",
    "item_descriptions",
    "primary_doc_description",
    "headline_text",
    "news_source",
)

EVENTS_PHASE_B_COLUMNS: tuple[str, ...] = (
    "ticker",
    "date",
    "split",
    "has_news",
    "news_count",
    "news_source",
    "forms",
    "item_codes",
    "titles_capped_5",
    "text_concat_capped_5",
)


def _clean_str(val: Any, default: str = "") -> str:
    """Convert optional scalar to clean string, treating None and NaN as default."""
    if val is None or pd.isna(val):
        return default
    text = str(val).strip()
    if not text or text.lower() == "nan":
        return default
    return str(val)


def _normalize_headline_key(text: str) -> str:
    """Return a whitespace-collapsed lowercase key for intra-day headline deduplication."""
    return re.sub(r"\s+", " ", _clean_str(text).strip().lower())


def _has_item_202(items_csv: str) -> bool:
    """Return True iff comma-separated SEC Form 8-K items include Item 2.02 (earnings)."""
    clean = _clean_str(items_csv)
    if not clean:
        return False
    return "2.02" in {tok.strip() for tok in clean.split(",") if tok.strip()}


def align_events_to_trading_days(
    events_raw_df: pd.DataFrame,
    trading_dates: Sequence[str],
    *,
    cutoff_tz: str = config.CUTOFF_TZ,
    cutoff_hour: int = config.CUTOFF_HOUR,
    early_close_dates: frozenset[str] = NYSE_EARLY_CLOSE_DATES,
    early_close_hour: int = EARLY_CLOSE_HOUR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align UTC corporate events strictly to NYSE trading days with zero post-close leakage.

    Cutoff rule:
    - Convert `acceptance_datetime_utc` to `America/New_York` (DST-aware).
    - If `et_date` is a normal NYSE trading day and `et_time < 16:00:00`: assign to `et_date`.
    - If `et_date` is an NYSE early-close day and `et_time < 13:00:00`: assign to `et_date`.
    - Otherwise (at/after 16:00 ET, at/after 13:00 ET on early-close day, or weekend/holiday):
      shift forward to the next NYSE trading day `t+1` strictly greater than `et_date`.
    - Events after the final calendar day's cutoff are dropped and counted explicitly.
    """
    cal_list = sorted({str(d) for d in trading_dates})
    if not cal_list:
        raise ValueError("trading_dates cannot be empty")
    cal_set = set(cal_list)
    tz = ZoneInfo(cutoff_tz)

    records: list[dict[str, Any]] = []
    stats: dict[str, int] = {
        "total_input_events": len(events_raw_df),
        "on_trading_day_before_cutoff": 0,
        "shifted_after_1600_et": 0,
        "shifted_due_to_early_close": 0,
        "shifted_non_trading_day": 0,
        "dropped_after_end_date": 0,
        "dst_edt_events": 0,
        "dst_est_events": 0,
    }
    early_close_shifted_examples: list[dict[str, str]] = []

    for row in events_raw_df.to_dict(orient="records"):
        utc_str = _clean_str(row["acceptance_datetime_utc"]).strip()
        if utc_str.endswith("Z"):
            utc_str = f"{utc_str[:-1]}+00:00"
        dt_utc = datetime.fromisoformat(utc_str)
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=UTC)
        dt_et = dt_utc.astimezone(tz)
        et_date = dt_et.strftime("%Y-%m-%d")
        tz_abbr = dt_et.tzname() or "ET"
        if tz_abbr == "EDT":
            stats["dst_edt_events"] += 1
        elif tz_abbr == "EST":
            stats["dst_est_events"] += 1

        is_early = et_date in early_close_dates
        effective_cutoff = early_close_hour if is_early else cutoff_hour
        shifted_early = False

        if et_date in cal_set:
            if dt_et.hour < effective_cutoff:
                aligned_date: str | None = et_date
                shift_reason = "same_day_pre_close"
                stats["on_trading_day_before_cutoff"] += 1
            else:
                idx = bisect.bisect_right(cal_list, et_date)
                aligned_date = cal_list[idx] if idx < len(cal_list) else None
                if aligned_date is None:
                    stats["dropped_after_end_date"] += 1
                    continue
                if is_early and dt_et.hour < cutoff_hour:
                    shifted_early = True
                    shift_reason = "shifted_early_close_1300_et"
                    stats["shifted_due_to_early_close"] += 1
                    early_close_shifted_examples.append(
                        {
                            "ticker": _clean_str(row["ticker"]),
                            "form": _clean_str(row["form"]),
                            "accession_number": _clean_str(row["accession_number"]),
                            "acceptance_datetime_et": dt_et.isoformat(timespec="seconds"),
                            "aligned_date": aligned_date,
                        }
                    )
                else:
                    shift_reason = "shifted_after_1600_et"
                    stats["shifted_after_1600_et"] += 1
        else:
            idx = bisect.bisect_right(cal_list, et_date)
            aligned_date = cal_list[idx] if idx < len(cal_list) else None
            if aligned_date is None:
                stats["dropped_after_end_date"] += 1
                continue
            shift_reason = "shifted_non_trading_day"
            stats["shifted_non_trading_day"] += 1

        records.append(
            {
                "ticker": _clean_str(row["ticker"]),
                "cik": _clean_str(row["cik"]),
                "accession_number": _clean_str(row["accession_number"]),
                "form": _clean_str(row["form"]),
                "filing_date": _clean_str(row["filing_date"]),
                "report_date": _clean_str(row.get("report_date", "")),
                "acceptance_datetime_utc": utc_str,
                "acceptance_datetime_et": dt_et.isoformat(timespec="seconds"),
                "et_date": et_date,
                "et_tz_abbr": tz_abbr,
                "aligned_date": aligned_date,
                "is_early_close_day": bool(is_early),
                "shifted_due_to_early_close": bool(shifted_early),
                "shift_reason": shift_reason,
                "items": _clean_str(row.get("items", "")),
                "item_descriptions": _clean_str(row.get("item_descriptions", "")),
                "primary_doc_description": _clean_str(row.get("primary_doc_description", "")),
                "headline_text": _clean_str(row["headline_text"]),
                "news_source": _clean_str(row.get("news_source"), default="sec_edgar"),
            }
        )

    aligned_df = pd.DataFrame(records, columns=list(EVENTS_ALIGNED_COLUMNS))
    if not aligned_df.empty:
        aligned_df = aligned_df.sort_values(
            ["ticker", "aligned_date", "acceptance_datetime_utc", "accession_number"],
            kind="mergesort",
        ).reset_index(drop=True)

    audit: dict[str, Any] = {
        **stats,
        "retained_after_alignment": len(aligned_df),
        "cutoff_tz": cutoff_tz,
        "cutoff_hour": cutoff_hour,
        "early_close_hour": early_close_hour,
        "early_close_dates_count": len(early_close_dates),
        "early_close_shifted_examples": early_close_shifted_examples,
    }
    return aligned_df, audit


def dedup_and_cap_events(
    events_aligned_df: pd.DataFrame,
    *,
    max_events_per_day: int = config.MAX_EVENTS_PER_DAY,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Deduplicate intra-day headlines and cap each (ticker, aligned_date) at <= 5 latest items."""
    if max_events_per_day <= 0:
        raise ValueError("max_events_per_day must be positive")
    if events_aligned_df.empty:
        return events_aligned_df.copy(), {
            "input_rows": 0,
            "dedup_accession_dropped": 0,
            "dedup_headline_dropped": 0,
            "capped_over_5_dropped": 0,
            "ticker_days_capped": 0,
            "output_rows": 0,
            "active_ticker_days": 0,
            "max_events_per_day_after_cap": 0,
        }

    df = events_aligned_df.copy()
    n_input = len(df)
    df = df.sort_values(
        ["ticker", "aligned_date", "acceptance_datetime_utc", "accession_number"],
        kind="mergesort",
    )

    # 1. Accession-level safety dedup
    df = df.drop_duplicates(subset=["ticker", "accession_number"], keep="last")
    n_after_acc = len(df)

    # 2. Intra-day normalized headline dedup keeping latest timestamp
    df["_hl_key"] = df["headline_text"].map(_normalize_headline_key)
    df = df.drop_duplicates(subset=["ticker", "aligned_date", "_hl_key"], keep="last")
    df = df.drop(columns=["_hl_key"])
    n_after_hl = len(df)

    # 3. Cap each (ticker, aligned_date) to the latest `max_events_per_day` events
    group_sizes = df.groupby(["ticker", "aligned_date"]).size()
    ticker_days_capped = int((group_sizes > max_events_per_day).sum())

    capped_df = (
        df.sort_values(
            ["ticker", "aligned_date", "acceptance_datetime_utc", "accession_number"],
            kind="mergesort",
        )
        .groupby(["ticker", "aligned_date"], sort=False, group_keys=False)
        .tail(max_events_per_day)
        .sort_values(
            ["ticker", "aligned_date", "acceptance_datetime_utc", "accession_number"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    n_output = len(capped_df)

    post_sizes = capped_df.groupby(["ticker", "aligned_date"]).size()
    max_after_cap = int(post_sizes.max()) if not post_sizes.empty else 0
    if max_after_cap > max_events_per_day:
        raise AssertionError(f"Capping invariant violated: max={max_after_cap}")

    audit = {
        "input_rows": n_input,
        "dedup_accession_dropped": n_input - n_after_acc,
        "dedup_headline_dropped": n_after_acc - n_after_hl,
        "capped_over_5_dropped": n_after_hl - n_output,
        "ticker_days_capped": ticker_days_capped,
        "output_rows": n_output,
        "active_ticker_days": int(len(post_sizes)),
        "max_events_per_day_after_cap": max_after_cap,
    }
    return capped_df, audit


def build_daily_event_features_and_phase_b_text(
    events_capped_df: pd.DataFrame,
    panel_keys_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate capped events to daily panel features and frozen Phase B text table.

    `panel_keys_df` must contain `[ticker, date, split]` sorted by `(ticker, date)`.
    """
    keys = (
        panel_keys_df[["ticker", "date", "split"]]
        .sort_values(["ticker", "date"], kind="mergesort")
        .reset_index(drop=True)
    )

    if events_capped_df.empty:
        daily_df = keys[["ticker", "date"]].copy()
        daily_df["has_news"] = 0
        daily_df["news_count"] = 0
        daily_df["has_8k_earnings"] = 0
        daily_df["news_source"] = "none"

        phase_b_df = keys.copy()
        phase_b_df["has_news"] = 0
        phase_b_df["news_count"] = 0
        phase_b_df["news_source"] = "none"
        phase_b_df["forms"] = ""
        phase_b_df["item_codes"] = ""
        phase_b_df["titles_capped_5"] = "[]"
        phase_b_df["text_concat_capped_5"] = ""
        return daily_df, phase_b_df[list(EVENTS_PHASE_B_COLUMNS)]

    agg_records: list[dict[str, Any]] = []
    for (ticker, aligned_date), grp in events_capped_df.groupby(
        ["ticker", "aligned_date"], sort=True
    ):
        g = grp.sort_values(
            ["acceptance_datetime_utc", "accession_number"], kind="mergesort"
        )
        titles = [str(t) for t in g["headline_text"].tolist()]
        forms_unique = list(dict.fromkeys(str(f) for f in g["form"].tolist() if str(f)))
        item_tokens: list[str] = []
        for raw_items in g["items"].tolist():
            for tok in str(raw_items or "").split(","):
                tok_c = tok.strip()
                if tok_c and tok_c not in item_tokens:
                    item_tokens.append(tok_c)
        has_202 = int(any(_has_item_202(str(x)) for x in g["items"].tolist()))
        sources = list(dict.fromkeys(str(s) for s in g["news_source"].tolist() if str(s)))
        news_src = sources[0] if len(sources) == 1 else ",".join(sources)

        agg_records.append(
            {
                "ticker": str(ticker),
                "date": str(aligned_date),
                "has_news": 1,
                "news_count": len(titles),
                "has_8k_earnings": has_202,
                "news_source": news_src or "sec_edgar",
                "forms": ",".join(forms_unique),
                "item_codes": ",".join(item_tokens),
                "titles_capped_5": json.dumps(titles, ensure_ascii=False),
                "text_concat_capped_5": " || ".join(titles),
            }
        )

    agg_df = pd.DataFrame(agg_records)
    merged = keys.merge(agg_df, on=["ticker", "date"], how="left")
    merged["has_news"] = merged["has_news"].fillna(0).astype(int)
    merged["news_count"] = merged["news_count"].fillna(0).astype(int)
    merged["has_8k_earnings"] = merged["has_8k_earnings"].fillna(0).astype(int)
    merged["news_source"] = merged["news_source"].fillna("none").astype(str)
    merged["forms"] = merged["forms"].fillna("").astype(str)
    merged["item_codes"] = merged["item_codes"].fillna("").astype(str)
    merged["titles_capped_5"] = merged["titles_capped_5"].fillna("[]").astype(str)
    merged["text_concat_capped_5"] = merged["text_concat_capped_5"].fillna("").astype(str)

    daily_features_df = merged[
        ["ticker", "date", "has_news", "news_count", "has_8k_earnings", "news_source"]
    ].copy()
    phase_b_df = merged[list(EVENTS_PHASE_B_COLUMNS)].copy()
    return daily_features_df, phase_b_df
