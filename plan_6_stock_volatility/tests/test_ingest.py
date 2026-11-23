"""Comprehensive unit, offline replay, and committed-artifact contract tests for Milestone M1."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from stock_volatility import config

# ---------------------------------------------------------------------------
# Layer 1: Pure Offline Unit Tests (Zero Network Calls)
# ---------------------------------------------------------------------------


def test_provenance_log_record_verify_and_tamper_detection(
    tmp_path: Path, block_network: None
) -> None:
    """Verify ProvenanceLog records file hashes/sizes and detects tampering offline."""
    from stock_volatility.ingest import ProvenanceLog

    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sample_file = raw_dir / "sample.csv"
    sample_file.write_text("ticker,price\nAAPL,150.0\n", encoding="utf-8")

    prov = ProvenanceLog(
        project_root=tmp_path,
        generated_at_utc="2026-11-22T00:00:00+00:00",
    )
    entry = prov.record_file(
        sample_file,
        source="unit_test",
        source_url="https://example.edu/sample.csv",
        http_status=200,
        fetched_at_utc="2026-11-22T00:00:00+00:00",
        row_count=1,
        description="Synthetic sample file",
    )
    assert entry["path"] == "data/raw/sample.csv"
    assert entry["byte_size"] == sample_file.stat().st_size
    assert entry["sha256"] == config.sha256_file(sample_file)
    assert entry["row_count"] == 1

    prov_path = raw_dir / "data_provenance.json"
    prov.save(prov_path)
    loaded = ProvenanceLog.load(prov_path, project_root=tmp_path)
    loaded.verify_files()

    # Tamper with sample_file and verify ValueError is raised
    sample_file.write_text("ticker,price\nAAPL,999.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        loaded.verify_files()


def test_resolve_universe_ciks_and_predecessor_linkages(
    tmp_path: Path, block_network: None
) -> None:
    """Verify 60-ticker CIK resolution, 10-digit zero-padding, BLK/XOM predecessors, and CSV I/O."""
    from stock_volatility.ingest.universe import (
        PREDECESSOR_CIK_MAP,
        build_universe_table,
        load_universe_csv,
        resolve_universe_ciks,
        write_universe_csv,
    )

    assert PREDECESSOR_CIK_MAP == {
        "BLK": ("0001364742",),
        "XOM": ("0000034088",),
    }

    payload: dict[str, dict[str, Any]] = {}
    for idx, ticker in enumerate(config.UNIVERSE_TICKERS, start=1):
        cik_int = 2012383 if ticker == "BLK" else (2115436 if ticker == "XOM" else 1000 + idx)
        payload[str(idx - 1)] = {
            "cik_str": cik_int,
            "ticker": ticker,
            "title": f"{ticker} Corp",
        }

    cik_map = resolve_universe_ciks(payload)
    assert len(cik_map) == 60
    assert cik_map["AAPL"]["cik"] == "0000001001"
    assert cik_map["BLK"]["cik"] == "0002012383"
    assert cik_map["BLK"]["predecessor_ciks"] == ["0001364742"]
    assert cik_map["XOM"]["cik"] == "0002115436"
    assert cik_map["XOM"]["predecessor_ciks"] == ["0000034088"]

    # Missing ticker must raise ValueError
    incomplete_payload = {k: v for k, v in payload.items() if v["ticker"] != "NVDA"}
    with pytest.raises(ValueError, match="NVDA"):
        resolve_universe_ciks(incomplete_payload)

    # Build synthetic OHLCV with 2 trading days per ticker
    rows = []
    for ticker in config.UNIVERSE_TICKERS:
        for dt in ("2019-01-02", "2019-01-03"):
            rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.5,
                    "close": 10.5,
                    "adj_close": 10.5,
                    "volume": 1000,
                    "dividends": 0.0,
                    "stock_splits": 0.0,
                }
            )
    ohlcv_df = pd.DataFrame(rows)
    uni_df = build_universe_table(ohlcv_df, cik_map, expected_trading_days=2)
    assert list(uni_df.columns) == [
        "ticker",
        "sector",
        "sub_group",
        "cik",
        "company_name",
        "first_trade_date",
        "last_trade_date",
        "n_trading_days",
    ]
    assert len(uni_df) == 60
    assert (uni_df["n_trading_days"] == 2).all()

    csv_path = tmp_path / "universe_60.csv"
    write_universe_csv(uni_df, csv_path)
    reloaded = load_universe_csv(csv_path)
    assert reloaded["cik"].dtype == object
    assert reloaded.loc[reloaded["ticker"] == "AAPL", "cik"].iloc[0] == "0000001001"


def test_normalize_yfinance_frame_multiindex_and_validation(block_network: None) -> None:
    """Verify normalize_yfinance_frame handles MultiIndex, flat columns, and validation errors."""
    from stock_volatility.ingest.prices import (
        OHLCV_COLUMNS,
        exclusive_end_date,
        normalize_yfinance_frame,
    )

    assert exclusive_end_date("2025-12-31") == "2026-01-01"

    idx = pd.to_datetime(["2019-01-02", "2019-01-03", "2026-01-01"]).tz_localize(
        "America/New_York"
    )
    # 1. (Field, Ticker) MultiIndex with extra Capital Gains column and NaN dividends/splits
    cols_ft = pd.MultiIndex.from_tuples(
        [
            ("Open", "MSFT"),
            ("High", "MSFT"),
            ("Low", "MSFT"),
            ("Close", "MSFT"),
            ("Adj Close", "MSFT"),
            ("Volume", "MSFT"),
            ("Capital Gains", "MSFT"),
            ("Open", "AAPL"),
            ("High", "AAPL"),
            ("Low", "AAPL"),
            ("Close", "AAPL"),
            ("Adj Close", "AAPL"),
            ("Volume", "AAPL"),
            ("Dividends", "AAPL"),
            ("Stock Splits", "AAPL"),
        ]
    )
    data_ft = np.array(
        [
            [100, 102, 99, 101, 95, 5000, 0, 150, 155, 149, 152, 148, 8000, np.nan, 0.0],
            [101, 103, 100, 102, 96, 5100, 0, 152, 156, 151, 154, 150, 8200, 0.75, 4.0],
            [102, 104, 101, 103, 97, 5200, 0, 154, 158, 153, 157, 153, 8300, 0.0, 0.0],
        ],
        dtype=float,
    )
    raw_ft = pd.DataFrame(data_ft, index=idx, columns=cols_ft)
    norm_ft = normalize_yfinance_frame(
        raw_ft,
        ["MSFT", "AAPL"],
        start_date="2019-01-02",
        end_date="2025-12-31",
    )
    assert tuple(norm_ft.columns) == OHLCV_COLUMNS
    # 2026-01-01 filtered out; sorted by (ticker, date) -> AAPL then MSFT
    assert len(norm_ft) == 4
    assert list(norm_ft["ticker"]) == ["AAPL", "AAPL", "MSFT", "MSFT"]
    assert list(norm_ft["date"]) == ["2019-01-02", "2019-01-03", "2019-01-02", "2019-01-03"]
    assert norm_ft["dividends"].isna().sum() == 0
    assert norm_ft.loc[0, "dividends"] == 0.0
    assert norm_ft.loc[1, "stock_splits"] == 4.0

    # 2. (Ticker, Field) MultiIndex
    raw_tf = raw_ft.swaplevel(0, 1, axis=1)
    norm_tf = normalize_yfinance_frame(
        raw_tf,
        ["MSFT", "AAPL"],
        start_date="2019-01-02",
        end_date="2025-12-31",
    )
    pd.testing.assert_frame_equal(norm_ft, norm_tf)

    # 3. Single-ticker flat DataFrame
    flat_df = pd.DataFrame(
        {
            "Open": [50.0, 51.0],
            "High": [52.0, 53.0],
            "Low": [49.0, 50.0],
            "Close": [51.0, 52.0],
            "Adj Close": [50.5, 51.5],
            "Volume": [1000, 1100],
            "Capital Gains": [0.0, 0.0],
        },
        index=pd.to_datetime(["2019-01-02", "2019-01-03"]),
    )
    norm_flat = normalize_yfinance_frame(
        flat_df, ["SPY"], start_date="2019-01-02", end_date="2025-12-31"
    )
    assert len(norm_flat) == 2
    assert (norm_flat["ticker"] == "SPY").all()

    # 4. Invalid price ordering (high < low) must raise ValueError
    bad_df = flat_df.copy()
    bad_df.loc[bad_df.index[0], "High"] = 48.0
    with pytest.raises(ValueError, match="high < low"):
        normalize_yfinance_frame(bad_df, ["SPY"], start_date="2019-01-02", end_date="2025-12-31")


def test_align_macro_to_calendar_backward_only_ffill_never_backfills(
    tmp_path: Path, block_network: None
) -> None:
    """Verify align_macro_to_calendar uses backward-only ffill and rejects leading NaN."""
    from stock_volatility.ingest.macro import (
        MACRO_COLUMNS,
        align_macro_to_calendar,
        parse_fred_csv,
        parse_yahoo_v8_chart_payload,
    )

    fred_t10_csv = tmp_path / "fred_t10y2y.csv"
    fred_t10_csv.write_text(
        "observation_date,T10Y2Y\n"
        "2018-12-31,0.19\n"
        "2019-01-02,\n"
        "2019-01-03,0.18\n"
        "2019-01-04,.\n",
        encoding="utf-8",
    )
    t10_df = parse_fred_csv(fred_t10_csv, "T10Y2Y")
    assert list(t10_df.columns) == ["date", "value"]
    assert np.isnan(t10_df.loc[t10_df["date"] == "2019-01-02", "value"].iloc[0])
    assert np.isnan(t10_df.loc[t10_df["date"] == "2019-01-04", "value"].iloc[0])

    vixcls_csv = tmp_path / "fred_vixcls.csv"
    vixcls_csv.write_text(
        "observation_date,VIXCLS\n"
        "2018-12-31,25.42\n"
        "2019-01-02,23.22\n"
        "2019-01-03,25.45\n"
        "2019-01-04,21.38\n",
        encoding="utf-8",
    )
    vix_fred_df = parse_fred_csv(vixcls_csv, "VIXCLS")

    spy_df = pd.DataFrame(
        {
            "date": ["2019-01-02", "2019-01-03", "2019-01-04"],
            "open": [245.0, 248.0, 247.0],
            "high": [250.0, 249.0, 253.0],
            "low": [244.0, 243.0, 246.0],
            "close": [250.0, 244.0, 252.0],
            "adj_close": [230.0, 224.48, 232.0],
            "volume": [100000, 110000, 120000],
            "dividends": [0.0, 0.0, 0.0],
            "stock_splits": [0.0, 0.0, 0.0],
        }
    )
    vix_df = pd.DataFrame(
        {
            "date": ["2019-01-02", "2019-01-03", "2019-01-04"],
            "open": [23.0, 25.0, 22.0],
            "high": [24.0, 26.0, 23.0],
            "low": [22.0, 24.0, 21.0],
            "close": [23.22, 25.45, 21.38],
            "adj_close": [23.22, 25.45, 21.38],
            "volume": [0, 0, 0],
        }
    )

    macro_df, summary = align_macro_to_calendar(spy_df, vix_df, t10_df, vix_fred_df)
    assert tuple(macro_df.columns) == MACRO_COLUMNS
    assert len(macro_df) == 3
    # 2019-01-02 must be forward-filled from 2018-12-31 (0.19), NOT backfilled from 2019-01-03!
    assert macro_df.loc[0, "t10y2y"] == pytest.approx(0.19)
    assert bool(macro_df.loc[0, "t10y2y_ffilled_flag"]) is True
    assert macro_df.loc[1, "t10y2y"] == pytest.approx(0.18)
    assert bool(macro_df.loc[1, "t10y2y_ffilled_flag"]) is False
    assert macro_df.loc[2, "t10y2y"] == pytest.approx(0.18)
    assert bool(macro_df.loc[2, "t10y2y_ffilled_flag"]) is True
    assert summary["t10y2y_ffilled_count"] == 2
    assert summary["vix_fred_ffilled_count"] == 0
    assert np.isnan(macro_df.loc[0, "spy_ret_1d"])
    assert macro_df.loc[1, "spy_ret_1d"] == pytest.approx(224.48 / 230.0 - 1.0)

    # Verify that if 2018-12-31 is removed so 2019-01-02 has no prior history,
    # align_macro_to_calendar NEVER backfills from 2019-01-03 and instead raises ValueError!
    t10_no_prior = t10_df[t10_df["date"] >= "2019-01-02"].copy()
    with pytest.raises(ValueError, match="backfill|missing"):
        align_macro_to_calendar(spy_df, vix_df, t10_no_prior, vix_fred_df)

    # Verify Yahoo v8 chart JSON parser handles America/Chicago timezone and null holiday quotes
    v8_payload = {
        "chart": {
            "result": [
                {
                    "meta": {"exchangeTimezoneName": "America/Chicago"},
                    "timestamp": [1546439400, 1546525800],  # 2019-01-02 & 2019-01-03 08:30 CST
                    "indicators": {
                        "quote": [
                            {
                                "open": [23.0, None],
                                "high": [24.0, None],
                                "low": [22.0, None],
                                "close": [23.22, None],
                                "volume": [0, None],
                            }
                        ]
                    },
                }
            ]
        }
    }
    parsed_v8 = parse_yahoo_v8_chart_payload(
        v8_payload, start_date="2019-01-02", end_date="2025-12-31"
    )
    assert len(parsed_v8) == 1
    assert parsed_v8.loc[0, "date"] == "2019-01-02"
    assert parsed_v8.loc[0, "close"] == pytest.approx(23.22)


def test_sec_edgar_normalizers_archive_selection_and_predecessor_dedup(
    tmp_path: Path, block_network: None
) -> None:
    """Verify SEC timestamp/item normalizers, archive selection, parser, and BLK/XOM dedup."""
    from stock_volatility.ingest.sec_edgar import (
        EVENTS_RAW_COLUMNS,
        SEC_8K_ITEM_DESCRIPTIONS,
        build_events_raw_from_snapshot,
        build_headline_text,
        deduplicate_filings,
        describe_filing_items,
        normalize_acceptance_datetime,
        normalize_sec_items,
        parse_submissions_block,
        select_overlapping_archives,
    )

    # 1. Complete 33-item official Form 8-K dictionary including 1.05 & 2.02
    assert len(SEC_8K_ITEM_DESCRIPTIONS) == 33
    assert SEC_8K_ITEM_DESCRIPTIONS["1.05"] == "Material Cybersecurity Incidents"
    assert SEC_8K_ITEM_DESCRIPTIONS["2.02"] == "Results of Operations and Financial Condition"

    # 2. normalize_acceptance_datetime
    assert (
        normalize_acceptance_datetime("2025-12-05T21:31:42.000Z")
        == "2025-12-05T21:31:42+00:00"
    )
    assert (
        normalize_acceptance_datetime("2024-05-01T12:05:12-04:00")
        == "2024-05-01T16:05:12+00:00"
    )
    assert (
        normalize_acceptance_datetime("20190501160512")
        == "2019-05-01T16:05:12+00:00"
    )
    assert (
        normalize_acceptance_datetime("", fallback_filing_date="2021-03-15")
        == "2021-03-15T21:00:00+00:00"
    )
    with pytest.raises(ValueError, match="acceptanceDateTime"):
        normalize_acceptance_datetime("")

    # 3. normalize_sec_items & describe_filing_items
    assert normalize_sec_items(["2.02", "9.01", "2.02"]) == "2.02,9.01"
    assert normalize_sec_items(" 1.05, 8.01 ") == "1.05,8.01"
    assert normalize_sec_items("") == ""
    assert normalize_sec_items(None) == ""
    assert (
        describe_filing_items("8-K", "2.02,9.01")
        == "Results of Operations and Financial Condition; Financial Statements and Exhibits"
    )
    assert describe_filing_items("10-Q", "") == "Quarterly Report (Form 10-Q)"
    assert describe_filing_items("10-K", "") == "Annual Report (Form 10-K)"

    headline = build_headline_text(
        ticker="AAPL",
        form="8-K",
        items="2.02,9.01",
        item_descriptions=describe_filing_items("8-K", "2.02,9.01"),
        report_date="2025-09-27",
        primary_doc_description="8-K",
    )
    assert "[AAPL]" in headline
    assert "2.02,9.01" in headline
    assert "Results of Operations and Financial Condition" in headline

    # 4. select_overlapping_archives
    archives = [
        {
            "name": "CIK0000019617-submissions-001.json",
            "filingCount": 2000,
            "filingFrom": "2025-10-01",
            "filingTo": "2025-11-19",
        },
        {
            "name": "CIK0000019617-submissions-051.json",
            "filingCount": 2000,
            "filingFrom": "2018-11-01",
            "filingTo": "2018-12-26",
        },
        {
            "name": "CIK0000019617-submissions-052.json",
            "filingCount": 2000,
            "filingFrom": "2015-01-01",
            "filingTo": "2018-12-10",
        },
    ]
    selected = select_overlapping_archives(
        archives, window_start="2018-12-25", window_end="2026-01-05"
    )
    assert [a["name"] for a in selected] == [
        "CIK0000019617-submissions-001.json",
        "CIK0000019617-submissions-051.json",
    ]

    # 5. parse_submissions_block (works identically for filings.recent and top-level archive dict)
    block = {
        "accessionNumber": ["0001-24-000001", "0001-24-000002", "0001-18-000003"],
        "filingDate": ["2024-02-25", "2024-02-26", "2018-12-30"],
        "reportDate": ["2024-02-25", "2024-02-26", "2018-12-30"],
        "acceptanceDateTime": [
            "2024-02-25T21:15:00.000Z",
            "2024-02-26T14:00:00.000Z",
            "2018-12-30T18:00:00.000Z",
        ],
        "form": ["8-K", "424B2", "10-K"],
        "items": ["2.02,9.01", "", ""],
        "primaryDocument": ["doc1.htm", "doc2.htm", "doc3.htm"],
        "primaryDocDescription": ["EARNINGS RELEASE", "NOTE", "10-K"],
    }
    parsed = parse_submissions_block(
        block,
        ticker="BLK",
        cik="0002012383",
        source_file="CIK0002012383.json",
        start_date="2019-01-01",
        end_date="2025-12-31",
    )
    assert len(parsed) == 1
    assert parsed[0]["accession_number"] == "0001-24-000001"
    assert parsed[0]["form"] == "8-K"

    # 6. Predecessor CIK deduplication & snapshot -> events_raw builder
    dup_pred = dict(parsed[0], cik="0001364742", source_file="CIK0001364742.json")
    older_pred = {
        "ticker": "BLK",
        "cik": "0001364742",
        "accession_number": "0001-20-000099",
        "form": "10-Q",
        "filing_date": "2020-05-01",
        "report_date": "2020-03-31",
        "acceptance_datetime_raw": "2020-05-01T11:30:00.000Z",
        "acceptance_datetime_utc": "2020-05-01T11:30:00+00:00",
        "items_raw": "",
        "items": "",
        "primary_document": "q1.htm",
        "primary_doc_description": "10-Q",
        "source_file": "CIK0001364742-submissions-005.json",
    }
    deduped = deduplicate_filings([parsed[0], dup_pred, older_pred])
    assert len(deduped) == 2

    snap_path = tmp_path / "edgar_filings_2019_2025.parquet"
    pd.DataFrame(deduped).to_parquet(snap_path, index=False)
    events_df = build_events_raw_from_snapshot(snap_path)
    assert tuple(events_df.columns) == EVENTS_RAW_COLUMNS
    assert len(events_df) == 2
    assert list(events_df["accession_number"]) == ["0001-20-000099", "0001-24-000001"]
    assert (events_df["news_source"] == "sec_edgar").all()


def test_day1_audit_pure_functions_and_determinism(block_network: None) -> None:
    """Verify audit.py coverage table, metrics JSON, and Markdown rendering are deterministic."""
    from stock_volatility.ingest.audit import (
        build_news_coverage_table,
        compute_day1_audit_metrics,
        render_day1_audit_markdown,
    )

    uni_rows = []
    ohlcv_rows = []
    event_rows = []
    for idx, (ticker, sector, sub_group) in enumerate(config.UNIVERSE_60, start=1):
        cik = f"{idx:010d}"
        uni_rows.append(
            {
                "ticker": ticker,
                "sector": sector,
                "sub_group": sub_group,
                "cik": cik,
                "company_name": f"{ticker} Inc.",
                "first_trade_date": "2019-01-02",
                "last_trade_date": "2024-01-03",
                "n_trading_days": 2,
            }
        )
        for dt in ("2019-01-02", "2024-01-03"):
            ohlcv_rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "open": 100.0,
                    "high": 105.0,
                    "low": 99.0,
                    "close": 102.0,
                    "adj_close": 101.0,
                    "volume": 500_000,
                    "dividends": 0.5 if dt == "2019-01-02" and idx == 1 else 0.0,
                    "stock_splits": 4.0 if dt == "2024-01-03" and idx == 1 else 0.0,
                }
            )
            event_rows.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "accession_number": f"{cik}-{dt}",
                    "form": "8-K",
                    "filing_date": dt,
                    "report_date": dt,
                    "acceptance_datetime_utc": f"{dt}T21:00:00+00:00",
                    "items": "2.02,9.01",
                    "item_descriptions": (
                        "Results of Operations and Financial Condition; "
                        "Financial Statements and Exhibits"
                    ),
                    "primary_doc_description": "8-K",
                    "headline_text": f"[{ticker}] Form 8-K | Items 2.02,9.01",
                    "news_source": "sec_edgar",
                }
            )

    uni_df = pd.DataFrame(uni_rows)
    ohlcv_df = pd.DataFrame(ohlcv_rows)
    events_df = pd.DataFrame(event_rows)
    macro_df = pd.DataFrame(
        {
            "date": ["2019-01-02", "2024-01-03"],
            "spy_close": [250.0, 470.0],
            "spy_adj_close": [230.0, 465.0],
            "spy_ret_1d": [np.nan, 0.01],
            "vix_close": [23.22, 14.04],
            "vix_fred": [23.22, 14.04],
            "t10y2y": [0.19, -0.35],
            "t10y2y_ffilled_flag": [False, True],
        }
    )
    pagination_manifest = {
        "total_ciks_queried": 62,
        "primary_ciks": 60,
        "predecessor_ciks": 2,
        "total_archives_listed": 351,
        "overlapping_archives_fetched": 195,
        "total_http_requests": 258,
        "requests": [],
    }
    provenance_dict = {
        "generated_at_utc": "2026-11-22T00:00:00+00:00",
        "files": {},
    }

    cov_df = build_news_coverage_table(uni_df, events_df)
    assert len(cov_df) == 60
    assert (cov_df["filings_2019_2023"] == 1).all()
    assert (cov_df["filings_2024_2025"] == 1).all()

    metrics_1 = compute_day1_audit_metrics(
        uni_df=uni_df,
        ohlcv_df=ohlcv_df,
        macro_df=macro_df,
        events_df=events_df,
        coverage_df=cov_df,
        pagination_manifest=pagination_manifest,
        provenance_dict=provenance_dict,
    )
    metrics_2 = compute_day1_audit_metrics(
        uni_df=uni_df,
        ohlcv_df=ohlcv_df,
        macro_df=macro_df,
        events_df=events_df,
        coverage_df=cov_df,
        pagination_manifest=pagination_manifest,
        provenance_dict=provenance_dict,
    )
    assert metrics_1 == metrics_2
    assert metrics_1["generated_at_utc"] == "2026-11-22T00:00:00+00:00"

    md_1 = render_day1_audit_markdown(metrics_1, cov_df)
    md_2 = render_day1_audit_markdown(metrics_2, cov_df)
    assert md_1 == md_2
    assert "0001364742" in md_1
    assert "0000034088" in md_1


# ---------------------------------------------------------------------------
# Layer 2 & 3: Committed M1 Artifact Contract & Offline Replay Tests
# ---------------------------------------------------------------------------


def test_committed_m1_raw_and_interim_artifact_contracts() -> None:
    """Verify all committed M1 raw, interim, provenance, and audit files satisfy exact contracts."""
    from stock_volatility.ingest import ProvenanceLog
    from stock_volatility.ingest.macro import MACRO_COLUMNS
    from stock_volatility.ingest.prices import OHLCV_COLUMNS
    from stock_volatility.ingest.sec_edgar import EVENTS_RAW_COLUMNS
    from stock_volatility.ingest.universe import load_universe_csv

    # 1. Verify data_provenance.json integrity for every committed file in data/raw/
    assert config.PROVENANCE_PATH.is_file(), f"Missing {config.PROVENANCE_PATH}"
    prov = ProvenanceLog.load(config.PROVENANCE_PATH, project_root=config.PROJECT_ROOT)
    prov.verify_files()

    raw_files_on_disk = sorted(
        p.relative_to(config.PROJECT_ROOT).as_posix()
        for p in config.DATA_RAW_DIR.rglob("*")
        if p.is_file() and p != config.PROVENANCE_PATH
    )
    assert len(raw_files_on_disk) == 9
    assert sorted(prov.files.keys()) == raw_files_on_disk

    # 2. Verify universe_60.csv contract
    uni_path = config.DATA_RAW_DIR / "universe_60.csv"
    uni_df = load_universe_csv(uni_path)
    assert list(uni_df.columns) == [
        "ticker",
        "sector",
        "sub_group",
        "cik",
        "company_name",
        "first_trade_date",
        "last_trade_date",
        "n_trading_days",
    ]
    assert len(uni_df) == 60
    assert tuple(uni_df["ticker"]) == config.UNIVERSE_TICKERS
    assert uni_df["cik"].str.fullmatch(r"\d{10}").all()
    assert (uni_df["first_trade_date"] == "2019-01-02").all()
    assert (uni_df["last_trade_date"] == "2025-12-31").all()
    assert (uni_df["n_trading_days"] == 1760).all()

    # 3. Verify ohlcv_daily.parquet contract
    ohlcv_path = config.DATA_RAW_DIR / "prices" / "ohlcv_daily.parquet"
    ohlcv_df = pd.read_parquet(ohlcv_path)
    assert tuple(ohlcv_df.columns) == OHLCV_COLUMNS
    assert len(ohlcv_df) == 105_600
    assert int(ohlcv_df.isna().sum().sum()) == 0
    assert (ohlcv_df["open"] > 0).all()
    assert (ohlcv_df["high"] >= ohlcv_df["low"]).all()
    assert (ohlcv_df["low"] > 0).all()
    assert (ohlcv_df["close"] > 0).all()
    assert (ohlcv_df["adj_close"] > 0).all()
    assert (ohlcv_df["volume"] > 0).all()
    assert int((ohlcv_df["stock_splits"] > 0).sum()) == 9
    assert int((ohlcv_df["dividends"] > 0).sum()) == 1608

    # 4. Verify macro_daily.parquet contract
    macro_path = config.DATA_INTERIM_DIR / "macro_daily.parquet"
    macro_df = pd.read_parquet(macro_path)
    assert tuple(macro_df.columns) == MACRO_COLUMNS
    assert len(macro_df) == 1760
    assert macro_df["date"].iloc[0] == "2019-01-02"
    assert macro_df["date"].iloc[-1] == "2025-12-31"
    assert macro_df["date"].is_monotonic_increasing
    non_null_macro_cols = (
        "spy_close",
        "spy_adj_close",
        "vix_close",
        "vix_fred",
        "t10y2y",
        "t10y2y_ffilled_flag",
    )
    for col in non_null_macro_cols:
        assert int(macro_df[col].isna().sum()) == 0, f"Unexpected NaN in {col}"
    assert int(macro_df["spy_ret_1d"].isna().sum()) == 1
    assert int(macro_df["t10y2y_ffilled_flag"].sum()) == 13
    corr_vix = float(macro_df["vix_close"].corr(macro_df["vix_fred"]))
    assert corr_vix > 0.999999

    # 5. Verify events_raw.parquet contract
    events_path = config.DATA_INTERIM_DIR / "events_raw.parquet"
    events_df = pd.read_parquet(events_path)
    assert tuple(events_df.columns) == EVENTS_RAW_COLUMNS
    assert len(events_df) == 7703
    assert set(events_df["ticker"]) == set(config.UNIVERSE_TICKERS)
    assert set(events_df["form"]).issubset({"8-K", "8-K/A", "10-Q", "10-K"})
    assert events_df["acceptance_datetime_utc"].str.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00"
    ).all()
    assert (events_df["headline_text"].str.len() > 10).all()
    assert (events_df["news_source"] == "sec_edgar").all()

    # Check all 60 tickers have filings in BOTH 2019-2023 and 2024-2025
    e_1923 = events_df[events_df["filing_date"] <= "2023-12-31"]
    e_2425 = events_df[events_df["filing_date"] >= "2024-01-01"]
    assert len(e_1923) == 5665
    assert len(e_2425) == 2038
    assert len(set(e_1923["ticker"])) == 60
    assert len(set(e_2425["ticker"])) == 60

    # Check BLK & XOM predecessor CIK presence
    blk_ciks = set(events_df.loc[events_df["ticker"] == "BLK", "cik"])
    assert {"0001364742", "0002012383"}.issubset(blk_ciks)
    xom_ciks = set(events_df.loc[events_df["ticker"] == "XOM", "cik"])
    assert "0000034088" in xom_ciks

    # 6. Verify Day-1 audit outputs exist and match metrics
    audit_json_path = config.OUTPUTS_METRICS_DIR / "day1_data_audit.json"
    audit_md_path = config.OUTPUTS_TABLES_DIR / "day1_data_audit.md"
    cov_csv_path = config.OUTPUTS_TABLES_DIR / "news_coverage_audit.csv"
    for p in (audit_json_path, audit_md_path, cov_csv_path):
        assert p.is_file(), f"Missing audit output: {p}"

    audit_metrics = json.loads(audit_json_path.read_text(encoding="utf-8"))
    assert audit_metrics["ohlcv"]["total_rows"] == 105_600
    assert audit_metrics["macro"]["total_rows"] == 1760
    assert audit_metrics["macro"]["t10y2y_ffilled_count"] == 13
    assert audit_metrics["sec_edgar"]["total_unique_filings"] == 7703
    assert audit_metrics["sec_edgar"]["active_tickers_2019_2023"] == 60
    assert audit_metrics["sec_edgar"]["active_tickers_2024_2025"] == 60

    cov_df = pd.read_csv(cov_csv_path, dtype={"cik": str})
    assert len(cov_df) == 60
    assert (cov_df["filings_2019_2023"] > 0).all()
    assert (cov_df["filings_2024_2025"] > 0).all()


def test_skip_download_offline_replay_is_network_free_and_bit_identical(
    block_network: None,
) -> None:
    """Verify --skip-download runs offline, mutates 0 raw files, and produces identical hashes."""
    from stock_volatility.ingest.audit import run_skip_download_pipeline

    raw_paths = sorted(p for p in config.DATA_RAW_DIR.rglob("*") if p.is_file())
    raw_hashes_before = {p: config.sha256_file(p) for p in raw_paths}

    target_outputs = [
        config.DATA_INTERIM_DIR / "macro_daily.parquet",
        config.DATA_INTERIM_DIR / "events_raw.parquet",
        config.OUTPUTS_METRICS_DIR / "day1_data_audit.json",
        config.OUTPUTS_TABLES_DIR / "day1_data_audit.md",
        config.OUTPUTS_TABLES_DIR / "news_coverage_audit.csv",
    ]
    hashes_before = {p: config.sha256_file(p) for p in target_outputs}

    # Execute skip-download pipeline with network blocked by block_network fixture
    summary = run_skip_download_pipeline()
    assert summary["mode"] == "skip-download"

    raw_hashes_after = {p: config.sha256_file(p) for p in raw_paths}
    assert raw_hashes_after == raw_hashes_before, (
        "data/raw/ files were mutated during --skip-download!"
    )

    hashes_after = {p: config.sha256_file(p) for p in target_outputs}
    assert hashes_after == hashes_before, (
        "Rebuilt interim/output hashes differed after --skip-download replay!"
    )


def test_milestone_delivery_docs_exist_complete_linked_and_not_gitignored() -> None:
    """Verify M0 & M1 milestone delivery docs exist, are complete, linked, and tracked by git."""
    repo_root = config.PROJECT_ROOT.parent
    milestones_dir = config.PROJECT_ROOT / "docs" / "milestones"
    assert milestones_dir.is_dir(), f"Missing milestones directory: {milestones_dir}"

    readme_doc = milestones_dir / "README.md"
    m0_doc = milestones_dir / "M0_repo_cleanup_and_scaffold.md"
    m1_doc = milestones_dir / "M1_data_ingest_and_day1_audit.md"

    for doc_path in (readme_doc, m0_doc, m1_doc):
        assert doc_path.is_file(), f"Missing milestone documentation file: {doc_path}"
        assert doc_path.stat().st_size > 1500, f"Milestone doc too short: {doc_path}"

    # 1. Verify docs/milestones/README.md ledger, policy, and M2-M6 template
    ledger_text = readme_doc.read_text(encoding="utf-8")
    for token in (
        "M0_repo_cleanup_and_scaffold.md",
        "M1_data_ingest_and_day1_audit.md",
        "M2_panel_splits_and_freeze.md",
        "M3_eda_and_h1_panel_har_rv.md",
        "M4_h2_forecasting_calibration_and_shap.md",
        "M5_figures_table1_and_cs482_report.md",
        "M6_e2e_and_adversarial_hardening.md",
        "124fc67eb8d578a0495733092598b60aa2b763e0",
        "Template",
        "里程碑",
    ):
        assert token in ledger_text, f"Missing {token!r} in docs/milestones/README.md"

    # 2. Verify M0 milestone doc records both M0 deliverables and AI execution history
    m0_text = m0_doc.read_text(encoding="utf-8")
    for token in (
        "124fc67eb8d578a0495733092598b60aa2b763e0",
        "numba>=0.60",
        "pdflatex",
        "UNIVERSE_60",
        "test_smoke.py",
        "26 passed",
        "CLEAN",
        "APPROVE",
    ):
        assert token in m0_text, f"Missing {token!r} in M0_repo_cleanup_and_scaffold.md"

    # 3. Verify M1 milestone doc records both M1 deliverables and AI discoveries/execution
    m1_text = m1_doc.read_text(encoding="utf-8")
    for token in (
        "universe_60.csv",
        "ohlcv_daily.parquet",
        "macro_daily.parquet",
        "events_raw.parquet",
        "data_provenance.json",
        "day1_data_audit.json",
        "day1_data_audit.md",
        "news_coverage_audit.csv",
        "01_download_and_audit.py",
        "--skip-download",
        "0001364742",  # BLK predecessor CIK
        "0000034088",  # XOM predecessor CIK
        "VIXCLS",
        "T10Y2Y",
        "filings.files",
    ):
        assert token in m1_text, f"Missing {token!r} in M1_data_ingest_and_day1_audit.md"

    # 4. Verify links from Plan 6 README.md and root README.md
    plan6_readme = (config.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    for rel_link in (
        "docs/milestones/README.md",
        "docs/milestones/M0_repo_cleanup_and_scaffold.md",
        "docs/milestones/M1_data_ingest_and_day1_audit.md",
    ):
        assert rel_link in plan6_readme, (
            f"Missing link {rel_link!r} in plan_6_stock_volatility/README.md"
        )

    root_readme = (repo_root / "README.md").read_text(encoding="utf-8")
    assert "plan_6_stock_volatility/docs/milestones/README.md" in root_readme

    # 5. Verify none of the milestone docs are ignored by .gitignore or .git/info/exclude
    proc = subprocess.run(
        [
            "git",
            "check-ignore",
            "-v",
            "--no-index",
            "plan_6_stock_volatility/docs/milestones/README.md",
            "plan_6_stock_volatility/docs/milestones/M0_repo_cleanup_and_scaffold.md",
            "plan_6_stock_volatility/docs/milestones/M1_data_ingest_and_day1_audit.md",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert proc.returncode == 1, (
        f"Milestone docs must NOT be gitignored:\n{proc.stdout}\n{proc.stderr}"
    )
    assert proc.stdout.strip() == ""
