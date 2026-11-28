"""Comprehensive unit, anti-leakage, freeze tamper, and contract tests for Milestone M2."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_volatility import config

# ---------------------------------------------------------------------------
# Layer 1: Pure Offline Unit & Anti-Leakage Tests (Zero Network Calls)
# ---------------------------------------------------------------------------


def test_event_cutoff_dst_and_early_close_alignment(block_network: None) -> None:
    """Verify DST-aware 16:00 ET cutoff, 13:00 ET early-close cutoff, and weekend shifting."""
    from stock_volatility.panel.events import (
        NYSE_EARLY_CLOSE_DATES,
        align_events_to_trading_days,
    )

    assert "2019-07-03" in NYSE_EARLY_CLOSE_DATES
    assert "2019-11-29" in NYSE_EARLY_CLOSE_DATES
    assert len(NYSE_EARLY_CLOSE_DATES) == 15

    trading_dates = [
        "2019-07-03",  # EDT early close (13:00 EDT = 17:00 UTC)
        "2019-07-05",  # EDT normal day (16:00 EDT = 20:00 UTC)
        "2019-07-08",  # Monday
        "2019-11-29",  # EST early close (13:00 EST = 18:00 UTC)
        "2019-12-02",  # EST normal day (16:00 EST = 21:00 UTC)
        "2019-12-03",
    ]
    events_raw = pd.DataFrame(
        [
            # 1. EDT early-close day BEFORE 13:00 EDT (12:59:59 EDT = 16:59:59 UTC) -> stays 07-03
            {
                "ticker": "WFC",
                "cik": "0000072971",
                "accession_number": "acc-1",
                "form": "8-K",
                "filing_date": "2019-07-03",
                "report_date": "2019-07-03",
                "acceptance_datetime_utc": "2019-07-03T16:59:59+00:00",
                "items": "8.01",
                "item_descriptions": "Other Events",
                "primary_doc_description": "8-K",
                "headline_text": "[WFC] Early close before 13:00 EDT",
                "news_source": "sec_edgar",
            },
            # 2. EDT early-close day AFTER 13:00 EDT (14:55:29 EDT = 18:55:29 UTC) -> shifts 07-05
            {
                "ticker": "WFC",
                "cik": "0000072971",
                "accession_number": "acc-2",
                "form": "8-K",
                "filing_date": "2019-07-03",
                "report_date": "2019-07-03",
                "acceptance_datetime_utc": "2019-07-03T18:55:29+00:00",
                "items": "8.01",
                "item_descriptions": "Other Events",
                "primary_doc_description": "8-K",
                "headline_text": "[WFC] Early close after 13:00 EDT",
                "news_source": "sec_edgar",
            },
            # 3. EDT normal day at EXACT 16:00:00 EDT (= 20:00:00 UTC) -> shifts to 07-08
            {
                "ticker": "AAPL",
                "cik": "0000320193",
                "accession_number": "acc-3",
                "form": "8-K",
                "filing_date": "2019-07-05",
                "report_date": "2019-07-05",
                "acceptance_datetime_utc": "2019-07-05T20:00:00+00:00",
                "items": "2.02",
                "item_descriptions": "Results of Operations and Financial Condition",
                "primary_doc_description": "8-K",
                "headline_text": "[AAPL] Exact 16:00:00 EDT boundary",
                "news_source": "sec_edgar",
            },
            # 4. EDT normal day at 15:59:59 EDT (= 19:59:59 UTC) -> stays 07-05
            {
                "ticker": "AAPL",
                "cik": "0000320193",
                "accession_number": "acc-4",
                "form": "8-K",
                "filing_date": "2019-07-05",
                "report_date": "2019-07-05",
                "acceptance_datetime_utc": "2019-07-05T19:59:59+00:00",
                "items": "7.01",
                "item_descriptions": "Regulation FD Disclosure",
                "primary_doc_description": "8-K",
                "headline_text": "[AAPL] Just before 16:00 EDT",
                "news_source": "sec_edgar",
            },
            # 5. Weekend (Saturday 2019-07-06) -> shifts to Monday 07-08
            {
                "ticker": "MSFT",
                "cik": "0000789019",
                "accession_number": "acc-5",
                "form": "8-K",
                "filing_date": "2019-07-08",
                "report_date": "2019-07-06",
                "acceptance_datetime_utc": "2019-07-06T14:00:00+00:00",
                "items": "8.01",
                "item_descriptions": "Other Events",
                "primary_doc_description": "8-K",
                "headline_text": "[MSFT] Weekend filing",
                "news_source": "sec_edgar",
            },
            # 6. EST early-close day AFTER 13:00 EST (13:02:31 EST = 18:02:31 UTC) -> shifts 12-02
            {
                "ticker": "WFC",
                "cik": "0000072971",
                "accession_number": "acc-6",
                "form": "8-K",
                "filing_date": "2019-11-29",
                "report_date": "2019-11-29",
                "acceptance_datetime_utc": "2019-11-29T18:02:31+00:00",
                "items": "8.01",
                "item_descriptions": "Other Events",
                "primary_doc_description": "8-K",
                "headline_text": "[WFC] Early close after 13:00 EST",
                "news_source": "sec_edgar",
            },
            # 7. EST normal day at 15:30:00 EST (= 20:30:00 UTC) -> stays 12-02
            {
                "ticker": "JPM",
                "cik": "0000019617",
                "accession_number": "acc-7",
                "form": "8-K",
                "filing_date": "2019-12-02",
                "report_date": "2019-12-02",
                "acceptance_datetime_utc": "2019-12-02T20:30:00+00:00",
                "items": "8.01",
                "item_descriptions": "Other Events",
                "primary_doc_description": "8-K",
                "headline_text": "[JPM] EST 15:30 stays same day",
                "news_source": "sec_edgar",
            },
            # 8. After last trading day cutoff (2019-12-03 16:05 EST = 21:05 UTC) -> dropped
            {
                "ticker": "JPM",
                "cik": "0000019617",
                "accession_number": "acc-8",
                "form": "8-K",
                "filing_date": "2019-12-03",
                "report_date": "2019-12-03",
                "acceptance_datetime_utc": "2019-12-03T21:05:00+00:00",
                "items": "8.01",
                "item_descriptions": "Other Events",
                "primary_doc_description": "8-K",
                "headline_text": "[JPM] After final calendar day cutoff",
                "news_source": "sec_edgar",
            },
        ]
    )

    aligned, audit = align_events_to_trading_days(events_raw, trading_dates)
    assert len(aligned) == 7
    assert audit["dropped_after_end_date"] == 1
    assert audit["shifted_due_to_early_close"] == 2
    assert audit["shifted_non_trading_day"] == 1
    assert audit["shifted_after_1600_et"] == 1
    assert audit["on_trading_day_before_cutoff"] == 3

    by_acc = aligned.set_index("accession_number")["aligned_date"].to_dict()
    assert by_acc["acc-1"] == "2019-07-03"
    assert by_acc["acc-2"] == "2019-07-05"
    assert by_acc["acc-3"] == "2019-07-08"
    assert by_acc["acc-4"] == "2019-07-05"
    assert by_acc["acc-5"] == "2019-07-08"
    assert by_acc["acc-6"] == "2019-12-02"
    assert by_acc["acc-7"] == "2019-12-02"
    assert "acc-8" not in by_acc


def test_event_dedup_and_cap_at_5_latest(block_network: None) -> None:
    """Verify duplicate headlines are deduplicated and >5 events/day are capped to latest 5."""
    from stock_volatility.panel.events import dedup_and_cap_events

    rows = []
    # 7 unique headlines + 1 duplicate headline on same (AAPL, 2024-03-01)
    for idx in range(1, 8):
        rows.append(
            {
                "ticker": "AAPL",
                "cik": "0000320193",
                "accession_number": f"acc-{idx:02d}",
                "form": "8-K",
                "filing_date": "2024-03-01",
                "report_date": "2024-03-01",
                "acceptance_datetime_utc": f"2024-03-01T1{idx}:00:00+00:00",
                "aligned_date": "2024-03-01",
                "items": "2.02" if idx == 7 else "8.01",
                "item_descriptions": "Desc",
                "primary_doc_description": "8-K",
                "headline_text": f"[AAPL] Unique headline #{idx}",
                "news_source": "sec_edgar",
                "shift_reason": "same_day_pre_close",
            }
        )
    # Exact duplicate headline text of #6 at an earlier timestamp -> must keep latest #6
    rows.append(
        {
            **rows[5],
            "accession_number": "acc-06-dup",
            "acceptance_datetime_utc": "2024-03-01T15:30:00+00:00",
            "headline_text": "  [aapl]   unique HEADLINE #6  ",
        }
    )

    capped, audit = dedup_and_cap_events(pd.DataFrame(rows), max_events_per_day=5)
    assert len(capped) == 5
    assert audit["dedup_headline_dropped"] == 1
    assert audit["capped_over_5_dropped"] == 2
    assert audit["ticker_days_capped"] == 1
    assert audit["max_events_per_day_after_cap"] == 5
    # Must retain the 5 latest unique events: #3, #4, #5, #6 (latest acc-06), #7
    assert list(capped["accession_number"]) == [
        "acc-03",
        "acc-04",
        "acc-05",
        "acc-06",
        "acc-07",
    ]


def test_features_and_targets_strict_no_leakage_and_train_only_params(
    block_network: None,
) -> None:
    """Verify exact Parkinson/CC RV formulas, strict <=t features, and Train-only thresholds."""
    from stock_volatility.panel.features import (
        build_panel_features_and_targets,
        fit_train_preprocessing_params,
    )

    # Create 40 trading days spanning train (30 days) and val (10 days) for 1 ticker
    dates_train = pd.bdate_range("2023-11-01", periods=30).strftime("%Y-%m-%d").tolist()
    dates_val = pd.bdate_range("2024-01-02", periods=10).strftime("%Y-%m-%d").tolist()
    all_dates = dates_train + dates_val
    n = len(all_dates)

    base_close = 100.0 * np.exp(np.linspace(0.0, 0.05, n))
    high = base_close * 1.02
    low = base_close * 0.98
    # Inject a sharp -10% drop on days 25..29 (indices 25..29) so day 24 has y_tail == 1
    adj_close = base_close.copy()
    adj_close[29] = adj_close[24] * 0.90  # exactly -10% 5-day return from index 24 -> 29

    ohlcv = pd.DataFrame(
        {
            "date": all_dates,
            "ticker": ["AAPL"] * n,
            "open": base_close,
            "high": high,
            "low": low,
            "close": base_close,
            "adj_close": adj_close,
            "volume": np.full(n, 1_000_000.0),
            "dividends": np.zeros(n),
            "stock_splits": np.zeros(n),
        }
    )
    universe = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "sector": "Technology",
                "sub_group": "Technology Hardware",
                "cik": "0000320193",
                "company_name": "Apple Inc.",
                "first_trade_date": all_dates[0],
                "last_trade_date": all_dates[-1],
                "n_trading_days": n,
            }
        ]
    )
    macro = pd.DataFrame(
        {
            "date": all_dates,
            "spy_close": np.linspace(400.0, 420.0, n),
            "spy_adj_close": np.linspace(400.0, 420.0, n),
            "spy_ret_1d": np.r_[np.nan, np.full(n - 1, 0.001)],
            "vix_close": np.full(n, 18.0),
            "vix_fred": np.full(n, 18.0),
            "t10y2y": np.full(n, -0.30),
            "t10y2y_ffilled_flag": np.zeros(n, dtype=bool),
        }
    )
    daily_events = pd.DataFrame(
        {
            "ticker": ["AAPL"] * n,
            "date": all_dates,
            "has_news": [1 if i == 24 else 0 for i in range(n)],
            "news_count": [2 if i == 24 else 0 for i in range(n)],
            "has_8k_earnings": [1 if i == 24 else 0 for i in range(n)],
            "news_source": ["sec_edgar" if i == 24 else "none" for i in range(n)],
        }
    )
    splits = ["train"] * 30 + ["val"] * 10
    split_df = pd.DataFrame({"date": all_dates, "split": splits})

    panel_a, params_a = build_panel_features_and_targets(
        ohlcv_df=ohlcv,
        universe_df=universe,
        macro_df=macro,
        daily_events_df=daily_events,
        split_df=split_df,
    )

    # Check exact analytical Parkinson RV on index 24 (high/low = 1.02/0.98 constant)
    hl_sq = math.log(1.02 / 0.98) ** 2
    expected_rv = math.sqrt((252.0 / (4.0 * math.log(2.0))) * hl_sq)
    assert math.isclose(panel_a.loc[24, "rv_d"], expected_rv, rel_tol=1e-12)
    assert math.isclose(panel_a.loc[24, "rv_w"], expected_rv, rel_tol=1e-12)
    assert math.isclose(panel_a.loc[24, "rv_m"], expected_rv, rel_tol=1e-12)
    assert math.isclose(panel_a.loc[24, "rv_5d_fwd"], expected_rv, rel_tol=1e-12)
    assert math.isclose(
        panel_a.loc[24, "y_log_rv"], math.log(expected_rv + config.RV_EPS), rel_tol=1e-12
    )
    assert math.isclose(panel_a.loc[24, "cum_ret_5d_fwd"], -0.10, abs_tol=1e-12)
    assert int(panel_a.loc[24, "y_tail"]) == 1

    # Last 5 rows must have has_full_target == False and NA targets (never 0)
    assert not panel_a.tail(5)["has_full_target"].any()
    assert panel_a.tail(5)["y_tail"].isna().all()
    assert panel_a.tail(5)["y_tail_rv90"].isna().all()

    # Anti-leakage perturbation test: shock Val prices at indices 30..39 by 5x
    ohlcv_shocked = ohlcv.copy()
    ohlcv_shocked.loc[30:, "high"] = ohlcv_shocked.loc[30:, "close"] * 1.50
    ohlcv_shocked.loc[30:, "low"] = ohlcv_shocked.loc[30:, "close"] * 0.50
    ohlcv_shocked.loc[30:, "adj_close"] *= 0.50
    panel_b, params_b = build_panel_features_and_targets(
        ohlcv_df=ohlcv_shocked,
        universe_df=universe,
        macro_df=macro,
        daily_events_df=daily_events,
        split_df=split_df,
    )

    # Every <=t feature on rows 0..29 must be 100% IDENTICAL despite future Val shock
    feature_cols = [
        "rv_d",
        "rv_w",
        "rv_m",
        "log_rv_d",
        "log_rv_w",
        "log_rv_m",
        "ret_1d",
        "neg_ret_1d",
        "ret_5d",
        "vol_ratio_20d",
        "daily_range_pct",
    ]
    pd.testing.assert_frame_equal(
        panel_a.loc[:29, feature_cols], panel_b.loc[:29, feature_cols]
    )

    # Train-only preprocessing parameters (rv90_threshold, Winsor bounds, mean/std) must be
    # computed strictly on Train rows whose targets also stay within Train (here indices 21..24)
    # Verify fit_train_preprocessing_params is invariant to Val rows
    params_check_a = fit_train_preprocessing_params(panel_a[panel_a["date"] <= dates_train[24]])
    params_check_b = fit_train_preprocessing_params(panel_b[panel_b["date"] <= dates_train[24]])
    assert params_check_a == params_check_b
    assert params_a["rv90_threshold"] > 0


def test_purged_embargo_splits_enforce_zero_target_overlap(block_network: None) -> None:
    """Verify split assignment and zero target-window leakage across Train/Val/Test boundaries."""
    from stock_volatility.panel.splits import (
        assign_purged_splits,
        validate_purged_embargo_splits,
    )

    macro_df = pd.read_parquet(config.DATA_INTERIM_DIR / "macro_daily.parquet")
    cal_dates = macro_df["date"].tolist()
    split_df = assign_purged_splits(cal_dates)
    counts = split_df["split"].value_counts().to_dict()
    assert counts == {
        "train": 1252,
        "embargo_1": 6,
        "val": 246,
        "embargo_2": 6,
        "test": 250,
    }

    # Construct minimal panel_df matching calendar to test validator
    dummy_panel = split_df.copy()
    dummy_panel["ticker"] = "AAPL"
    dummy_panel["has_full_features"] = True
    dummy_panel.loc[:20, "has_full_features"] = False
    dummy_panel["has_full_target"] = True
    dummy_panel.loc[len(dummy_panel) - 5 :, "has_full_target"] = False

    meta = validate_purged_embargo_splits(dummy_panel, cal_dates, horizon_days=5)
    assert meta["zero_target_overlap_verified"] is True
    assert meta["embargo_1_trading_days"] == 6
    assert meta["embargo_2_trading_days"] == 6
    assert meta["train_last_target_date"] < config.VAL_START
    assert meta["val_last_target_date"] < config.TEST_START
    assert meta["test_scorable_trading_days"] == 245
    assert meta["test_incomplete_tail_days"] == 5


def test_freeze_manifest_verify_and_tamper_detection(
    tmp_path: Path, block_network: None
) -> None:
    """Verify verify_freeze_manifest passes on clean bundle and raises ValueError on tampering."""
    from stock_volatility.panel.freeze import (
        FREEZE_ARTIFACT_NAMES,
        verify_freeze_manifest,
        write_freeze_manifest,
    )

    frozen_dir = tmp_path / "data" / "frozen"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    for name in FREEZE_ARTIFACT_NAMES:
        (frozen_dir / name).write_text(f"synthetic payload for {name}\n", encoding="utf-8")

    manifest = write_freeze_manifest(
        frozen_dir=frozen_dir,
        project_root=tmp_path,
        generated_at_utc="2026-11-22T14:04:28+00:00",
    )
    assert len(manifest["files"]) == len(FREEZE_ARTIFACT_NAMES)

    # Clean verification must pass
    verified = verify_freeze_manifest(frozen_dir=frozen_dir, project_root=tmp_path)
    assert verified["status"] == "verified"

    # Tamper with one file -> must raise ValueError mentioning SHA-256
    target = frozen_dir / "analysis_spec.json"
    target.write_text("tampered content\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_freeze_manifest(frozen_dir=frozen_dir, project_root=tmp_path)


# ---------------------------------------------------------------------------
# Layer 2: Committed M2 Frozen Artifacts, Replay & Milestone Doc Contracts
# ---------------------------------------------------------------------------


def test_committed_m2_frozen_artifacts_and_contracts(block_network: None) -> None:
    """Verify all committed M2 interim/frozen artifacts satisfy Plan 6 contracts and SHA-256."""
    from stock_volatility.panel.events import EVENTS_PHASE_B_COLUMNS
    from stock_volatility.panel.features import PANEL_DAILY_COLUMNS
    from stock_volatility.panel.freeze import verify_freeze_manifest

    # 1. Cryptographic freeze manifest must verify cleanly
    manifest = verify_freeze_manifest()
    assert manifest["status"] == "verified"
    assert len(manifest["files"]) == 6

    # 2. Verify data/interim/events_aligned.parquet
    events_aligned_path = config.DATA_INTERIM_DIR / "events_aligned.parquet"
    assert events_aligned_path.is_file()
    ev_aligned = pd.read_parquet(events_aligned_path)
    assert len(ev_aligned) > 7000
    assert (ev_aligned.groupby(["ticker", "aligned_date"]).size() <= 5).all()
    assert int(ev_aligned["shifted_due_to_early_close"].sum()) == 2

    # 3. Verify data/frozen/panel_daily.parquet
    panel_path = config.DATA_FROZEN_DIR / "panel_daily.parquet"
    panel = pd.read_parquet(panel_path)
    assert tuple(panel.columns) == PANEL_DAILY_COLUMNS
    assert len(panel) == 105_600
    assert not panel.duplicated(subset=["ticker", "date"]).any()
    assert set(panel["ticker"]) == set(config.UNIVERSE_TICKERS)
    assert set(panel["split"]) == {"train", "embargo_1", "val", "embargo_2", "test"}

    # Exact split day counts
    split_days = panel.groupby("split")["date"].nunique().to_dict()
    assert split_days == {
        "train": 1252,
        "embargo_1": 6,
        "val": 246,
        "embargo_2": 6,
        "test": 250,
    }

    # Check warmup & tail flags
    train_complete = panel[(panel["split"] == "train") & panel["has_full_features"]]
    assert train_complete["date"].nunique() == 1231
    assert train_complete["has_full_target"].all()

    val_complete = panel[(panel["split"] == "val") & panel["has_full_features"]]
    assert val_complete["date"].nunique() == 246
    assert val_complete["has_full_target"].all()

    test_scorable = panel[(panel["split"] == "test") & panel["has_full_target"]]
    test_tail = panel[(panel["split"] == "test") & (~panel["has_full_target"])]
    assert test_scorable["date"].nunique() == 245
    assert test_tail["date"].nunique() == 5
    assert test_tail["y_tail"].isna().all()
    assert test_tail["y_tail_rv90"].isna().all()
    assert test_tail["rv_5d_fwd"].isna().all()

    # Zero NaNs on feature-complete + target-complete rows
    modeling_rows = panel[panel["has_full_features"] & panel["has_full_target"]]
    assert int(modeling_rows.isna().sum().sum()) == 0
    assert (modeling_rows["rv_5d_fwd"] > 0).all()
    assert (modeling_rows["news_count"] <= 5).all()
    assert set(modeling_rows["y_tail"].astype(int).unique()) == {0, 1}
    assert set(modeling_rows["y_tail_rv90"].astype(int).unique()) == {0, 1}

    # 4. Verify data/frozen/events_text_phase_b.parquet
    phase_b_path = config.DATA_FROZEN_DIR / "events_text_phase_b.parquet"
    phase_b = pd.read_parquet(phase_b_path)
    assert tuple(phase_b.columns) == EVENTS_PHASE_B_COLUMNS
    assert len(phase_b) == 105_600
    assert (phase_b["news_count"] <= 5).all()
    assert (phase_b["news_count"] >= 0).all()
    active_pb = phase_b[phase_b["has_news"] == 1]
    assert len(active_pb) > 6000
    assert (active_pb["text_concat_capped_5"].str.len() > 10).all()
    # Every titles_capped_5 JSON list length must match news_count
    parsed_lens = phase_b["titles_capped_5"].map(lambda s: len(json.loads(s)))
    assert (parsed_lens == phase_b["news_count"]).all()

    # 5. Verify JSON specs & Markdown dictionary
    spec = json.loads((config.DATA_FROZEN_DIR / "analysis_spec.json").read_text(encoding="utf-8"))
    assert spec["random_seed"] == 42
    assert spec["horizon_days"] == 5
    assert spec["cutoff_tz"] == "America/New_York"
    assert spec["cutoff_hour"] == 16
    assert spec["max_events_per_day"] == 5
    assert "H1" in spec["hypotheses"] and "H2" in spec["hypotheses"] and "H3" in spec["hypotheses"]

    split_meta = json.loads(
        (config.DATA_FROZEN_DIR / "split_metadata.json").read_text(encoding="utf-8")
    )
    assert split_meta["zero_target_overlap_verified"] is True
    assert split_meta["event_alignment_audit"]["shifted_due_to_early_close"] == 2

    feat_manifest = json.loads(
        (config.DATA_FROZEN_DIR / "feature_manifest.json").read_text(encoding="utf-8")
    )
    assert feat_manifest["train_preprocessing"]["fitted_split"] == "train"
    assert 0.20 < feat_manifest["train_preprocessing"]["rv90_threshold"] < 0.60

    dict_md = (config.DATA_FROZEN_DIR / "data_dictionary.md").read_text(encoding="utf-8")
    for token in (
        "panel_daily.parquet",
        "events_text_phase_b.parquet",
        "16:00",
        "13:00",
        "rv_5d_fwd",
    ):
        assert token in dict_md


def test_m2_refreeze_is_offline_and_bit_identical(block_network: None) -> None:
    """Verify --refreeze runs strictly offline and produces bit-identical frozen SHA-256 hashes."""
    from stock_volatility.panel.freeze import (
        FREEZE_ARTIFACT_NAMES,
        build_and_freeze_panel,
        verify_freeze_manifest,
    )

    tracked_paths = [
        config.DATA_INTERIM_DIR / "events_aligned.parquet",
        config.FREEZE_MANIFEST_PATH,
        *(config.DATA_FROZEN_DIR / name for name in FREEZE_ARTIFACT_NAMES),
    ]
    hashes_before = {p: config.sha256_file(p) for p in tracked_paths}

    summary = build_and_freeze_panel(refreeze=True)
    assert summary["mode"] == "refreeze"
    verify_freeze_manifest()

    hashes_after = {p: config.sha256_file(p) for p in tracked_paths}
    assert hashes_after == hashes_before, (
        "M2 --refreeze produced non-deterministic artifact hashes!"
    )


def test_m2_milestone_delivery_doc_complete_linked_and_not_gitignored() -> None:
    """Verify M2 milestone doc exists, records deliverables + AI log, is linked & tracked."""
    repo_root = config.PROJECT_ROOT.parent
    milestones_dir = config.PROJECT_ROOT / "docs" / "milestones"
    m2_doc = milestones_dir / "M2_panel_splits_and_freeze.md"
    assert m2_doc.is_file(), f"Missing M2 milestone doc: {m2_doc}"
    assert m2_doc.stat().st_size > 2000

    m2_text = m2_doc.read_text(encoding="utf-8")
    for token in (
        "5370d1835215f5463b38f3855bffa96211ceba69",
        "panel_daily.parquet",
        "events_text_phase_b.parquet",
        "analysis_spec.json",
        "split_metadata.json",
        "feature_manifest.json",
        "data_dictionary.md",
        "FREEZE_MANIFEST.json",
        "02_build_panel_and_freeze.py",
        "16:00",
        "13:00",
        "embargo_1",
        "embargo_2",
        "verify_freeze_manifest",
    ):
        assert token in m2_text, f"Missing {token!r} in M2_panel_splits_and_freeze.md"

    ledger_text = (milestones_dir / "README.md").read_text(encoding="utf-8")
    assert "5370d1835215f5463b38f3855bffa96211ceba69" in ledger_text
    assert "M2_panel_splits_and_freeze.md" in ledger_text

    plan6_readme = (config.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/milestones/M2_panel_splits_and_freeze.md" in plan6_readme

    proc = subprocess.run(
        [
            "git",
            "check-ignore",
            "-v",
            "--no-index",
            "plan_6_stock_volatility/docs/milestones/M2_panel_splits_and_freeze.md",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert proc.returncode == 1, f"M2 milestone doc must NOT be gitignored:\n{proc.stdout}"
    assert proc.stdout.strip() == ""
