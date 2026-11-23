"""Deterministic 60-stock cross-sector universe resolution and CIK mapping for Plan 6."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from stock_volatility import config

UNIVERSE_COLUMNS: tuple[str, ...] = (
    "ticker",
    "sector",
    "sub_group",
    "cik",
    "company_name",
    "first_trade_date",
    "last_trade_date",
    "n_trading_days",
)

# Pre-registered SEC EDGAR holding-company reorganization predecessor CIK linkages:
# - BLK: CIK 0002012383 (BlackRock, Inc., filings from 2024-02-20+) succeeded predecessor
#        CIK 0001364742 (BlackRock Finance, Inc., filings 2006-06-09..2025-04-04).
# - XOM: CIK 0002115436 (ExxonMobil Holdings Corp, filings from 2026-07-01+) succeeded
#        predecessor CIK 0000034088 (EXXON MOBIL CORP, filings 1994-03-04..2026-10-27).
PREDECESSOR_CIK_MAP: dict[str, tuple[str, ...]] = {
    "BLK": ("0001364742",),
    "XOM": ("0000034088",),
}

CIK_PREDECESSOR_LINKAGE_METADATA: dict[str, dict[str, Any]] = {
    "BLK": {
        "ticker": "BLK",
        "primary_cik": "0002012383",
        "primary_entity_name": "BlackRock, Inc.",
        "predecessor_ciks": ["0001364742"],
        "predecessor_entity_names": ["BlackRock Finance, Inc."],
        "reorganization_note": (
            "BlackRock completed its Global Infrastructure Partners (GIP) holding-company "
            "reorganization in 2024; 2019-2023 and early-2024 filings live under predecessor "
            "CIK 0001364742 while post-2024-02-20 filings live under CIK 0002012383."
        ),
    },
    "XOM": {
        "ticker": "XOM",
        "primary_cik": "0002115436",
        "primary_entity_name": "ExxonMobil Holdings Corp",
        "predecessor_ciks": ["0000034088"],
        "predecessor_entity_names": ["EXXON MOBIL CORP"],
        "reorganization_note": (
            "ExxonMobil registered holding-company CIK 0002115436 in July 2026; 100% of "
            "2019-2025 filings live under predecessor CIK 0000034088."
        ),
    },
}


def resolve_universe_ciks(
    company_tickers_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Map all 60 UNIVERSE_TICKERS to zero-padded 10-digit primary and predecessor CIKs."""
    by_ticker: dict[str, dict[str, Any]] = {}
    for entry in company_tickers_payload.values():
        if not isinstance(entry, dict):
            continue
        raw_ticker = str(entry.get("ticker", "")).strip().upper()
        if not raw_ticker or raw_ticker in by_ticker:
            continue
        cik_int = int(entry["cik_str"])
        primary_cik = f"{cik_int:010d}"
        by_ticker[raw_ticker] = {
            "ticker": raw_ticker,
            "cik": primary_cik,
            "company_name": str(entry.get("title", raw_ticker)).strip(),
        }

    missing = [t for t in config.UNIVERSE_TICKERS if t not in by_ticker]
    if missing:
        raise ValueError(f"Missing universe tickers in SEC company_tickers.json: {missing}")

    resolved: dict[str, dict[str, Any]] = {}
    for ticker, sector, sub_group in config.UNIVERSE_60:
        info = by_ticker[ticker]
        pred_ciks = list(PREDECESSOR_CIK_MAP.get(ticker, ()))
        primary_cik = info["cik"]
        all_ciks = [primary_cik, *[c for c in pred_ciks if c != primary_cik]]
        resolved[ticker] = {
            "ticker": ticker,
            "sector": sector,
            "sub_group": sub_group,
            "cik": primary_cik,
            "company_name": info["company_name"],
            "predecessor_ciks": pred_ciks,
            "all_ciks": all_ciks,
        }
    return resolved


def build_universe_table(
    ohlcv_df: pd.DataFrame,
    cik_map: dict[str, dict[str, Any]],
    *,
    expected_trading_days: int = 1760,
) -> pd.DataFrame:
    """Construct `universe_60.csv` table and verify 100% OHLCV trading-day continuity."""
    if ohlcv_df.empty:
        raise ValueError("Cannot build universe table from empty OHLCV DataFrame")

    grouped = (
        ohlcv_df.groupby("ticker", sort=False)["date"]
        .agg(
            first_trade_date="min",
            last_trade_date="max",
            n_trading_days="count",
        )
        .to_dict(orient="index")
    )

    rows: list[dict[str, Any]] = []
    for ticker, sector, sub_group in config.UNIVERSE_60:
        if ticker not in cik_map:
            raise ValueError(f"Ticker {ticker} missing from cik_map")
        if ticker not in grouped:
            raise ValueError(f"Ticker {ticker} missing from ohlcv_df")
        stats = grouped[ticker]
        n_days = int(stats["n_trading_days"])
        if n_days != expected_trading_days:
            raise ValueError(
                f"Ticker {ticker} has {n_days} trading days, expected {expected_trading_days}"
            )
        cik_str = str(cik_map[ticker]["cik"]).strip()
        if len(cik_str) != 10 or not cik_str.isdigit():
            raise ValueError(f"Invalid 10-digit CIK for {ticker}: {cik_str!r}")
        rows.append(
            {
                "ticker": ticker,
                "sector": sector,
                "sub_group": sub_group,
                "cik": cik_str,
                "company_name": str(cik_map[ticker]["company_name"]).strip(),
                "first_trade_date": str(stats["first_trade_date"]),
                "last_trade_date": str(stats["last_trade_date"]),
                "n_trading_days": n_days,
            }
        )

    df = pd.DataFrame(rows, columns=list(UNIVERSE_COLUMNS))
    if len(df) != 60:
        raise ValueError(f"Expected 60 universe rows, got {len(df)}")
    return df


def write_universe_csv(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Write `data/raw/universe_60.csv` with LF line endings and 10-digit zero-padded CIKs."""
    out_path = Path(path) if path is not None else (config.DATA_RAW_DIR / "universe_60.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, lineterminator="\n")
    return out_path


def load_universe_csv(path: Path | None = None) -> pd.DataFrame:
    """Load `data/raw/universe_60.csv` preserving leading zeros in 10-digit `cik` strings."""
    in_path = Path(path) if path is not None else (config.DATA_RAW_DIR / "universe_60.csv")
    return pd.read_csv(in_path, dtype={"cik": str})
