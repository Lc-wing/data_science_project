"""Daily equity OHLCV, adjusted close, dividends, and stock splits ingestion for Plan 6."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from stock_volatility import config

OHLCV_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividends",
    "stock_splits",
)

_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Dividends",
        "Stock Splits",
        "Capital Gains",
    }
)


def exclusive_end_date(end_date: str) -> str:
    """Return `end_date + 1 day` in ISO format (`yfinance.download(..., end=...)` is exclusive)."""
    return (date.fromisoformat(end_date) + timedelta(days=1)).isoformat()


def _extract_single_ticker_frame(
    raw: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """Extract a single ticker's flat OHLCV frame from a MultiIndex or flat yfinance DataFrame."""
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = {str(x) for x in raw.columns.get_level_values(0)}
        lvl1 = {str(x) for x in raw.columns.get_level_values(1)}
        if ticker in lvl1 and (lvl0 & _FIELD_NAMES):
            sub = raw.xs(ticker, axis=1, level=1, drop_level=True).copy()
        elif ticker in lvl0 and (lvl1 & _FIELD_NAMES):
            sub = raw.xs(ticker, axis=1, level=0, drop_level=True).copy()
        else:
            raise ValueError(f"Ticker {ticker!r} not found in MultiIndex columns {raw.columns}")
    else:
        sub = raw.copy()
    return sub


def normalize_yfinance_frame(
    raw: pd.DataFrame,
    tickers: Sequence[str],
    *,
    start_date: str = config.START_DATE,
    end_date: str = config.END_DATE,
) -> pd.DataFrame:
    """Normalize raw `yfinance.download` output into canonical `(ticker, date)` sorted OHLCV."""
    if raw is None or raw.empty:
        raise ValueError("Cannot normalize empty yfinance DataFrame")

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        sub = _extract_single_ticker_frame(raw, ticker)
        if "Capital Gains" in sub.columns:
            sub = sub.drop(columns=["Capital Gains"])

        required_price_cols = ("Open", "High", "Low", "Close", "Volume")
        for col in required_price_cols:
            if col not in sub.columns:
                raise ValueError(f"Missing column {col!r} for ticker {ticker}")

        adj_col = "Adj Close" if "Adj Close" in sub.columns else "Close"
        div_series = (
            pd.to_numeric(sub["Dividends"], errors="coerce").fillna(0.0)
            if "Dividends" in sub.columns
            else pd.Series(0.0, index=sub.index, dtype=float)
        )
        split_series = (
            pd.to_numeric(sub["Stock Splits"], errors="coerce").fillna(0.0)
            if "Stock Splits" in sub.columns
            else pd.Series(0.0, index=sub.index, dtype=float)
        )

        dt_idx = pd.to_datetime(sub.index)
        if dt_idx.tz is not None:
            dt_idx = dt_idx.tz_localize(None)
        date_str = dt_idx.strftime("%Y-%m-%d")

        t_df = pd.DataFrame(
            {
                "date": date_str,
                "ticker": ticker,
                "open": pd.to_numeric(sub["Open"], errors="coerce").to_numpy(dtype=float),
                "high": pd.to_numeric(sub["High"], errors="coerce").to_numpy(dtype=float),
                "low": pd.to_numeric(sub["Low"], errors="coerce").to_numpy(dtype=float),
                "close": pd.to_numeric(sub["Close"], errors="coerce").to_numpy(dtype=float),
                "adj_close": pd.to_numeric(sub[adj_col], errors="coerce").to_numpy(dtype=float),
                "volume": pd.to_numeric(sub["Volume"], errors="coerce").to_numpy(dtype=float),
                "dividends": div_series.to_numpy(dtype=float),
                "stock_splits": split_series.to_numpy(dtype=float),
            }
        )
        t_df = t_df[(t_df["date"] >= start_date) & (t_df["date"] <= end_date)].copy()
        if t_df.empty:
            raise ValueError(f"No rows in [{start_date}, {end_date}] for ticker {ticker}")

        price_cols = ["open", "high", "low", "close", "adj_close"]
        if t_df[price_cols + ["volume"]].isna().any().any():
            raise ValueError(f"NaN values detected in OHLCV columns for ticker {ticker}")
        if not np.isfinite(t_df[price_cols + ["volume"]].to_numpy()).all():
            raise ValueError(f"Non-finite values detected in OHLCV columns for ticker {ticker}")
        if (t_df[price_cols] <= 0).any().any():
            raise ValueError(f"Non-positive prices detected for ticker {ticker}")
        if (t_df["high"] < t_df["low"]).any():
            raise ValueError(f"Invalid price ordering (high < low) detected for ticker {ticker}")
        if (t_df["volume"] < 0).any():
            raise ValueError(f"Negative volume detected for ticker {ticker}")

        frames.append(t_df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    if combined.duplicated(subset=["ticker", "date"]).any():
        raise ValueError("Duplicate (ticker, date) rows found after normalization")
    return combined[list(OHLCV_COLUMNS)]


def _clear_yfinance_lru_cache() -> None:
    """Clear `YfData().cache_get` LRU cache before retrying a failed ticker."""
    try:
        from yfinance.data import YfData

        yf_data = YfData()
        if hasattr(yf_data, "cache_get") and hasattr(yf_data.cache_get, "cache_clear"):
            yf_data.cache_get.cache_clear()
    except Exception:
        return


def download_ohlcv_universe(
    tickers: Sequence[str] = config.UNIVERSE_TICKERS,
    *,
    out_path: Path | None = None,
    start_date: str = config.START_DATE,
    end_date: str = config.END_DATE,
    batch_size: int = 15,
    expected_days_per_ticker: int = 1760,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download 60-ticker daily OHLCV with `yf.config.network.retries = 5` and per-ticker retry."""
    target_path = (
        Path(out_path)
        if out_path is not None
        else (config.DATA_RAW_DIR / "prices" / "ohlcv_daily.parquet")
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.is_file() and not force:
        cached_df = pd.read_parquet(target_path)
        if len(cached_df) == len(tickers) * expected_days_per_ticker:
            return cached_df, {
                "fetched_at_utc": config.utc_now_iso(),
                "http_status": 200,
                "row_count": len(cached_df),
                "retried_tickers": [],
                "from_disk_cache": True,
            }

    # Activate yfinance built-in exponential backoff retry on transient socket errors
    yf.config.network.retries = 5
    end_exclusive = exclusive_end_date(end_date)

    ticker_frames: list[pd.DataFrame] = []
    retried_tickers: list[str] = []

    ticker_list = list(tickers)
    for start_idx in range(0, len(ticker_list), batch_size):
        batch = ticker_list[start_idx : start_idx + batch_size]
        raw_batch = yf.download(
            batch,
            start=start_date,
            end=end_exclusive,
            auto_adjust=False,
            actions=True,
            progress=False,
            threads=False,
        )
        for ticker in batch:
            try:
                norm_single = normalize_yfinance_frame(
                    raw_batch,
                    [ticker],
                    start_date=start_date,
                    end_date=end_date,
                )
                if len(norm_single) != expected_days_per_ticker:
                    raise ValueError(
                        f"Ticker {ticker} returned {len(norm_single)} rows, "
                        f"expected {expected_days_per_ticker}"
                    )
                ticker_frames.append(norm_single)
            except Exception:
                retried_tickers.append(ticker)
                _clear_yfinance_lru_cache()
                raw_single = yf.download(
                    ticker,
                    start=start_date,
                    end=end_exclusive,
                    auto_adjust=False,
                    actions=True,
                    progress=False,
                    threads=False,
                )
                norm_single = normalize_yfinance_frame(
                    raw_single,
                    [ticker],
                    start_date=start_date,
                    end_date=end_date,
                )
                if len(norm_single) != expected_days_per_ticker:
                    raise ValueError(
                        f"Retry for {ticker} returned {len(norm_single)} rows, "
                        f"expected {expected_days_per_ticker}"
                    ) from None
                ticker_frames.append(norm_single)

    full_df = pd.concat(ticker_frames, ignore_index=True)
    full_df = full_df.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    full_df = full_df[list(OHLCV_COLUMNS)]

    expected_total = len(ticker_list) * expected_days_per_ticker
    if len(full_df) != expected_total:
        raise ValueError(f"Expected {expected_total} OHLCV rows, got {len(full_df)}")

    fetched_at = config.utc_now_iso()
    full_df.to_parquet(target_path, index=False, engine="pyarrow", compression="snappy")
    return full_df, {
        "fetched_at_utc": fetched_at,
        "http_status": 200,
        "row_count": len(full_df),
        "retried_tickers": retried_tickers,
        "from_disk_cache": False,
    }
