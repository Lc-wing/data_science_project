"""Macro & implied volatility ingestion (SPY, Yahoo ^VIX, FRED T10Y2Y & VIXCLS) and alignment."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from stock_volatility import config
from stock_volatility.ingest import build_retry_session
from stock_volatility.ingest.prices import (
    OHLCV_COLUMNS,
    exclusive_end_date,
    normalize_yfinance_frame,
)

MACRO_COLUMNS: tuple[str, ...] = (
    "date",
    "spy_close",
    "spy_adj_close",
    "spy_ret_1d",
    "vix_close",
    "vix_fred",
    "t10y2y",
    "t10y2y_ffilled_flag",
)

VIX_RAW_COLUMNS: tuple[str, ...] = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
)

FRED_URLS: dict[str, str] = {
    "T10Y2Y": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y",
    "VIXCLS": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS",
}

YAHOO_V8_CHART_URL_TEMPLATE: str = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?period1=1546300800&period2=1767312000&interval=1d"
)


def parse_yahoo_v8_chart_payload(
    payload: dict[str, Any],
    *,
    start_date: str = config.START_DATE,
    end_date: str = config.END_DATE,
) -> pd.DataFrame:
    """Parse Yahoo Finance v8 Chart JSON response into a daily OHLCV DataFrame."""
    results = payload.get("chart", {}).get("result") or []
    if not results:
        raise ValueError("Yahoo v8 chart payload missing chart.result")
    res = results[0]
    meta = res.get("meta") or {}
    tz_name = str(meta.get("exchangeTimezoneName", "America/New_York"))
    tz = ZoneInfo(tz_name)

    timestamps = res.get("timestamp") or []
    quotes = (res.get("indicators", {}).get("quote") or [{}])[0]
    opens = quotes.get("open") or []
    highs = quotes.get("high") or []
    lows = quotes.get("low") or []
    closes = quotes.get("close") or []
    volumes = quotes.get("volume") or []

    rows: list[dict[str, Any]] = []
    for idx, ts in enumerate(timestamps):
        close_val = closes[idx] if idx < len(closes) else None
        if close_val is None or (isinstance(close_val, float) and np.isnan(close_val)):
            continue
        dt_local = datetime.fromtimestamp(int(ts), tz=UTC).astimezone(tz)
        dt_str = dt_local.strftime("%Y-%m-%d")
        if dt_str < start_date or dt_str > end_date:
            continue
        c_float = float(close_val)
        o_val = opens[idx] if idx < len(opens) and opens[idx] is not None else c_float
        h_val = highs[idx] if idx < len(highs) and highs[idx] is not None else c_float
        l_val = lows[idx] if idx < len(lows) and lows[idx] is not None else c_float
        v_val = volumes[idx] if idx < len(volumes) and volumes[idx] is not None else 0.0
        rows.append(
            {
                "date": dt_str,
                "open": float(o_val),
                "high": float(h_val),
                "low": float(l_val),
                "close": c_float,
                "adj_close": c_float,
                "volume": float(v_val),
            }
        )

    if not rows:
        raise ValueError("Yahoo v8 chart payload produced zero valid daily rows")
    df = pd.DataFrame(rows, columns=list(VIX_RAW_COLUMNS))
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    return df


def fetch_yahoo_v8_daily(
    symbol: str = "^VIX",
    *,
    start_date: str = config.START_DATE,
    end_date: str = config.END_DATE,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch daily series via Yahoo Finance v8 Chart API fallback."""
    sess = session or build_retry_session()
    encoded = quote(symbol, safe="")
    url = YAHOO_V8_CHART_URL_TEMPLATE.format(symbol=encoded)
    resp = sess.get(url, timeout=30)
    resp.raise_for_status()
    df = parse_yahoo_v8_chart_payload(resp.json(), start_date=start_date, end_date=end_date)
    return df, {
        "url": url,
        "http_status": int(resp.status_code),
        "fetched_at_utc": config.utc_now_iso(),
        "row_count": len(df),
        "method": "yahoo_v8_chart_api",
    }


def parse_fred_csv(path: Path, series_id: str) -> pd.DataFrame:
    """Parse a raw FRED CSV (`observation_date,<SERIES>`) coercing `''` and `'.'` to `NaN`."""
    raw_df = pd.read_csv(path, dtype=str)
    date_col = "observation_date" if "observation_date" in raw_df.columns else raw_df.columns[0]
    val_col = series_id if series_id in raw_df.columns else raw_df.columns[1]
    out = pd.DataFrame(
        {
            "date": raw_df[date_col].astype(str).str.strip(),
            "value": pd.to_numeric(raw_df[val_col].astype(str).str.strip(), errors="coerce"),
        }
    )
    out = out.dropna(subset=["date"]).sort_values("date", kind="mergesort").reset_index(drop=True)
    return out


def align_macro_to_calendar(
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    fred_t10y2y_df: pd.DataFrame,
    fred_vixcls_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align SPY, Yahoo ^VIX, FRED T10Y2Y, and FRED VIXCLS onto SPY's canonical NYSE calendar.

    Strictly applies backward-only forward-fill across the full sorted FRED history BEFORE
    left-joining onto the NYSE trading calendar. Never backfills.
    """
    if spy_df.empty:
        raise ValueError("spy_df cannot be empty")

    spy = spy_df.sort_values("date", kind="mergesort").reset_index(drop=True).copy()
    if spy["date"].duplicated().any():
        raise ValueError("Duplicate dates in spy_df canonical calendar")

    cal = pd.DataFrame({"date": spy["date"].astype(str)})
    cal["spy_close"] = pd.to_numeric(spy["close"], errors="coerce").astype(float)
    cal["spy_adj_close"] = pd.to_numeric(spy["adj_close"], errors="coerce").astype(float)
    cal["spy_ret_1d"] = cal["spy_adj_close"].pct_change()

    # Align Yahoo ^VIX with backward-only ffill across sorted VIX history (never bfill)
    vix_sorted = vix_df.sort_values("date", kind="mergesort").drop_duplicates("date", keep="last")
    vix_raw_map = dict(
        zip(
            vix_sorted["date"].astype(str),
            pd.to_numeric(vix_sorted["close"], errors="coerce"),
            strict=True,
        )
    )
    vix_union_dates = sorted(set(cal["date"]).union(vix_raw_map.keys()))
    vix_union = pd.Series(
        [vix_raw_map.get(d, np.nan) for d in vix_union_dates],
        index=vix_union_dates,
    )
    vix_raw_on_cal = vix_union.reindex(cal["date"]).to_numpy()
    vix_ffilled = vix_union.ffill().reindex(cal["date"]).to_numpy(dtype=float)
    if np.isnan(vix_ffilled).any():
        raise ValueError("Missing leading vix_close value on NYSE calendar; backfill is forbidden")
    vix_close_ffilled_count = int(np.isnan(vix_raw_on_cal).sum())
    cal["vix_close"] = vix_ffilled

    # Backward-only forward-fill on full FRED VIXCLS history BEFORE joining to NYSE calendar
    vixcls_sorted = (
        fred_vixcls_df.sort_values("date", kind="mergesort")
        .drop_duplicates("date", keep="last")
        .copy()
    )
    vixcls_raw_map = dict(
        zip(
            vixcls_sorted["date"].astype(str),
            pd.to_numeric(vixcls_sorted["value"], errors="coerce"),
            strict=True,
        )
    )
    vixcls_union_dates = sorted(set(cal["date"]).union(vixcls_raw_map.keys()))
    vixcls_union = pd.Series(
        [vixcls_raw_map.get(d, np.nan) for d in vixcls_union_dates],
        index=vixcls_union_dates,
        dtype=float,
    )
    vixcls_raw_on_cal = vixcls_union.reindex(cal["date"]).to_numpy(dtype=float)
    vixcls_ffilled_on_cal = vixcls_union.ffill().reindex(cal["date"]).to_numpy(dtype=float)
    if np.isnan(vixcls_ffilled_on_cal).any():
        raise ValueError(
            "Missing leading FRED VIXCLS value on NYSE calendar; backfill is forbidden"
        )
    vix_fred_ffilled_count = int(np.isnan(vixcls_raw_on_cal).sum())
    cal["vix_fred"] = vixcls_ffilled_on_cal

    # Backward-only forward-fill on full FRED T10Y2Y history BEFORE joining to NYSE calendar
    t10_sorted = (
        fred_t10y2y_df.sort_values("date", kind="mergesort")
        .drop_duplicates("date", keep="last")
        .copy()
    )
    t10_raw_map = dict(
        zip(
            t10_sorted["date"].astype(str),
            pd.to_numeric(t10_sorted["value"], errors="coerce"),
            strict=True,
        )
    )
    t10_union_dates = sorted(set(cal["date"]).union(t10_raw_map.keys()))
    t10_union = pd.Series(
        [t10_raw_map.get(d, np.nan) for d in t10_union_dates],
        index=t10_union_dates,
        dtype=float,
    )
    t10_raw_on_cal = t10_union.reindex(cal["date"]).to_numpy(dtype=float)
    t10_ffilled_on_cal = t10_union.ffill().reindex(cal["date"]).to_numpy(dtype=float)
    if np.isnan(t10_ffilled_on_cal).any():
        raise ValueError(
            "Missing leading FRED T10Y2Y value on NYSE calendar; backfill is forbidden"
        )
    t10_ffilled_mask = np.isnan(t10_raw_on_cal)
    cal["t10y2y"] = t10_ffilled_on_cal
    cal["t10y2y_ffilled_flag"] = t10_ffilled_mask.astype(bool)

    cal = cal[list(MACRO_COLUMNS)]

    abs_diff = np.abs(cal["vix_close"] - cal["vix_fred"])
    max_idx = int(abs_diff.idxmax())
    corr_val = (
        float(cal["vix_close"].corr(cal["vix_fred"]))
        if len(cal) >= 2 and cal["vix_close"].std() > 0 and cal["vix_fred"].std() > 0
        else 1.0
    )
    ffilled_dates = list(cal.loc[cal["t10y2y_ffilled_flag"], "date"].astype(str))

    summary: dict[str, Any] = {
        "n_trading_days": len(cal),
        "first_date": str(cal["date"].iloc[0]),
        "last_date": str(cal["date"].iloc[-1]),
        "spy_ret_1d_nan_count": int(cal["spy_ret_1d"].isna().sum()),
        "vix_close_ffilled_count": vix_close_ffilled_count,
        "vix_fred_ffilled_count": vix_fred_ffilled_count,
        "t10y2y_ffilled_count": int(cal["t10y2y_ffilled_flag"].sum()),
        "t10y2y_ffilled_share": float(cal["t10y2y_ffilled_flag"].mean()),
        "t10y2y_ffilled_dates": ffilled_dates,
        "vix_close_vs_fred_pearson_r": corr_val,
        "vix_close_vs_fred_mean_abs_diff": float(abs_diff.mean()),
        "vix_close_vs_fred_max_abs_diff": float(abs_diff.max()),
        "vix_close_vs_fred_max_abs_diff_date": str(cal.loc[max_idx, "date"]),
    }
    return cal, summary


def download_macro_raw(
    raw_macro_dir: Path | None = None,
    *,
    start_date: str = config.START_DATE,
    end_date: str = config.END_DATE,
    expected_trading_days: int = 1760,
    session: requests.Session | None = None,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """Download raw SPY, Yahoo ^VIX (with v8 chart fallback), and FRED T10Y2Y & VIXCLS snapshots."""
    macro_dir = (
        Path(raw_macro_dir)
        if raw_macro_dir is not None
        else (config.DATA_RAW_DIR / "macro")
    )
    macro_dir.mkdir(parents=True, exist_ok=True)
    sess = session or build_retry_session()

    yf.config.network.retries = 5
    end_exclusive = exclusive_end_date(end_date)

    # 1. SPY canonical NYSE trading calendar & benchmark
    spy_path = macro_dir / "spy_daily.parquet"
    if not (spy_path.is_file() and not force):
        raw_spy = yf.download(
            "SPY",
            start=start_date,
            end=end_exclusive,
            auto_adjust=False,
            actions=True,
            progress=False,
            threads=False,
        )
        norm_spy = normalize_yfinance_frame(
            raw_spy,
            ["SPY"],
            start_date=start_date,
            end_date=end_date,
        )
        spy_cols = [c for c in OHLCV_COLUMNS if c != "ticker"]
        spy_df = norm_spy[spy_cols].reset_index(drop=True)
        if len(spy_df) != expected_trading_days:
            raise ValueError(
                f"SPY download returned {len(spy_df)} rows, expected {expected_trading_days}"
            )
        spy_df.to_parquet(spy_path, index=False, engine="pyarrow", compression="snappy")
        spy_fetched_at = config.utc_now_iso()
    else:
        spy_df = pd.read_parquet(spy_path)
        spy_fetched_at = config.utc_now_iso()

    # 2. Yahoo ^VIX with automatic fallback to Yahoo v8 Chart API
    vix_path = macro_dir / "vix_daily.parquet"
    vix_method = "yfinance_with_retries_5"
    vix_url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?interval=1d"
    if not (vix_path.is_file() and not force):
        vix_df: pd.DataFrame | None = None
        try:
            raw_vix = yf.download(
                "^VIX",
                start=start_date,
                end=end_exclusive,
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )
            norm_vix = normalize_yfinance_frame(
                raw_vix,
                ["^VIX"],
                start_date=start_date,
                end_date=end_date,
            )
            if len(norm_vix) == expected_trading_days:
                vix_df = norm_vix[list(VIX_RAW_COLUMNS)].reset_index(drop=True)
        except Exception:
            vix_df = None

        if vix_df is None or len(vix_df) != expected_trading_days:
            vix_df, v8_meta = fetch_yahoo_v8_daily(
                "^VIX",
                start_date=start_date,
                end_date=end_date,
                session=sess,
            )
            vix_method = str(v8_meta["method"])
            vix_url = str(v8_meta["url"])

        if len(vix_df) != expected_trading_days:
            raise ValueError(
                f"^VIX download returned {len(vix_df)} rows, expected {expected_trading_days}"
            )
        vix_df.to_parquet(vix_path, index=False, engine="pyarrow", compression="snappy")
        vix_fetched_at = config.utc_now_iso()
    else:
        vix_df = pd.read_parquet(vix_path)
        vix_fetched_at = config.utc_now_iso()

    # 3. Full FRED CSV downloads (retaining pre-2019 history for backward-only ffill)
    t10_path = macro_dir / "fred_t10y2y.csv"
    if not (t10_path.is_file() and not force):
        resp_t10 = sess.get(FRED_URLS["T10Y2Y"], timeout=30)
        resp_t10.raise_for_status()
        t10_path.write_bytes(resp_t10.content)
        t10_status = int(resp_t10.status_code)
    else:
        t10_status = 200
    t10_fetched_at = config.utc_now_iso()
    t10_df = parse_fred_csv(t10_path, "T10Y2Y")

    vixcls_path = macro_dir / "fred_vixcls.csv"
    if not (vixcls_path.is_file() and not force):
        resp_vixcls = sess.get(FRED_URLS["VIXCLS"], timeout=30)
        resp_vixcls.raise_for_status()
        vixcls_path.write_bytes(resp_vixcls.content)
        vixcls_status = int(resp_vixcls.status_code)
    else:
        vixcls_status = 200
    vixcls_fetched_at = config.utc_now_iso()
    vixcls_df = parse_fred_csv(vixcls_path, "VIXCLS")

    return {
        "spy_daily": {
            "path": spy_path,
            "source": "yahoo_finance_yfinance",
            "source_url": "https://query1.finance.yahoo.com/v8/finance/chart/SPY?interval=1d",
            "http_status": 200,
            "fetched_at_utc": spy_fetched_at,
            "row_count": len(spy_df),
        },
        "vix_daily": {
            "path": vix_path,
            "source": f"yahoo_finance_{vix_method}",
            "source_url": vix_url,
            "http_status": 200,
            "fetched_at_utc": vix_fetched_at,
            "row_count": len(vix_df),
        },
        "fred_t10y2y": {
            "path": t10_path,
            "source": "fred_http_csv",
            "source_url": FRED_URLS["T10Y2Y"],
            "http_status": t10_status,
            "fetched_at_utc": t10_fetched_at,
            "row_count": len(t10_df),
        },
        "fred_vixcls": {
            "path": vixcls_path,
            "source": "fred_http_csv",
            "source_url": FRED_URLS["VIXCLS"],
            "http_status": vixcls_status,
            "fetched_at_utc": vixcls_fetched_at,
            "row_count": len(vixcls_df),
        },
    }


def build_macro_daily_from_raw(
    raw_macro_dir: Path | None = None,
    *,
    out_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild `data/interim/macro_daily.parquet` offline from committed `data/raw/macro/`."""
    macro_dir = (
        Path(raw_macro_dir)
        if raw_macro_dir is not None
        else (config.DATA_RAW_DIR / "macro")
    )
    target_path = (
        Path(out_path)
        if out_path is not None
        else (config.DATA_INTERIM_DIR / "macro_daily.parquet")
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)

    spy_df = pd.read_parquet(macro_dir / "spy_daily.parquet")
    vix_df = pd.read_parquet(macro_dir / "vix_daily.parquet")
    t10_df = parse_fred_csv(macro_dir / "fred_t10y2y.csv", "T10Y2Y")
    vixcls_df = parse_fred_csv(macro_dir / "fred_vixcls.csv", "VIXCLS")

    macro_df, summary = align_macro_to_calendar(spy_df, vix_df, t10_df, vixcls_df)
    macro_df.to_parquet(target_path, index=False, engine="pyarrow", compression="snappy")
    return macro_df, summary
