"""Paginated SEC EDGAR Form 8-K, 8-K/A, 10-Q, and 10-K corporate event ingestion for Plan 6."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from stock_volatility import config
from stock_volatility.ingest import build_retry_session
from stock_volatility.ingest.universe import (
    CIK_PREDECESSOR_LINKAGE_METADATA,
    PREDECESSOR_CIK_MAP,
    resolve_universe_ciks,
)

SEC_COMPANY_TICKERS_URL: str = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_TEMPLATE: str = "https://data.sec.gov/submissions/{filename}"

TARGET_FORMS: tuple[str, ...] = ("8-K", "8-K/A", "10-Q", "10-K")

EVENTS_RAW_COLUMNS: tuple[str, ...] = (
    "ticker",
    "cik",
    "accession_number",
    "form",
    "filing_date",
    "report_date",
    "acceptance_datetime_utc",
    "items",
    "item_descriptions",
    "primary_doc_description",
    "headline_text",
    "news_source",
)

SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "ticker",
    "cik",
    "accession_number",
    "form",
    "filing_date",
    "report_date",
    "acceptance_datetime_raw",
    "acceptance_datetime_utc",
    "items_raw",
    "items",
    "primary_document",
    "primary_doc_description",
    "source_file",
)

# Complete 33-item official SEC Form 8-K item taxonomy (Sections 1 through 9, including 1.05)
SEC_8K_ITEM_DESCRIPTIONS: dict[str, str] = {
    # Section 1 — Registrant's Business and Operations
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "1.04": "Mine Safety - Reporting of Shutdowns and Patterns of Violations",
    "1.05": "Material Cybersecurity Incidents",
    # Section 2 — Financial Information
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": (
        "Creation of a Direct Financial Obligation or an Obligation under an "
        "Off-Balance Sheet Arrangement of a Registrant"
    ),
    "2.04": (
        "Triggering Events That Accelerate or Increase a Direct Financial Obligation "
        "or an Obligation under an Off-Balance Sheet Arrangement"
    ),
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    # Section 3 — Securities and Trading Markets
    "3.01": (
        "Notice of Delisting or Failure to Satisfy a Continued Listing Rule or "
        "Standard; Transfer of Listing"
    ),
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    # Section 4 — Matters Related to Accountants and Financial Statements
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": (
        "Non-Reliance on Previously Issued Financial Statements or a Related Audit "
        "Report or Completed Interim Review"
    ),
    # Section 5 — Corporate Governance and Management
    "5.01": "Changes in Control of Registrant",
    "5.02": (
        "Departure of Directors or Certain Officers; Election of Directors; "
        "Appointment of Certain Officers; Compensatory Arrangements of Certain Officers"
    ),
    "5.03": "Amendments to Articles of Incorporation or Bylaws; Change in Fiscal Year",
    "5.04": "Temporary Suspension of Trading Under Registrant's Employee Benefit Plans",
    "5.05": (
        "Amendments to the Registrant's Code of Ethics, or Waiver of a Provision of "
        "the Code of Ethics"
    ),
    "5.06": "Change in Shell Company Status",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "5.08": "Shareholder Director Nominations",
    # Section 6 — Asset-Backed Securities
    "6.01": "ABS Informational and Computational Material",
    "6.02": "Change of Servicer or Trustee",
    "6.03": "Change in Credit Enhancement or Other External Support",
    "6.04": "Failure to Make a Required Distribution",
    "6.05": "Securities Act Updating Disclosure",
    "6.06": "Static Pool",
    # Section 7 — Regulation FD
    "7.01": "Regulation FD Disclosure",
    # Section 8 — Other Events
    "8.01": "Other Events",
    # Section 9 — Financial Statements and Exhibits
    "9.01": "Financial Statements and Exhibits",
}


class RateLimiter:
    """Monotonic-clock rate limiter enforcing `<= 7.5 req/s` (`<= 8.0 req/s` SEC budget)."""

    def __init__(self, max_per_sec: float = 7.5) -> None:
        effective = min(float(max_per_sec), config.SEC_MAX_REQUESTS_PER_SEC)
        if effective <= 0:
            raise ValueError("max_per_sec must be positive")
        self.min_interval: float = 1.0 / effective
        self._last_call: float = 0.0

    def wait(self) -> None:
        """Sleep if needed so consecutive calls are spaced by at least `min_interval` seconds."""
        now = time.monotonic()
        elapsed = now - self._last_call
        if self._last_call > 0.0 and elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()


def _clean_str(val: Any) -> str:
    """Convert optional string/scalar to stripped string, treating None and NaN as empty."""
    if val is None or (isinstance(val, float) and pd.isna(val)) or pd.isna(val):
        return ""
    text = str(val).strip()
    return "" if text.lower() == "nan" else text


def normalize_acceptance_datetime(
    raw_value: str | None,
    *,
    fallback_filing_date: str = "",
) -> str:
    """Normalize SEC `acceptanceDateTime` to canonical second-precision ISO-8601 UTC."""
    text = _clean_str(raw_value)
    if not text:
        if fallback_filing_date:
            return f"{fallback_filing_date}T21:00:00+00:00"
        raise ValueError("Empty acceptanceDateTime without fallback_filing_date")
    if re.fullmatch(r"\d{14}", text):
        dt = datetime.strptime(text, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        return dt.isoformat(timespec="seconds")
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt.isoformat(timespec="seconds")


def normalize_sec_items(raw_items: Any) -> str:
    """Normalize polymorphic SEC `items` (`str`, `list[str]`, `None`) to comma-separated codes."""
    if raw_items is None:
        return ""
    tokens: list[str] = []
    if isinstance(raw_items, str):
        parts = re.split(r"[,;\s]+", raw_items.strip())
        tokens = [p.strip() for p in parts if p.strip()]
    elif isinstance(raw_items, Iterable):
        for item in raw_items:
            if item is None:
                continue
            for sub in re.split(r"[,;\s]+", str(item).strip()):
                if sub.strip():
                    tokens.append(sub.strip())
    else:
        text = str(raw_items).strip()
        if text and text.lower() != "nan":
            tokens = [p.strip() for p in re.split(r"[,;\s]+", text) if p.strip()]

    seen: set[str] = set()
    ordered: list[str] = []
    for tok in tokens:
        if tok not in seen:
            seen.add(tok)
            ordered.append(tok)
    return ",".join(ordered)


def describe_filing_items(form: str, items_csv: str) -> str:
    """Map form type and comma-separated Form 8-K item codes to official descriptions."""
    clean_form = form.strip().upper()
    if clean_form == "10-Q":
        return "Quarterly Report (Form 10-Q)"
    if clean_form == "10-K":
        return "Annual Report (Form 10-K)"
    codes = [c.strip() for c in items_csv.split(",") if c.strip()]
    if not codes:
        return f"Current Report (Form {clean_form})"
    descriptions = [SEC_8K_ITEM_DESCRIPTIONS.get(code, f"Item {code}") for code in codes]
    return "; ".join(descriptions)


def build_headline_text(
    *,
    ticker: str,
    form: str,
    items: str,
    item_descriptions: str,
    report_date: str,
    primary_doc_description: str,
) -> str:
    """Construct rich, non-empty structured corporate event text for Phase A & Phase B."""
    parts: list[str] = [f"[{ticker}] Form {form}"]
    if items:
        parts.append(f"Items {items}: {item_descriptions}")
    else:
        parts.append(item_descriptions)
    if report_date:
        parts.append(f"Report {report_date}")
    if primary_doc_description:
        parts.append(f"Desc: {primary_doc_description}")
    return " | ".join(parts)


def select_overlapping_archives(
    files_meta: list[dict[str, Any]],
    *,
    window_start: str = "2018-12-25",
    window_end: str = "2026-01-05",
) -> list[dict[str, Any]]:
    """Select all `filings.files` archive entries overlapping `[window_start, window_end]`."""
    selected: list[dict[str, Any]] = []
    for meta in files_meta or []:
        if not isinstance(meta, dict):
            continue
        name = str(meta.get("name", "")).strip()
        if not name:
            continue
        ff = str(meta.get("filingFrom", "") or "").strip()
        ft = str(meta.get("filingTo", "") or "").strip()
        if not ff or not ft or (ft >= window_start and ff <= window_end):
            selected.append(meta)
    return selected


def parse_submissions_block(
    block: dict[str, Any],
    *,
    ticker: str,
    cik: str,
    source_file: str = "",
    start_date: str = "2019-01-01",
    end_date: str = config.END_DATE,
    target_forms: tuple[str, ...] = TARGET_FORMS,
) -> list[dict[str, Any]]:
    """Parse columnar arrays from `payload['filings']['recent']` or a top-level archive JSON."""
    if not block or "accessionNumber" not in block:
        return []

    accessions = block.get("accessionNumber") or []
    filing_dates = block.get("filingDate") or []
    report_dates = block.get("reportDate") or []
    acceptance_dts = block.get("acceptanceDateTime") or []
    forms = block.get("form") or []
    items_arr = block.get("items") or []
    primary_docs = block.get("primaryDocument") or []
    primary_descs = block.get("primaryDocDescription") or []

    n_rows = len(accessions)
    target_set = frozenset(target_forms)
    cik_10 = f"{int(cik):010d}" if str(cik).isdigit() else str(cik).strip()

    records: list[dict[str, Any]] = []
    for idx in range(n_rows):
        form_val = str(forms[idx] if idx < len(forms) and forms[idx] is not None else "").strip()
        if form_val not in target_set:
            continue
        f_date = str(
            filing_dates[idx] if idx < len(filing_dates) and filing_dates[idx] is not None else ""
        ).strip()
        if not f_date or f_date < start_date or f_date > end_date:
            continue

        acc_num = str(accessions[idx] or "").strip()
        if not acc_num:
            continue
        r_date = str(
            report_dates[idx] if idx < len(report_dates) and report_dates[idx] is not None else ""
        ).strip()
        raw_acc_dt = str(
            acceptance_dts[idx]
            if idx < len(acceptance_dts) and acceptance_dts[idx] is not None
            else ""
        ).strip()
        norm_acc_dt = normalize_acceptance_datetime(raw_acc_dt, fallback_filing_date=f_date)
        raw_item = items_arr[idx] if idx < len(items_arr) else ""
        norm_items = normalize_sec_items(raw_item)
        p_doc = str(
            primary_docs[idx] if idx < len(primary_docs) and primary_docs[idx] is not None else ""
        ).strip()
        p_desc = str(
            primary_descs[idx]
            if idx < len(primary_descs) and primary_descs[idx] is not None
            else ""
        ).strip()

        records.append(
            {
                "ticker": ticker,
                "cik": cik_10,
                "accession_number": acc_num,
                "form": form_val,
                "filing_date": f_date,
                "report_date": r_date,
                "acceptance_datetime_raw": raw_acc_dt,
                "acceptance_datetime_utc": norm_acc_dt,
                "items_raw": (
                    raw_item if isinstance(raw_item, str) else json.dumps(raw_item)
                ),
                "items": norm_items,
                "primary_document": p_doc,
                "primary_doc_description": p_desc,
                "source_file": source_file,
            }
        )
    return records


def deduplicate_filings(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate filings by `(ticker, accession_number)` and sort deterministically."""
    if not records:
        return []
    sorted_recs = sorted(
        records,
        key=lambda r: (
            str(r["ticker"]),
            str(r["acceptance_datetime_utc"]),
            str(r["accession_number"]),
            str(r["cik"]),
        ),
    )
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    for rec in sorted_recs:
        key = (str(rec["ticker"]), str(rec["accession_number"]))
        if key not in seen:
            seen.add(key)
            unique.append(rec)
    return unique


def _fetch_sec_json(
    url: str,
    cache_path: Path,
    *,
    session: requests.Session,
    limiter: RateLimiter,
    force: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fetch a SEC EDGAR JSON file with rate limiting and `.cache/sec_edgar/` disk caching."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.is_file() and not force:
        raw_bytes = cache_path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
        return payload, {
            "url": url,
            "http_status": 200,
            "raw_bytes": len(raw_bytes),
            "raw_sha256": config.sha256_bytes(raw_bytes),
            "from_cache": True,
        }

    limiter.wait()
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    raw_bytes = resp.content
    cache_path.write_bytes(raw_bytes)
    payload = json.loads(raw_bytes.decode("utf-8"))
    return payload, {
        "url": url,
        "http_status": int(resp.status_code),
        "raw_bytes": len(raw_bytes),
        "raw_sha256": config.sha256_bytes(raw_bytes),
        "from_cache": False,
    }


def download_sec_edgar_all(
    *,
    raw_sec_dir: Path | None = None,
    cache_sec_dir: Path | None = None,
    session: requests.Session | None = None,
    window_start: str = "2018-12-25",
    window_end: str = "2026-01-05",
    filing_start_date: str = "2019-01-01",
    filing_end_date: str = config.END_DATE,
    force: bool = False,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    """Query `company_tickers.json`, all 62 CIKs, and 195 overlapping archives (<= 7.5 req/s)."""
    sec_raw = (
        Path(raw_sec_dir)
        if raw_sec_dir is not None
        else (config.DATA_RAW_DIR / "sec_edgar")
    )
    sec_cache = (
        Path(cache_sec_dir)
        if cache_sec_dir is not None
        else (config.CACHE_DIR / "sec_edgar")
    )
    sec_raw.mkdir(parents=True, exist_ok=True)
    sec_cache.mkdir(parents=True, exist_ok=True)

    owns_session = session is None
    sess = session or build_retry_session(user_agent=config.SEC_USER_AGENT)
    limiter = RateLimiter(max_per_sec=7.5)
    try:
        # 1. Fetch company_tickers.json
        tickers_payload, tickers_meta = _fetch_sec_json(
            SEC_COMPANY_TICKERS_URL,
            sec_cache / "company_tickers.json",
            session=sess,
            limiter=limiter,
            force=force,
        )
        cik_map = resolve_universe_ciks(tickers_payload)

        # Write compact committable snapshot data/raw/sec_edgar/company_tickers_60.json
        company_tickers_60_path = sec_raw / "company_tickers_60.json"
        company_tickers_60_obj = {
            "source_url": SEC_COMPANY_TICKERS_URL,
            "http_status": tickers_meta["http_status"],
            "raw_byte_size": tickers_meta["raw_bytes"],
            "raw_sha256": tickers_meta["raw_sha256"],
            "universe_count": len(cik_map),
            "predecessor_cik_map": {k: list(v) for k, v in PREDECESSOR_CIK_MAP.items()},
            "cik_predecessor_linkages": CIK_PREDECESSOR_LINKAGE_METADATA,
            "tickers": cik_map,
        }
        company_tickers_60_path.write_text(
            json.dumps(company_tickers_60_obj, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # 2. Fetch main submissions + overlapping historical archives across all 62 CIKs
        all_raw_records: list[dict[str, Any]] = []
        manifest_requests: list[dict[str, Any]] = [
            {
                "request_type": "company_tickers",
                "ticker": "ALL_60",
                "cik": "",
                "filename": "company_tickers.json",
                "url": SEC_COMPANY_TICKERS_URL,
                "http_status": tickers_meta["http_status"],
                "raw_byte_size": tickers_meta["raw_bytes"],
                "raw_sha256": tickers_meta["raw_sha256"],
                "filingFrom": "",
                "filingTo": "",
                "total_rows": len(tickers_payload),
                "target_rows_2019_2025": 60,
            }
        ]

        total_ciks_queried = 0
        total_archives_listed = 0
        overlapping_archives_fetched = 0
        filings_from_recent_raw = 0
        filings_from_archives_raw = 0

        for ticker, _, _ in config.UNIVERSE_60:
            info = cik_map[ticker]
            for role_idx, cik_10 in enumerate(info["all_ciks"]):
                role = "primary" if role_idx == 0 else "predecessor"
                total_ciks_queried += 1
                main_filename = f"CIK{cik_10}.json"
                main_url = SEC_SUBMISSIONS_URL_TEMPLATE.format(filename=main_filename)
                main_payload, main_meta = _fetch_sec_json(
                    main_url,
                    sec_cache / main_filename,
                    session=sess,
                    limiter=limiter,
                    force=force,
                )

                recent_block = main_payload.get("filings", {}).get("recent", {})
                recent_total_rows = len(recent_block.get("accessionNumber") or [])
                recent_Dates = [str(d) for d in (recent_block.get("filingDate") or []) if d]
                recent_from = min(recent_Dates) if recent_Dates else ""
                recent_to = max(recent_Dates) if recent_Dates else ""

                recent_recs = parse_submissions_block(
                    recent_block,
                    ticker=ticker,
                    cik=cik_10,
                    source_file=main_filename,
                    start_date=filing_start_date,
                    end_date=filing_end_date,
                )
                filings_from_recent_raw += len(recent_recs)
                all_raw_records.extend(recent_recs)

                manifest_requests.append(
                    {
                        "request_type": f"submissions_main_{role}",
                        "ticker": ticker,
                        "cik": cik_10,
                        "filename": main_filename,
                        "url": main_url,
                        "http_status": main_meta["http_status"],
                        "raw_byte_size": main_meta["raw_bytes"],
                        "raw_sha256": main_meta["raw_sha256"],
                        "filingFrom": recent_from,
                        "filingTo": recent_to,
                        "total_rows": recent_total_rows,
                        "target_rows_2019_2025": len(recent_recs),
                    }
                )

                files_meta = main_payload.get("filings", {}).get("files") or []
                total_archives_listed += len(files_meta)
                overlapping = select_overlapping_archives(
                    files_meta,
                    window_start=window_start,
                    window_end=window_end,
                )
                for arch_meta in overlapping:
                    overlapping_archives_fetched += 1
                    arch_name = str(arch_meta["name"]).strip()
                    arch_url = SEC_SUBMISSIONS_URL_TEMPLATE.format(filename=arch_name)
                    arch_payload, arch_http_meta = _fetch_sec_json(
                        arch_url,
                        sec_cache / arch_name,
                        session=sess,
                        limiter=limiter,
                        force=force,
                    )
                    arch_total_rows = len(arch_payload.get("accessionNumber") or [])
                    arch_recs = parse_submissions_block(
                        arch_payload,
                        ticker=ticker,
                        cik=cik_10,
                        source_file=arch_name,
                        start_date=filing_start_date,
                        end_date=filing_end_date,
                    )
                    filings_from_archives_raw += len(arch_recs)
                    all_raw_records.extend(arch_recs)

                    manifest_requests.append(
                        {
                            "request_type": f"submissions_archive_{role}",
                            "ticker": ticker,
                            "cik": cik_10,
                            "filename": arch_name,
                            "url": arch_url,
                            "http_status": arch_http_meta["http_status"],
                            "raw_byte_size": arch_http_meta["raw_bytes"],
                            "raw_sha256": arch_http_meta["raw_sha256"],
                            "filingFrom": str(arch_meta.get("filingFrom", "")),
                            "filingTo": str(arch_meta.get("filingTo", "")),
                            "total_rows": arch_total_rows,
                            "target_rows_2019_2025": len(arch_recs),
                        }
                    )
    finally:
        if owns_session:
            sess.close()

    unique_records = deduplicate_filings(all_raw_records)
    duplicate_count = len(all_raw_records) - len(unique_records)

    snapshot_df = pd.DataFrame(unique_records, columns=list(SNAPSHOT_COLUMNS))
    snapshot_path = sec_raw / "edgar_filings_2019_2025.parquet"
    snapshot_df.to_parquet(snapshot_path, index=False, engine="pyarrow", compression="snappy")

    filings_from_recent_unique = int(
        (~snapshot_df["source_file"].str.contains("-submissions-")).sum()
    )
    filings_from_archives_unique = int(
        snapshot_df["source_file"].str.contains("-submissions-").sum()
    )
    filings_1923 = int((snapshot_df["filing_date"] <= "2023-12-31").sum())
    filings_2425 = int((snapshot_df["filing_date"] >= "2024-01-01").sum())

    pagination_manifest_path = sec_raw / "pagination_manifest.json"
    pagination_manifest_obj: dict[str, Any] = {
        "schema_version": "1.0",
        "window_start": window_start,
        "window_end": window_end,
        "filing_start_date": filing_start_date,
        "filing_end_date": filing_end_date,
        "target_forms": list(TARGET_FORMS),
        "max_requests_per_sec_limit": config.SEC_MAX_REQUESTS_PER_SEC,
        "effective_rate_limit_per_sec": 7.5,
        "total_ciks_queried": total_ciks_queried,
        "primary_ciks": 60,
        "predecessor_ciks": total_ciks_queried - 60,
        "total_archives_listed": total_archives_listed,
        "overlapping_archives_fetched": overlapping_archives_fetched,
        "total_http_requests": len(manifest_requests),
        "raw_target_filings_before_dedup": len(all_raw_records),
        "duplicate_accession_rows_dropped": duplicate_count,
        "total_unique_target_filings_2019_2025": len(snapshot_df),
        "filings_from_recent": filings_from_recent_unique,
        "filings_from_archives": filings_from_archives_unique,
        "filings_2019_2023": filings_1923,
        "filings_2024_2025": filings_2425,
        "requests": manifest_requests,
    }
    pagination_manifest_path.write_text(
        json.dumps(pagination_manifest_obj, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    fetched_at = config.utc_now_iso()
    summary = {
        "fetched_at_utc": fetched_at,
        "company_tickers_60_path": company_tickers_60_path,
        "pagination_manifest_path": pagination_manifest_path,
        "snapshot_path": snapshot_path,
        "pagination_summary": {
            k: v for k, v in pagination_manifest_obj.items() if k != "requests"
        },
    }
    return cik_map, snapshot_df, summary


def build_events_raw_from_snapshot(
    snapshot_path: Path | None = None,
    *,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Build `data/interim/events_raw.parquet` offline from `edgar_filings_2019_2025.parquet`."""
    in_path = (
        Path(snapshot_path)
        if snapshot_path is not None
        else (config.DATA_RAW_DIR / "sec_edgar" / "edgar_filings_2019_2025.parquet")
    )
    snap_df = pd.read_parquet(in_path)
    if snap_df.empty:
        raise ValueError(f"SEC EDGAR snapshot at {in_path} is empty")

    rows: list[dict[str, Any]] = []
    for rec in snap_df.to_dict(orient="records"):
        ticker = _clean_str(rec["ticker"])
        cik_raw = _clean_str(rec["cik"])
        cik_10 = f"{int(cik_raw):010d}" if cik_raw.isdigit() else cik_raw
        acc_num = _clean_str(rec["accession_number"])
        form = _clean_str(rec["form"])
        f_date = _clean_str(rec["filing_date"])
        r_date = _clean_str(rec.get("report_date", ""))

        raw_acc_dt = _clean_str(rec.get("acceptance_datetime_raw")) or _clean_str(
            rec.get("acceptance_datetime_utc")
        )
        acc_dt_utc = normalize_acceptance_datetime(raw_acc_dt, fallback_filing_date=f_date)

        raw_items = rec.get("items") if "items" in rec else rec.get("items_raw", "")
        norm_items = normalize_sec_items(raw_items)
        item_descs = describe_filing_items(form, norm_items)
        p_desc = _clean_str(rec.get("primary_doc_description", ""))
        headline = build_headline_text(
            ticker=ticker,
            form=form,
            items=norm_items,
            item_descriptions=item_descs,
            report_date=r_date,
            primary_doc_description=p_desc,
        )
        rows.append(
            {
                "ticker": ticker,
                "cik": cik_10,
                "accession_number": acc_num,
                "form": form,
                "filing_date": f_date,
                "report_date": r_date,
                "acceptance_datetime_utc": acc_dt_utc,
                "items": norm_items,
                "item_descriptions": item_descs,
                "primary_doc_description": p_desc,
                "headline_text": headline,
                "news_source": "sec_edgar",
            }
        )

    deduped = deduplicate_filings(rows)
    events_df = pd.DataFrame(deduped, columns=list(EVENTS_RAW_COLUMNS))
    events_df = (
        events_df.sort_values(
            ["ticker", "acceptance_datetime_utc", "accession_number"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if out_path is not None:
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        events_df.to_parquet(target, index=False, engine="pyarrow", compression="snappy")

    return events_df
