"""Data ingestion, provenance tracking, and Day-1 availability & coverage audit for Plan 6."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from stock_volatility import config

DEFAULT_HTTP_USER_AGENT: str = config.SEC_USER_AGENT


def build_retry_session(
    user_agent: str | None = None,
    *,
    max_retries: int = 5,
    backoff_factor: float = 0.6,
) -> requests.Session:
    """Build a persistent HTTP/1.1 session with exponential-backoff retries."""
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": user_agent or DEFAULT_HTTP_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        }
    )
    return session


class ProvenanceLog:
    """Cryptographic provenance tracker for all committed `data/raw/` snapshots."""

    def __init__(
        self,
        *,
        project_root: Path | None = None,
        generated_at_utc: str | None = None,
        news_source_decision: dict[str, Any] | None = None,
        ticker_replacements: list[dict[str, Any]] | None = None,
        cik_predecessor_linkages: dict[str, Any] | None = None,
        sec_pagination_summary: dict[str, Any] | None = None,
        anomaly_counts: dict[str, Any] | None = None,
        files: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.project_root: Path = (
            Path(project_root) if project_root is not None else config.PROJECT_ROOT
        )
        self.generated_at_utc: str = generated_at_utc or config.utc_now_iso()
        self.news_source_decision: dict[str, Any] = news_source_decision or {}
        self.ticker_replacements: list[dict[str, Any]] = (
            list(ticker_replacements) if ticker_replacements is not None else []
        )
        self.cik_predecessor_linkages: dict[str, Any] = cik_predecessor_linkages or {}
        self.sec_pagination_summary: dict[str, Any] = sec_pagination_summary or {}
        self.anomaly_counts: dict[str, Any] = anomaly_counts or {}
        self.files: dict[str, dict[str, Any]] = dict(files) if files is not None else {}

    def record_file(
        self,
        path: Path,
        *,
        source: str,
        source_url: str,
        http_status: int,
        fetched_at_utc: str,
        row_count: int,
        description: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a committed file under `data/raw/` with exact byte size and SHA-256 digest."""
        target = Path(path).resolve()
        if not target.is_file():
            raise FileNotFoundError(f"Cannot record missing file in provenance: {target}")
        rel_posix = target.relative_to(self.project_root.resolve()).as_posix()
        entry: dict[str, Any] = {
            "path": rel_posix,
            "source": source,
            "source_url": source_url,
            "url": source_url,
            "fetched_at_utc": fetched_at_utc,
            "http_status": int(http_status),
            "byte_size": int(target.stat().st_size),
            "sha256": config.sha256_file(target),
            "row_count": int(row_count),
            "description": description,
        }
        if extra:
            entry.update(extra)
        self.files[rel_posix] = entry
        return entry

    def verify_files(self) -> None:
        """Verify every recorded file exists on disk with matching byte size and SHA-256 digest."""
        if not self.files:
            raise ValueError("ProvenanceLog contains zero recorded files")
        root = self.project_root.resolve()
        for rel_path, meta in sorted(self.files.items()):
            full_path = root / rel_path
            if not full_path.is_file():
                raise FileNotFoundError(f"Missing raw file recorded in provenance: {full_path}")
            actual_size = full_path.stat().st_size
            expected_size = int(meta["byte_size"])
            actual_sha = config.sha256_file(full_path)
            expected_sha = str(meta["sha256"])
            if actual_sha != expected_sha or actual_size != expected_size:
                raise ValueError(
                    f"SHA-256 or byte_size mismatch for {rel_path}: "
                    f"expected ({expected_sha}, {expected_size} B), "
                    f"got ({actual_sha}, {actual_size} B)"
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize provenance metadata and sorted file manifests to a dictionary."""
        return {
            "schema_version": "1.0",
            "project": "plan_6_stock_volatility",
            "milestone": "M1",
            "generated_at_utc": self.generated_at_utc,
            "random_seed": config.RANDOM_SEED,
            "date_window": {
                "start_date": config.START_DATE,
                "end_date": config.END_DATE,
                "n_trading_days": 1760,
            },
            "news_source_decision": self.news_source_decision,
            "ticker_replacements": self.ticker_replacements,
            "cik_predecessor_linkages": self.cik_predecessor_linkages,
            "sec_pagination_summary": self.sec_pagination_summary,
            "anomaly_counts": self.anomaly_counts,
            "files": dict(sorted(self.files.items())),
        }

    def save(self, path: Path | None = None) -> Path:
        """Write deterministic JSON provenance file to disk."""
        out_path = (
            Path(path)
            if path is not None
            else (self.project_root / "data" / "raw" / "data_provenance.json")
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return out_path

    @classmethod
    def load(
        cls,
        path: Path | None = None,
        *,
        project_root: Path | None = None,
    ) -> ProvenanceLog:
        """Load an existing provenance manifest from JSON."""
        root = Path(project_root) if project_root is not None else config.PROJECT_ROOT
        in_path = (
            Path(path)
            if path is not None
            else (root / "data" / "raw" / "data_provenance.json")
        )
        payload = json.loads(in_path.read_text(encoding="utf-8"))
        return cls(
            project_root=root,
            generated_at_utc=payload.get("generated_at_utc"),
            news_source_decision=payload.get("news_source_decision", {}),
            ticker_replacements=payload.get("ticker_replacements", []),
            cik_predecessor_linkages=payload.get("cik_predecessor_linkages", {}),
            sec_pagination_summary=payload.get("sec_pagination_summary", {}),
            anomaly_counts=payload.get("anomaly_counts", {}),
            files=payload.get("files", {}),
        )


__all__ = [
    "DEFAULT_HTTP_USER_AGENT",
    "ProvenanceLog",
    "build_retry_session",
]
