"""Canonical paths, pre-registered constants, deterministic universe, and helpers for Plan 6."""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Canonical repository paths (anchored to plan_6_stock_volatility/)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_INTERIM_DIR: Path = DATA_DIR / "interim"
DATA_FROZEN_DIR: Path = DATA_DIR / "frozen"
CACHE_DIR: Path = PROJECT_ROOT / ".cache"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
OUTPUTS_TABLES_DIR: Path = OUTPUTS_DIR / "tables"
OUTPUTS_METRICS_DIR: Path = OUTPUTS_DIR / "metrics"
OUTPUTS_FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
REPORT_DIR: Path = PROJECT_ROOT / "report"

PROVENANCE_PATH: Path = DATA_RAW_DIR / "data_provenance.json"
FREEZE_MANIFEST_PATH: Path = DATA_FROZEN_DIR / "FREEZE_MANIFEST.json"

# Pre-registered modeling, split, and anti-leakage constants
RANDOM_SEED: int = 42

START_DATE: str = "2019-01-02"
END_DATE: str = "2025-12-31"
TRAIN_END: str = "2023-12-20"
VAL_START: str = "2024-01-02"
VAL_END: str = "2024-12-20"
TEST_START: str = "2025-01-02"
TEST_END: str = "2025-12-31"

HORIZON_DAYS: int = 5
RV_EPS: float = 1e-6
TAIL_RETURN_THRESHOLD: float = -0.05
MAX_EVENTS_PER_DAY: int = 5
CUTOFF_TZ: str = "America/New_York"
CUTOFF_HOUR: int = 16
TRADING_DAYS_PER_YEAR: int = 252

# SEC EDGAR ingestion & report artifact configuration
SEC_MAX_REQUESTS_PER_SEC: float = 8.0
SEC_USER_AGENT: str = os.environ.get(
    "SEC_EDGAR_USER_AGENT",
    "CS482-Academic-Research Plan6-StockVolatility (cs482-research@example.edu)",
)
REPORT_PDF_NAME: str = "CS482_GroupX_StockVolatilityLLM_PhaseA_Draft.pdf"

# Deterministic 60-stock cross-sector universe (5 GICS sectors x 12 pre-2019 liquid S&P 500 names)
SECTORS: tuple[str, ...] = (
    "Technology",
    "Financials",
    "Healthcare",
    "Consumer",
    "Energy",
)

UNIVERSE_60: tuple[tuple[str, str, str], ...] = (
    # Technology (12)
    ("AAPL", "Technology", "Technology Hardware"),
    ("MSFT", "Technology", "Software & Services"),
    ("NVDA", "Technology", "Semiconductors"),
    ("AVGO", "Technology", "Semiconductors"),
    ("ORCL", "Technology", "Software & Services"),
    ("CSCO", "Technology", "Technology Hardware"),
    ("ADBE", "Technology", "Software & Services"),
    ("CRM", "Technology", "Software & Services"),
    ("INTC", "Technology", "Semiconductors"),
    ("TXN", "Technology", "Semiconductors"),
    ("QCOM", "Technology", "Semiconductors"),
    ("IBM", "Technology", "Software & Services"),
    # Financials (12)
    ("JPM", "Financials", "Diversified Banks"),
    ("BAC", "Financials", "Diversified Banks"),
    ("WFC", "Financials", "Diversified Banks"),
    ("C", "Financials", "Diversified Banks"),
    ("GS", "Financials", "Capital Markets"),
    ("MS", "Financials", "Capital Markets"),
    ("BLK", "Financials", "Capital Markets"),
    ("SCHW", "Financials", "Capital Markets"),
    ("AXP", "Financials", "Consumer Finance"),
    ("USB", "Financials", "Regional Banks"),
    ("PNC", "Financials", "Regional Banks"),
    ("CB", "Financials", "Insurance"),
    # Healthcare (12)
    ("JNJ", "Healthcare", "Pharmaceuticals"),
    ("UNH", "Healthcare", "Managed Care & Services"),
    ("PFE", "Healthcare", "Pharmaceuticals"),
    ("MRK", "Healthcare", "Pharmaceuticals"),
    ("ABT", "Healthcare", "MedTech & Life Sciences"),
    ("TMO", "Healthcare", "MedTech & Life Sciences"),
    ("LLY", "Healthcare", "Pharmaceuticals"),
    ("AMGN", "Healthcare", "Biotechnology"),
    ("BMY", "Healthcare", "Pharmaceuticals"),
    ("GILD", "Healthcare", "Biotechnology"),
    ("CVS", "Healthcare", "Managed Care & Services"),
    ("CI", "Healthcare", "Managed Care & Services"),
    # Consumer (12: 6 Staples + 6 Discretionary under pre-2019 GICS taxonomy, zero lookahead)
    ("WMT", "Consumer", "Consumer Staples"),
    ("PG", "Consumer", "Consumer Staples"),
    ("KO", "Consumer", "Consumer Staples"),
    ("PEP", "Consumer", "Consumer Staples"),
    ("COST", "Consumer", "Consumer Staples"),
    ("CL", "Consumer", "Consumer Staples"),
    ("AMZN", "Consumer", "Consumer Discretionary"),
    ("HD", "Consumer", "Consumer Discretionary"),
    ("MCD", "Consumer", "Consumer Discretionary"),
    ("NKE", "Consumer", "Consumer Discretionary"),
    ("SBUX", "Consumer", "Consumer Discretionary"),
    # Pre-2019 GICS classification (reclassified to Consumer Staples in March 2023)
    ("TGT", "Consumer", "Consumer Discretionary"),
    # Energy (12)
    ("XOM", "Energy", "Integrated Oil & Gas"),
    ("CVX", "Energy", "Integrated Oil & Gas"),
    ("COP", "Energy", "Exploration & Production"),
    ("EOG", "Energy", "Exploration & Production"),
    ("SLB", "Energy", "Oilfield Services"),
    ("MPC", "Energy", "Refining & Marketing"),
    ("PSX", "Energy", "Refining & Marketing"),
    ("VLO", "Energy", "Refining & Marketing"),
    ("OXY", "Energy", "Exploration & Production"),
    ("HAL", "Energy", "Oilfield Services"),
    ("KMI", "Energy", "Midstream"),
    ("WMB", "Energy", "Midstream"),
)

UNIVERSE_TICKERS: tuple[str, ...] = tuple(ticker for ticker, _, _ in UNIVERSE_60)
SECTOR_MAP: dict[str, str] = {ticker: sector for ticker, sector, _ in UNIVERSE_60}
SUB_GROUP_MAP: dict[str, str] = {ticker: sub_group for ticker, _, sub_group in UNIVERSE_60}


def ensure_project_dirs(root: Path | None = None) -> tuple[Path, ...]:
    """Create canonical data, cache, outputs, and report directories on demand."""
    base = Path(root) if root is not None else PROJECT_ROOT
    dirs = (
        base / "data" / "raw",
        base / "data" / "interim",
        base / "data" / "frozen",
        base / ".cache",
        base / "outputs" / "tables",
        base / "outputs" / "metrics",
        base / "outputs" / "figures",
        base / "report",
    )
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase hexadecimal SHA-256 digest of a byte sequence."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the lowercase hexadecimal SHA-256 digest of a file streamed in 64 KiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    """Return the current UTC timestamp in second-precision ISO-8601 format."""
    return datetime.now(UTC).isoformat(timespec="seconds")
