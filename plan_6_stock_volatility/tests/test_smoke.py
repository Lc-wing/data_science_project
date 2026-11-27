"""M0 smoke, contract, and toolchain verification tests for Plan 6."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

EXPECTED_SECTORS = ("Technology", "Financials", "Healthcare", "Consumer", "Energy")

EXPECTED_UNIVERSE_BY_SECTOR: dict[str, tuple[str, ...]] = {
    "Technology": (
        "AAPL",
        "MSFT",
        "NVDA",
        "AVGO",
        "ORCL",
        "CSCO",
        "ADBE",
        "CRM",
        "INTC",
        "TXN",
        "QCOM",
        "IBM",
    ),
    "Financials": (
        "JPM",
        "BAC",
        "WFC",
        "C",
        "GS",
        "MS",
        "BLK",
        "SCHW",
        "AXP",
        "USB",
        "PNC",
        "CB",
    ),
    "Healthcare": (
        "JNJ",
        "UNH",
        "PFE",
        "MRK",
        "ABT",
        "TMO",
        "LLY",
        "AMGN",
        "BMY",
        "GILD",
        "CVS",
        "CI",
    ),
    "Consumer": (
        "WMT",
        "PG",
        "KO",
        "PEP",
        "COST",
        "CL",
        "AMZN",
        "HD",
        "MCD",
        "NKE",
        "SBUX",
        "TGT",
    ),
    "Energy": (
        "XOM",
        "CVX",
        "COP",
        "EOG",
        "SLB",
        "MPC",
        "PSX",
        "VLO",
        "OXY",
        "HAL",
        "KMI",
        "WMB",
    ),
}

EXPECTED_CONSUMER_STAPLES = {"WMT", "PG", "KO", "PEP", "COST", "CL"}
EXPECTED_CONSUMER_DISCRETIONARY = {"AMZN", "HD", "MCD", "NKE", "SBUX", "TGT"}

CORE_IMPORT_MODULES = [
    "pandas",
    "numpy",
    "scipy",
    "statsmodels",
    "sklearn",
    "lightgbm",
    "arch",
    "shap",
    "yfinance",
    "pyarrow",
    "requests",
    "matplotlib",
    "seaborn",
    "pypdf",
]


def _count_nyse_weekdays_between(start_exclusive: dt.date, end_exclusive: dt.date) -> int:
    """Count weekdays strictly between two dates excluding fixed Dec 25 and Jan 1 NYSE holidays."""
    day = start_exclusive + dt.timedelta(days=1)
    trading_days = 0
    while day < end_exclusive:
        if day.weekday() < 5 and (day.month, day.day) not in {(12, 25), (1, 1)}:
            trading_days += 1
        day += dt.timedelta(days=1)
    return trading_days


def test_package_import_and_version() -> None:
    import stock_volatility

    assert stock_volatility.__version__ == "0.1.0"
    assert "__version__" in stock_volatility.__all__


def test_project_root_and_directory_layout(tmp_path: Path) -> None:
    from stock_volatility import config

    assert config.PROJECT_ROOT.is_dir()
    assert config.PROJECT_ROOT.name == "plan_6_stock_volatility"

    for required_file in ("pyproject.toml", "README.md", ".gitignore", "uv.lock"):
        candidate = config.PROJECT_ROOT / required_file
        assert candidate.is_file(), f"Missing required M0 file: {candidate}"
    assert (config.PROJECT_ROOT / "uv.lock").stat().st_size > 100

    assert config.DATA_DIR == config.PROJECT_ROOT / "data"
    assert config.DATA_RAW_DIR == config.PROJECT_ROOT / "data" / "raw"
    assert config.DATA_INTERIM_DIR == config.PROJECT_ROOT / "data" / "interim"
    assert config.DATA_FROZEN_DIR == config.PROJECT_ROOT / "data" / "frozen"
    assert config.CACHE_DIR == config.PROJECT_ROOT / ".cache"
    assert config.OUTPUTS_DIR == config.PROJECT_ROOT / "outputs"
    assert config.OUTPUTS_TABLES_DIR == config.PROJECT_ROOT / "outputs" / "tables"
    assert config.OUTPUTS_METRICS_DIR == config.PROJECT_ROOT / "outputs" / "metrics"
    assert config.OUTPUTS_FIGURES_DIR == config.PROJECT_ROOT / "outputs" / "figures"
    assert config.OUTPUTS_PREDICTIONS_DIR == config.PROJECT_ROOT / "outputs" / "predictions"
    assert config.REPORT_DIR == config.PROJECT_ROOT / "report"
    assert config.PROVENANCE_PATH == config.DATA_RAW_DIR / "data_provenance.json"
    assert config.FREEZE_MANIFEST_PATH == config.DATA_FROZEN_DIR / "FREEZE_MANIFEST.json"

    created = config.ensure_project_dirs(root=tmp_path)
    expected_dirs = (
        tmp_path / "data" / "raw",
        tmp_path / "data" / "interim",
        tmp_path / "data" / "frozen",
        tmp_path / ".cache",
        tmp_path / "outputs" / "tables",
        tmp_path / "outputs" / "metrics",
        tmp_path / "outputs" / "figures",
        tmp_path / "outputs" / "predictions",
        tmp_path / "report",
    )
    assert created == expected_dirs
    for directory in expected_dirs:
        assert directory.is_dir()
    # Idempotency check
    assert config.ensure_project_dirs(root=tmp_path) == expected_dirs


def test_gitignore_and_readme_contract() -> None:
    from stock_volatility import config

    gitignore_text = (config.PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in (
        ".venv/",
        "__pycache__/",
        ".pytest_cache/",
        ".env",
        ".agents/",
        ".cache/",
        "*.safetensors",
        "data/raw/*.zip",
        "*.aux",
    ):
        assert pattern in gitignore_text, f"Missing pattern {pattern!r} in Plan 6 .gitignore"

    readme_text = (config.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "uv sync --all-groups" in readme_text
    assert "pdflatex" in readme_text
    assert "CS482" in readme_text
    assert "不构成任何投资或交易建议" in readme_text
    assert "investment" in readme_text.lower()


def test_universe_60_invariants() -> None:
    from stock_volatility import config

    assert isinstance(config.UNIVERSE_60, tuple)
    assert len(config.UNIVERSE_60) == 60
    assert config.SECTORS == EXPECTED_SECTORS

    tickers: list[str] = []
    sector_counts: Counter[str] = Counter()
    observed_by_sector: dict[str, list[str]] = {s: [] for s in EXPECTED_SECTORS}
    consumer_staples: set[str] = set()
    consumer_discretionary: set[str] = set()

    for entry in config.UNIVERSE_60:
        assert isinstance(entry, tuple) and len(entry) == 3
        ticker, sector, sub_group = entry
        assert all(isinstance(x, str) and x == x.strip() and x != "" for x in entry)
        assert re.fullmatch(r"[A-Z]{1,5}", ticker), f"Invalid ticker format: {ticker!r}"
        assert sector in EXPECTED_SECTORS
        tickers.append(ticker)
        sector_counts[sector] += 1
        observed_by_sector[sector].append(ticker)
        if sector == "Consumer":
            if sub_group == "Consumer Staples":
                consumer_staples.add(ticker)
            elif sub_group == "Consumer Discretionary":
                consumer_discretionary.add(ticker)
            else:
                pytest.fail(f"Unexpected Consumer sub_group for {ticker}: {sub_group!r}")

    assert len(set(tickers)) == 60
    assert dict(sector_counts) == {sector: 12 for sector in EXPECTED_SECTORS}
    for sector, expected_tickers in EXPECTED_UNIVERSE_BY_SECTOR.items():
        assert tuple(observed_by_sector[sector]) == expected_tickers

    assert consumer_staples == EXPECTED_CONSUMER_STAPLES
    assert consumer_discretionary == EXPECTED_CONSUMER_DISCRETIONARY

    assert config.UNIVERSE_TICKERS == tuple(tickers)
    assert config.SECTOR_MAP == {t: s for t, s, _ in config.UNIVERSE_60}
    assert config.SUB_GROUP_MAP == {t: g for t, _, g in config.UNIVERSE_60}


def test_split_dates_chronology_embargo_and_timezone() -> None:
    from stock_volatility import config

    start = dt.date.fromisoformat(config.START_DATE)
    train_end = dt.date.fromisoformat(config.TRAIN_END)
    val_start = dt.date.fromisoformat(config.VAL_START)
    val_end = dt.date.fromisoformat(config.VAL_END)
    test_start = dt.date.fromisoformat(config.TEST_START)
    test_end = dt.date.fromisoformat(config.TEST_END)
    end = dt.date.fromisoformat(config.END_DATE)

    assert start < train_end < val_start < val_end < test_start < test_end
    assert end == test_end

    for label, boundary in (
        ("START_DATE", start),
        ("TRAIN_END", train_end),
        ("VAL_START", val_start),
        ("VAL_END", val_end),
        ("TEST_START", test_start),
        ("TEST_END", test_end),
    ):
        assert boundary.weekday() < 5, f"{label}={boundary} must be a weekday"

    embargo_1_days = _count_nyse_weekdays_between(train_end, val_start)
    embargo_2_days = _count_nyse_weekdays_between(val_end, test_start)
    assert embargo_1_days == 6
    assert embargo_2_days == 6
    assert embargo_1_days >= config.HORIZON_DAYS
    assert embargo_2_days >= config.HORIZON_DAYS

    tz = ZoneInfo(config.CUTOFF_TZ)
    winter_cutoff = dt.datetime(2024, 1, 2, config.CUTOFF_HOUR, 0, tzinfo=tz)
    summer_cutoff = dt.datetime(2024, 7, 2, config.CUTOFF_HOUR, 0, tzinfo=tz)
    assert winter_cutoff.utcoffset() == dt.timedelta(hours=-5)
    assert summer_cutoff.utcoffset() == dt.timedelta(hours=-4)


def test_numeric_constants_and_hash_helpers(tmp_path: Path) -> None:
    from stock_volatility import config

    assert config.RANDOM_SEED == 42
    assert config.HORIZON_DAYS == 5
    assert config.RV_EPS == 1e-6
    assert config.RV_EPS > 0
    assert config.TAIL_RETURN_THRESHOLD == -0.05
    assert -1.0 < config.TAIL_RETURN_THRESHOLD < 0.0
    assert config.MAX_EVENTS_PER_DAY == 5
    assert config.CUTOFF_TZ == "America/New_York"
    assert config.CUTOFF_HOUR == 16
    assert config.TRADING_DAYS_PER_YEAR == 252
    assert config.SEC_MAX_REQUESTS_PER_SEC == 8.0
    assert isinstance(config.SEC_USER_AGENT, str) and len(config.SEC_USER_AGENT.strip()) > 0
    assert config.REPORT_PDF_NAME == "CS482_GroupX_StockVolatilityLLM.pdf"

    payload = b"plan6-stock-volatility-m0" * 5000  # > 64 KiB to exercise chunked file hashing
    expected_digest = hashlib.sha256(payload).hexdigest()
    assert config.sha256_bytes(payload) == expected_digest
    assert len(expected_digest) == 64

    sample_file = tmp_path / "payload.bin"
    sample_file.write_bytes(payload)
    assert config.sha256_file(sample_file) == expected_digest

    ts = dt.datetime.fromisoformat(config.utc_now_iso())
    assert ts.tzinfo is not None
    assert ts.utcoffset() == dt.timedelta(0)


@pytest.mark.parametrize("module_name", CORE_IMPORT_MODULES)
def test_core_ds_ml_library_imports(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_runtime_ml_econometrics_sanity() -> None:
    import arch
    import lightgbm as lgb
    import matplotlib
    import numpy as np
    import pandas as pd
    import shap

    assert pd.__version__.startswith("2.")
    assert int(np.__version__.split(".")[0]) >= 2
    assert matplotlib.get_backend().lower() == "agg"

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(80, 3)), columns=["f1", "f2", "f3"])
    y = 0.7 * X["f1"].to_numpy() - 0.4 * X["f2"].to_numpy() + rng.normal(scale=0.1, size=80)

    model = lgb.LGBMRegressor(
        n_estimators=12,
        num_leaves=4,
        min_child_samples=5,
        n_jobs=1,
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)
    preds = model.predict(X)

    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(X), dtype=float)
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    max_additivity_err = float(np.max(np.abs(base_value + shap_values.sum(axis=1) - preds)))
    assert max_additivity_err < 1e-6, f"SHAP additivity error too high: {max_additivity_err}"

    # Tiny GJR-GARCH(1,1) fit on synthetic 100*r percentage returns
    returns_pct = rng.normal(loc=0.02, scale=1.5, size=260)
    garch_model = arch.arch_model(
        returns_pct,
        vol="Garch",
        p=1,
        o=1,
        q=1,
        mean="Constant",
        dist="Normal",
    )
    res = garch_model.fit(disp="off")
    var_fcst = res.forecast(horizon=5).variance.iloc[-1].to_numpy(dtype=float)
    assert var_fcst.shape == (5,)
    assert np.all(np.isfinite(var_fcst))
    assert np.all(var_fcst > 0.0)


@pytest.mark.parametrize(
    ("cmd", "flag"),
    [
        ("pdflatex", "--version"),
        ("bibtex", "--version"),
        ("pdfinfo", "-v"),
        ("pdftotext", "-v"),
    ],
)
def test_latex_and_poppler_cli_available(cmd: str, flag: str) -> None:
    assert shutil.which(cmd) is not None, f"Required CLI not found on PATH: {cmd}"
    proc = subprocess.run(
        [cmd, flag],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert proc.returncode == 0, f"{cmd} {flag} exited {proc.returncode}: {proc.stderr}"
    assert (proc.stdout + proc.stderr).strip() != ""


def test_pdflatex_and_poppler_a4_compile_smoke(tmp_path: Path) -> None:
    tex_source = r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[a4paper,margin=2.15cm]{geometry}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{setspace}
\usepackage[numbers]{natbib}
\usepackage{hyperref}
\onehalfspacing
\begin{document}
Plan 6 M0 A4 LaTeX smoke test: $\sigma^2_{t+1}$.
\begin{table}[h]
\centering
\begin{tabular}{lr}
\toprule
Metric & Value \\
\midrule
Horizon & 5 \\
\bottomrule
\end{tabular}
\end{table}
\end{document}
"""
    tex_path = tmp_path / "m0_smoke.tex"
    tex_path.write_text(tex_source, encoding="utf-8")

    compile_proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert compile_proc.returncode == 0, (
        f"pdflatex failed:\nSTDOUT:\n{compile_proc.stdout}\nSTDERR:\n{compile_proc.stderr}"
    )

    pdf_path = tmp_path / "m0_smoke.pdf"
    assert pdf_path.is_file() and pdf_path.stat().st_size > 0

    info_proc = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    assert "A4" in info_proc.stdout
    assert re.search(r"Pages:\s+1\b", info_proc.stdout)

    text_proc = subprocess.run(
        ["pdftotext", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    assert "Plan 6 M0 A4 LaTeX smoke test:" in text_proc.stdout
    assert "Horizon" in text_proc.stdout
