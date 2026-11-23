"""Pytest configuration for Plan 6 (`stock_volatility`).

Imports `stock_volatility.config` at collection time so `MPLBACKEND=Agg` and
`OMP_NUM_THREADS=1` are set even when running isolated tests without shell exports.
Provides an optional `block_network` fixture for offline unit and `--skip-download`
replay tests.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import requests
import yfinance as yf

import stock_volatility.config as _config


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register optional `--project-root` CLI flag for opaque-box E2E and Tier 5 tests."""
    parser.addoption(
        "--project-root",
        action="store",
        default=None,
        help="Path to the Plan 6 project root under test (defaults to plan_6_stock_volatility/).",
    )


@pytest.fixture(scope="session")
def project_root(request: pytest.FixtureRequest) -> Path:
    """Return the resolved Plan 6 project root for E2E and adversarial fault-injection runs."""
    cli_opt = request.config.getoption("--project-root")
    if cli_opt:
        return Path(cli_opt).resolve()
    env_opt = os.environ.get("STOCK_VOL_PROJECT_ROOT")
    if env_opt:
        return Path(env_opt).resolve()
    return _config.PROJECT_ROOT.resolve()


@pytest.fixture()
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hard-block outbound HTTP and yfinance network calls during offline tests."""

    def _forbidden_request(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"Forbidden network call in offline mode: args={args!r}")

    monkeypatch.setattr(requests.Session, "request", _forbidden_request)
    monkeypatch.setattr(urllib.request, "urlopen", _forbidden_request)
    monkeypatch.setattr(yf, "download", _forbidden_request)
