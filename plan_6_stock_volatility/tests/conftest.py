"""Pytest configuration for Plan 6 (`stock_volatility`).

Imports `stock_volatility.config` at collection time so `MPLBACKEND=Agg` and
`OMP_NUM_THREADS=1` are set even when running isolated tests without shell exports.
Provides an optional `block_network` fixture for offline unit and `--skip-download`
replay tests.
"""

from __future__ import annotations

import urllib.request
from typing import Any

import pytest
import requests
import yfinance as yf

import stock_volatility.config as _config  # noqa: F401


@pytest.fixture()
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hard-block outbound HTTP and yfinance network calls during offline tests."""

    def _forbidden_request(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"Forbidden network call in offline mode: args={args!r}")

    monkeypatch.setattr(requests.Session, "request", _forbidden_request)
    monkeypatch.setattr(urllib.request, "urlopen", _forbidden_request)
    monkeypatch.setattr(yf, "download", _forbidden_request)
