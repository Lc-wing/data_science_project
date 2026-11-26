"""Plan 6: US equity realized volatility and downside tail-risk forecasting (Phase A & Phase B)."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

__version__ = "0.1.0"
__all__ = ["__version__"]

