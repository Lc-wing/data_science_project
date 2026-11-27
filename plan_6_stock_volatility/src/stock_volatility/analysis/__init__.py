"""Section 3 Data Analysis: EDA, Panel HAR-RV, Baselines, LightGBM, TreeSHAP & Deep Learning."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "baselines",
    "dl",
    "eda",
    "lgbm",
    "panel_har_rv",
    "treeshap",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
