"""Plan 6 Phase B: Deep Time-Series (PatchTST), Chronos, FinBERT & Gemma-4 / Qwen QLoRA."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "data_contract",
    "patchtst",
    "route_b_finbert",
    "route_b_llm",
    "route_b_chronos",
    "fusion_and_eval",
    "report_integration",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

