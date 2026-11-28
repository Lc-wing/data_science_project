"""Section 3.3 Deep Time-Series (PatchTST, Chronos) & Event-Anchored LLMs (FinBERT, QLoRA)."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "chronos",
    "datasets",
    "finbert",
    "fusion",
    "patchtst",
    "qlora_llm",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
