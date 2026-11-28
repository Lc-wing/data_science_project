"""Writing result tables as CSV and as Markdown."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macao_tourism.config import OUTPUTS_TABLES_DIR


def write_csv(frame: pd.DataFrame, name: str, directory: Path = OUTPUTS_TABLES_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def markdown_table(frame: pd.DataFrame, decimals: int = 2) -> str:
    def cell(value: object) -> str:
        if isinstance(value, float):
            return f"{value:,.{decimals}f}"
        return str(value)

    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.to_numpy()]
    return "\n".join(lines)
