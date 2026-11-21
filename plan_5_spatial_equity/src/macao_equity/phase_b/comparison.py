"""One comparison table for Phase A baselines and Phase B models.

Phase A numbers are copied from the frozen metric file. They are not refit.
Route B rows stay in the table when that run has not been done, with an
explicit status, so the pre-specified arms are not dropped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from macao_equity.phase_b.spec import (
    PHASE_A_MODELS_IN_TABLE,
    PRIMARY_METRICS,
    PROTOCOLS,
    ROUTE_B_ARMS,
)

ROUTE_B_BLOCKER = (
    "Not run in this environment. Qwen3-Embedding-4B and BGE-M3 are not "
    "downloaded or fine-tuned without a CUDA GPU. Run scripts/colab_phase_b.py "
    "on a Colab L4 or A100."
)


def _blank_metric_row(model: str, protocol: str, status: str, note: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model,
        "protocol": protocol,
        "n_seeds": 0,
        "status": status,
        "std_definition": "",
        "note": note,
    }
    for metric in PRIMARY_METRICS:
        row[f"{metric}_mean"] = np_nan()
        row[f"{metric}_std"] = np_nan()
    return row


def np_nan() -> float:
    return float("nan")


def phase_a_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Copy fold-mean metrics. Std is the Phase A fold standard deviation."""
    rows = []
    for name in PHASE_A_MODELS_IN_TABLE:
        block = payload["models"][name]
        row: dict[str, Any] = {
            "model": name,
            "protocol": "spatial_block_cv",
            "n_seeds": 1,
            "status": "copied_from_phase_a",
            "std_definition": "ddof=0 across the 5 frozen folds; one Phase A seed",
            "note": "Phase A seed 42. Association under the frozen blocks, not a siting effect.",
        }
        for metric in PRIMARY_METRICS:
            row[f"{metric}_mean"] = block[f"mean_{metric}"]
            row[f"{metric}_std"] = block[f"std_{metric}"]
        rows.append(row)
    return rows


def route_b_placeholder_rows() -> list[dict[str, Any]]:
    rows = []
    for name in ROUTE_B_ARMS:
        for protocol in PROTOCOLS:
            rows.append(_blank_metric_row(name, protocol, "not_run", ROUTE_B_BLOCKER))
    return rows


def format_mean_std(mean: float, std: float) -> str:
    if pd.isna(mean) or pd.isna(std):
        return ""
    return f"{mean:.3f} ± {std:.3f}"


def comparison_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "model",
        "protocol",
        "n_seeds",
        "status",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "pr_auc_std",
        "top20_hit_rate_mean",
        "top20_hit_rate_std",
        "std_definition",
        "note",
    ]
    return pd.DataFrame(rows, columns=columns)


def to_markdown(frame: pd.DataFrame) -> str:
    """Plan table: ROC-AUC, PR-AUC, and top-20% hit rate, mean ± std."""
    lines = [
        "# Phase B comparison",
        "",
        "Primary metrics are ROC-AUC, PR-AUC, and the top-20% hit rate.",
        "Empty Route B cells are blocked runs, not omitted models.",
        "Scores are associations under the frozen spatial blocks.",
        "Phase A ± is the standard deviation across the five folds for seed 42.",
        "Route A ± is the standard deviation across the ten seeds of that fold-mean.",
        "Those two dispersions are not interchangeable.",
        "",
        "| 模型 | 协议 | ROC-AUC | PR-AUC | 前 20% 命中率 | 状态 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            "| {model} | {protocol} | {roc} | {pr} | {hit} | {status} |".format(
                model=row.model,
                protocol=row.protocol,
                roc=format_mean_std(row.roc_auc_mean, row.roc_auc_std),
                pr=format_mean_std(row.pr_auc_mean, row.pr_auc_std),
                hit=format_mean_std(row.top20_hit_rate_mean, row.top20_hit_rate_std),
                status=row.status,
            )
        )
    lines.append("")
    return "\n".join(lines)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_comparison_rows(
    phase_a_payload: dict[str, Any],
    route_a_summary_rows: list[dict[str, Any]],
    route_b_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows = phase_a_rows(phase_a_payload)
    rows.extend(route_a_summary_rows)
    if route_b_payload is None:
        rows.extend(route_b_placeholder_rows())
        return rows
    found = {(str(row["model"]), str(row["protocol"])): row for row in route_b_payload["rows"]}
    for name in ROUTE_B_ARMS:
        for protocol in PROTOCOLS:
            key = (name, protocol)
            if key not in found:
                rows.append(_blank_metric_row(name, protocol, "not_run", ROUTE_B_BLOCKER))
            else:
                rows.append(found[key])
    return rows
