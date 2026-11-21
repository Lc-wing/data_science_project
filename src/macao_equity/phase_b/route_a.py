"""Route A study: ten seeds, inductive and transductive, three pre-specified nets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from macao_equity.phase_b.candidates import attention_neighbor_rows, select_candidates
from macao_equity.phase_b.comparison import (
    build_comparison_rows,
    comparison_frame,
    load_json,
    to_markdown,
)
from macao_equity.phase_b.data import PhaseBBundle
from macao_equity.phase_b.metrics import mean_std
from macao_equity.phase_b.models import build_model, parameter_count
from macao_equity.phase_b.spec import (
    ATTENTION_SEED,
    CANDIDATE_MODEL,
    CANDIDATE_PROTOCOL,
    PARAM_BUDGET_MAX,
    PARAM_BUDGET_MIN,
    PHASE_B_SEEDS,
    PRIMARY_METRICS,
    PROTOCOLS,
    ROUTE_A_MODELS,
    SAGE_PARAM_MAX,
    SAGE_PARAM_MIN,
    SPEC_BUDGET,
    EpochBudget,
)
from macao_equity.phase_b.train import train_one_fold


def _parameter_report(in_dim: int) -> dict[str, int]:
    counts = {name: parameter_count(build_model(name, in_dim)) for name in ROUTE_A_MODELS}
    for name in ("gatv2", "graph_transformer"):
        count = counts[name]
        if count < PARAM_BUDGET_MIN or count > PARAM_BUDGET_MAX:
            raise RuntimeError(f"{name} has {count} parameters, outside 0.2M–1.5M")
    sage = counts["graphsage_5k"]
    if sage < SAGE_PARAM_MIN or sage > SAGE_PARAM_MAX:
        raise RuntimeError(f"graphsage_5k has {sage} parameters, outside 4k–6k")
    return counts


def _summary_rows(fold_frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Headline is the mean across seeds of the within-seed fold mean."""
    rows: list[dict[str, Any]] = []
    for model in ROUTE_A_MODELS:
        for protocol in PROTOCOLS:
            part = fold_frame[(fold_frame["model"] == model) & (fold_frame["protocol"] == protocol)]
            n_seeds = int(part["seed"].nunique())
            row: dict[str, Any] = {
                "model": model,
                "protocol": protocol,
                "n_seeds": n_seeds,
                "status": "ran_on_cpu",
                "std_definition": (
                    "mean and ddof=1 std across seeds of the ddof=0 mean across 5 folds"
                ),
                "note": (
                    "From-scratch Route A. Inductive training drops test nodes and "
                    "cross edges. Transductive training keeps those edges and still "
                    "masks test labels."
                ),
            }
            for metric in PRIMARY_METRICS:
                per_seed = []
                for _seed, seed_part in part.groupby("seed"):
                    per_seed.append(float(seed_part[metric].mean()))
                mean, std = mean_std(per_seed, ddof=1)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
            rows.append(row)
    return rows


def run_route_a(
    bundle: PhaseBBundle,
    *,
    seeds: tuple[int, ...] = PHASE_B_SEEDS,
    models: tuple[str, ...] = ROUTE_A_MODELS,
    protocols: tuple[str, ...] = PROTOCOLS,
    budget: EpochBudget = SPEC_BUDGET,
    phase_a_metrics: Path,
    route_b_metrics: Path | None = None,
) -> dict[str, Any]:
    """Train every pre-specified Route A cell and assemble tables."""
    counts = _parameter_report(bundle.x.shape[1])
    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    oof_probs: dict[tuple[str, str, int], np.ndarray] = {}
    n_jobs = len(models) * len(protocols) * len(seeds) * 5
    done = 0
    for model_name in models:
        for protocol in protocols:
            for seed in seeds:
                probs = np.full(bundle.n_nodes, np.nan, dtype=np.float64)
                for fold in range(5):
                    capture = (
                        model_name == "gatv2"
                        and protocol == "inductive"
                        and int(seed) == int(ATTENTION_SEED)
                    )
                    fit = train_one_fold(
                        bundle.x,
                        bundle.y,
                        bundle.blocks,
                        bundle.src,
                        bundle.dst,
                        model_name=model_name,
                        protocol=protocol,
                        fold=fold,
                        seed=int(seed),
                        budget=budget,
                        capture_attention=capture,
                    )
                    done += 1
                    print(
                        f"[{done}/{n_jobs}] {model_name} {protocol} seed={seed} "
                        f"fold={fold} roc={fit.metrics['roc_auc']:.3f} epoch={fit.best_epoch}",
                        flush=True,
                    )
                    test = np.isfinite(fit.probabilities)
                    probs[test] = fit.probabilities[test]
                    fold_rows.append(
                        {
                            "model": model_name,
                            "protocol": protocol,
                            "seed": int(seed),
                            "fold": fold,
                            "roc_auc": fit.metrics["roc_auc"],
                            "pr_auc": fit.metrics["pr_auc"],
                            "top20_hit_rate": fit.metrics["top20_hit_rate"],
                            "best_epoch": fit.best_epoch,
                            "n_train_edges": fit.n_train_edges,
                            "n_eval_edges": fit.n_eval_edges,
                            "n_loss": fit.n_loss,
                            "n_test": fit.n_test,
                        }
                    )
                    if capture:
                        attention_rows.extend(attention_neighbor_rows(bundle.cell_id, fit))
                if not np.isfinite(probs).all():
                    raise RuntimeError("out-of-fold probabilities were not filled")
                oof_probs[(model_name, protocol, int(seed))] = probs
                for index, cell in enumerate(bundle.cell_id.tolist()):
                    oof_rows.append(
                        {
                            "cell_id": cell,
                            "spatial_block": int(bundle.blocks[index]),
                            "has_ev_charger": int(bundle.y[index]),
                            "model": model_name,
                            "protocol": protocol,
                            "seed": int(seed),
                            "probability": float(np.round(probs[index], 8)),
                        }
                    )

    fold_frame = pd.DataFrame(fold_rows)
    summary_rows = _summary_rows(fold_frame)
    candidate_key = [
        (CANDIDATE_MODEL, CANDIDATE_PROTOCOL, int(seed)) for seed in seeds
    ]
    stacked = np.vstack([oof_probs[key] for key in candidate_key])
    mean_prob = stacked.mean(axis=0)
    candidates = select_candidates(bundle.cell_id, bundle.y, bundle.acc15_ev, mean_prob)
    phase_a_payload = load_json(phase_a_metrics)
    route_b_payload = None
    if route_b_metrics is not None and route_b_metrics.exists():
        route_b_payload = load_json(route_b_metrics)
    rows = build_comparison_rows(phase_a_payload, summary_rows, route_b_payload)
    table = comparison_frame(rows)
    payload = {
        "task": "has_ev_charger node classification",
        "feature_columns": list(bundle.feature_names),
        "seeds": [int(seed) for seed in seeds],
        "protocols": list(protocols),
        "models": list(models),
        "parameter_counts": counts,
        "epoch_budget": {
            "max_epochs": budget.max_epochs,
            "min_epochs": budget.min_epochs,
            "patience": budget.patience,
        },
        "aggregation": (
            "For each seed, average the five fold scores with ddof=0. "
            "Across seeds, report the mean and the ddof=1 standard deviation. "
            "The test block is not used to choose the architecture or the epoch."
        ),
        "language": "correlational",
        "summary_rows": summary_rows,
        "n_cells": bundle.n_nodes,
        "n_pos": int(np.sum(bundle.y)),
    }
    return {
        "payload": payload,
        "fold_frame": fold_frame,
        "oof_frame": pd.DataFrame(oof_rows),
        "candidates": candidates,
        "attention": pd.DataFrame(attention_rows),
        "comparison": table,
        "comparison_md": to_markdown(table),
    }


def write_route_a_artifacts(
    result: dict[str, Any],
    metrics_path: Path,
    tables_dir: Path,
) -> dict[str, Path]:
    metrics_path = Path(metrics_path)
    tables_dir = Path(tables_dir)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(result["payload"], indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    fold_path = tables_dir / "phase_b_route_a_folds.csv"
    oof_path = tables_dir / "phase_b_route_a_oof.csv"
    candidate_path = tables_dir / "phase_b_candidates.csv"
    attention_path = tables_dir / "phase_b_attention_neighbors.csv"
    comparison_csv = tables_dir / "phase_b_comparison.csv"
    comparison_md = tables_dir / "phase_b_comparison.md"
    result["fold_frame"].to_csv(fold_path, index=False)
    result["oof_frame"].to_csv(oof_path, index=False)
    result["candidates"].to_csv(candidate_path, index=False)
    result["attention"].to_csv(attention_path, index=False)
    result["comparison"].to_csv(comparison_csv, index=False)
    comparison_md.write_text(result["comparison_md"], encoding="utf-8")
    return {
        "metrics": metrics_path,
        "folds": fold_path,
        "oof": oof_path,
        "candidates": candidate_path,
        "attention": attention_path,
        "comparison_csv": comparison_csv,
        "comparison_md": comparison_md,
    }
