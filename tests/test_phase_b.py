"""Phase B contracts that do not download Qwen3-Embedding-4B or BGE-M3.

Route A leakage checks use a tiny synthetic graph. The frozen Macao bundle is
used for schema, edge-mask, and POI-text checks. Full ten-seed metrics, when
present, are re-scored from the saved fold table.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from macao_equity.config import DATA_FROZEN_DIR, EV_REGEX_EXT, EV_REGEX_SPEC, OUTPUTS_METRICS_DIR
from macao_equity.ml.baselines import top20_hit_rate
from macao_equity.phase_b.comparison import ROUTE_B_BLOCKER, build_comparison_rows, comparison_frame
from macao_equity.phase_b.data import load_phase_b_bundle
from macao_equity.phase_b.metrics import score_binary
from macao_equity.phase_b.models import build_model, parameter_count
from macao_equity.phase_b.route_b import (
    ensure_route_b_runtime,
    fuse_features,
    last_token_pool,
    mrl_truncate,
    pca_project,
    project_text,
    qwen_prompt,
)
from macao_equity.phase_b.spec import (
    PARAM_BUDGET_MAX,
    PARAM_BUDGET_MIN,
    PHASE_B_SEEDS,
    PRIMARY_METRICS,
    QWEN_LORA_R,
    QWEN_LORA_TARGET_MODULES,
    ROUTE_A_MODELS,
    ROUTE_B_ARMS,
    SAGE_PARAM_MAX,
    SAGE_PARAM_MIN,
    TEXT_DIM,
    TOTAL_PARAM_LIMIT,
    EpochBudget,
)
from macao_equity.phase_b.splits import make_fold_split
from macao_equity.phase_b.train import train_one_fold


def _colab_main():
    path = Path(__file__).resolve().parents[1] / "scripts" / "colab_phase_b.py"
    spec = importlib.util.spec_from_file_location("colab_phase_b_cli", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load colab_phase_b.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main

FROZEN = DATA_FROZEN_DIR
ML_FEATURES = [
    "pop_total",
    "pop_65plus",
    "pop_density",
    "elderly_share",
    "intersection_density",
    "road_density_km_per_km2",
    "res_bldg_share",
    "com_bldg_share",
    "bldg_coverage_ratio",
    "dist_nearest_lrt_m",
    "dist_nearest_bus_m",
]
SHORT_BUDGET = EpochBudget(max_epochs=2, min_epochs=2, patience=10)


@pytest.fixture(scope="module")
def bundle():
    return load_phase_b_bundle(
        FROZEN / "cells.gpkg",
        FROZEN / "graph_edges.csv",
        FROZEN / "spatial_folds.csv",
        FROZEN / "feature_manifest.json",
        FROZEN / "cell_poi_texts.csv",
    )


def _synthetic_graph(seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_per = 6
    n = 5 * n_per
    blocks = np.repeat(np.arange(5), n_per)
    x = rng.normal(size=(n, 11))
    y = np.zeros(n, dtype=np.int64)
    for block in range(5):
        start = block * n_per
        y[start] = 1
        y[start + 1] = 1
    src: list[int] = []
    dst: list[int] = []
    for index in range(n - 1):
        src.extend([index, index + 1])
        dst.extend([index + 1, index])
    # Extra cross-block edge between block 0 and block 1.
    src.extend([0, 6])
    dst.extend([6, 0])
    return {
        "x": x,
        "y": y,
        "blocks": blocks,
        "src": np.asarray(src, dtype=np.int64),
        "dst": np.asarray(dst, dtype=np.int64),
    }


def _fit(graph: dict[str, np.ndarray], **kwargs):
    return train_one_fold(
        kwargs.pop("x", graph["x"]),
        kwargs.pop("y", graph["y"]),
        graph["blocks"],
        graph["src"],
        graph["dst"],
        **kwargs,
    )


def _state_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    if left.keys() != right.keys():
        return False
    return all(torch.equal(left[key], right[key]) for key in left)


def test_seeds_and_primary_metrics_are_pre_specified() -> None:
    assert len(PHASE_B_SEEDS) == 10
    assert PHASE_B_SEEDS[0] == 42
    assert PRIMARY_METRICS == ("roc_auc", "pr_auc", "top20_hit_rate")
    assert "brier" not in PRIMARY_METRICS


def test_parameter_caps() -> None:
    gat = parameter_count(build_model("gatv2", 11))
    transformer = parameter_count(build_model("graph_transformer", 11))
    sage = parameter_count(build_model("graphsage_5k", 11))
    fused = parameter_count(build_model("gatv2", 11 + TEXT_DIM))
    assert PARAM_BUDGET_MIN <= gat <= PARAM_BUDGET_MAX
    assert PARAM_BUDGET_MIN <= transformer <= PARAM_BUDGET_MAX
    assert SAGE_PARAM_MIN <= sage <= SAGE_PARAM_MAX
    assert PARAM_BUDGET_MIN <= fused <= PARAM_BUDGET_MAX
    assert fused < TOTAL_PARAM_LIMIT


def test_frozen_features_are_exactly_ml_features(bundle) -> None:
    assert bundle.feature_names == ML_FEATURES
    assert bundle.x.shape == (bundle.n_nodes, len(ML_FEATURES))
    assert bundle.crs_epsg == 32649
    assert bundle.n_nodes == 287
    manifest = json.loads((FROZEN / "feature_manifest.json").read_text(encoding="utf-8"))
    assert manifest["ml_features"] == ML_FEATURES
    assert manifest["random_seed"] == 42


def test_poi_text_has_no_ev_token(bundle) -> None:
    for text in bundle.poi_text:
        assert EV_REGEX_SPEC.search(text) is None
        assert EV_REGEX_EXT.search(text) is None


def test_inductive_edges_drop_test_nodes_and_keep_other_cross_edges(bundle) -> None:
    split = make_fold_split(bundle.blocks, bundle.src, bundle.dst, fold=0, protocol="inductive")
    assert split.train_src.size > 0
    assert not split.test_mask[split.train_src].any()
    assert not split.test_mask[split.train_dst].any()
    train_blocks = bundle.blocks[~split.test_mask]
    assert train_blocks.min() != train_blocks.max()
    cross = bundle.blocks[split.train_src] != bundle.blocks[split.train_dst]
    assert bool(np.any(cross))
    eval_src_test = split.test_mask[split.eval_src]
    eval_dst_test = split.test_mask[split.eval_dst]
    assert not np.any(eval_src_test & eval_dst_test)
    assert not np.any(eval_src_test)
    assert np.any(eval_dst_test)


def test_transductive_keeps_cross_edges_and_masks_test_labels(bundle) -> None:
    split = make_fold_split(bundle.blocks, bundle.src, bundle.dst, fold=0, protocol="transductive")
    assert split.train_src.size == bundle.src.size
    assert np.any(split.test_mask[split.train_src] | split.test_mask[split.train_dst])
    assert not np.any(split.loss_mask & split.test_mask)
    assert not np.any(split.loss_mask & split.val_mask)
    assert int(split.val_mask.sum()) > 0


def test_top20_definition_matches_phase_a() -> None:
    y = np.asarray([1, 0, 1, 0, 0])
    scores = np.asarray([0.9, 0.1, 0.2, 0.0, 0.05])
    scored = score_binary(y, scores)
    assert scored["top20_hit_rate"] == pytest.approx(top20_hit_rate(y, scores))


def test_scaler_uses_loss_rows_only() -> None:
    graph = _synthetic_graph()
    fit = _fit(
        graph,
        model_name="graphsage_5k",
        protocol="inductive",
        fold=0,
        seed=42,
        budget=SHORT_BUDGET,
    )
    split = make_fold_split(
        graph["blocks"],
        graph["src"],
        graph["dst"],
        fold=0,
        protocol="inductive",
    )
    expected = graph["x"][split.loss_mask].mean(axis=0)
    assert fit.scaler_mean == pytest.approx(expected)
    assert not np.allclose(fit.scaler_mean, graph["x"].mean(axis=0))


def test_test_labels_do_not_change_weights_and_inductive_ignores_test_features() -> None:
    graph = _synthetic_graph()
    kwargs = {
        "model_name": "graphsage_5k",
        "fold": 0,
        "seed": 42,
        "budget": SHORT_BUDGET,
    }
    base = _fit(graph, protocol="inductive", **kwargs)
    y_flip = graph["y"].copy()
    test = graph["blocks"] == 0
    y_flip[test] = 1 - y_flip[test]
    flipped = _fit(graph, y=y_flip, protocol="transductive", **kwargs)
    flipped_labels_inductive = _fit(graph, y=y_flip, protocol="inductive", **kwargs)
    x_corrupt = graph["x"].copy()
    x_corrupt[test] += 50.0
    corrupt_inductive = _fit(graph, x=x_corrupt, protocol="inductive", **kwargs)
    corrupt_trans = _fit(graph, x=x_corrupt, protocol="transductive", **kwargs)
    trans_base = _fit(graph, protocol="transductive", **kwargs)
    assert _state_equal(base.state_dict, flipped_labels_inductive.state_dict)
    assert _state_equal(trans_base.state_dict, flipped.state_dict)
    assert _state_equal(base.state_dict, corrupt_inductive.state_dict)
    assert not _state_equal(trans_base.state_dict, corrupt_trans.state_dict)


def test_training_is_deterministic_for_a_seed() -> None:
    graph = _synthetic_graph()
    kwargs = {
        "model_name": "gatv2",
        "protocol": "transductive",
        "fold": 1,
        "seed": 7,
        "budget": SHORT_BUDGET,
    }
    first = _fit(graph, **kwargs)
    second = _fit(graph, **kwargs)
    assert _state_equal(first.state_dict, second.state_dict)
    transformer = _fit(
        graph,
        model_name="graph_transformer",
        protocol="inductive",
        fold=2,
        seed=42,
        budget=SHORT_BUDGET,
    )
    assert np.isfinite(transformer.probabilities[graph["blocks"] == 2]).all()


def test_mrl_ignores_held_out_rows_and_pca_uses_fit_rows_only() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(40, 16))
    mask = np.zeros(40, dtype=bool)
    mask[:25] = True
    projected = project_text(matrix, method="mrl", fit_mask=mask, dim=4, seed=42)
    tampered = matrix.copy()
    tampered[mask] += 3.0
    assert np.allclose(mrl_truncate(matrix, 4)[~mask], mrl_truncate(tampered, 4)[~mask])
    pca_base = pca_project(matrix, mask, 4, 42)
    held_out = matrix.copy()
    held_out[~mask] += 10.0
    pca_held = pca_project(held_out, mask, 4, 42)
    assert np.allclose(pca_base[mask], pca_held[mask])
    assert projected.shape[1] == 4


def test_fuse_keeps_numeric_block_untouched() -> None:
    numeric = np.arange(12, dtype=float).reshape(3, 4)
    text = np.ones((3, 2))
    fused = fuse_features(numeric, text)
    assert fused.shape == (3, 6)
    assert np.allclose(fused[:, :4], numeric)


def test_last_token_pool_respects_padding() -> None:
    hidden = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[4.0, 0.0], [5.0, 0.0], [6.0, 0.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    pooled = last_token_pool(hidden, mask)
    assert torch.equal(pooled[0], hidden[0, 1])
    assert torch.equal(pooled[1], hidden[1, 2])


def test_qwen_lora_spec_and_download_refusal(tmp_path: Path) -> None:
    assert QWEN_LORA_R == 8
    assert QWEN_LORA_TARGET_MODULES == ("q_proj", "k_proj", "v_proj", "o_proj")
    assert "charger" not in qwen_prompt("park").lower()
    with pytest.raises(RuntimeError, match="Refusing to download"):
        ensure_route_b_runtime(False)
    with pytest.raises(RuntimeError, match="Refusing to download"):
        ensure_route_b_runtime(True)
    code = _colab_main()(
        [
            "--assemble-only",
            "--assembly-out",
            str(tmp_path / "assembly.json"),
        ]
    )
    assert code == 0
    payload = json.loads((tmp_path / "assembly.json").read_text(encoding="utf-8"))
    assert payload["feature_names"] == ML_FEATURES
    assert payload["n_cells"] == 287
    refused = _colab_main()([])
    assert refused == 2


def test_comparison_keeps_route_b_rows_when_unrun() -> None:
    phase_a_path = OUTPUTS_METRICS_DIR / "phase_a_ml_baselines.json"
    phase_a = json.loads(phase_a_path.read_text(encoding="utf-8"))
    route_a_rows = [
        {
            "model": model,
            "protocol": protocol,
            "n_seeds": 10,
            "status": "ran_on_cpu",
            "std_definition": "test",
            "note": "",
            "roc_auc_mean": 0.5,
            "roc_auc_std": 0.0,
            "pr_auc_mean": 0.2,
            "pr_auc_std": 0.0,
            "top20_hit_rate_mean": 0.2,
            "top20_hit_rate_std": 0.0,
        }
        for model in ROUTE_A_MODELS
        for protocol in ("inductive", "transductive")
    ]
    frame = comparison_frame(build_comparison_rows(phase_a, route_a_rows, None))
    for arm in ROUTE_B_ARMS:
        part = frame[frame["model"] == arm]
        assert len(part) == 2
        assert set(part["status"]) == {"not_run"}
        assert ROUTE_B_BLOCKER in set(part["note"])
    assert set(frame["model"]).issuperset({"logistic_regression", "random_forest", "naive_prior"})


def test_committed_route_a_artifacts_match_fold_table() -> None:
    metrics_path = OUTPUTS_METRICS_DIR / "phase_b_route_a.json"
    fold_path = OUTPUTS_METRICS_DIR.parent / "tables" / "phase_b_route_a_folds.csv"
    if not metrics_path.exists() or not fold_path.exists():
        pytest.skip("Route A artifacts are written by scripts/run_phase_b_route_a.py")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    folds = pd.read_csv(fold_path)
    assert payload["feature_columns"] == ML_FEATURES
    assert payload["seeds"] == list(PHASE_B_SEEDS)
    assert set(payload["models"]) == set(ROUTE_A_MODELS)
    assert set(folds["protocol"]) == {"inductive", "transductive"}
    assert folds["seed"].nunique() == 10
    assert len(folds) == 10 * 5 * len(ROUTE_A_MODELS) * 2
    for row in payload["summary_rows"]:
        part = folds[(folds["model"] == row["model"]) & (folds["protocol"] == row["protocol"])]
        per_seed = [float(seed_part["roc_auc"].mean()) for _, seed_part in part.groupby("seed")]
        assert row["roc_auc_mean"] == pytest.approx(float(np.mean(per_seed)))
        assert row["status"] == "ran_on_cpu"
