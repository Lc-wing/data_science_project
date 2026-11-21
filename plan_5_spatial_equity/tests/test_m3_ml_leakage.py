"""Milestone M3 leakage and metric tests (A13).

Binary target is ``has_ev_charger`` on H3 r9. Model inputs are exactly
``feature_manifest.json`` ``ml_features``. Scores are out-of-fold under the
frozen 5 spatial blocks. Top-20% hit rate is recall among the top
``ceil(0.20 * n)`` scores (ties broken by row order), matching the Phase A
metric contract. These checks describe predictive association under spatial
block CV; they do not support a causal or siting-policy claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from macao_equity.config import DATA_FROZEN_DIR, RANDOM_SEED
from macao_equity.ml.baselines import (
    MODEL_NAMES,
    assert_leak_free_features,
    load_baseline_inputs,
    run_spatial_block_baselines,
    top20_hit_rate,
    write_baseline_artifacts,
)

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
METRIC_NAMES = ("roc_auc", "pr_auc", "brier", "top20_hit_rate")
CELLS_R9 = DATA_FROZEN_DIR / "cells_base_res9.gpkg"
FOLDS_CSV = DATA_FROZEN_DIR / "spatial_folds.csv"
MANIFEST_JSON = DATA_FROZEN_DIR / "feature_manifest.json"
EXPECTED_LR_C = (0.01, 0.1, 1.0, 10.0)
EXPECTED_RF_DEPTH = (4, None)
EXPECTED_RF_LEAF = (2, 8)
EXPECTED_RF_TREES = 200


def _oracle_top20(y_true: list[int], y_score: list[float], frac: float = 0.20) -> float:
    """Recall at the top ceil(frac * n) scores; ties keep the smaller row index."""
    n = len(y_true)
    k = max(1, int(np.ceil(frac * n)))
    order = sorted(range(n), key=lambda i: (-float(y_score[i]), i))
    hits = sum(int(y_true[i]) for i in order[:k])
    total_pos = sum(int(y) for y in y_true)
    return (hits / total_pos) if total_pos else 0.0


def _make_synthetic(n_per_block: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for block in range(5):
        for i in range(n_per_block):
            feats = rng.normal(loc=block / 10.0, scale=1.0, size=len(ML_FEATURES))
            logit = 0.9 * float(feats[0]) - 0.4 * float(feats[3]) + (0.2 if block < 3 else -0.6)
            y = int(rng.random() < 1.0 / (1.0 + np.exp(-logit)))
            row: dict[str, object] = {name: float(feats[j]) for j, name in enumerate(ML_FEATURES)}
            row["cell_id"] = f"b{block}_{i:02d}"
            row["spatial_block"] = block
            row["has_ev_charger"] = y
            rows.append(row)
    frame = pd.DataFrame(rows)
    for block in range(5):
        mask = frame["spatial_block"] == block
        if frame.loc[mask, "has_ev_charger"].nunique() < 2:
            idx = frame.index[mask][:2]
            frame.loc[idx[0], "has_ev_charger"] = 0
            frame.loc[idx[1], "has_ev_charger"] = 1
    return frame


@pytest.fixture(scope="module")
def manifest() -> dict[str, object]:
    return json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def synthetic_frame() -> pd.DataFrame:
    return _make_synthetic()


@pytest.fixture(scope="module")
def synthetic_report(synthetic_frame: pd.DataFrame):
    return run_spatial_block_baselines(
        synthetic_frame,
        feature_columns=list(ML_FEATURES),
        seed=RANDOM_SEED,
    )


@pytest.fixture(scope="module")
def real_bundle(tmp_path_factory: pytest.TempPathFactory):
    frame, features, manifest_obj = load_baseline_inputs(CELLS_R9, FOLDS_CSV, MANIFEST_JSON)
    report = run_spatial_block_baselines(frame, feature_columns=features, seed=RANDOM_SEED)
    out = tmp_path_factory.mktemp("m3_real")
    paths = write_baseline_artifacts(
        report,
        out / "phase_a_ml_baselines.json",
        out,
    )
    payload = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    cv = pd.read_csv(paths["cv"])
    oof = pd.read_csv(paths["oof"])
    return {
        "frame": frame,
        "features": features,
        "manifest": manifest_obj,
        "report": report,
        "payload": payload,
        "cv": cv,
        "oof": oof,
    }


def test_top20_hit_rate_is_recall_with_index_tie_break() -> None:
    y = [1, 0, 1, 0, 0]
    scores = [0.2, 0.9, 0.8, 0.1, 0.0]
    assert top20_hit_rate(y, scores) == pytest.approx(_oracle_top20(y, scores))
    assert top20_hit_rate(y, scores) == pytest.approx(0.0)

    # n=5 keeps k=1, so one captured positive out of two is recall 0.5.
    scores_pos = [0.9, 0.1, 0.2, 0.0, 0.05]
    assert top20_hit_rate(y, scores_pos) == pytest.approx(0.5)
    assert top20_hit_rate(y, scores_pos) == pytest.approx(_oracle_top20(y, scores_pos))

    tied = [1, 0, 0]
    tied_scores = [0.5, 0.5, 0.5]
    assert _oracle_top20(tied, tied_scores) == pytest.approx(1.0)
    assert top20_hit_rate(tied, tied_scores) == pytest.approx(1.0)


def test_feature_list_must_equal_manifest_and_reject_leak_columns(
    manifest: dict[str, object],
) -> None:
    assert manifest["ml_features"] == ML_FEATURES
    assert_leak_free_features(list(ML_FEATURES), manifest)

    with pytest.raises(ValueError):
        assert_leak_free_features(ML_FEATURES[:-1], manifest)

    for leaked in (
        "centroid_x",
        "centroid_y",
        "poi_text",
        "ev_station_count",
        "ev_charger_capacity",
        "has_ev_charger",
        "acc10_ev_charger",
        "acc15_park",
        "lisa_i_acc15_composite",
    ):
        bad = dict(manifest)
        bad["ml_features"] = [*ML_FEATURES, leaked]
        with pytest.raises(ValueError):
            assert_leak_free_features(bad["ml_features"], bad)


def test_spatial_folds_must_match_cells_one_to_one(tmp_path: Path) -> None:
    folds = pd.read_csv(FOLDS_CSV)
    flipped = folds.copy()
    flipped.loc[0, "spatial_block"] = (int(flipped.loc[0, "spatial_block"]) + 1) % 5
    bad = tmp_path / "spatial_folds.csv"
    flipped.to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_baseline_inputs(CELLS_R9, bad, MANIFEST_JSON)


def test_non_finite_or_single_class_block_raises(synthetic_frame: pd.DataFrame) -> None:
    broken = synthetic_frame.copy()
    broken.loc[0, "pop_density"] = np.inf
    with pytest.raises(ValueError):
        run_spatial_block_baselines(broken, feature_columns=list(ML_FEATURES), seed=RANDOM_SEED)

    one_class = synthetic_frame.copy()
    one_class.loc[one_class["spatial_block"] == 4, "has_ev_charger"] = 0
    with pytest.raises(ValueError):
        run_spatial_block_baselines(one_class, feature_columns=list(ML_FEATURES), seed=RANDOM_SEED)


def test_poison_columns_do_not_change_predictions(
    synthetic_frame: pd.DataFrame, synthetic_report
) -> None:
    poisoned = synthetic_frame.copy()
    poisoned["ev_station_count"] = 10_000
    poisoned["ev_charger_capacity"] = 99_999
    poisoned["centroid_x"] = -1.0e6
    poisoned["centroid_y"] = 1.0e6
    poisoned["acc10_ev_charger"] = 123.0
    poisoned["acc15_park"] = 8.0
    poisoned["lisa_i_acc15_ev_charger"] = 1.0
    poisoned["poi_text"] = "ev charger charging station 充電 充电"
    poisoned["geometry"] = "POINT (0 0)"
    again = run_spatial_block_baselines(
        poisoned,
        feature_columns=list(ML_FEATURES),
        seed=RANDOM_SEED,
    )
    for name in MODEL_NAMES:
        np.testing.assert_allclose(
            again.oof_proba[name],
            synthetic_report.oof_proba[name],
            atol=1e-12,
        )


def test_scaler_and_pipeline_fit_only_on_training_blocks(
    synthetic_frame: pd.DataFrame,
    synthetic_report,
) -> None:
    blocks = synthetic_frame["spatial_block"].to_numpy()
    x_all = synthetic_frame[ML_FEATURES].to_numpy(dtype=float)
    for name in MODEL_NAMES:
        fitted = synthetic_report.fitted_pipelines[name]
        assert set(fitted) == {0, 1, 2, 3, 4}
        for fold, pipe in fitted.items():
            assert isinstance(pipe, Pipeline)
            assert list(pipe.named_steps) == ["scaler", "clf"]
            scaler = pipe.named_steps["scaler"]
            assert isinstance(scaler, StandardScaler)
            train = blocks != fold
            assert int(scaler.n_samples_seen_) == int(train.sum())
            expected = StandardScaler().fit(x_all[train])
            np.testing.assert_allclose(scaler.mean_, expected.mean_)
            np.testing.assert_allclose(scaler.scale_, expected.scale_)
            holdout = blocks == fold
            proba = pipe.predict_proba(x_all[holdout])
            classes = list(pipe.named_steps["clf"].classes_)
            positive = proba[:, classes.index(1)]
            np.testing.assert_allclose(
                positive,
                synthetic_report.oof_proba[name][holdout],
                atol=1e-12,
            )


def test_held_out_block_does_not_change_the_model_that_scores_it(
    synthetic_frame: pd.DataFrame,
    synthetic_report,
) -> None:
    corrupted = synthetic_frame.copy()
    rng = np.random.default_rng(0)
    holdout = corrupted["spatial_block"] == 0
    for col in ML_FEATURES:
        corrupted.loc[holdout, col] = rng.normal(size=int(holdout.sum()))
    corrupted.loc[holdout, "has_ev_charger"] = 1 - corrupted.loc[holdout, "has_ev_charger"].astype(
        int
    )
    # Keep both classes inside the corrupted block so the other folds still train.
    idx = corrupted.index[holdout]
    corrupted.loc[idx[0], "has_ev_charger"] = 0
    corrupted.loc[idx[1], "has_ev_charger"] = 1
    refit = run_spatial_block_baselines(
        corrupted,
        feature_columns=list(ML_FEATURES),
        seed=RANDOM_SEED,
    )
    x_train = synthetic_frame.loc[~holdout, ML_FEATURES].to_numpy(dtype=float)

    lr_a = synthetic_report.fitted_pipelines["logistic_regression"][0].named_steps["clf"]
    lr_b = refit.fitted_pipelines["logistic_regression"][0].named_steps["clf"]
    np.testing.assert_allclose(lr_a.coef_, lr_b.coef_, atol=1e-8)
    np.testing.assert_allclose(lr_a.intercept_, lr_b.intercept_, atol=1e-8)

    rf_a = synthetic_report.fitted_pipelines["random_forest"][0]
    rf_b = refit.fitted_pipelines["random_forest"][0]
    np.testing.assert_allclose(rf_a.predict_proba(x_train), rf_b.predict_proba(x_train), atol=1e-8)

    prior_a = synthetic_report.oof_proba["naive_prior"][holdout.to_numpy()]
    prior_b = refit.oof_proba["naive_prior"][holdout.to_numpy()]
    np.testing.assert_allclose(prior_a, prior_b, atol=1e-12)

    # The same corruption is inside the training set of the model for block 1.
    lr_other_a = synthetic_report.fitted_pipelines["logistic_regression"][1].named_steps["clf"]
    lr_other_b = refit.fitted_pipelines["logistic_regression"][1].named_steps["clf"]
    assert not np.allclose(lr_other_a.coef_, lr_other_b.coef_, atol=1e-6)


def test_dummy_prior_is_training_prevalence_and_fold_auc_is_half(
    synthetic_frame: pd.DataFrame,
    synthetic_report,
) -> None:
    y = synthetic_frame["has_ev_charger"].to_numpy(dtype=int)
    blocks = synthetic_frame["spatial_block"].to_numpy(dtype=int)
    payload = synthetic_report.to_json_dict()
    for fold in range(5):
        train = blocks != fold
        prior = float(y[train].mean())
        scored = blocks == fold
        np.testing.assert_allclose(
            synthetic_report.oof_proba["naive_prior"][scored], prior, atol=1e-12
        )
        fold_row = payload["models"]["naive_prior"]["fold_metrics"][fold]
        assert fold_row["fold"] == fold
        assert fold_row["roc_auc"] == pytest.approx(0.5, abs=1e-8)
        pipe = synthetic_report.fitted_pipelines["naive_prior"][fold]
        clf = pipe.named_steps["clf"]
        assert isinstance(clf, DummyClassifier)
        assert clf.strategy == "prior"
        assert clf.random_state == RANDOM_SEED


def test_logistic_and_forest_hyperparameters_come_from_the_inner_grid(synthetic_report) -> None:
    for fold, pipe in synthetic_report.fitted_pipelines["logistic_regression"].items():
        clf = pipe.named_steps["clf"]
        assert isinstance(clf, LogisticRegression)
        assert clf.random_state == RANDOM_SEED
        assert any(clf.C == pytest.approx(c) for c in EXPECTED_LR_C)
        assert fold in range(5)
    for pipe in synthetic_report.fitted_pipelines["random_forest"].values():
        clf = pipe.named_steps["clf"]
        assert isinstance(clf, RandomForestClassifier)
        assert clf.n_estimators == EXPECTED_RF_TREES
        assert clf.random_state == RANDOM_SEED
        assert clf.max_depth in EXPECTED_RF_DEPTH
        assert clf.min_samples_leaf in EXPECTED_RF_LEAF


def test_tuning_does_not_use_a_random_row_split(
    monkeypatch: pytest.MonkeyPatch, synthetic_frame: pd.DataFrame
) -> None:
    import sklearn.model_selection as model_selection

    def _forbid(self: object, *args: object, **kwargs: object) -> None:
        raise AssertionError(f"random splitter is forbidden: {type(self).__name__}")

    for cls_name in ("KFold", "StratifiedKFold", "ShuffleSplit", "StratifiedShuffleSplit"):
        monkeypatch.setattr(getattr(model_selection, cls_name), "__init__", _forbid)
    monkeypatch.setattr(model_selection, "train_test_split", _forbid)
    run_spatial_block_baselines(
        synthetic_frame,
        feature_columns=list(ML_FEATURES),
        seed=RANDOM_SEED,
    )


def test_oof_metrics_mean_std_and_published_tables(
    synthetic_frame: pd.DataFrame,
    synthetic_report,
    tmp_path: Path,
) -> None:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y = synthetic_frame["has_ev_charger"].to_numpy(dtype=int)
    blocks = synthetic_frame["spatial_block"].to_numpy(dtype=int)
    paths = write_baseline_artifacts(
        synthetic_report,
        tmp_path / "phase_a_ml_baselines.json",
        tmp_path,
    )
    payload = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert payload["seed"] == RANDOM_SEED
    assert payload["n_cells"] == len(synthetic_frame)
    assert payload["n_pos"] == int(y.sum())
    assert payload["prevalence"] == pytest.approx(payload["n_pos"] / payload["n_cells"], abs=1e-12)
    assert payload["feature_columns"] == ML_FEATURES
    assert list(payload["models"]) == list(MODEL_NAMES)

    oof_table = pd.read_csv(paths["oof"])
    assert list(oof_table.columns) == [
        "cell_id",
        "spatial_block",
        "has_ev_charger",
        "prob_naive_prior",
        "prob_logistic_regression",
        "prob_random_forest",
    ]
    assert oof_table["cell_id"].tolist() == synthetic_frame["cell_id"].tolist()
    assert oof_table["has_ev_charger"].tolist() == y.tolist()
    assert set(oof_table["spatial_block"].tolist()) == {0, 1, 2, 3, 4}

    cv_table = pd.read_csv(paths["cv"])
    assert list(cv_table["model"]) == list(MODEL_NAMES)
    assert set(MODEL_NAMES) == set(cv_table["model"])

    for name in MODEL_NAMES:
        block = payload["models"][name]
        folds = block["fold_metrics"]
        assert [row["fold"] for row in folds] == [0, 1, 2, 3, 4]
        probs = oof_table[f"prob_{name}"].to_numpy(dtype=float)
        assert np.isfinite(probs).all()
        assert ((probs >= 0.0) & (probs <= 1.0)).all()
        assert block["oof_roc_auc"] == pytest.approx(float(roc_auc_score(y, probs)), abs=1e-6)
        assert block["oof_pr_auc"] == pytest.approx(
            float(average_precision_score(y, probs)), abs=1e-6
        )
        assert block["oof_brier"] == pytest.approx(float(brier_score_loss(y, probs)), abs=1e-6)
        assert block["oof_top20_hit_rate"] == pytest.approx(
            _oracle_top20(y.tolist(), probs.tolist()),
            abs=1e-6,
        )
        for row in folds:
            mask = blocks == int(row["fold"])
            y_fold = y[mask]
            p_fold = probs[mask]
            assert row["roc_auc"] == pytest.approx(float(roc_auc_score(y_fold, p_fold)), abs=1e-6)
            assert row["pr_auc"] == pytest.approx(
                float(average_precision_score(y_fold, p_fold)), abs=1e-6
            )
            assert row["brier"] == pytest.approx(float(brier_score_loss(y_fold, p_fold)), abs=1e-6)
            assert row["top20_hit_rate"] == pytest.approx(
                _oracle_top20(y_fold.tolist(), p_fold.tolist()),
                abs=1e-6,
            )
            for metric in METRIC_NAMES:
                assert 0.0 <= float(row[metric]) <= 1.0
        for metric in METRIC_NAMES:
            vals = [float(row[metric]) for row in folds]
            assert block[f"mean_{metric}"] == pytest.approx(float(np.mean(vals)), abs=1e-7)
            assert block[f"std_{metric}"] == pytest.approx(float(np.std(vals, ddof=0)), abs=1e-7)
            assert 0.0 <= float(block[f"mean_{metric}"]) <= 1.0
            assert 0.0 <= float(block[f"std_{metric}"]) <= 1.0
            assert 0.0 <= float(block[f"oof_{metric}"]) <= 1.0
        cv_row = cv_table.loc[cv_table["model"] == name].iloc[0]
        for metric in METRIC_NAMES:
            for prefix in ("mean", "std", "oof"):
                key = f"{prefix}_{metric}"
                assert float(cv_row[key]) == pytest.approx(float(block[key]), abs=1e-8)

    assert payload["models"]["naive_prior"]["mean_roc_auc"] == pytest.approx(0.5, abs=1e-8)


def test_real_h3_r9_contract_matches_frozen_inputs(real_bundle: dict[str, object]) -> None:
    import h3

    frame = real_bundle["frame"]
    payload = real_bundle["payload"]
    manifest_obj = real_bundle["manifest"]
    meta = json.loads((DATA_FROZEN_DIR / "split_metadata.json").read_text(encoding="utf-8"))
    assert list(frame["cell_id"].astype(str))
    assert all(h3.get_resolution(str(cid)) == 9 for cid in frame["cell_id"])
    assert int(payload["n_cells"]) == int(meta["n_cells_total"]) == len(frame)
    assert int(payload["n_pos"]) == int(meta["n_pos_ev_total"])
    assert int(payload["n_pos"]) == int(frame["has_ev_charger"].sum())
    assert payload["feature_columns"] == list(manifest_obj["ml_features"]) == ML_FEATURES
    assert payload["seed"] == manifest_obj["random_seed"] == RANDOM_SEED
    forbidden = set(manifest_obj["excluded_target_and_audit"]) | {
        "centroid_x",
        "centroid_y",
        "poi_text",
    }
    assert forbidden.isdisjoint(payload["feature_columns"])
    for col in payload["feature_columns"]:
        assert not str(col).startswith(tuple(manifest_obj["analysis_prefix_excluded"]))

    oof = real_bundle["oof"]
    assert len(oof) == len(frame)
    assert oof["cell_id"].astype(str).tolist() == frame["cell_id"].astype(str).tolist()
    merged = frame[["cell_id", "spatial_block", "has_ev_charger"]].copy()
    merged["cell_id"] = merged["cell_id"].astype(str)
    got = oof[["cell_id", "spatial_block", "has_ev_charger"]].copy()
    got["cell_id"] = got["cell_id"].astype(str)
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True),
        merged.reset_index(drop=True),
        check_dtype=False,
    )
    prev = float(payload["prevalence"])
    naive = payload["models"]["naive_prior"]
    assert abs(float(naive["oof_brier"]) - prev * (1.0 - prev)) < 0.05
    for row in naive["fold_metrics"]:
        assert abs(float(row["roc_auc"]) - 0.5) < 1e-6
    assert abs(float(naive["mean_roc_auc"]) - 0.5) < 1e-6


def test_script_targets_res9_contract_outputs() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "03_run_ml_baselines.py").read_text(
        encoding="utf-8"
    )
    assert "cells_base_res9.gpkg" in source
    assert "cells_base_res8" not in source
    assert "feature_manifest.json" in source
    assert "spatial_folds.csv" in source
    assert "phase_a_ml_baselines.json" in source
    assert "ml_cv_results.csv" in source
    assert "ml_oof_predictions.csv" in source
