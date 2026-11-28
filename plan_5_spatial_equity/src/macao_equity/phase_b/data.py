"""Load the frozen Phase A bundle for Phase B.

The feature matrix is exactly ``feature_manifest.json`` ``ml_features``.
Coordinates, EV counts, accessibility, LISA, and POI text stay out of that
matrix. POI text is returned separately for the Colab text arm, after the same
EV-token check used when the file was frozen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from macao_equity.config import EV_REGEX_EXT, EV_REGEX_SPEC
from macao_equity.ml.baselines import assert_leak_free_features
from macao_equity.phase_b.spec import CANDIDATE_ACCESS_COLUMN

BANNED_FEATURE_COLUMNS = frozenset(
    {
        "centroid_x",
        "centroid_y",
        "geometry",
        "poi_text",
        "lat",
        "lon",
        "latitude",
        "longitude",
        "has_ev_charger",
        "ev_station_count",
        "ev_charger_capacity",
        "spatial_block",
    }
)


@dataclass(frozen=True)
class PhaseBBundle:
    """In-memory graph. ``x`` is the only matrix a numeric GNN may read."""

    cell_id: np.ndarray
    x: np.ndarray
    feature_names: list[str]
    y: np.ndarray
    blocks: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    edge_weight: np.ndarray
    poi_text: np.ndarray
    acc15_ev: np.ndarray
    crs_epsg: int

    @property
    def n_nodes(self) -> int:
        return int(self.y.shape[0])

    def assembly_summary(self) -> dict[str, Any]:
        """Counts a reviewer can check without training a model."""
        blocks = self.blocks
        per_block = []
        for fold in range(5):
            mask = blocks == fold
            per_block.append(
                {
                    "block": fold,
                    "n_cells": int(np.sum(mask)),
                    "n_pos": int(np.sum(self.y[mask])),
                    "n_neg": int(np.sum(mask) - np.sum(self.y[mask])),
                }
            )
        return {
            "n_cells": self.n_nodes,
            "n_pos": int(np.sum(self.y)),
            "n_edges": int(self.src.shape[0]),
            "n_features": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "crs_epsg": int(self.crs_epsg),
            "blocks": per_block,
            "poi_text_nonempty": int(np.sum(self.poi_text != "")),
        }


def _crs_epsg(cells: gpd.GeoDataFrame) -> int:
    crs = cells.crs
    if crs is None:
        raise ValueError("cells.gpkg has no CRS")
    epsg = crs.to_epsg()
    if epsg != 32649:
        raise ValueError(f"cells.gpkg CRS must be EPSG:32649, got {epsg}")
    return int(epsg)


def _check_poi_text(texts: pd.Series) -> None:
    for text in texts.astype(str):
        if EV_REGEX_SPEC.search(text) or EV_REGEX_EXT.search(text):
            raise ValueError("EV or charging token in cell_poi_texts.csv")


def load_phase_b_bundle(
    cells_path: Path,
    edges_path: Path,
    folds_path: Path,
    manifest_path: Path,
    poi_path: Path,
) -> PhaseBBundle:
    """Join frozen cells, folds, edges, and EV-stripped POI text on ``cell_id``."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    features = [str(col) for col in manifest["ml_features"]]
    assert_leak_free_features(features, manifest)
    for name in features:
        if name in BANNED_FEATURE_COLUMNS or name.startswith(("acc10_", "acc15_", "lisa_")):
            raise ValueError(f"forbidden model input: {name}")

    cells = gpd.read_file(cells_path)
    epsg = _crs_epsg(cells)
    folds = pd.read_csv(folds_path)
    edges = pd.read_csv(edges_path)
    poi = pd.read_csv(poi_path)
    for frame in (cells, folds, edges, poi):
        if "cell_id" in frame.columns:
            frame["cell_id"] = frame["cell_id"].astype(str)
    edges["src_cell_id"] = edges["src_cell_id"].astype(str)
    edges["dst_cell_id"] = edges["dst_cell_id"].astype(str)

    if cells["cell_id"].duplicated().any() or folds["cell_id"].duplicated().any():
        raise ValueError("duplicate cell_id")
    if set(cells["cell_id"]) != set(folds["cell_id"]):
        raise ValueError("cell_id mismatch between cells and spatial_folds.csv")
    if set(cells["cell_id"]) != set(poi["cell_id"]):
        raise ValueError("cell_id mismatch between cells and cell_poi_texts.csv")
    _check_poi_text(poi["poi_text"].fillna(""))

    cells = cells.sort_values("cell_id").reset_index(drop=True)
    fold_map = dict(zip(folds["cell_id"], folds["spatial_block"].astype(int), strict=True))
    poi_map = dict(zip(poi["cell_id"], poi["poi_text"].fillna("").astype(str), strict=True))
    mapped = cells["cell_id"].map(fold_map).to_numpy(dtype=int)
    if not np.array_equal(mapped, cells["spatial_block"].to_numpy(dtype=int)):
        raise ValueError("spatial_block mismatch between cells and spatial_folds.csv")
    if set(np.unique(mapped).tolist()) != {0, 1, 2, 3, 4}:
        raise ValueError("spatial_block must be the frozen blocks 0..4")

    missing = [col for col in features if col not in cells.columns]
    if missing:
        raise ValueError(f"cells missing ml_features: {missing}")
    if CANDIDATE_ACCESS_COLUMN not in cells.columns:
        raise ValueError(f"cells missing {CANDIDATE_ACCESS_COLUMN}")
    x = cells.loc[:, features].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("ML features contain NaN or Inf")
    y_raw = cells["has_ev_charger"].to_numpy()
    if not np.isin(y_raw, [0, 1]).all():
        raise ValueError("has_ev_charger must be binary {0, 1}")
    y = y_raw.astype(np.int64)
    for fold in range(5):
        y_fold = y[mapped == fold]
        if int(y_fold.min()) == int(y_fold.max()):
            raise ValueError(f"spatial block {fold} must contain both classes")

    index = {cid: i for i, cid in enumerate(cells["cell_id"].tolist())}
    src = edges["src_cell_id"].map(index)
    dst = edges["dst_cell_id"].map(index)
    if src.isna().any() or dst.isna().any():
        raise ValueError("graph_edges.csv contains an unknown cell_id")
    src_arr = src.to_numpy(dtype=np.int64)
    dst_arr = dst.to_numpy(dtype=np.int64)
    if np.any(src_arr == dst_arr):
        raise ValueError("graph_edges.csv contains a self-loop")
    if "weight" not in edges.columns:
        raise ValueError("graph_edges.csv is missing required weight column")
    weight_arr = edges["weight"].to_numpy(dtype=np.float64)
    if not np.isfinite(weight_arr).all() or np.any(weight_arr <= 0.0):
        raise ValueError("graph_edges.csv weight column must be finite and strictly positive")
    acc = cells[CANDIDATE_ACCESS_COLUMN].to_numpy(dtype=np.float64)
    if not np.isfinite(acc).all():
        raise ValueError(f"{CANDIDATE_ACCESS_COLUMN} contains NaN or Inf")

    texts = np.asarray([poi_map[cid] for cid in cells["cell_id"].tolist()], dtype=object)
    return PhaseBBundle(
        cell_id=cells["cell_id"].to_numpy(),
        x=x,
        feature_names=features,
        y=y,
        blocks=mapped,
        src=src_arr,
        dst=dst_arr,
        edge_weight=weight_arr,
        poi_text=texts.astype(str),
        acc15_ev=acc,
        crs_epsg=epsg,
    )
