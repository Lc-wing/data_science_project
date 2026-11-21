"""Island-safe spatial weights, Global Moran's I, and FDR-adjusted LISA.

H1 weights are k-nearest neighbors computed separately inside each landmass, so
a peninsula cell is never joined to Taipa or Coloane across the harbour and no
unit is left without a neighbor when its landmass has at least two cells.
The H3 sensitivity weight is Queen contiguity plus the shortest centroid bridges
needed to connect components.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from esda.moran import Moran, Moran_Local
from libpysal.weights import KNN, Queen, W

from macao_equity.analysis.inequality import bh_fdr
from macao_equity.config import RANDOM_SEED

_QUADRANTS = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}


def knn_weights(frame: object, k: int = 4) -> W:
    """Row-standardized k-nearest neighbors on projected centroids."""
    geodata = _reset(frame)
    if len(geodata) < 2:
        raise ValueError("k-nearest neighbors needs at least two units")
    k_use = min(k, len(geodata) - 1)
    weights = KNN.from_dataframe(geodata, k=k_use)
    weights.transform = "r"
    return weights


def island_safe_knn_weights(frame: object, landmass: np.ndarray, k: int = 4) -> W:
    """k-NN within each landmass, row-standardized, with no cross-harbour edges."""
    geodata = _reset(frame)
    labels = np.asarray(landmass).astype(str).ravel()
    if labels.size != len(geodata):
        raise ValueError("landmass labels must align with rows")
    neighbors: dict[int, list[int]] = {i: [] for i in range(len(geodata))}
    for label in dict.fromkeys(labels.tolist()):
        positions = np.flatnonzero(labels == label)
        if positions.size < 2:
            continue
        subset = geodata.iloc[positions].reset_index(drop=True)
        k_use = min(k, len(subset) - 1)
        local = KNN.from_dataframe(subset, k=k_use)
        for local_i, global_i in enumerate(positions):
            neighbors[int(global_i)] = [int(positions[j]) for j in local.neighbors[local_i]]
    weights = W(neighbors, silence_warnings=True)
    weights.transform = "r"
    return weights


def bridge_queen_weights(frame: object) -> W:
    """Queen contiguity, then shortest centroid links until one component remains.

    Existing Queen neighbors are kept. Bridges are added only between different
    components, which is how the peninsula and the islands become connected for
    the H3 sensitivity specification.
    """
    geodata = _reset(frame)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        queen = Queen.from_dataframe(geodata, use_index=True)
    neighbors: dict[int, list[int]] = {
        i: list(queen.neighbors.get(i, [])) for i in range(len(geodata))
    }
    centroids = np.column_stack([geodata.geometry.centroid.x, geodata.geometry.centroid.y])
    components = _components(neighbors)
    while len(components) > 1:
        membership = np.empty(len(geodata), dtype=int)
        for comp_id, members in enumerate(components):
            membership[members] = comp_id
        best_pair: tuple[int, int] | None = None
        best_distance = math.inf
        for left in range(len(geodata)):
            for right in range(left + 1, len(geodata)):
                if membership[left] == membership[right]:
                    continue
                gap = float(np.hypot(*(centroids[left] - centroids[right])))
                if gap < best_distance:
                    best_distance = gap
                    best_pair = (left, right)
        if best_pair is None:
            break
        origin, dest = best_pair
        neighbors[origin].append(dest)
        neighbors[dest].append(origin)
        components = _components(neighbors)
    cleaned = {i: sorted(set(neigh)) for i, neigh in neighbors.items()}
    weights = W(cleaned, silence_warnings=True)
    weights.transform = "r"
    return weights


def weights_diagnostics(weights: W) -> dict[str, int]:
    """Neighbor counts, isolates, and undirected components of ``weights``."""
    cardinalities = [len(weights.neighbors[i]) for i in weights.neighbors]
    return {
        "n_islands": int(len(weights.islands)),
        "min_neighbors": int(min(cardinalities) if cardinalities else 0),
        "max_neighbors": int(max(cardinalities) if cardinalities else 0),
        "n_components": int(weights.n_components),
    }


def global_moran(
    values: np.ndarray,
    weights: W,
    permutations: int = 999,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """Global Moran's I and a two-sided permutation p-value.

    With 999 permutations the smallest p-value is 0.001. A constant map is
    reported as I = 0 and p = 1 because the statistic is undefined.
    """
    y = np.asarray(values, dtype=float).ravel()
    if y.size < 3 or float(np.std(y)) <= 1e-15:
        return 0.0, 1.0
    np.random.seed(seed)
    stat = Moran(y, weights, transformation="r", permutations=permutations, two_tailed=True)
    p_value = float(stat.p_sim)
    if not np.isfinite(p_value):
        p_value = 1.0
    return float(stat.I), float(min(1.0, max(p_value, 0.0)))


def lisa_fdr(
    values: np.ndarray,
    weights: W,
    permutations: int = 999,
    seed: int = RANDOM_SEED,
    alpha: float = 0.05,
) -> dict[str, np.ndarray]:
    """Local Moran's I, permutation p-values, BH q-values, and cluster labels.

    Labels are HH, LL, LH, HL when the Benjamini–Hochberg q-value is at most
    ``alpha``, and NS otherwise. Quadrants follow esda (1 HH, 2 LH, 3 LL, 4 HL).
    """
    y = np.asarray(values, dtype=float).ravel()
    n = y.size
    if n == 0 or float(np.std(y)) <= 1e-15:
        return {
            "i": np.zeros(n, dtype=float),
            "p": np.ones(n, dtype=float),
            "q_fdr": np.ones(n, dtype=float),
            "cluster": np.array(["NS"] * n, dtype=object),
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        local = Moran_Local(
            y,
            weights,
            transformation="r",
            permutations=permutations,
            seed=seed,
            alternative="two-sided",
        )
    pvals = np.clip(np.asarray(local.p_sim, dtype=float), 0.0, 1.0)
    qvals = np.asarray(bh_fdr(pvals), dtype=float)
    clusters = np.empty(n, dtype=object)
    for i, (q_value, quadrant) in enumerate(zip(qvals, local.q, strict=True)):
        if float(q_value) <= alpha:
            clusters[i] = _QUADRANTS[int(quadrant)]
        else:
            clusters[i] = "NS"
    return {
        "i": np.asarray(local.Is, dtype=float),
        "p": pvals,
        "q_fdr": qvals,
        "cluster": clusters,
    }


def _reset(frame: object) -> object:
    geodata = frame.reset_index(drop=True)  # type: ignore[attr-defined]
    if geodata.crs is None:
        raise ValueError("spatial weights require a projected CRS")
    return geodata


def _components(neighbors: dict[int, list[int]]) -> list[list[int]]:
    seen: set[int] = set()
    components: list[list[int]] = []
    for start in neighbors:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        members: list[int] = []
        while stack:
            node = stack.pop()
            members.append(node)
            for nxt in neighbors[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(members)
    return components
