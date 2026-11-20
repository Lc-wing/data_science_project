"""Gaussian two-step floating catchment area (Dai 2010) on the walk network.

Travel time is network distance plus the straight-line snap from each demand
or supply point to its nearest walk node, at 75 m/min. A supply location whose
Gaussian-weighted demand is zero receives R_j = 0 (no division by zero, and
that capacity is not allocated).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from macao_equity.config import ANALYSIS_CRS, WALK_SPEED_M_PER_MIN, WGS84_CRS, make_transformer

TRADITIONAL_FACILITY_TYPES: tuple[str, ...] = (
    "park",
    "health_centre",
    "library",
    "bus_stop",
)
_GAUSS_FLOOR = float(np.exp(-0.5))
_GAUSS_DENOM = 1.0 - _GAUSS_FLOOR


def gaussian_weight(distance_min: np.ndarray, cutoff_min: float) -> np.ndarray:
    """Dai Gaussian decay: 1 at distance 0 and 0 at and beyond ``cutoff_min``."""
    if cutoff_min <= 0.0:
        raise ValueError("cutoff_min must be positive")
    distance = np.asarray(distance_min, dtype=float)
    weights = np.zeros(distance.shape, dtype=float)
    inside = np.isfinite(distance) & (distance < cutoff_min)
    if np.any(inside):
        ratio = distance[inside] / cutoff_min
        weights[inside] = np.maximum(
            (np.exp(-0.5 * ratio * ratio) - _GAUSS_FLOOR) / _GAUSS_DENOM,
            0.0,
        )
    return weights


def gaussian_2sfca(
    population: np.ndarray,
    capacity: np.ndarray,
    distance_min: np.ndarray,
    cutoff_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(accessibility, supply_to_demand_ratios)``.

    ``distance_min`` has shape ``(n_demand, n_supply)``. Non-finite distances are
    outside every cutoff. When the weighted demand at supply j is 0, R_j is 0.
    For every supply with R_j > 0, population-weighted accessibility sums back to
    that supply's capacity.
    """
    pop = np.asarray(population, dtype=float).ravel()
    cap = np.asarray(capacity, dtype=float).ravel()
    distance = np.asarray(distance_min, dtype=float)
    if distance.ndim != 2 or distance.shape != (pop.size, cap.size):
        raise ValueError("distance_min must have shape (n_demand, n_supply)")
    kernel = gaussian_weight(distance, cutoff_min)
    weighted_demand = pop @ kernel
    ratios = np.zeros(cap.size, dtype=float)
    positive = weighted_demand > 0.0
    ratios[positive] = cap[positive] / weighted_demand[positive]
    accessibility = kernel @ ratios
    return accessibility, ratios


def composite_accessibility(access_by_type: dict[str, np.ndarray]) -> np.ndarray:
    """Min-max average of park, health centre, library, and bus access.

    A facility type with no variation contributes zeros, so the index stays on
    [0, 1] and does not invent spread. EV access is not included.
    """
    scaled: list[np.ndarray] = []
    n_obs: int | None = None
    for facility in TRADITIONAL_FACILITY_TYPES:
        values = np.asarray(access_by_type[facility], dtype=float).ravel()
        if n_obs is None:
            n_obs = values.size
        elif values.size != n_obs:
            raise ValueError("accessibility series differ in length")
        span = float(values.max() - values.min()) if values.size else 0.0
        if span <= 1e-15:
            scaled.append(np.zeros(values.size, dtype=float))
        else:
            scaled.append((values - float(values.min())) / span)
    if not scaled or n_obs is None:
        raise ValueError("composite_accessibility received no facility series")
    return np.mean(np.vstack(scaled), axis=0)


@dataclass(frozen=True)
class WalkNetwork:
    """Symmetrized pedestrian graph in EPSG:32649, keyed by original node order."""

    node_xy: np.ndarray
    graph: csr_matrix
    tree: cKDTree

    def minutes(
        self,
        demand_xy: np.ndarray,
        supply_xy: np.ndarray,
        max_minutes: float = 15.0,
    ) -> np.ndarray:
        """Minutes between demand and supply, including snap access and egress.

        Pairs whose network distance exceeds ``max_minutes`` (before snap) are
        infinite. Snap time can only make a pair slower, so the network limit
        does not drop a pair that would still fall inside the cutoff.
        """
        demand = np.asarray(demand_xy, dtype=float)
        supply = np.asarray(supply_xy, dtype=float)
        if demand.ndim != 2 or supply.ndim != 2:
            raise ValueError("demand_xy and supply_xy must have shape (n, 2)")
        snap_demand, demand_nodes = self.tree.query(demand, k=1)
        snap_supply, supply_nodes = self.tree.query(supply, k=1)
        unique_nodes, inverse = np.unique(demand_nodes, return_inverse=True)
        limit_m = max_minutes * WALK_SPEED_M_PER_MIN
        distances = dijkstra(
            self.graph,
            directed=True,
            indices=unique_nodes,
            limit=limit_m,
        )
        network_m = np.asarray(distances, dtype=float)[inverse][:, supply_nodes]
        minutes = (snap_demand[:, None] + network_m + snap_supply[None, :]) / WALK_SPEED_M_PER_MIN
        minutes[~np.isfinite(minutes)] = np.inf
        return minutes


def prepare_walk_network(graph: object) -> WalkNetwork:
    """Build a bidirectional meter graph from an OSMnx walk MultiDiGraph.

    Pedestrians are not bound by motor one-way tags, so each edge is stored in
    both directions at its OSM length. The shorter parallel edge is kept.
    """
    nodes = list(graph.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    lon = np.array([float(graph.nodes[node]["x"]) for node in nodes], dtype=float)
    lat = np.array([float(graph.nodes[node]["y"]) for node in nodes], dtype=float)
    transformer = make_transformer(WGS84_CRS, ANALYSIS_CRS)
    easting, northing = transformer.transform(lon, lat)
    node_xy = np.column_stack([easting, northing])
    best: dict[tuple[int, int], float] = {}
    for origin, dest, data in graph.edges(data=True):
        length = float(data["length"])
        arc = (index[origin], index[dest])
        reverse = (arc[1], arc[0])
        if length < best.get(arc, math.inf):
            best[arc] = length
        if length < best.get(reverse, math.inf):
            best[reverse] = length
    if not best:
        matrix = csr_matrix((len(nodes), len(nodes)), dtype=float)
    else:
        rows, cols, lengths = zip(
            *((origin, dest, length) for (origin, dest), length in best.items()),
            strict=True,
        )
        matrix = csr_matrix((lengths, (rows, cols)), shape=(len(nodes), len(nodes)))
    return WalkNetwork(node_xy=node_xy, graph=matrix, tree=cKDTree(node_xy))


def travel_minutes_from_graph(
    demand_xy: np.ndarray,
    supply_xy: np.ndarray,
    graph: object,
    max_minutes: float = 15.0,
) -> np.ndarray:
    """Convenience wrapper that prepares ``graph`` and returns travel minutes."""
    return prepare_walk_network(graph).minutes(demand_xy, supply_xy, max_minutes=max_minutes)
