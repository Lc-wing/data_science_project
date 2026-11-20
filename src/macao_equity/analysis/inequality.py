"""Population-weighted Gini, Theil T, Lorenz curves, and paired bootstrap.

Theil's T uses the GE(1) convention 0 ln 0 = 0, so cells with zero access do
not break the within-parish + between-parish identity. Bootstrap resamples are
indices of analysis units (H3 cells), shared across facility types so Gini
differences stay paired.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from macao_equity.config import RANDOM_SEED


def weighted_gini(values: np.ndarray, weights: np.ndarray) -> float:
    """Population-weighted Gini from the trapezoidal area under the Lorenz curve."""
    accessibility, population = _clean_pairs(values, weights)
    total_w = float(population.sum())
    total_wy = float(np.sum(population * accessibility))
    if total_w <= 0.0:
        raise ValueError("total population weight must be positive")
    if total_wy <= 0.0:
        return 0.0
    order = np.argsort(accessibility, kind="mergesort")
    accessibility = accessibility[order]
    population = population[order]
    cum_p = 0.0
    cum_q = 0.0
    area = 0.0
    for value, weight in zip(accessibility, population, strict=True):
        next_p = cum_p + weight / total_w
        next_q = cum_q + (weight * value) / total_wy
        area += 0.5 * (next_p - cum_p) * (cum_q + next_q)
        cum_p = next_p
        cum_q = next_q
    return float(min(1.0, max(0.0, 1.0 - 2.0 * area)))


def weighted_theil_t(
    values: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    """Population-weighted Theil T with an exact within/between parish split.

    A parish whose residents all have zero access is omitted from both parts.
    Those residents also contribute nothing to the total (0 ln 0 = 0), so
    ``total == within + between`` still holds.
    """
    accessibility, population, labels = _clean_grouped(values, weights, groups)
    total_w = float(population.sum())
    total_wy = float(np.sum(population * accessibility))
    if total_w <= 0.0:
        raise ValueError("total population weight must be positive")
    if total_wy <= 0.0:
        return {"total": 0.0, "within": 0.0, "between": 0.0, "between_share": 0.0}

    mu = total_wy / total_w
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for value, weight, label in zip(accessibility, population, labels, strict=True):
        grouped[label].append((float(value), float(weight)))

    total_t = 0.0
    within_t = 0.0
    between_t = 0.0
    for items in grouped.values():
        group_w = sum(weight for _, weight in items)
        group_wy = sum(weight * value for value, weight in items)
        if group_w <= 0.0 or group_wy <= 0.0:
            continue
        mu_g = group_wy / group_w
        share_g = group_wy / total_wy
        between_t += share_g * math.log(mu_g / mu)
        within_group = 0.0
        for value, weight in items:
            if value > 0.0:
                total_t += ((weight * value) / total_wy) * math.log(value / mu)
                within_group += ((weight * value) / group_wy) * math.log(value / mu_g)
        within_t += share_g * within_group
    between_share = (between_t / total_t) if total_t > 1e-15 else 0.0
    return {
        "total": float(total_t),
        "within": float(within_t),
        "between": float(between_t),
        "between_share": float(between_share),
    }


def lorenz_points(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative population share and access share, starting at (0, 0)."""
    accessibility, population = _clean_pairs(values, weights)
    total_w = float(population.sum())
    total_wy = float(np.sum(population * accessibility))
    order = np.argsort(accessibility, kind="mergesort")
    accessibility = accessibility[order]
    population = population[order]
    pop_share = [0.0]
    acc_share = [0.0]
    cum_p = 0.0
    cum_q = 0.0
    for value, weight in zip(accessibility, population, strict=True):
        cum_p += weight / total_w
        if total_wy > 0.0:
            cum_q += (weight * value) / total_wy
        pop_share.append(min(1.0, cum_p))
        acc_share.append(min(1.0, cum_q))
    return np.asarray(pop_share, dtype=float), np.asarray(acc_share, dtype=float)


def bootstrap_indices(n: int, n_boot: int, seed: int = RANDOM_SEED) -> np.ndarray:
    """Draw ``n_boot`` resamples of ``n`` unit indices with replacement."""
    if n <= 0 or n_boot <= 0:
        raise ValueError("n and n_boot must be positive")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def bootstrap_gini_theil(
    values: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    """Point estimates plus percentile 95% intervals on a shared index matrix."""
    accessibility = np.asarray(values, dtype=float).ravel()
    population = np.asarray(weights, dtype=float).ravel()
    labels = np.asarray(groups).astype(str).ravel()
    point_gini = weighted_gini(accessibility, population)
    point = weighted_theil_t(accessibility, population, labels)
    gini_samples = np.empty(len(indices), dtype=float)
    theil_samples = np.empty(len(indices), dtype=float)
    for replicate, drawn in enumerate(indices):
        gini_samples[replicate] = weighted_gini(accessibility[drawn], population[drawn])
        theil_samples[replicate] = weighted_theil_t(
            accessibility[drawn], population[drawn], labels[drawn]
        )["total"]
    gini_low, gini_high = _percentile_interval(gini_samples, 0.0, 1.0)
    theil_low, theil_high = _percentile_interval(theil_samples, 0.0, None)
    return {
        "gini": point_gini,
        "gini_ci_low": gini_low,
        "gini_ci_high": gini_high,
        "theil_total": point["total"],
        "theil_within": point["within"],
        "theil_between": point["between"],
        "theil_between_share": point["between_share"],
        "theil_total_ci_low": theil_low,
        "theil_total_ci_high": theil_high,
    }


def paired_gini_diff_bootstrap(
    values_by_facility: dict[str, np.ndarray],
    weights: np.ndarray,
    comparisons: tuple[str, ...] | list[str],
    indices: np.ndarray,
    seed: int = RANDOM_SEED,
) -> list[dict[str, float | int | str]]:
    """Paired ``Gini_EV - Gini_other`` using one shared set of resampled cells.

    Holm adjustment is applied inside this comparison family, in ``comparisons``
    order. Callers that stack several families should recompute Holm on the
    combined raw p-values.
    """
    population = np.asarray(weights, dtype=float).ravel()
    ev = np.asarray(values_by_facility["ev_charger"], dtype=float).ravel()
    gini_ev = weighted_gini(ev, population)
    boot_ev = _bootstrap_gini(ev, population, indices)
    rows: list[dict[str, float | int | str]] = []
    for facility in comparisons:
        other = np.asarray(values_by_facility[facility], dtype=float).ravel()
        gini_other = weighted_gini(other, population)
        diffs = boot_ev - _bootstrap_gini(other, population, indices)
        ci_low, ci_high = _percentile_interval(diffs, -1.0, 1.0)
        rows.append(
            {
                "comparison": f"ev_charger_vs_{facility}",
                "gini_ev": gini_ev,
                "gini_other": gini_other,
                "gini_diff": gini_ev - gini_other,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_boot_raw": _two_sided_bootstrap_pvalue(diffs),
                "n_boot": int(len(indices)),
                "seed": int(seed),
            }
        )
    adjusted = holm_bonferroni([float(row["p_boot_raw"]) for row in rows])
    for row, p_holm in zip(rows, adjusted, strict=True):
        row["p_holm"] = p_holm
    return rows


def bh_fdr(pvals: list[float] | np.ndarray) -> list[float]:
    """Benjamini–Hochberg step-up q-values. Each q-value is at least its raw p."""
    raw = [float(p) for p in pvals]
    m = len(raw)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: raw[i])
    q_sorted = [0.0] * m
    running_min = 1.0
    for rank_idx in range(m - 1, -1, -1):
        rank = rank_idx + 1
        raw_q = (m / rank) * raw[order[rank_idx]]
        running_min = min(running_min, raw_q, 1.0)
        q_sorted[rank_idx] = max(0.0, running_min)
    adjusted = [0.0] * m
    for pos, original in enumerate(order):
        adjusted[original] = q_sorted[pos]
    return adjusted


def holm_bonferroni(pvals: list[float] | np.ndarray) -> list[float]:
    """Holm step-down adjusted p-values, monotone in the sorted raw p-values."""
    raw = [float(p) for p in pvals]
    m = len(raw)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: raw[i])
    adjusted_sorted = [0.0] * m
    running_max = 0.0
    for idx, original in enumerate(order):
        adjusted = min(1.0, (m - idx) * raw[original])
        running_max = max(running_max, adjusted)
        adjusted_sorted[idx] = running_max
    result = [0.0] * m
    for pos, original in enumerate(order):
        result[original] = adjusted_sorted[pos]
    return result


def _clean_pairs(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    accessibility = np.asarray(values, dtype=float).ravel()
    population = np.asarray(weights, dtype=float).ravel()
    if accessibility.size == 0 or accessibility.size != population.size:
        raise ValueError("values and weights must be non-empty and the same length")
    if not np.isfinite(accessibility).all() or not np.isfinite(population).all():
        raise ValueError("values and weights must be finite")
    if np.any(population < 0.0) or np.any(accessibility < -1e-12):
        raise ValueError("negative weight or accessibility")
    accessibility = np.maximum(accessibility, 0.0)
    keep = population > 0.0
    return accessibility[keep], population[keep]


def _clean_grouped(
    values: np.ndarray, weights: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(groups).astype(str).ravel()
    accessibility = np.asarray(values, dtype=float).ravel()
    population = np.asarray(weights, dtype=float).ravel()
    if not (accessibility.size == population.size == labels.size) or accessibility.size == 0:
        raise ValueError("values, weights, and groups must be non-empty and aligned")
    if not np.isfinite(accessibility).all() or not np.isfinite(population).all():
        raise ValueError("values and weights must be finite")
    if np.any(population < 0.0) or np.any(accessibility < -1e-12):
        raise ValueError("negative weight or accessibility")
    accessibility = np.maximum(accessibility, 0.0)
    keep = population > 0.0
    return accessibility[keep], population[keep], labels[keep]


def _bootstrap_gini(values: np.ndarray, weights: np.ndarray, indices: np.ndarray) -> np.ndarray:
    samples = np.empty(len(indices), dtype=float)
    for replicate, drawn in enumerate(indices):
        samples[replicate] = weighted_gini(values[drawn], weights[drawn])
    return samples


def _two_sided_bootstrap_pvalue(samples: np.ndarray) -> float:
    """Two-sided p-value for H0: difference = 0, with the +1 permutation correction."""
    draws = np.asarray(samples, dtype=float)
    n_boot = draws.size
    n_le = int(np.sum(draws <= 0.0))
    n_ge = int(np.sum(draws >= 0.0))
    p_value = 2.0 * min((n_le + 1) / (n_boot + 1), (n_ge + 1) / (n_boot + 1))
    return float(min(1.0, p_value))


def _percentile_interval(
    samples: np.ndarray,
    lower_bound: float,
    upper_bound: float | None,
) -> tuple[float, float]:
    """Percentile interval. A zero-width draw is opened by 1e-6 so lo < hi."""
    low, high = np.quantile(np.asarray(samples, dtype=float), [0.025, 0.975])
    low_f = float(low)
    high_f = float(high)
    if upper_bound is None:
        low_f = max(lower_bound, low_f)
        if high_f < low_f:
            high_f = low_f
        if low_f < high_f:
            return low_f, high_f
        return low_f, low_f + 1e-6
    low_f = float(np.clip(low_f, lower_bound, upper_bound))
    high_f = float(np.clip(high_f, lower_bound, upper_bound))
    if low_f < high_f:
        return low_f, high_f
    mid = float(np.clip(np.median(samples), lower_bound, upper_bound))
    opened_low = max(lower_bound, mid - 1e-6)
    opened_high = min(upper_bound, mid + 1e-6)
    if opened_low < opened_high:
        return opened_low, opened_high
    return lower_bound, upper_bound
