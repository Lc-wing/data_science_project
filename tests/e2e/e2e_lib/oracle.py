"""Independent pure-Python 3.11 math, statistical, ML, and geometry oracles."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Sequence

_CAUSAL_CLAIM_RE = re.compile(
    r"(?i)\b("
    r"causes?|caused|causing|"
    r"proves?|proved|proving|"
    r"leads?\s+directly\s+to|"
    r"directly\s+determines?|"
    r"causal\s+(?:effect|impact)\s+of"
    r")\b"
)
_CAUSAL_HEDGE_RE = re.compile(
    r"(?i)\b("
    r"not|cannot|can't|no|never|neither|"
    r"non-causal|noncausal|correlational|associational|association|associated|"
    r"cross-sectional|observational|"
    r"does\s+not\s+(?:imply|establish|prove|demonstrate|indicate)|"
    r"rather\s+than|instead\s+of|caveat|limitation|avoid\s+causal|"
    r"without\s+claiming|cannot\s+infer|cannot\s+establish|do\s+not\s+imply|"
    r"ecological\s+fallacy"
    r")\b"
)


def weighted_gini(values: Sequence[float], weights: Sequence[float]) -> float:
    """Compute population-weighted Gini coefficient via exact trapezoidal Lorenz area."""
    if len(values) != len(weights) or not values:
        raise ValueError("values and weights must be non-empty sequences of equal length")
    pairs: list[tuple[float, float]] = []
    total_w = 0.0
    total_wy = 0.0
    for val, wgt in zip(values, weights, strict=True):
        v = float(val)
        w = float(wgt)
        if not (math.isfinite(v) and math.isfinite(w)):
            raise ValueError("Non-finite input to weighted_gini")
        if w < 0.0 or v < -1e-12:
            raise ValueError(f"Negative weight/value in weighted_gini: v={v}, w={w}")
        v = max(0.0, v)
        if w == 0.0:
            continue
        pairs.append((v, w))
        total_w += w
        total_wy += w * v
    if total_w <= 0.0:
        raise ValueError("Total weight must be positive")
    if total_wy <= 0.0:
        return 0.0
    pairs.sort(key=lambda item: item[0])
    cum_p = 0.0
    cum_q = 0.0
    area_under_lorenz = 0.0
    for v, w in pairs:
        next_p = cum_p + w / total_w
        next_q = cum_q + (w * v) / total_wy
        area_under_lorenz += 0.5 * (next_p - cum_p) * (cum_q + next_q)
        cum_p = next_p
        cum_q = next_q
    gini = 1.0 - 2.0 * area_under_lorenz
    return max(0.0, min(1.0, gini))


def weighted_theil_t(
    values: Sequence[float],
    weights: Sequence[float],
    groups: Sequence[str],
) -> dict[str, float]:
    """Compute population-weighted Theil's T (GE(1)) with 0*ln(0)=0 parish decomposition."""
    n = len(values)
    if not (n == len(weights) == len(groups)) or n == 0:
        raise ValueError("values, weights, groups must be non-empty of equal length")
    total_w = 0.0
    total_wy = 0.0
    by_group: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for val, wgt, grp in zip(values, weights, groups, strict=True):
        v = float(val)
        w = float(wgt)
        if not (math.isfinite(v) and math.isfinite(w)):
            raise ValueError("Non-finite input to weighted_theil_t")
        if w < 0.0 or v < -1e-12:
            raise ValueError(f"Negative input to weighted_theil_t: v={v}, w={w}")
        v = max(0.0, v)
        if w == 0.0:
            continue
        total_w += w
        total_wy += w * v
        by_group[str(grp)].append((v, w))
    if total_w <= 0.0:
        raise ValueError("Total weight must be positive")
    if total_wy <= 0.0:
        return {"total": 0.0, "within": 0.0, "between": 0.0, "between_share": 0.0}

    mu = total_wy / total_w
    total_t = 0.0
    within_t = 0.0
    between_t = 0.0
    for items in by_group.values():
        gw = sum(w for _, w in items)
        gwy = sum(w * v for v, w in items)
        if gw <= 0.0 or gwy <= 0.0:
            continue
        mu_g = gwy / gw
        share_g = gwy / total_wy
        between_t += share_g * math.log(mu_g / mu)
        t_g = 0.0
        for v, w in items:
            if v > 0.0:
                ratio_all = v / mu
                total_t += ((w * v) / total_wy) * math.log(ratio_all)
                t_g += ((w * v) / gwy) * math.log(v / mu_g)
        within_t += share_g * t_g

    between_share = (between_t / total_t) if total_t > 1e-15 else 0.0
    return {
        "total": total_t,
        "within": within_t,
        "between": between_t,
        "between_share": between_share,
    }


def bh_fdr(pvals: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg step-up adjusted q-values."""
    m = len(pvals)
    if m == 0:
        return []
    indexed = sorted(enumerate(float(p) for p in pvals), key=lambda pair: pair[1])
    q_sorted = [0.0] * m
    running_min = 1.0
    for rank_idx in range(m - 1, -1, -1):
        rank = rank_idx + 1
        raw_q = (m / rank) * indexed[rank_idx][1]
        running_min = min(running_min, raw_q, 1.0)
        q_sorted[rank_idx] = max(0.0, running_min)
    result = [0.0] * m
    for pos, (orig_idx, _) in enumerate(indexed):
        result[orig_idx] = q_sorted[pos]
    return result


def holm_bonferroni(pvals: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    m = len(pvals)
    if m == 0:
        return []
    indexed = sorted(enumerate(float(p) for p in pvals), key=lambda pair: pair[1])
    adj_sorted = [0.0] * m
    running_max = 0.0
    for idx, (_, pval) in enumerate(indexed):
        adj = min(1.0, (m - idx) * pval)
        running_max = max(running_max, adj)
        adj_sorted[idx] = running_max
    result = [0.0] * m
    for pos, (orig_idx, _) in enumerate(indexed):
        result[orig_idx] = adj_sorted[pos]
    return result


def _invert_matrix(mat: list[list[float]]) -> list[list[float]]:
    """Invert a square matrix via Gauss-Jordan elimination with partial pivoting."""
    n = len(mat)
    aug = [
        [float(mat[i][j]) for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)]
        for i in range(n)
    ]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        pivot = aug[pivot_row][col]
        if abs(pivot) < 1e-14:
            raise ValueError("Singular matrix in linear algebra oracle")
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        inv_pivot = 1.0 / aug[col][col]
        aug[col] = [v * inv_pivot for v in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor != 0.0:
                row_col = aug[col]
                aug[r] = [aug[r][c] - factor * row_col[c] for c in range(2 * n)]
    return [row[n:] for row in aug]


def _mat_vec(mat: list[list[float]], vec: list[float]) -> list[float]:
    return [sum(a * b for a, b in zip(row, vec, strict=True)) for row in mat]


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows = len(a)
    inner = len(b)
    cols = len(b[0])
    return [
        [sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)] for i in range(rows)
    ]


def _ols_r2(y: Sequence[float], x_cols: Sequence[Sequence[float]]) -> float:
    n = len(y)
    k = len(x_cols) + 1
    x_rows = [[1.0, *(float(x_cols[j][i]) for j in range(k - 1))] for i in range(n)]
    xtx = [[sum(x_rows[i][a] * x_rows[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    xty = [sum(x_rows[i][a] * float(y[i]) for i in range(n)) for a in range(k)]
    beta = _mat_vec(_invert_matrix(xtx), xty)
    y_mean = sum(float(v) for v in y) / n
    sst = sum((float(v) - y_mean) ** 2 for v in y)
    if sst <= 1e-15:
        return 0.0
    sse = sum((float(y[i]) - sum(x_rows[i][j] * beta[j] for j in range(k))) ** 2 for i in range(n))
    return max(0.0, min(1.0, 1.0 - sse / sst))


def ols_hc3_and_vif(
    y: Sequence[float],
    x_cols: Sequence[Sequence[float]],
    col_names: Sequence[str],
) -> dict[str, object]:
    """Compute exact OLS coefficients, HC3 sandwich SEs, R^2, and predictor VIFs."""
    n = len(y)
    p = len(x_cols)
    if p != len(col_names) or n <= p + 1:
        raise ValueError("Invalid dimensions for ols_hc3_and_vif")
    k = p + 1
    x_rows = [[1.0, *(float(x_cols[j][i]) for j in range(p))] for i in range(n)]
    y_vec = [float(v) for v in y]

    xtx = [[sum(x_rows[i][a] * x_rows[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    xty = [sum(x_rows[i][a] * y_vec[i] for i in range(n)) for a in range(k)]
    xtx_inv = _invert_matrix(xtx)
    beta = _mat_vec(xtx_inv, xty)

    residuals = [y_vec[i] - sum(x_rows[i][j] * beta[j] for j in range(k)) for i in range(n)]
    hc3_weights: list[float] = []
    for i in range(n):
        xi = x_rows[i]
        xtx_inv_xi = _mat_vec(xtx_inv, xi)
        h_ii = sum(xi[j] * xtx_inv_xi[j] for j in range(k))
        denom = max((1.0 - h_ii) ** 2, 1e-18)
        hc3_weights.append((residuals[i] ** 2) / denom)

    meat = [
        [sum(x_rows[i][a] * hc3_weights[i] * x_rows[i][b] for i in range(n)) for b in range(k)]
        for a in range(k)
    ]
    cov_hc3 = _mat_mul(_mat_mul(xtx_inv, meat), xtx_inv)
    se_hc3 = [math.sqrt(max(cov_hc3[j][j], 0.0)) for j in range(k)]

    y_mean = sum(y_vec) / n
    sst = sum((v - y_mean) ** 2 for v in y_vec)
    sse = sum(e * e for e in residuals)
    r2 = (1.0 - sse / sst) if sst > 1e-15 else 0.0

    vifs: dict[str, float] = {}
    for j, name in enumerate(col_names):
        if p == 1:
            vifs[name] = 1.0
        else:
            others = [x_cols[m] for m in range(p) if m != j]
            r2_j = _ols_r2(x_cols[j], others)
            vifs[name] = 1.0 / max(1.0 - r2_j, 1e-12)

    names_full = ["const", *col_names]
    return {
        "coefs": dict(zip(names_full, beta, strict=True)),
        "se_hc3": dict(zip(names_full, se_hc3, strict=True)),
        "r2": r2,
        "vif": vifs,
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    n = len(values)
    order = sorted(range(n), key=lambda idx: float(values[idx]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        vi = float(values[order[i]])
        while j + 1 < n and float(values[order[j + 1]]) == vi:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for pos in range(i, j + 1):
            ranks[order[pos]] = avg_rank
        i = j + 1
    return ranks


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Pearson correlation coefficient (0.0 if either series has zero variance)."""
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0
    mx = sum(float(v) for v in x) / n
    my = sum(float(v) for v in y) / n
    num = sum((float(a) - mx) * (float(b) - my) for a, b in zip(x, y, strict=True))
    dx = math.sqrt(sum((float(a) - mx) ** 2 for a in x))
    dy = math.sqrt(sum((float(b) - my) ** 2 for b in y))
    if dx <= 1e-15 or dy <= 1e-15:
        return 0.0
    return num / (dx * dy)


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute tie-aware Spearman rank correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    return pearson_corr(_average_ranks(x), _average_ranks(y))


def roc_auc_score(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Exact tie-aware Mann-Whitney U ROC-AUC matching scikit-learn."""
    if len(y_true) != len(y_score) or not y_true:
        raise ValueError("y_true and y_score must be non-empty of equal length")
    y_bin = [int(v) for v in y_true]
    n_pos = sum(y_bin)
    n_neg = len(y_bin) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC requires both positive and negative classes")
    ranks = _average_ranks(y_score)
    sum_pos_ranks = sum(r for r, y in zip(ranks, y_bin, strict=True) if y == 1)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision_score(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Exact threshold-grouped step Average Precision (PR-AUC) matching scikit-learn."""
    if len(y_true) != len(y_score) or not y_true:
        raise ValueError("y_true and y_score must be non-empty of equal length")
    pairs = sorted(
        ((float(s), int(y)) for s, y in zip(y_score, y_true, strict=True)),
        key=lambda item: item[0],
        reverse=True,
    )
    total_pos = sum(y for _, y in pairs)
    if total_pos == 0:
        return 0.0
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    n = len(pairs)
    i = 0
    while i < n:
        score = pairs[i][0]
        while i < n and pairs[i][0] == score:
            if pairs[i][1] == 1:
                tp += 1
            else:
                fp += 1
            i += 1
        recall = tp / total_pos
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def brier_score(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Mean squared probability error."""
    if len(y_true) != len(y_score) or not y_true:
        raise ValueError("y_true and y_score must be non-empty of equal length")
    return sum((int(y) - float(p)) ** 2 for y, p in zip(y_true, y_score, strict=True)) / len(y_true)


def top20_hit_rates(
    y_true: Sequence[int],
    y_score: Sequence[float],
    frac: float = 0.20,
) -> tuple[float, float]:
    """Return (recall_at_top_k, precision_at_top_k) for top-frac highest-scored items."""
    n = len(y_true)
    if n != len(y_score) or n == 0:
        raise ValueError("y_true and y_score must be non-empty of equal length")
    k = max(1, int(math.ceil(frac * n)))
    order = sorted(range(n), key=lambda idx: (-float(y_score[idx]), idx))
    hits = sum(int(y_true[idx]) for idx in order[:k])
    total_pos = sum(int(y) for y in y_true)
    recall_k = (hits / total_pos) if total_pos > 0 else 0.0
    precision_k = hits / k
    return recall_k, precision_k


def fit_predict_ridge_prob(
    x_train: Sequence[Sequence[float]],
    y_train: Sequence[int],
    x_test: Sequence[Sequence[float]],
    alpha: float = 1.0,
) -> list[float]:
    """Pure-Python leak-free standardized ridge linear scorer mapped to [0, 1] probabilities."""
    n_tr = len(x_train)
    if n_tr == 0 or not x_test:
        return []
    d = len(x_train[0])
    means = [sum(row[j] for row in x_train) / n_tr for j in range(d)]
    stds = []
    for j in range(d):
        var = sum((row[j] - means[j]) ** 2 for row in x_train) / max(n_tr, 1)
        s = math.sqrt(var)
        stds.append(s if s > 1e-12 else 1.0)

    def _std_row(row: Sequence[float]) -> list[float]:
        return [1.0, *((row[j] - means[j]) / stds[j] for j in range(d))]

    z_tr = [_std_row(r) for r in x_train]
    z_te = [_std_row(r) for r in x_test]
    k = d + 1
    ztz = [
        [
            sum(z_tr[i][a] * z_tr[i][b] for i in range(n_tr)) + (alpha if a == b and a > 0 else 0.0)
            for b in range(k)
        ]
        for a in range(k)
    ]
    zty = [sum(z_tr[i][a] * float(y_train[i]) for i in range(n_tr)) for a in range(k)]
    w = _mat_vec(_invert_matrix(ztz), zty)

    probs: list[float] = []
    for r in z_te:
        raw = sum(r[j] * w[j] for j in range(k))
        probs.append(max(0.001, min(0.999, raw)))
    return probs


def _point_to_segment_dist(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    seg_len2 = dx * dx + dy * dy
    if seg_len2 <= 1e-18:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len2
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _point_in_ring(px: float, py: float, ring: Sequence[tuple[float, float]]) -> bool:
    inside = False
    n = len(ring)
    if n < 3:
        return False
    for i in range(n - 1):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        if _point_to_segment_dist(px, py, x1, y1, x2, y2) <= 1e-9:
            return True
        if (y1 > py) != (y2 > py):
            x_intersect = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < x_intersect:
                inside = not inside
    return inside


def point_distance_to_polygons(
    px: float,
    py: float,
    polygons: Sequence[
        tuple[
            tuple[float, float, float, float],
            Sequence[tuple[float, float]],
            Sequence[Sequence[tuple[float, float]]],
        ]
    ],
) -> float:
    """Exact Euclidean distance from (px, py) to union of polygons (0.0 if inside)."""
    best = float("inf")
    for bbox, outer, holes in polygons:
        minx, miny, maxx, maxy = bbox
        dx_box = max(minx - px, 0.0, px - maxx)
        dy_box = max(miny - py, 0.0, py - maxy)
        box_lb = math.hypot(dx_box, dy_box)
        if box_lb > best:
            continue
        if box_lb == 0.0 and _point_in_ring(px, py, outer):
            if not any(_point_in_ring(px, py, h) for h in holes):
                return 0.0
        for ring in (outer, *holes):
            for i in range(len(ring) - 1):
                x1, y1 = ring[i]
                x2, y2 = ring[i + 1]
                d = _point_to_segment_dist(px, py, x1, y1, x2, y2)
                if d < best:
                    best = d
    return best


def sentence_aware_unhedged_causal_hits(text: str) -> list[str]:
    """Return sentences making unhedged causal claims while ignoring negated/caveated ones."""
    hits: list[str] = []
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    for sent in sentences:
        cleaned = sent.strip()
        if not cleaned:
            continue
        if _CAUSAL_CLAIM_RE.search(cleaned) and not _CAUSAL_HEDGE_RE.search(cleaned):
            hits.append(cleaned)
    return hits
