"""Panel regression for H1 and H2: least squares with Driscoll–Kraay standard errors.

The model, the tests, and the sensitivity runs are those of data/frozen/analysis_spec.json.
Diagnostics are computed after the fit and change nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan

from macao_tourism.config import MAIN_SAMPLE_PERIODS, PANEL_START, TIERS
from macao_tourism.panel import logit_pct

ALPHA = 0.05
FOCAL = "log_visitors_overnight"
SHARE = "same_day_share_pct"
OUTCOMES: tuple[str, ...] = ("occupancy_pct", "logit_occupancy")
PANDEMIC_PERIODS: tuple[str, ...] = ("transition_2020_01", "pandemic")
SENSITIVITY_BANDWIDTHS: tuple[int, ...] = (2, 8)
LJUNG_BOX_LAGS: tuple[int, ...] = (12, 24)


def month_index(periods: pd.Series) -> np.ndarray:
    """Calendar months since the first month of the panel."""
    start_year, start_month = (int(part) for part in PANEL_START.split("-"))
    years = periods.str.slice(0, 4).astype(int)
    months = periods.str.slice(5, 7).astype(int)
    return ((years - start_year) * 12 + (months - start_month)).to_numpy()


def bandwidth_rule(n_months: int) -> int:
    """Highest lag of the Bartlett kernel: floor(4 × (T / 100)^(2/9))."""
    return math.floor(4 * (n_months / 100) ** (2 / 9))


@dataclass(frozen=True)
class Design:
    y: np.ndarray
    x: np.ndarray
    names: tuple[str, ...]
    rows: pd.DataFrame
    outcome: str

    @property
    def time(self) -> np.ndarray:
        return month_index(self.rows["period"])


def build_design(
    panel: pd.DataFrame,
    controls: list[str],
    *,
    outcome: str = "occupancy_pct",
    include_pandemic: bool = False,
    trends: bool = False,
    fully_interacted: bool = False,
) -> Design:
    """Design matrix of the specified model, or of one of its sensitivity variants."""
    if outcome not in OUTCOMES:
        raise ValueError(f"Unknown outcome {outcome!r}")
    keep = list(MAIN_SAMPLE_PERIODS) + (list(PANDEMIC_PERIODS) if include_pandemic else [])
    rows = panel[panel["period_label"].isin(keep)].sort_values(["period", "tier"])
    rows = rows.reset_index(drop=True)
    regimes = {"post": rows["post"].to_numpy(dtype=float)}
    if include_pandemic:
        regimes["pand"] = rows["period_label"].isin(PANDEMIC_PERIODS).to_numpy(dtype=float)

    columns: dict[str, np.ndarray] = {}
    is_tier = {tier: (rows["tier"] == tier).to_numpy(dtype=float) for tier in TIERS}
    focal = rows[FOCAL].to_numpy(dtype=float)
    share = rows[SHARE].to_numpy(dtype=float)
    for tier in TIERS:
        columns[f"a[{tier}]"] = is_tier[tier]
    for regime, flag in regimes.items():
        for tier in TIERS:
            columns[f"a_{regime}[{tier}]"] = is_tier[tier] * flag
    for month in range(2, 13):
        columns[f"month[{month}]"] = (rows["month"] == month).to_numpy(dtype=float)
    for tier in TIERS:
        columns[f"b[{tier}]"] = is_tier[tier] * focal
    for regime, flag in regimes.items():
        for tier in TIERS:
            columns[f"b_{regime}[{tier}]"] = is_tier[tier] * flag * focal
    columns["c"] = share
    for regime, flag in regimes.items():
        columns[f"c_{regime}"] = flag * share
    for control in controls:
        columns[control] = rows[control].to_numpy(dtype=float)
    if trends:
        years = month_index(rows["period"]) / 12
        for tier in TIERS:
            columns[f"trend[{tier}]"] = is_tier[tier] * years
    if fully_interacted:
        post = regimes["post"]
        for month in range(2, 13):
            columns[f"month_post[{month}]"] = columns[f"month[{month}]"] * post
        for control in controls:
            columns[f"{control}_post"] = columns[control] * post

    occupancy = rows["occupancy_pct"]
    y = occupancy if outcome == "occupancy_pct" else logit_pct(occupancy)
    x = np.column_stack(list(columns.values()))
    if np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError("The design matrix does not have full column rank")
    return Design(np.asarray(y, dtype=float), x, tuple(columns), rows, outcome)


def driscoll_kraay_scores(
    x: np.ndarray, resid: np.ndarray, time: np.ndarray, bandwidth: int
) -> np.ndarray:
    """Bartlett-weighted long-run covariance of the monthly sums of x·e.

    Lags are counted in calendar months: a pair of months enters lag j only if the two are
    exactly j months apart and both are in the sample.
    """
    scores = x * resid[:, None]
    months = np.unique(time)
    sums = {int(month): scores[time == month].sum(axis=0) for month in months}
    k = x.shape[1]
    total = np.zeros((k, k))
    for here in sums.values():
        total += np.outer(here, here)
    for lag in range(1, bandwidth + 1):
        weight = 1 - lag / (bandwidth + 1)
        gamma = np.zeros((k, k))
        for month, here in sums.items():
            earlier = sums.get(month - lag)
            if earlier is not None:
                gamma += np.outer(here, earlier)
        total += weight * (gamma + gamma.T)
    return total


@dataclass(frozen=True)
class Fit:
    design: Design
    beta: np.ndarray
    cov: np.ndarray
    resid: np.ndarray
    bandwidth: int

    @property
    def names(self) -> tuple[str, ...]:
        return self.design.names

    @property
    def n(self) -> int:
        return len(self.resid)

    @property
    def k(self) -> int:
        return len(self.beta)

    @property
    def n_months(self) -> int:
        return int(self.design.rows["period"].nunique())

    @property
    def r_squared(self) -> float:
        y = self.design.y
        return float(1 - (self.resid @ self.resid) / ((y - y.mean()) @ (y - y.mean())))

    @property
    def fitted(self) -> np.ndarray:
        return self.design.y - self.resid


def fit(design: Design, bandwidth: int | None = None) -> Fit:
    """Least squares with Driscoll–Kraay covariance, scaled by n / (n − k)."""
    x, y = design.x, design.y
    n, k = x.shape
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    if bandwidth is None:
        bandwidth = bandwidth_rule(int(design.rows["period"].nunique()))
    bread = np.linalg.inv(x.T @ x)
    meat = driscoll_kraay_scores(x, resid, design.time, bandwidth)
    cov = bread @ meat @ bread * (n / (n - k))
    return Fit(design, beta, cov, resid, bandwidth)


def weights(names: tuple[str, ...], terms: dict[str, float]) -> np.ndarray:
    row = np.zeros(len(names))
    for name, weight in terms.items():
        row[names.index(name)] = weight
    return row


def normal_interval(estimate: float, se: float) -> tuple[float, float]:
    half = stats.norm.ppf(1 - ALPHA / 2) * se
    return estimate - half, estimate + half


def contrast(result: Fit, terms: dict[str, float]) -> dict[str, float]:
    """A linear combination of coefficients with its z test against zero and 95% CI."""
    row = weights(result.names, terms)
    estimate = float(row @ result.beta)
    se = float(math.sqrt(row @ result.cov @ row))
    z = estimate / se
    low, high = normal_interval(estimate, se)
    return {
        "estimate": estimate,
        "se": se,
        "z": z,
        "p_value": float(2 * stats.norm.sf(abs(z))),
        "ci_low": low,
        "ci_high": high,
    }


def wald(result: Fit, restrictions: list[dict[str, float]]) -> dict[str, float]:
    """Joint test that every listed linear combination is zero; chi-square reference."""
    r = np.vstack([weights(result.names, terms) for terms in restrictions])
    value = r @ result.beta
    statistic = float(value @ np.linalg.solve(r @ result.cov @ r.T, value))
    df = len(restrictions)
    return {"chi2": statistic, "df": df, "p_value": float(stats.chi2.sf(statistic, df))}


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the order given."""
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def post_slope(tier: str) -> dict[str, float]:
    return {f"b[{tier}]": 1.0, f"b_post[{tier}]": 1.0}


def h1_terms() -> dict[str, float]:
    """(b[5star] + b_post[5star]) − (b[3star] + b_post[3star])."""
    return {**post_slope("5star"), **{name: -1.0 for name in post_slope("3star")}}


def h2_terms() -> dict[str, float]:
    return {"c_post": 1.0}


def confirmatory_tests(result: Fit) -> pd.DataFrame:
    """H1 and H2 with Holm-adjusted p-values over the two."""
    tests = {"H1": h1_terms(), "H2": h2_terms()}
    rows = [{"hypothesis": name, **contrast(result, terms)} for name, terms in tests.items()]
    frame = pd.DataFrame(rows)
    frame["p_holm"] = holm(frame["p_value"].tolist())
    frame["reject_at_0.05"] = frame["p_holm"] < ALPHA
    return frame


def secondary_tests(result: Fit) -> pd.DataFrame:
    equal_post_slopes = [
        {**post_slope("5star"), **{name: -1.0 for name in post_slope(tier)}} for tier in TIERS[1:]
    ]
    chow = [{f"b_post[{tier}]": 1.0} for tier in TIERS] + [{"c_post": 1.0}]
    rows = [
        {"test": "equal_post_pandemic_slopes", **wald(result, equal_post_slopes)},
        {"test": "chow_post_interactions_on_focal_regressors", **wald(result, chow)},
    ]
    return pd.DataFrame(rows)


def coefficient_table(result: Fit) -> pd.DataFrame:
    rows = [{"term": name, **contrast(result, {name: 1.0})} for name in result.names]
    return pd.DataFrame(rows)


def slopes_by_class(result: Fit) -> pd.DataFrame:
    """Slopes on the two focal regressors, before and after the pandemic, and their change."""
    rows = []
    for tier in TIERS:
        parts = {
            "pre_pandemic": {f"b[{tier}]": 1.0},
            "post_pandemic": post_slope(tier),
            "change": {f"b_post[{tier}]": 1.0},
        }
        for label, terms in parts.items():
            rows.append(
                {"regressor": FOCAL, "tier": tier, "quantity": label, **contrast(result, terms)}
            )
    parts = {
        "pre_pandemic": {"c": 1.0},
        "post_pandemic": {"c": 1.0, "c_post": 1.0},
        "change": {"c_post": 1.0},
    }
    for label, terms in parts.items():
        rows.append(
            {"regressor": SHARE, "tier": "all", "quantity": label, **contrast(result, terms)}
        )
    return pd.DataFrame(rows)


def sensitivity_row(name: str, description: str, result: Fit) -> dict[str, Any]:
    tests = confirmatory_tests(result).set_index("hypothesis")
    row: dict[str, Any] = {
        "run": name,
        "description": description,
        "outcome": result.design.outcome,
        "n": result.n,
        "months": result.n_months,
        "parameters": result.k,
        "bandwidth": result.bandwidth,
        "r_squared": result.r_squared,
    }
    for hypothesis in ("H1", "H2"):
        for column in ("estimate", "se", "ci_low", "ci_high", "p_value", "p_holm"):
            row[f"{hypothesis}_{column}"] = float(tests.loc[hypothesis, column])
    return row


def _with_constant(design: Design) -> np.ndarray:
    """The design with its first class indicator replaced by a constant: same column space."""
    if not design.names[0].startswith("a["):
        raise ValueError("The first column is expected to be a star-class indicator")
    return np.column_stack([np.ones(len(design.y)), design.x[:, 1:]])


def diagnostics(result: Fit) -> pd.DataFrame:
    """Ljung–Box per star class and contiguous block, and Breusch–Pagan. Reported only."""
    rows = []
    frame = result.design.rows.assign(resid=result.resid)
    for tier in TIERS:
        for block in MAIN_SAMPLE_PERIODS:
            part = frame[(frame["tier"] == tier) & (frame["period_label"] == block)]
            series = part.sort_values("period")["resid"].to_numpy()
            table = acorr_ljungbox(series, lags=list(LJUNG_BOX_LAGS))
            for lag in LJUNG_BOX_LAGS:
                rows.append(
                    {
                        "test": "ljung_box",
                        "tier": tier,
                        "block": block,
                        "lag": lag,
                        "n": len(series),
                        "statistic": float(table.loc[lag, "lb_stat"]),
                        "p_value": float(table.loc[lag, "lb_pvalue"]),
                    }
                )
    lm, lm_p, _, _ = het_breuschpagan(result.resid, _with_constant(result.design))
    rows.append(
        {
            "test": "breusch_pagan",
            "tier": "all",
            "block": "main_sample",
            "lag": 0,
            "n": result.n,
            "statistic": float(lm),
            "p_value": float(lm_p),
        }
    )
    return pd.DataFrame(rows)


def residual_table(result: Fit) -> pd.DataFrame:
    rows = result.design.rows[["tier", "period", "period_label"]].copy()
    rows["observed"] = result.design.y
    rows["fitted"] = result.fitted
    rows["residual"] = result.resid
    return rows
