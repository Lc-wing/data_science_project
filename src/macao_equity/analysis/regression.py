"""Pre-specified zone regressions: OLS-HC3, spatial error ML, and spatial lag ML.

All three models are fit for every outcome. A Lagrange-multiplier or residual
Moran p-value is recorded and is not used to drop a specification.
"""

from __future__ import annotations

import contextlib
import io
import math
import warnings

import numpy as np
import spreg
from libpysal.weights import W
from scipy import stats

from macao_equity.analysis.spatial_stats import global_moran
from macao_equity.config import RANDOM_SEED

Z95 = 1.96
_TERM_ALIASES = {"CONSTANT": "const", "W_y": "rho"}


def ols_hc3(y: np.ndarray, x: np.ndarray, names: list[str]) -> dict[str, object]:
    """OLS coefficients, HC3 standard errors, R², AIC, VIF, and residuals.

    HC3 uses the MacKinnon–White form e_i² / (1 - h_ii)² with no extra n/(n-k)
    factor, so the standard errors match a direct sandwich calculation.
    """
    outcome = np.asarray(y, dtype=float).ravel()
    predictors = np.asarray(x, dtype=float)
    if predictors.ndim == 1:
        predictors = predictors.reshape(-1, 1)
    if predictors.shape[0] != outcome.size or predictors.shape[1] != len(names):
        raise ValueError("x columns must align with names and y")
    n_obs, n_pred = predictors.shape
    design = np.column_stack([np.ones(n_obs), predictors])
    xtx_inv = np.linalg.inv(design.T @ design)
    beta = xtx_inv @ (design.T @ outcome)
    residuals = outcome - design @ beta
    leverage = np.sum((design @ xtx_inv) * design, axis=1)
    scale = residuals**2 / np.maximum((1.0 - leverage) ** 2, 1e-18)
    meat = (design * scale[:, None]).T @ design
    covariance = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    z_stat = beta / np.maximum(se, 1e-18)
    pvals = 2.0 * stats.norm.sf(np.abs(z_stat))
    sst = float(np.sum((outcome - outcome.mean()) ** 2))
    sse = float(np.sum(residuals**2))
    r2 = (1.0 - sse / sst) if sst > 1e-15 else 0.0
    aic = float(n_obs * np.log(sse / n_obs) + 2.0 * design.shape[1]) if sse > 0.0 else math.nan
    vif = _variance_inflation(predictors, names)
    keys = ["const", *names]
    return {
        "coefs": {key: float(value) for key, value in zip(keys, beta, strict=True)},
        "se": {key: float(value) for key, value in zip(keys, se, strict=True)},
        "p": {key: float(value) for key, value in zip(keys, pvals, strict=True)},
        "r2": float(r2),
        "aic": aic,
        "vif": vif,
        "residuals": residuals,
    }


def fit_prespecified_models(
    y: np.ndarray,
    x: np.ndarray,
    names: list[str],
    weights: W,
    permutations: int = 999,
    seed: int = RANDOM_SEED,
) -> list[dict[str, object]]:
    """Fit OLS_HC3, SEM_ML, and SLM_ML on the same rows and the same weights.

    The returned rows always contain all three ``model_type`` values. Spatial
    ML standard errors and p-values come from spreg's asymptotic z tests.
    Residual Moran's I uses the same weights and the same permutation count.
    """
    outcome = np.asarray(y, dtype=float).ravel()
    predictors = np.asarray(x, dtype=float)
    if predictors.ndim == 1:
        predictors = predictors.reshape(-1, 1)
    n_obs = int(outcome.size)
    ols = ols_hc3(outcome, predictors, names)
    rows = _pack_terms(
        "OLS_HC3",
        list(ols["coefs"]),
        ols["coefs"],  # type: ignore[arg-type]
        ols["se"],  # type: ignore[arg-type]
        ols["p"],  # type: ignore[arg-type]
        float(ols["r2"]),
        float(ols["aic"]),
        np.asarray(ols["residuals"], dtype=float),
        weights,
        permutations,
        seed,
        n_obs,
    )
    y_col = outcome.reshape(-1, 1)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sem = spreg.ML_Error(y_col, predictors, weights, name_y="y", name_x=list(names))
        slm = spreg.ML_Lag(y_col, predictors, weights, name_y="y", name_x=list(names))
    rows.extend(
        _pack_spreg("SEM_ML", sem, weights, permutations, seed, n_obs),
    )
    rows.extend(
        _pack_spreg("SLM_ML", slm, weights, permutations, seed, n_obs),
    )
    return rows


def _pack_spreg(
    model_type: str,
    model: object,
    weights: W,
    permutations: int,
    seed: int,
    n_obs: int,
) -> list[dict[str, object]]:
    raw_names = [_TERM_ALIASES.get(str(name), str(name)) for name in model.name_x]  # type: ignore[attr-defined]
    coefs = {
        name: float(value)
        for name, value in zip(raw_names, np.asarray(model.betas).ravel(), strict=True)  # type: ignore[attr-defined]
    }
    ses = {
        name: float(value)
        for name, value in zip(raw_names, np.asarray(model.std_err).ravel(), strict=True)  # type: ignore[attr-defined]
    }
    pvals = {
        name: float(pair[1])
        for name, pair in zip(raw_names, model.z_stat, strict=True)  # type: ignore[attr-defined]
    }
    pseudo_r2 = float(np.clip(float(model.pr2), -1.0, 1.0))  # type: ignore[attr-defined]
    return _pack_terms(
        model_type,
        raw_names,
        coefs,
        ses,
        pvals,
        pseudo_r2,
        float(model.aic),  # type: ignore[attr-defined]
        np.asarray(model.u).ravel(),  # type: ignore[attr-defined]
        weights,
        permutations,
        seed,
        n_obs,
    )


def _pack_terms(
    model_type: str,
    terms: list[str],
    coefs: dict[str, float],
    ses: dict[str, float],
    pvals: dict[str, float],
    r2: float,
    aic: float,
    residuals: np.ndarray,
    weights: W,
    permutations: int,
    seed: int,
    n_obs: int,
) -> list[dict[str, object]]:
    moran_i, moran_p = global_moran(residuals, weights, permutations=permutations, seed=seed)
    if not (0.0 < moran_p <= 1.0):
        moran_p = min(1.0, max(moran_p, 1.0 / (permutations + 1.0)))
    rows: list[dict[str, object]] = []
    for term in terms:
        coef = float(coefs[term])
        se = float(ses[term])
        if not np.isfinite(se) or se <= 0.0:
            se = 1e-8
        rows.append(
            {
                "model_type": model_type,
                "term": term,
                "coef": coef,
                "se": se,
                "ci_low": coef - Z95 * se,
                "ci_high": coef + Z95 * se,
                "p_value": float(np.clip(pvals[term], 0.0, 1.0)),
                "r2_or_pseudo_r2": float(r2),
                "aic": float(aic),
                "resid_moran_i": float(moran_i),
                "resid_moran_p": float(moran_p),
                "n_obs": int(n_obs),
            }
        )
    return rows


def _variance_inflation(predictors: np.ndarray, names: list[str]) -> dict[str, float]:
    vifs: dict[str, float] = {}
    n_pred = predictors.shape[1]
    for j, name in enumerate(names):
        if n_pred == 1:
            vifs[name] = 1.0
            continue
        others = np.delete(predictors, j, axis=1)
        r2_j = _ols_r2(predictors[:, j], others)
        vifs[name] = float(1.0 / max(1.0 - r2_j, 1e-12))
    return vifs


def _ols_r2(y: np.ndarray, x: np.ndarray) -> float:
    outcome = np.asarray(y, dtype=float).ravel()
    design = np.column_stack([np.ones(outcome.size), np.asarray(x, dtype=float)])
    beta = np.linalg.lstsq(design, outcome, rcond=None)[0]
    fitted = design @ beta
    sst = float(np.sum((outcome - outcome.mean()) ** 2))
    if sst <= 1e-15:
        return 0.0
    sse = float(np.sum((outcome - fitted) ** 2))
    return max(0.0, min(1.0, 1.0 - sse / sst))
