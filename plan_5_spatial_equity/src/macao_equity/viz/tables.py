"""Table 1 built only by copying cells from the committed M2 and M3 outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

H1_FACILITIES: tuple[str, ...] = (
    "park",
    "health_centre",
    "library",
    "bus_stop",
    "ev_charger",
    "composite",
)
H3_OUTCOMES: tuple[str, ...] = ("acc15_park", "acc15_composite")
H3_MODELS: tuple[str, ...] = ("OLS_HC3", "SEM_ML", "SLM_ML")
H3_TERMS: tuple[str, ...] = ("elderly_share", "log_pop_density")
ML_MODELS: tuple[str, ...] = ("naive_prior", "logistic_regression", "random_forest")
ML_METRICS: tuple[str, ...] = (
    "oof_roc_auc",
    "oof_pr_auc",
    "oof_brier",
    "oof_top20_hit_rate",
)
TABLE_COLUMNS: tuple[str, ...] = (
    "panel",
    "item",
    "metric",
    "estimate",
    "ci_low",
    "ci_high",
    "aux_metric",
    "aux_value",
)


@dataclass(frozen=True)
class Claim:
    """One reported number and the source cell it was copied from."""

    claim_id: str
    reported: str
    source_file: str
    locator: str
    source_value: float
    tolerance: float


def format_number(value: float, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def parse_reported(text: str) -> float:
    return float(text.replace(",", "").replace("%", "").strip())


def make_claim(
    claim_id: str,
    raw: object,
    *,
    source_file: str,
    locator: str,
    digits: int,
    thousands: bool = False,
) -> Claim:
    value = float(raw)
    if thousands:
        reported = f"{int(round(value)):,}"
        tolerance = 0.51
    elif digits <= 0:
        reported = str(int(round(value)))
        tolerance = 0.51
    else:
        reported = format_number(value, digits)
        tolerance = 0.5 * (10 ** (-digits)) + 1e-9
    return Claim(
        claim_id=claim_id,
        reported=reported,
        source_file=source_file,
        locator=locator,
        source_value=value,
        tolerance=tolerance,
    )


def _h1_row(frame: pd.DataFrame, resolution: str, threshold: int, facility: str) -> pd.Series:
    mask = (
        (frame["resolution"].astype(str) == resolution)
        & (frame["threshold_min"].astype(int) == threshold)
        & (frame["facility_type"].astype(str) == facility)
    )
    hits = frame.loc[mask]
    if len(hits) != 1:
        raise ValueError(f"Expected one H1 row for {resolution} {threshold} {facility}")
    return hits.iloc[0]


def _h3_row(
    frame: pd.DataFrame,
    *,
    outcome: str,
    model: str,
    term: str,
    weights: str = "knn4",
    threshold: int = 15,
) -> pd.Series:
    mask = (
        (frame["threshold_min"].astype(int) == threshold)
        & (frame["outcome"].astype(str) == outcome)
        & (frame["weights_spec"].astype(str) == weights)
        & (frame["model_type"].astype(str) == model)
        & (frame["term"].astype(str) == term)
    )
    hits = frame.loc[mask]
    if len(hits) != 1:
        raise ValueError(f"Expected one H3 row for {outcome} {weights} {model} {term}")
    return hits.iloc[0]


def _first_h3_elderly_ols(frame: pd.DataFrame) -> pd.Series:
    """First 15-minute OLS-HC3 elderly_share row, in committed file order."""
    mask = (
        (frame["threshold_min"].astype(int) == 15)
        & (frame["model_type"].astype(str) == "OLS_HC3")
        & (frame["term"].astype(str).str.lower() == "elderly_share")
    )
    hits = frame.loc[mask]
    if hits.empty:
        raise ValueError("H3 table has no 15-minute OLS-HC3 elderly_share row")
    return hits.iloc[0]


def build_table1_rows(root: Path) -> tuple[list[dict[str, str]], list[Claim]]:
    """Long Table 1. Every estimate string is a rounded copy of a source cell."""
    tables = root / "outputs" / "tables"
    h1 = pd.read_csv(tables / "h1_inequality_lisa.csv")
    h3 = pd.read_csv(tables / "h3_spatial_regression.csv")
    ml = json.loads(
        (root / "outputs" / "metrics" / "phase_a_ml_baselines.json").read_text(encoding="utf-8")
    )
    rows: list[dict[str, str]] = []
    claims: list[Claim] = []

    for facility in H1_FACILITIES:
        src = _h1_row(h1, "res9", 15, facility)
        gini = make_claim(
            f"h1_15_{facility}_gini",
            src["gini"],
            source_file="outputs/tables/h1_inequality_lisa.csv",
            locator=f"res9,15,{facility},gini",
            digits=4,
        )
        glo = make_claim(
            f"h1_15_{facility}_gini_lo",
            src["gini_ci_low"],
            source_file="outputs/tables/h1_inequality_lisa.csv",
            locator=f"res9,15,{facility},gini_ci_low",
            digits=4,
        )
        ghi = make_claim(
            f"h1_15_{facility}_gini_hi",
            src["gini_ci_high"],
            source_file="outputs/tables/h1_inequality_lisa.csv",
            locator=f"res9,15,{facility},gini_ci_high",
            digits=4,
        )
        theil = make_claim(
            f"h1_15_{facility}_theil",
            src["theil_total"],
            source_file="outputs/tables/h1_inequality_lisa.csv",
            locator=f"res9,15,{facility},theil_total",
            digits=4,
        )
        tlo = make_claim(
            f"h1_15_{facility}_theil_lo",
            src["theil_total_ci_low"],
            source_file="outputs/tables/h1_inequality_lisa.csv",
            locator=f"res9,15,{facility},theil_total_ci_low",
            digits=4,
        )
        thi = make_claim(
            f"h1_15_{facility}_theil_hi",
            src["theil_total_ci_high"],
            source_file="outputs/tables/h1_inequality_lisa.csv",
            locator=f"res9,15,{facility},theil_total_ci_high",
            digits=4,
        )
        for claim in (gini, glo, ghi, theil, tlo, thi):
            claims.append(claim)
        rows.append(
            {
                "panel": "H1_Gini_Theil_15min_res9",
                "item": facility,
                "metric": "gini",
                "estimate": gini.reported,
                "ci_low": glo.reported,
                "ci_high": ghi.reported,
                "aux_metric": "theil_total",
                "aux_value": theil.reported,
            }
        )
        rows.append(
            {
                "panel": "H1_Gini_Theil_15min_res9",
                "item": facility,
                "metric": "theil_total",
                "estimate": theil.reported,
                "ci_low": tlo.reported,
                "ci_high": thi.reported,
                "aux_metric": "gini",
                "aux_value": gini.reported,
            }
        )

    for outcome in H3_OUTCOMES:
        for model in H3_MODELS:
            for term in H3_TERMS:
                src = _h3_row(h3, outcome=outcome, model=model, term=term)
                coef = make_claim(
                    f"h3_{outcome}_{model}_{term}_coef",
                    src["coef"],
                    source_file="outputs/tables/h3_spatial_regression.csv",
                    locator=f"15,{outcome},knn4,{model},{term},coef",
                    digits=4,
                )
                lo = make_claim(
                    f"h3_{outcome}_{model}_{term}_lo",
                    src["ci_low"],
                    source_file="outputs/tables/h3_spatial_regression.csv",
                    locator=f"15,{outcome},knn4,{model},{term},ci_low",
                    digits=4,
                )
                hi = make_claim(
                    f"h3_{outcome}_{model}_{term}_hi",
                    src["ci_high"],
                    source_file="outputs/tables/h3_spatial_regression.csv",
                    locator=f"15,{outcome},knn4,{model},{term},ci_high",
                    digits=4,
                )
                pval = make_claim(
                    f"h3_{outcome}_{model}_{term}_p",
                    src["p_value"],
                    source_file="outputs/tables/h3_spatial_regression.csv",
                    locator=f"15,{outcome},knn4,{model},{term},p_value",
                    digits=4,
                )
                for claim in (coef, lo, hi, pval):
                    claims.append(claim)
                rows.append(
                    {
                        "panel": "H3_OLS_SEM_SLM_N23_knn4",
                        "item": f"{outcome}|{model}|{term}",
                        "metric": "coef",
                        "estimate": coef.reported,
                        "ci_low": lo.reported,
                        "ci_high": hi.reported,
                        "aux_metric": "p_value",
                        "aux_value": pval.reported,
                    }
                )

    for model in ML_MODELS:
        block = ml["models"][model]
        metric_claims = {
            metric: make_claim(
                f"ml_{model}_{metric}",
                block[metric],
                source_file="outputs/metrics/phase_a_ml_baselines.json",
                locator=f"models.{model}.{metric}",
                digits=4,
            )
            for metric in ML_METRICS
        }
        claims.extend(metric_claims.values())
        roc = metric_claims["oof_roc_auc"]
        pr = metric_claims["oof_pr_auc"]
        brier = metric_claims["oof_brier"]
        top20 = metric_claims["oof_top20_hit_rate"]
        rows.append(
            {
                "panel": "ML_SpatialBlock_CV",
                "item": model,
                "metric": "oof_roc_auc",
                "estimate": roc.reported,
                "ci_low": "",
                "ci_high": "",
                "aux_metric": "oof_pr_auc",
                "aux_value": pr.reported,
            }
        )
        rows.append(
            {
                "panel": "ML_SpatialBlock_CV",
                "item": model,
                "metric": "oof_brier",
                "estimate": brier.reported,
                "ci_low": "",
                "ci_high": "",
                "aux_metric": "oof_top20_hit_rate",
                "aux_value": top20.reported,
            }
        )

    first = _first_h3_elderly_ols(h3)
    first_claim = make_claim(
        "h3_first_ols_elderly_share_coef",
        first["coef"],
        source_file="outputs/tables/h3_spatial_regression.csv",
        locator=(
            f"15,{first['outcome']},{first['weights_spec']},OLS_HC3,elderly_share,coef"
        ),
        digits=4,
    )
    if first_claim.reported not in {row["estimate"] for row in rows}:
        claims.append(first_claim)
        rows.append(
            {
                "panel": "H3_OLS_SEM_SLM_N23_knn4",
                "item": f"{first['outcome']}|OLS_HC3|elderly_share",
                "metric": "coef",
                "estimate": first_claim.reported,
                "ci_low": format_number(float(first["ci_low"]), 4),
                "ci_high": format_number(float(first["ci_high"]), 4),
                "aux_metric": "p_value",
                "aux_value": format_number(float(first["p_value"]), 4),
            }
        )
    else:
        claims.append(first_claim)
    return rows, claims


def table1_frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(TABLE_COLUMNS))


def render_table1_md(rows: list[dict[str, str]]) -> str:
    header = (
        "| Panel | Item | Metric | Estimate | CI low | CI high | Aux metric | Aux value |\n"
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: |\n"
    )
    body = []
    for row in rows:
        body.append(
            "| {panel} | {item} | {metric} | {estimate} | {ci_low} | {ci_high} | "
            "{aux_metric} | {aux_value} |".format(**row)
        )
    title = (
        "# Table 1. Phase A summary\n\n"
        "Copied from `h1_inequality_lisa.csv`, `h3_spatial_regression.csv`, and "
        "`phase_a_ml_baselines.json`. "
        "H3 reports OLS_HC3, SEM_ML, and SLM_ML side by side for the same "
        "15-minute outcomes on the k=4 nearest-neighbour weights. "
        "No model is selected from a diagnostic p-value.\n\n"
    )
    return title + header + "\n".join(body) + "\n"


def _tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def render_table1_tex(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{center}",
        r"{\small",
        r"\textbf{Table 1.} Gini and Theil T with bootstrap intervals, "
        r"pre-specified H3 coefficients (OLS\_HC3, SEM\_ML, and SLM\_ML side by side), "
        r"and spatial-block CV scores. "
        r"Values are copied from the committed CSV and JSON outputs. "
        r"A diagnostic p-value does not select a model.",
        r"\par\vspace{1em}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrlr}",
        r"\toprule",
        r"Panel & Item & Metric & Estimate & CI low & CI high & Aux & Aux value \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _tex_escape(row["panel"]),
                    _tex_escape(row["item"]),
                    _tex_escape(row["metric"]),
                    row["estimate"],
                    row["ci_low"],
                    row["ci_high"],
                    _tex_escape(row["aux_metric"]),
                    row["aux_value"],
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"}",
            r"\end{center}",
            "",
        ]
    )
    return "\n".join(lines)


def write_table1(root: Path) -> tuple[pd.DataFrame, list[Claim]]:
    rows, claims = build_table1_rows(root)
    frame = table1_frame(rows)
    out_dir = root / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "table1_summary.csv", index=False)
    (out_dir / "table1_summary.md").write_text(render_table1_md(rows), encoding="utf-8")
    (out_dir / "table1_summary.tex").write_text(render_table1_tex(rows), encoding="utf-8")
    return frame, claims


def claim_index(claims: list[Claim]) -> dict[str, Claim]:
    return {claim.claim_id: claim for claim in claims}
