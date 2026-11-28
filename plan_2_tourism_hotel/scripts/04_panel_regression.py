"""Phase A step 4: panel regression for H1 and H2.

    uv run python scripts/04_panel_regression.py

Reads data/frozen only, after checking it against FREEZE_MANIFEST.json, and fits the model
of data/frozen/analysis_spec.json: least squares with star-class indicators and
Driscoll–Kraay standard errors. Diagnostics are reported and change nothing.
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from macao_tourism import plots
from macao_tourism import regression as rg
from macao_tourism.config import (
    OUTPUTS_FIGURES_DIR,
    OUTPUTS_METRICS_DIR,
    OUTPUTS_TABLES_DIR,
    PROJECT_ROOT,
)
from macao_tourism.freeze import load_frozen
from macao_tourism.tables import markdown_table, write_csv

DECIMALS = 10
FIGURE = "fig_regression_slopes_by_class"
RUNS: tuple[tuple[str, str, dict, int | None], ...] = (
    ("main", "the specified model", {}, None),
    (
        "S1_pandemic_included",
        "2020-01 to 2022-12 added with their own indicator",
        {"include_pandemic": True},
        None,
    ),
    ("S2_logit_outcome", "outcome logit(occupancy / 100)", {"outcome": "logit_occupancy"}, None),
    *(
        (f"S3_bandwidth_{lags}", f"Driscoll–Kraay bandwidth {lags}", {}, lags)
        for lags in rg.SENSITIVITY_BANDWIDTHS
    ),
    ("S4_trends", "class-specific linear time trends added", {"trends": True}, None),
    (
        "S5_fully_interacted",
        "controls and month indicators also interacted with post",
        {"fully_interacted": True},
        None,
    ),
)
LEFT_OPEN = [
    "Bandwidth is the highest lag with a non-zero Bartlett weight; lag j has weight "
    "1 − j / (bandwidth + 1).",
    "The covariance is scaled by n / (n − k).",
    "Joint tests are Wald tests with a chi-square reference.",
    "S1 uses the bandwidth rule with its own number of months.",
    "S4 trends are linear in calendar time, one per star class.",
    "Ljung–Box is computed per star class and per contiguous block (pre-pandemic, "
    "post-pandemic), because the residual series has a 36-month gap.",
    "Breusch–Pagan is the studentised (Koenker) LM test on the regressors of the model.",
]


def p_text(value: float) -> str:
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def shown(frame: pd.DataFrame, decimals: int = 3) -> str:
    out = frame.copy()
    for column in out.columns:
        if column.startswith("p_") or column.endswith("p_value") or column.endswith("p_holm"):
            out[column] = out[column].map(p_text)
    return markdown_table(out, decimals=decimals)


def verdict(row: pd.Series) -> str:
    interval = f"95% CI {row['ci_low']:.2f} to {row['ci_high']:.2f}"
    numbers = (
        f"estimate {row['estimate']:.2f} (SE {row['se']:.2f}, {interval}), "
        f"p = {p_text(row['p_value'])}, Holm-adjusted p = {p_text(row['p_holm'])}"
    )
    if row["reject_at_0.05"]:
        return f"{row['hypothesis']}: {numbers}. The null is rejected at the 0.05 level."
    return (
        f"{row['hypothesis']}: {numbers}. The null is not rejected at the 0.05 level. This "
        "is not evidence that the difference is zero; the interval gives the differences "
        "the data are compatible with."
    )


def robustness(sensitivity: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """Runs in which each null is, and is not, rejected at 0.05 after Holm."""
    out = {}
    for hypothesis in ("H1", "H2"):
        rejected = sensitivity[f"{hypothesis}_p_holm"] < rg.ALPHA
        out[hypothesis] = {
            "rejected": sensitivity.loc[rejected, "run"].tolist(),
            "not_rejected": sensitivity.loc[~rejected, "run"].tolist(),
        }
    return out


def render_markdown(summary: dict, tables: dict[str, pd.DataFrame]) -> str:
    spec = summary["specification"]
    main = summary["main_model"]
    sensitivity = tables["sensitivity"].drop(columns=["description", "outcome"])
    return "\n\n".join(
        [
            "# Panel regression for H1 and H2 (Plan 2, Phase A step 4)",
            "Ex-post description of the same-month association between visitor variables "
            "and occupancy. The coefficients are conditional associations. They are not "
            "causal effects, and they are not forecasts.",
            f"Frozen panel SHA-256: `{summary['panel_sha256']}`  \n"
            f"Analysis specification SHA-256: `{summary['analysis_spec_sha256']}` "
            f"(version {summary['spec_version']})  \n"
            f"Deviations from the specification: {len(summary['deviations_from_spec'])}",
            "## Model",
            f"`{spec['model']}`",
            f"- Sample: {spec['sample']}: {main['n']} rows, {main['months']} months.\n"
            f"- {main['parameters']} coefficients. R² = {main['r_squared']:.3f}.\n"
            f"- Standard errors: Driscoll–Kraay, Bartlett kernel, bandwidth "
            f"{main['bandwidth']} ({spec['standard_errors']['bandwidth_rule']}); lags "
            f"counted {spec['standard_errors']['lag_counting']}.\n"
            f"- Inference: {spec['inference']}.",
            "## Confirmatory tests",
            f"- H1 null: `{spec['confirmatory_tests']['H1']['null']}`. Unit: percentage "
            "points of occupancy per unit of log overnight visitors.\n"
            f"- H2 null: `{spec['confirmatory_tests']['H2']['null']}`. Unit: percentage "
            "points of occupancy per percentage point of same-day share.\n"
            f"- Multiplicity: {spec['confirmatory_tests']['multiplicity']}.",
            shown(tables["confirmatory"]),
            "\n".join(f"- {verdict(row)}" for _, row in tables["confirmatory"].iterrows()),
            "## Slopes before and after the pandemic",
            "A slope b on log overnight visitors means: a month with 10% more overnight "
            "visitors has b × ln(1.1) ≈ 0.095 × b points higher occupancy, the other "
            "regressors being equal.",
            shown(tables["slopes"]),
            "## Secondary tests",
            shown(tables["secondary"]),
            "## Sensitivity runs",
            "S2 is on the logit scale, so its estimates are not comparable in size with the "
            "other rows; signs and tests are. AIC and BIC are not compared across scales.",
            shown(sensitivity),
            "\n".join(
                f"- {name}: the null is rejected after Holm in {len(runs['rejected'])} of "
                f"{len(runs['rejected']) + len(runs['not_rejected'])} runs"
                + (f"; not in {', '.join(runs['not_rejected'])}." if runs["rejected"] else ".")
                + (f" Rejected in {', '.join(runs['rejected'])}." if runs["rejected"] else "")
                for name, runs in summary["robustness"].items()
            ),
            "## Diagnostics (reported only)",
            spec["diagnostics"]["rule"].capitalize() + ".",
            shown(tables["diagnostics"]),
            f"Ljung–Box has p < 0.05 in {summary['diagnostics_count']['ljung_box_below_0.05']} "
            f"of {summary['diagnostics_count']['ljung_box_cells']} cells; Breusch–Pagan has "
            f"p {p_text(summary['diagnostics_count']['breusch_pagan_p'])}. The Driscoll–Kraay "
            "standard errors allow for heteroskedasticity, for correlation between star "
            "classes in a month, and for serial correlation up to the bandwidth; the S3 "
            "rows show the tests at bandwidth 2 and 8. The normal reference rests on 165 "
            "months and may be optimistic.",
            "## All coefficients of the main model",
            shown(tables["coefficients"]),
            "## Choices the specification left open",
            "Fixed before the first fit on the real outcome:",
            "\n".join(f"- {item}" for item in summary["left_open_by_the_specification"]),
            "## Figure",
            f"`outputs/figures/{FIGURE}.png`: slopes before and after the pandemic with "
            "95% confidence intervals.",
        ]
    )


def main() -> int:
    frozen = load_frozen()
    spec = frozen.spec["regression_h1_h2"]
    controls = list(spec["terms"]["controls"])

    fits = {
        name: rg.fit(rg.build_design(frozen.panel, controls, **options), bandwidth)
        for name, _, options, bandwidth in RUNS
    }
    main_fit = fits["main"]
    tables = {
        "coefficients": rg.coefficient_table(main_fit),
        "confirmatory": rg.confirmatory_tests(main_fit),
        "secondary": rg.secondary_tests(main_fit),
        "slopes": rg.slopes_by_class(main_fit),
        "sensitivity": pd.DataFrame(
            [rg.sensitivity_row(name, text, fits[name]) for name, text, _, _ in RUNS]
        ),
        "diagnostics": rg.diagnostics(main_fit),
        "residuals": rg.residual_table(main_fit),
    }
    tables = {name: frame.round(DECIMALS) for name, frame in tables.items()}
    for name, frame in tables.items():
        write_csv(frame, f"regression_{name}.csv")
    figures = plots.save(
        plots.figure_slopes(tables["slopes"], rg.FOCAL, rg.SHARE), OUTPUTS_FIGURES_DIR, FIGURE
    )

    ljung_box = tables["diagnostics"][tables["diagnostics"]["test"] == "ljung_box"]
    breusch_pagan = tables["diagnostics"][tables["diagnostics"]["test"] == "breusch_pagan"]
    summary = {
        "schema_version": "1.0.0",
        "step": "Phase A step 4: panel regression for H1 and H2",
        "role": spec["role"],
        "frozen_at_utc8": frozen.manifest["frozen_at_utc8"],
        "panel_sha256": frozen.panel_sha256,
        "analysis_spec_sha256": frozen.spec_sha256,
        "spec_version": frozen.spec["spec_version"],
        "specification": spec,
        "deviations_from_spec": [],
        "left_open_by_the_specification": LEFT_OPEN,
        "main_model": {
            "n": main_fit.n,
            "months": main_fit.n_months,
            "parameters": main_fit.k,
            "bandwidth": main_fit.bandwidth,
            "r_squared": round(main_fit.r_squared, DECIMALS),
        },
        "confirmatory_tests": tables["confirmatory"].to_dict("records"),
        "robustness": robustness(tables["sensitivity"]),
        "secondary_tests": tables["secondary"].to_dict("records"),
        "slopes": tables["slopes"].to_dict("records"),
        "sensitivity": tables["sensitivity"].to_dict("records"),
        "diagnostics": tables["diagnostics"].to_dict("records"),
        "diagnostics_count": {
            "ljung_box_cells": int(len(ljung_box)),
            "ljung_box_below_0.05": int((ljung_box["p_value"] < rg.ALPHA).sum()),
            "breusch_pagan_p": float(breusch_pagan["p_value"].iloc[0]),
        },
        "figures": [str(path.relative_to(PROJECT_ROOT)) for path in figures],
    }
    OUTPUTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_METRICS_DIR / "regression_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUTS_TABLES_DIR / "regression_summary.md").write_text(
        render_markdown(summary, tables) + "\n", encoding="utf-8"
    )

    print(shown(tables["confirmatory"]))
    print(shown(tables["slopes"]))
    print(shown(tables["secondary"]))
    print(shown(tables["sensitivity"].drop(columns=["description", "outcome"])))
    print(shown(tables["diagnostics"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
