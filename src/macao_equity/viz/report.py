"""Phase A LaTeX draft, bibliography, and numeric Go/No-Go audit.

Prose numbers are substituted from claim strings. This module does not
recompute Gini, Theil, regression coefficients, or ROC-AUC.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import geopandas as gpd
import pandas as pd

from macao_equity.config import (
    DSEC_TERRESTRIAL_POP_65PLUS,
    DSEC_TERRESTRIAL_POP_TOTAL,
    RANDOM_SEED,
    WALK_SPEED_KMH,
)
from macao_equity.viz.tables import Claim, claim_index, make_claim

_VIZ_DIR = Path(__file__).resolve().parent
_TEMPLATE = (_VIZ_DIR / "phase_a_template.tex").read_text(encoding="utf-8")
REFERENCES_BIB = (_VIZ_DIR / "references_template.bib").read_text(encoding="utf-8")


def _h1(frame: pd.DataFrame, threshold: int, facility: str) -> pd.Series:
    mask = (
        (frame["resolution"].astype(str) == "res9")
        & (frame["threshold_min"].astype(int) == threshold)
        & (frame["facility_type"].astype(str) == facility)
    )
    hits = frame.loc[mask]
    if len(hits) != 1:
        raise ValueError(f"Missing H1 res9 {threshold} {facility}")
    return hits.iloc[0]


def _h2(frame: pd.DataFrame, threshold: int, comparison: str) -> pd.Series:
    mask = (frame["threshold_min"].astype(int) == threshold) & (
        frame["comparison"].astype(str) == comparison
    )
    hits = frame.loc[mask]
    if len(hits) != 1:
        raise ValueError(f"Missing H2 {threshold} {comparison}")
    return hits.iloc[0]


def _extra_claim(claim_id: str, raw: object, source: str, locator: str, digits: int) -> Claim:
    return make_claim(
        claim_id,
        raw,
        source_file=source,
        locator=locator,
        digits=digits,
    )


def build_prose_claims(root: Path) -> list[Claim]:
    """Numbers that appear in the prose and are not already Table 1 cells."""
    tables = root / "outputs" / "tables"
    h1 = pd.read_csv(tables / "h1_inequality_lisa.csv")
    h2 = pd.read_csv(tables / "h2_gini_diff_bootstrap.csv")
    maup = pd.read_csv(tables / "maup_r8_vs_r9.csv")
    diagnostics_path = tables / "h3_regression_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    ml = json.loads(
        (root / "outputs" / "metrics" / "phase_a_ml_baselines.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (root / "outputs" / "metrics" / "data_provenance.json").read_text(encoding="utf-8")
    )
    claims: list[Claim] = []

    def push(claim: Claim) -> Claim:
        claims.append(claim)
        return claim

    comp = _h1(h1, 15, "composite")
    ev15 = _h1(h1, 15, "ev_charger")
    ev10 = _h1(h1, 10, "ev_charger")
    park10 = _h1(h1, 10, "park")
    push(
        _extra_claim(
            "comp_within_15",
            comp["theil_within"],
            "outputs/tables/h1_inequality_lisa.csv",
            "res9,15,composite,theil_within",
            4,
        )
    )
    push(
        _extra_claim(
            "comp_between_15",
            comp["theil_between"],
            "outputs/tables/h1_inequality_lisa.csv",
            "res9,15,composite,theil_between",
            4,
        )
    )
    push(
        _extra_claim(
            "comp_moran_15",
            comp["moran_i"],
            "outputs/tables/h1_inequality_lisa.csv",
            "res9,15,composite,moran_i",
            4,
        )
    )
    push(
        _extra_claim(
            "comp_moran_p_15",
            comp["moran_p"],
            "outputs/tables/h1_inequality_lisa.csv",
            "res9,15,composite,moran_p",
            4,
        )
    )
    push(
        _extra_claim(
            "comp_n_ll",
            comp["n_ll_fdr"],
            "outputs/tables/h1_inequality_lisa.csv",
            "res9,15,composite,n_ll_fdr",
            0,
        )
    )
    push(
        _extra_claim(
            "comp_n_hh",
            comp["n_hh_fdr"],
            "outputs/tables/h1_inequality_lisa.csv",
            "res9,15,composite,n_hh_fdr",
            0,
        )
    )
    push(
        _extra_claim(
            "ev_moran_15",
            ev15["moran_i"],
            "outputs/tables/h1_inequality_lisa.csv",
            "res9,15,ev_charger,moran_i",
            4,
        )
    )
    push(
        _extra_claim(
            "ev_zero_15",
            ev15["zero_acc_pct"],
            "outputs/tables/h1_inequality_lisa.csv",
            "res9,15,ev_charger,zero_acc_pct",
            2,
        )
    )
    push(
        _extra_claim(
            "ev_gini_10",
            ev10["gini"],
            "outputs/tables/h1_inequality_lisa.csv",
            "res9,10,ev_charger,gini",
            4,
        )
    )
    push(
        _extra_claim(
            "park_gini_10",
            park10["gini"],
            "outputs/tables/h1_inequality_lisa.csv",
            "res9,10,park,gini",
            4,
        )
    )

    for key, comparison, digits_map in (
        ("h2_park", "ev_charger_vs_park", ("diff", "lo", "hi", "holm")),
        ("h2_lib", "ev_charger_vs_library", ("diff", "holm")),
        ("h2_bus", "ev_charger_vs_bus_stop", ("diff", "holm")),
        ("h2_health", "ev_charger_vs_health_centre", ("diff", "lo", "hi", "holm")),
    ):
        src = _h2(h2, 15, comparison)
        fields = {
            "diff": "gini_diff",
            "lo": "ci_low",
            "hi": "ci_high",
            "holm": "p_holm",
        }
        for suffix in digits_map:
            push(
                _extra_claim(
                    f"{key}_{suffix}",
                    src[fields[suffix]],
                    "outputs/tables/h2_gini_diff_bootstrap.csv",
                    f"15,{comparison},{fields[suffix]}",
                    4,
                )
            )

    maup_row = maup[
        (maup["threshold_min"].astype(int) == 15)
        & (maup["facility_type"].astype(str) == "composite")
    ].iloc[0]
    push(
        _extra_claim(
            "maup_r8",
            maup_row["gini_r8"],
            "outputs/tables/maup_r8_vs_r9.csv",
            "15,composite,gini_r8",
            4,
        )
    )
    push(
        _extra_claim(
            "maup_r9",
            maup_row["gini_r9"],
            "outputs/tables/maup_r8_vs_r9.csv",
            "15,composite,gini_r9",
            4,
        )
    )
    push(
        _extra_claim(
            "maup_delta",
            maup_row["gini_delta_r9_minus_r8"],
            "outputs/tables/maup_r8_vs_r9.csv",
            "15,composite,gini_delta_r9_minus_r8",
            4,
        )
    )
    vif = diagnostics["vif"]
    push(
        _extra_claim(
            "vif_eld",
            vif["elderly_share"],
            "outputs/tables/h3_regression_diagnostics.json",
            "vif.elderly_share",
            4,
        )
    )
    push(
        _extra_claim(
            "vif_dens",
            vif["log_pop_density"],
            "outputs/tables/h3_regression_diagnostics.json",
            "vif.log_pop_density",
            4,
        )
    )
    for model, prefix in (
        ("random_forest", "rf"),
        ("logistic_regression", "lr"),
    ):
        block = ml["models"][model]
        push(
            _extra_claim(
                f"{prefix}_mean_roc",
                block["mean_roc_auc"],
                "outputs/metrics/phase_a_ml_baselines.json",
                f"models.{model}.mean_roc_auc",
                4,
            )
        )
    push(
        _extra_claim(
            "rf_std_roc",
            ml["models"]["random_forest"]["std_roc_auc"],
            "outputs/metrics/phase_a_ml_baselines.json",
            "models.random_forest.std_roc_auc",
            4,
        )
    )

    reason = ""
    for source in provenance["sources"]:
        text = str(source.get("fallback_reason") or "")
        if "9488798" in text and "12.0" in text:
            reason = text
            break
    if not reason:
        raise ValueError("data_provenance.json is missing the station 9488798 capacity note")
    push(
        _extra_claim(
            "imputed_cap",
            12.0,
            "outputs/metrics/data_provenance.json",
            "cem fallback_reason capacity=12.0",
            1,
        )
    )

    cells = gpd.read_file(root / "data" / "frozen" / "cells_base_res9.gpkg")
    cells8 = gpd.read_file(root / "data" / "frozen" / "cells_base_res8.gpkg")
    facilities = gpd.read_file(root / "data" / "frozen" / "facilities_clean.gpkg")
    pop = float(cells["pop_total"].sum())
    pop65 = float(cells["pop_65plus"].sum())
    if abs(pop - DSEC_TERRESTRIAL_POP_TOTAL) / DSEC_TERRESTRIAL_POP_TOTAL > 0.0005:
        raise ValueError(f"Cell population sum {pop} drifted from the census control")
    if abs(pop65 - DSEC_TERRESTRIAL_POP_65PLUS) / DSEC_TERRESTRIAL_POP_65PLUS > 0.001:
        raise ValueError(f"Cell age-65 sum {pop65} drifted from the census control")
    push(
        make_claim(
            "pop_total",
            DSEC_TERRESTRIAL_POP_TOTAL,
            source_file="data/frozen/cells_base_res9.gpkg",
            locator="sum(pop_total) matches DSEC terrestrial control",
            digits=0,
            thousands=True,
        )
    )
    push(
        make_claim(
            "pop_65",
            DSEC_TERRESTRIAL_POP_65PLUS,
            source_file="data/frozen/cells_base_res9.gpkg",
            locator="sum(pop_65plus) matches DSEC terrestrial control",
            digits=0,
            thousands=True,
        )
    )
    push(
        _extra_claim(
            "n_cells",
            ml["n_cells"],
            "outputs/metrics/phase_a_ml_baselines.json",
            "n_cells",
            0,
        )
    )
    if int(ml["n_cells"]) != len(cells):
        raise ValueError("ML n_cells does not match the resolution-9 grid")
    push(
        _extra_claim(
            "n_pos",
            ml["n_pos"],
            "outputs/metrics/phase_a_ml_baselines.json",
            "n_pos",
            0,
        )
    )
    push(
        _extra_claim(
            "prevalence",
            ml["prevalence"],
            "outputs/metrics/phase_a_ml_baselines.json",
            "prevalence",
            4,
        )
    )
    push(
        _extra_claim(
            "n_res8",
            len(cells8),
            "data/frozen/cells_base_res8.gpkg",
            "row_count",
            0,
        )
    )
    counts = facilities["facility_type"].astype(str).value_counts()
    for claim_id, facility in (
        ("n_park", "park"),
        ("n_health", "health_centre"),
        ("n_library", "library"),
        ("n_bus", "bus_stop"),
        ("n_ev", "ev_charger"),
    ):
        push(
            _extra_claim(
                claim_id,
                int(counts[facility]),
                "data/frozen/facilities_clean.gpkg",
                f"count(facility_type={facility})",
                0,
            )
        )
    push(
        _extra_claim(
            "walk_kmh",
            WALK_SPEED_KMH,
            "src/macao_equity/config.py",
            "WALK_SPEED_KMH",
            1,
        )
    )
    push(
        _extra_claim(
            "seed",
            ml["seed"],
            "outputs/metrics/phase_a_ml_baselines.json",
            "seed",
            0,
        )
    )
    if int(ml["seed"]) != RANDOM_SEED:
        raise ValueError("ML seed does not match the project seed")
    push(
        _extra_claim(
            "n_boot",
            h2["n_boot"].iloc[0],
            "outputs/tables/h2_gini_diff_bootstrap.csv",
            "n_boot",
            0,
        )
    )
    push(
        _extra_claim(
            "n_perm",
            diagnostics["permutations"],
            "outputs/tables/h3_regression_diagnostics.json",
            "permutations",
            0,
        )
    )
    n_zones = len(gpd.read_file(root / "data" / "frozen" / "zones_23.gpkg"))
    n_parishes = len(gpd.read_file(root / "data" / "frozen" / "parishes_7.gpkg"))
    push(
        _extra_claim(
            "n_zones",
            n_zones,
            "data/frozen/zones_23.gpkg",
            "row_count",
            0,
        )
    )
    push(
        _extra_claim(
            "n_parishes",
            n_parishes,
            "data/frozen/parishes_7.gpkg",
            "row_count",
            0,
        )
    )
    return claims


def render_compact_table(by_id: dict[str, Claim]) -> str:
    """Short Table 1 for the draft. Every figure is a claim string."""

    def got(claim_id: str) -> str:
        return by_id[claim_id].reported

    facilities = (
        ("Park", "park"),
        ("Health centre", "health_centre"),
        ("Library", "library"),
        ("Bus stop", "bus_stop"),
        ("EV charger", "ev_charger"),
        ("Composite", "composite"),
    )
    models = (
        ("Park OLS-HC3", "h3_acc15_park_OLS_HC3_elderly_share_coef"),
        ("Park SEM", "h3_acc15_park_SEM_ML_elderly_share_coef"),
        ("Park SLM", "h3_acc15_park_SLM_ML_elderly_share_coef"),
        ("Composite OLS-HC3", "h3_acc15_composite_OLS_HC3_elderly_share_coef"),
        ("Composite SEM", "h3_acc15_composite_SEM_ML_elderly_share_coef"),
        ("Composite SLM", "h3_acc15_composite_SLM_ML_elderly_share_coef"),
    )
    learners = (
        ("Constant prior", "ml_naive_prior"),
        ("Logistic regression", "ml_logistic_regression"),
        ("Random forest", "ml_random_forest"),
    )
    lines = [
        r"\begin{center}",
        r"{\small",
        r"\textbf{Table 1.} 15-minute Gini and Theil T with bootstrap intervals,",
        r"pre-specified H3 elderly-share coefficients, and out-of-fold scores.",
        r"OLS, SEM, and SLM stay side by side. A diagnostic p-value does not select a model.",
        r"\par\vspace{1em}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Facility & Gini & CI low & CI high & Theil T \\",
        r"\midrule",
    ]
    for label, key in facilities:
        lines.append(
            f"{label} & {got(f'h1_15_{key}_gini')} & {got(f'h1_15_{key}_gini_lo')} & "
            f"{got(f'h1_15_{key}_gini_hi')} & {got(f'h1_15_{key}_theil')} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            r"H3 elderly share & Coef & & & \\",
            r"\midrule",
        ]
    )
    for label, claim_id in models:
        lines.append(f"{label} & {got(claim_id)} & & & \\\\")
    lines.extend(
        [
            r"\midrule",
            r"Classifier & ROC-AUC & PR-AUC & Brier & \\",
            r"\midrule",
        ]
    )
    for label, prefix in learners:
        lines.append(
            f"{label} & {got(prefix + '_oof_roc_auc')} & {got(prefix + '_oof_pr_auc')} & "
            f"{got(prefix + '_oof_brier')} & \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{center}"])
    return "\n".join(lines) + "\n"


def substitution_map(
    table_claims: list[Claim],
    prose_claims: list[Claim],
    *,
    fig_frac: float,
    pad_pt: int,
) -> dict[str, str]:
    by_id = claim_index([*table_claims, *prose_claims])

    def got(claim_id: str) -> str:
        if claim_id not in by_id:
            raise KeyError(claim_id)
        return by_id[claim_id].reported

    mapping = {
        "FIG_FRAC": f"{fig_frac:.2f}",
        "PAD_PT": str(int(pad_pt)),
        "COMPACT_TABLE": render_compact_table(by_id),
        "POP_TOTAL": got("pop_total"),
        "POP_65": got("pop_65"),
        "N_CELLS": got("n_cells"),
        "N_POS": got("n_pos"),
        "N_RES8": got("n_res8"),
        "N_ZONES": got("n_zones"),
        "N_PARISHES": got("n_parishes"),
        "N_BOOT": got("n_boot"),
        "N_PERM": got("n_perm"),
        "N_PARK": got("n_park"),
        "N_HEALTH": got("n_health"),
        "N_LIBRARY": got("n_library"),
        "N_BUS": got("n_bus"),
        "N_EV": got("n_ev"),
        "PREV": got("prevalence"),
        "WALK_KMH": got("walk_kmh"),
        "SEED": got("seed"),
        "IMPUTED_CAP": got("imputed_cap"),
        "VIF_ELD": got("vif_eld"),
        "VIF_DENS": got("vif_dens"),
        "EV_GINI_15": got("h1_15_ev_charger_gini"),
        "EV_GINI_15_LO": got("h1_15_ev_charger_gini_lo"),
        "EV_GINI_15_HI": got("h1_15_ev_charger_gini_hi"),
        "EV_THEIL_15": got("h1_15_ev_charger_theil"),
        "PARK_GINI_15": got("h1_15_park_gini"),
        "HEALTH_GINI_15": got("h1_15_health_centre_gini"),
        "LIBRARY_GINI_15": got("h1_15_library_gini"),
        "BUS_GINI_15": got("h1_15_bus_stop_gini"),
        "COMP_GINI_15": got("h1_15_composite_gini"),
        "COMP_THEIL_15": got("h1_15_composite_theil"),
        "COMP_WITHIN_15": got("comp_within_15"),
        "COMP_BETWEEN_15": got("comp_between_15"),
        "COMP_MORAN_15": got("comp_moran_15"),
        "COMP_MORAN_P_15": got("comp_moran_p_15"),
        "COMP_N_LL": got("comp_n_ll"),
        "COMP_N_HH": got("comp_n_hh"),
        "EV_MORAN_15": got("ev_moran_15"),
        "EV_ZERO_15": got("ev_zero_15"),
        "EV_GINI_10": got("ev_gini_10"),
        "PARK_GINI_10": got("park_gini_10"),
        "H2_PARK_DIFF": got("h2_park_diff"),
        "H2_PARK_LO": got("h2_park_lo"),
        "H2_PARK_HI": got("h2_park_hi"),
        "H2_PARK_HOLM": got("h2_park_holm"),
        "H2_LIB_DIFF": got("h2_lib_diff"),
        "H2_LIB_HOLM": got("h2_lib_holm"),
        "H2_BUS_DIFF": got("h2_bus_diff"),
        "H2_BUS_HOLM": got("h2_bus_holm"),
        "H2_HEALTH_DIFF": got("h2_health_diff"),
        "H2_HEALTH_LO": got("h2_health_lo"),
        "H2_HEALTH_HI": got("h2_health_hi"),
        "H2_HEALTH_HOLM": got("h2_health_holm"),
        "H3_COMP_OLS": got("h3_acc15_composite_OLS_HC3_elderly_share_coef"),
        "H3_COMP_OLS_P": got("h3_acc15_composite_OLS_HC3_elderly_share_p"),
        "H3_COMP_SEM": got("h3_acc15_composite_SEM_ML_elderly_share_coef"),
        "H3_COMP_SLM": got("h3_acc15_composite_SLM_ML_elderly_share_coef"),
        "H3_COMP_LOGPOP": got("h3_acc15_composite_OLS_HC3_log_pop_density_coef"),
        "H3_PARK_OLS": got("h3_acc15_park_OLS_HC3_elderly_share_coef"),
        "H3_PARK_OLS_P": got("h3_acc15_park_OLS_HC3_elderly_share_p"),
        "MAUP_R8": got("maup_r8"),
        "MAUP_R9": got("maup_r9"),
        "MAUP_DELTA": got("maup_delta"),
        "NAIVE_AUC": got("ml_naive_prior_oof_roc_auc"),
        "LR_AUC": got("ml_logistic_regression_oof_roc_auc"),
        "LR_PR": got("ml_logistic_regression_oof_pr_auc"),
        "LR_BRIER": got("ml_logistic_regression_oof_brier"),
        "LR_MEAN": got("lr_mean_roc"),
        "RF_AUC": got("ml_random_forest_oof_roc_auc"),
        "RF_PR": got("ml_random_forest_oof_pr_auc"),
        "RF_BRIER": got("ml_random_forest_oof_brier"),
        "RF_TOP": got("ml_random_forest_oof_top20_hit_rate"),
        "RF_MEAN": got("rf_mean_roc"),
        "RF_STD": got("rf_std_roc"),
    }
    return mapping


def render_tex(mapping: dict[str, str]) -> str:
    text = _TEMPLATE
    for key, value in mapping.items():
        token = f"__{key}__"
        if token not in text:
            raise KeyError(f"Template is missing {token}")
        text = text.replace(token, value)
    leftover = re.findall(r"__[A-Z0-9_]+__", text)
    if leftover:
        raise ValueError(f"Unreplaced placeholders: {leftover}")
    return text


def render_audit(claims: list[Claim], *, main_pages: float) -> str:
    bad = [claim for claim in claims if abs(_parsed(claim) - claim.source_value) > claim.tolerance]
    if bad:
        detail = ", ".join(claim.claim_id for claim in bad)
        raise ValueError(f"Audit refused to write a mismatched claim: {detail}")
    lines = [
        "# Phase A 12/03 go/no-go audit",
        "",
        "Checkpoint verdict: GO.",
        "Numeric audit: PASS.",
        "Every reported number below matches its source cell within rounding tolerance.",
        f"Main text length: {main_pages:.2f} A4 pages.",
        "",
        "## Minimum requirements",
        "",
        (
            "- MR1 Research question: H1, H2, and H3 are stated, with DSSCU, DSAT, "
            "and residents as the users of the description."
        ),
        (
            "- MR2 Data collection and preparation: DSEC, CEM, and OpenStreetMap "
            "in EPSG:32649, with 25 m cross-source dedup. Terrestrial population "
            "control 681,293 across 23 zones and 7 parishes."
        ),
        (
            "- MR3 Analysis: 2SFCA, Gini, Theil, LISA, side-by-side OLS / SEM / SLM, "
            "and spatial-block classifiers."
        ),
        (
            "- MR4 Evaluation: bootstrap intervals, FDR, Holm adjustment, MAUP, "
            "ROC-AUC, PR-AUC, and Brier score."
        ),
        (
            "- MR5 Results and discussion: Figure 1, Table 1, four limitations, "
            "correlational wording only."
        ),
        "",
        "## Model policy",
        "",
        (
            "H3 keeps OLS-HC3, SEM, and SLM side by side. "
            "A diagnostic p-value does not select a model."
        ),
        "",
        "## Limitations named in the draft",
        "",
        (
            "- Uniform building height in high-rise districts, including the residual "
            "dasymetric limit of building-group polygons."
        ),
        "- Incomplete facility capacity metadata, including imputed park area and station 9488798.",
        "- No slope, incomplete footbridges and overpasses, and no cross-harbour walk link.",
        "- Ecological fallacy: zone and cell associations are not individual associations.",
        "",
        "## Claim trace",
        "",
        "| Claim | Reported | Source value | Abs err | Status | Source | Locator |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for claim in claims:
        parsed = _parsed(claim)
        err = abs(parsed - claim.source_value)
        lines.append(
            f"| {claim.claim_id} | {claim.reported} | {claim.source_value:.10g} | "
            f"{err:.3g} | PASS | {claim.source_file} | {claim.locator} |"
        )
    lines.extend(
        [
            "",
            "Status of the numeric trace: PASS.",
            "The frozen bundle checksum is in `data/frozen/FREEZE_MANIFEST.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _parsed(claim: Claim) -> float:
    return float(claim.reported.replace(",", ""))


def write_report_sources(
    root: Path,
    mapping: dict[str, str],
) -> Path:
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    tex_path = report_dir / "phase_a_draft.tex"
    bib_path = report_dir / "references.bib"
    tex_path.write_text(render_tex(mapping), encoding="utf-8")
    bib_path.write_text(REFERENCES_BIB, encoding="utf-8")
    return tex_path


def compile_pdf(tex_path: Path) -> Path:
    work = tex_path.parent
    stem = tex_path.stem
    log_path = work / f"{stem}.log"
    commands = [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{stem}.tex"],
        ["bibtex", stem],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{stem}.tex"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{stem}.tex"],
    ]
    for command in commands:
        proc = subprocess.run(command, cwd=work, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            if log_path.exists():
                log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            else:
                log_tail = ""
            raise RuntimeError(
                f"{' '.join(command)} failed with code {proc.returncode}\n"
                f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}\n{log_tail}"
            )
    produced = work / f"{stem}.pdf"
    final = work / "CS482_GroupX_MacaoSpatialEquity_PhaseA_Draft.pdf"
    final.write_bytes(produced.read_bytes())
    if produced.resolve() != final.resolve():
        produced.unlink()
    text = subprocess.run(
        ["pdftotext", str(final), "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if "??" in text:
        raise RuntimeError("Compiled PDF still contains unresolved citations")
    return final


def measure_main_pages(pdf_path: Path, repo_root: Path) -> float:
    import sys

    e2e_dir = str(repo_root / "tests" / "e2e")
    if e2e_dir not in sys.path:
        sys.path.insert(0, e2e_dir)
    from e2e_lib.readers import inspect_pdf

    info = inspect_pdf(pdf_path)
    return float(info.main_fractional_pages)
