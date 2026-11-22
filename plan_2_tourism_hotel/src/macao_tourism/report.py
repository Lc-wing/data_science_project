"""Phase A report draft. Every number is copied from an output file and then checked."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from macao_tourism.config import OUTPUTS_METRICS_DIR, OUTPUTS_TABLES_DIR, PROJECT_ROOT

TEMPLATE_PATH = Path(__file__).resolve().parent / "phase_a_template.tex"
REPORT_DIR = PROJECT_ROOT / "report"
TEX_NAME = "phase_a_draft.tex"
PDF_NAME = "CS482_GroupX_MacaoTourismHotel_PhaseA_Draft.pdf"
AUDIT_NAME = "phase_a_audit.md"
SOURCE_DATE_EPOCH = "1763596800"
MAIN_PAGE_MIN = 3.0
MAIN_PAGE_MAX = 5.0
CAUSAL_WORDS = ("impact", "driver", "elasticity", "caused")

SHARE_LABELS = {
    "same_day_share_pct": "same-day share",
    "international_share_pct": "international share",
    "package_tour_share_pct": "package-tour share",
}


@dataclass(frozen=True)
class Claim:
    claim_id: str
    reported: str
    source_file: str
    locator: str
    source_value: float | None = None
    tolerance: float | None = None

    def matches_source(self) -> bool:
        if self.source_value is None or self.tolerance is None:
            return True
        parsed = float(self.reported.replace(",", "").lstrip("=").replace("< ", ""))
        return abs(parsed - self.source_value) <= self.tolerance + 1e-12


def format_number(value: float, digits: int) -> tuple[str, float]:
    reported = f"{float(value):,.{digits}f}"
    tolerance = 0.5 * (10 ** (-digits)) + 1e-9
    return reported, tolerance


def format_p(value: float) -> tuple[str, float]:
    if float(value) < 0.001:
        return "< 0.001", 0.001
    return format_number(value, 3)


class Claims:
    def __init__(self) -> None:
        self._claims: dict[str, Claim] = {}

    def number(self, token: str, value: float, digits: int, source: str, locator: str) -> str:
        reported, tolerance = format_number(value, digits)
        self._claims[token] = Claim(token, reported, source, locator, float(value), tolerance)
        return reported

    def p_value(
        self, token: str, value: float, source: str, locator: str, *, inline: bool = False
    ) -> str:
        reported, tolerance = format_p(value)
        if inline and reported.startswith("<"):
            reported = "<0.001"
        elif inline:
            reported = "=" + reported
        numeric = None if reported.startswith("<") else float(value)
        tol = None if numeric is None else tolerance
        self._claims[token] = Claim(token, reported, source, locator, numeric, tol)
        return reported

    def text(self, token: str, reported: str, source: str, locator: str) -> str:
        self._claims[token] = Claim(token, reported, source, locator)
        return reported

    def items(self) -> list[Claim]:
        return list(self._claims.values())

    def mapping(self) -> dict[str, str]:
        return {token: claim.reported for token, claim in self._claims.items()}


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_TABLES_DIR / name)


def _json(name: str, directory: Path = OUTPUTS_METRICS_DIR) -> dict:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def persistent_change_points(runs: pd.DataFrame) -> dict[str, list[str]]:
    """Change points that remain at every reported penalty, on the adjusted series."""
    adjusted = runs[runs["input"] == "seasonally_adjusted"]
    found: dict[str, list[str]] = {}
    for series, part in adjusted.groupby("series", sort=False):
        sets = []
        for points in part.sort_values("penalty_multiplier")["change_points"]:
            sets.append({month for month in str(points).split(";") if month})
        found[str(series)] = sorted(set.intersection(*sets))
    return found


def change_point_sentence(found: dict[str, list[str]]) -> str:
    pieces = []
    for series, months in found.items():
        listed = " and ".join(months) if months else "none"
        pieces.append(f"{listed} for the {SHARE_LABELS[series]}")
    body = "; ".join(pieces)
    return f"Change points that remain at every reported penalty are {body}."


def _tex_p(value: str) -> str:
    return rf"\({value}\)" if value.startswith("<") else value


def table_one(c: Claims) -> str:
    """One table: the confirmatory contrasts and the pooled forecast scores."""

    def row(cells: list[str]) -> str:
        return " & ".join(cells) + r" \\"

    m = c.mapping()
    return "\n".join(
        [
            r"\begin{center}",
            r"\begin{minipage}{\textwidth}",
            r"\centering",
            r"{\small Table 1. Confirmatory regression contrasts (top, in percentage points of "
            r"occupancy; Driscoll--Kraay SEs and Holm-adjusted \(p\)-values over H1 and H2) and "
            r"pooled test forecast accuracy (bottom; MAE and RMSE in percentage points, MASE "
            r"dimensionless, HLN-DM Holm-adjusted \(p\)-values vs.\ LightGBM over the 10-test "
            r"family; predictive comparison only).}\\[4pt]",
            r"{\small",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Quantity & Estimate & SE & 95\% CI & Holm \(p\) \\",
            r"\midrule",
            row(
                [
                    "H1, post slope, 5-star minus 3-star",
                    m["H1_EST"],
                    m["H1_SE"],
                    f"{m['H1_LO']} to {m['H1_HI']}",
                    _tex_p(m["H1_HOLM"]),
                ]
            ),
            row(
                [
                    "H2, change in the same-day slope",
                    m["H2_EST"],
                    m["H2_SE"],
                    f"{m['H2_LO']} to {m['H2_HI']}",
                    _tex_p(m["H2_HOLM"]),
                ]
            ),
            row(
                [
                    "5-star post slope on log overnight visitors",
                    m["B5_POST"],
                    "",
                    f"{m['B5_LO']} to {m['B5_HI']}",
                    "",
                ]
            ),
            row(
                [
                    "3-star post slope on log overnight visitors",
                    m["B3_POST"],
                    "",
                    f"{m['B3_LO']} to {m['B3_HI']}",
                    "",
                ]
            ),
            r"\midrule",
            r"Model (pooled test) & MAE & RMSE & MASE & Holm \(p\) \\",
            r"\midrule",
            row(["LightGBM", m["MAE_LGB"], m["RMSE_LGB"], m["MASE_LGB"], "--"]),
            row(["SARIMAX", m["MAE_SAR"], m["RMSE_SAR"], m["MASE_SAR"], _tex_p(m["DM_S_HOLM"])]),
            row(
                [
                    "Seasonal naive",
                    m["MAE_NAIVE"],
                    m["RMSE_NAIVE"],
                    m["MASE_NAIVE"],
                    _tex_p(m["DM_N_HOLM"]),
                ]
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{minipage}",
            r"\end{center}",
        ]
    )


def build_claims() -> Claims:
    c = Claims()
    audit = _json("day1_data_audit.json")
    window = audit["panel_window"]
    tier = audit["dsec"]["tier_classification"]
    jumps = tier["tier_months_with_room_change_of_10pct_or_more"]
    mix = _csv("eda_visitor_mix_by_period.csv").set_index("period_label")
    occupancy = _csv("eda_occupancy_by_period.csv")
    runs = _csv("eda_change_point_runs.csv")
    stl = _csv("eda_stl_summary.csv")
    confirmatory = _csv("regression_confirmatory.csv").set_index("hypothesis")
    slopes = _csv("regression_slopes.csv")
    secondary = _csv("regression_secondary.csv").set_index("test")
    sensitivity = _csv("regression_sensitivity.csv").set_index("run")
    regression = _json("regression_summary.json")
    metrics = _csv("forecast_test_metrics.csv")
    tests = _csv("forecast_dm_tests.csv")
    forecasts = _csv("forecast_test_forecasts.csv")
    shap = _csv("forecast_shap_importance.csv")
    manifest = _json("FREEZE_MANIFEST.json", PROJECT_ROOT / "data" / "frozen")
    day1 = "outputs/metrics/day1_data_audit.json"
    mix_file = "outputs/tables/eda_visitor_mix_by_period.csv"
    occ_file = "outputs/tables/eda_occupancy_by_period.csv"
    reg_file = "outputs/tables/regression_confirmatory.csv"
    slope_file = "outputs/tables/regression_slopes.csv"
    sec_file = "outputs/tables/regression_secondary.csv"
    sens_file = "outputs/tables/regression_sensitivity.csv"
    met_file = "outputs/tables/forecast_test_metrics.csv"
    dm_file = "outputs/tables/forecast_dm_tests.csv"
    shap_file = "outputs/tables/forecast_shap_importance.csv"
    fc_file = "outputs/tables/forecast_test_forecasts.csv"
    cp_file = "outputs/tables/eda_change_point_runs.csv"

    c.text("WINDOW", f"{window['start']} to {window['end']}", day1, "panel_window")
    c.number(
        "PANEL_ROWS", audit["dsec"]["tier_month_observations"], 0, day1, "tier_month_observations"
    )
    c.number("N_TIERS", len(window["tiers"]), 0, day1, "panel_window.tiers")
    c.number(
        "N_MONTHS", window["months_by_period"]["all_months"], 0, day1, "months_by_period.all_months"
    )
    c.number("PRE_MONTHS", window["months_by_period"]["pre_pandemic"], 0, day1, "pre_pandemic")
    c.number("PANDEMIC_MONTHS", window["months_by_period"]["pandemic"], 0, day1, "pandemic")
    c.number("POST_MONTHS", window["months_by_period"]["post_pandemic"], 0, day1, "post_pandemic")
    c.number(
        "MAIN_MONTHS",
        regression["main_model"]["months"],
        0,
        "outputs/metrics/regression_summary.json",
        "main_model.months",
    )
    c.number(
        "MAIN_ROWS",
        regression["main_model"]["n"],
        0,
        "outputs/metrics/regression_summary.json",
        "main_model.n",
    )
    c.number(
        "POST_ROWS",
        manifest["counts"]["post_pandemic_rows"],
        0,
        "data/frozen/FREEZE_MANIFEST.json",
        "post_pandemic_rows",
    )
    summary = _csv("panel_variable_summary.csv").set_index("variable")
    c.number(
        "OCC_MIN",
        summary.loc["occupancy_pct", "min"],
        1,
        "outputs/tables/panel_variable_summary.csv",
        "occupancy_pct.min",
    )
    c.number(
        "OCC_MAX",
        summary.loc["occupancy_pct", "max"],
        1,
        "outputs/tables/panel_variable_summary.csv",
        "occupancy_pct.max",
    )
    c.number("HOLIDAY_PERIODS", audit["cn_holidays"]["periods"], 0, day1, "cn_holidays.periods")
    c.number(
        "HOLIDAY_DAYS",
        audit["cn_holidays"]["holiday_days_in_panel_window"],
        0,
        day1,
        "holiday_days",
    )
    c.number(
        "TYPHOON_DAYS",
        audit["smg_typhoon_signals"]["typhoon_days_in_panel_window"],
        0,
        day1,
        "typhoon_days",
    )
    c.number(
        "EST_CHANGES",
        tier["months_with_establishment_count_change"],
        0,
        day1,
        "establishment changes",
    )
    c.number("ROOM_JUMPS", sum(jumps.values()), 0, day1, "room change of 10 percent")
    largest = tier["largest_room_change"]
    c.number("ROOMS_BEFORE", largest["rooms_before"], 0, day1, "largest_room_change.rooms_before")
    c.number("ROOMS_AFTER", largest["rooms_after"], 0, day1, "largest_room_change.rooms_after")
    c.number(
        "VAL_ORIGINS",
        manifest["counts"]["validation_origins"],
        0,
        "data/frozen/FREEZE_MANIFEST.json",
        "validation_origins",
    )
    c.number(
        "TEST_ORIGINS",
        manifest["counts"]["test_origins"],
        0,
        "data/frozen/FREEZE_MANIFEST.json",
        "test_origins",
    )
    c.number(
        "TEST_FORECASTS",
        manifest["counts"]["test_forecasts"],
        0,
        "data/frozen/FREEZE_MANIFEST.json",
        "test_forecasts",
    )
    panel_sha = manifest["files"]["data/frozen/panel_tier_month.csv"]
    c.text("PANEL_SHA", panel_sha[:12], "data/frozen/FREEZE_MANIFEST.json", "panel sha256")

    def mix_at(period: str, column: str, token: str, digits: int) -> None:
        c.number(token, mix.loc[period, column], digits, mix_file, f"{period}.{column}")

    mix_at("pre_pandemic", "mean_same_day_share_pct", "PRE_SAME", 2)
    mix_at("post_pandemic", "mean_same_day_share_pct", "POST_SAME", 2)
    mix_at("pre_pandemic", "mean_international_share_pct", "PRE_INTL", 2)
    mix_at("post_pandemic", "mean_international_share_pct", "POST_INTL", 2)
    mix_at("pre_pandemic", "mean_package_tour_share_pct", "PRE_TOUR", 2)
    mix_at("post_pandemic", "mean_package_tour_share_pct", "POST_TOUR", 2)
    mix_at("pre_pandemic", "mean_visitors_overnight", "PRE_ON", 0)
    mix_at("post_pandemic", "mean_visitors_overnight", "POST_ON", 0)
    mix_at("pandemic", "mean_international_share_pct", "PAN_INTL", 1)
    mix_at("pandemic", "mean_package_tour_share_pct", "PAN_TOUR", 1)
    c.text(
        "CHANGE_POINTS",
        change_point_sentence(persistent_change_points(runs)),
        cp_file,
        "all penalties",
    )

    def occ(tier_name: str, period: str, token: str) -> None:
        value = occupancy.loc[
            (occupancy["tier"] == tier_name) & (occupancy["period_label"] == period),
            "occupancy_mean",
        ].iloc[0]
        c.number(token, value, 1, occ_file, f"{tier_name}.{period}")

    occ("5star", "pre_pandemic", "OCC5_PRE")
    occ("5star", "post_pandemic", "OCC5_POST")
    occ("4star", "pre_pandemic", "OCC4_PRE")
    occ("4star", "post_pandemic", "OCC4_POST")
    occ("3star", "pre_pandemic", "OCC3_PRE")
    occ("3star", "post_pandemic", "OCC3_POST")
    occ("2star", "pre_pandemic", "OCC2_PRE")
    occ("2star", "post_pandemic", "OCC2_POST")
    post_peaks = stl.loc[stl["block"] == "post_pandemic", "seasonal_peak_month"]
    if not (post_peaks == 8).all():
        raise ValueError("The post-pandemic seasonal peak is not August in every class")

    c.number(
        "BANDWIDTH",
        regression["main_model"]["bandwidth"],
        0,
        "outputs/metrics/regression_summary.json",
        "bandwidth",
    )
    c.number(
        "N_COEF",
        regression["main_model"]["parameters"],
        0,
        "outputs/metrics/regression_summary.json",
        "parameters",
    )
    c.number(
        "R2",
        regression["main_model"]["r_squared"],
        3,
        "outputs/metrics/regression_summary.json",
        "r_squared",
    )

    def contrast(hypothesis: str, prefix: str) -> None:
        row = confirmatory.loc[hypothesis]
        c.number(f"{prefix}_EST", row["estimate"], 2, reg_file, f"{hypothesis}.estimate")
        c.number(f"{prefix}_SE", row["se"], 2, reg_file, f"{hypothesis}.se")
        c.number(f"{prefix}_LO", row["ci_low"], 2, reg_file, f"{hypothesis}.ci_low")
        c.number(f"{prefix}_HI", row["ci_high"], 2, reg_file, f"{hypothesis}.ci_high")
        c.p_value(f"{prefix}_HOLM", row["p_holm"], reg_file, f"{hypothesis}.p_holm")

    contrast("H1", "H1")
    contrast("H2", "H2")

    def slope(tier_name: str, prefix: str) -> None:
        row = slopes[
            (slopes["regressor"] == "log_visitors_overnight")
            & (slopes["tier"] == tier_name)
            & (slopes["quantity"] == "post_pandemic")
        ].iloc[0]
        c.number(f"{prefix}_POST", row["estimate"], 2, slope_file, f"{tier_name}.post")
        c.number(f"{prefix}_LO", row["ci_low"], 2, slope_file, f"{tier_name}.ci_low")
        c.number(f"{prefix}_HI", row["ci_high"], 2, slope_file, f"{tier_name}.ci_high")

    slope("5star", "B5")
    slope("3star", "B3")
    log_growth = math.log(1.1)
    c.number(
        "TEN_5",
        log_growth * slopes_post("5star", slopes),
        2,
        slope_file,
        "5star post times ln(1.1)",
    )
    c.number(
        "TEN_3",
        log_growth * slopes_post("3star", slopes),
        2,
        slope_file,
        "3star post times ln(1.1)",
    )

    eq, chow = (
        secondary.loc["equal_post_pandemic_slopes"],
        secondary.loc["chow_post_interactions_on_focal_regressors"],
    )
    c.number("DF_EQ", eq["df"], 0, sec_file, "equal slopes df")
    c.number("CHI_EQ", eq["chi2"], 1, sec_file, "equal slopes chi2")
    c.p_value("P_EQ", eq["p_value"], sec_file, "equal slopes p", inline=True)
    c.number("DF_CHOW", chow["df"], 0, sec_file, "chow df")
    c.number("CHI_CHOW", chow["chi2"], 1, sec_file, "chow chi2")
    c.p_value("P_CHOW", chow["p_value"], sec_file, "chow p", inline=True)
    c.p_value(
        "S5_HOLM",
        sensitivity.loc["S5_fully_interacted", "H1_p_holm"],
        sens_file,
        "S5 H1 holm",
        inline=True,
    )
    c.number("S4_EST", sensitivity.loc["S4_trends", "H2_estimate"], 2, sens_file, "S4 H2 estimate")
    c.p_value(
        "S4_HOLM", sensitivity.loc["S4_trends", "H2_p_holm"], sens_file, "S4 H2 holm", inline=True
    )
    diagnostics = regression["diagnostics_count"]
    c.number(
        "LB_LOW",
        diagnostics["ljung_box_below_0.05"],
        0,
        "outputs/metrics/regression_summary.json",
        "ljung_box_below_0.05",
    )
    c.number(
        "LB_N",
        diagnostics["ljung_box_cells"],
        0,
        "outputs/metrics/regression_summary.json",
        "ljung_box_cells",
    )
    c.p_value(
        "BP_P",
        diagnostics["breusch_pagan_p"],
        "outputs/metrics/regression_summary.json",
        "breusch_pagan_p",
        inline=True,
    )

    def pooled(run: str, model: str, column: str, token: str, digits: int = 2) -> None:
        row = metrics[
            (metrics["run"] == run) & (metrics["model"] == model) & (metrics["unit"] == "pooled")
        ]
        c.number(token, row.iloc[0][column], digits, met_file, f"{run}.{model}.{column}")

    pooled("main", "lightgbm", "mae", "MAE_LGB")
    pooled("main", "sarimax", "mae", "MAE_SAR")
    pooled("main", "seasonal_naive", "mae", "MAE_NAIVE")
    pooled("main", "lightgbm", "rmse", "RMSE_LGB")
    pooled("main", "sarimax", "rmse", "RMSE_SAR")
    pooled("main", "seasonal_naive", "rmse", "RMSE_NAIVE")
    pooled("main", "lightgbm", "mase", "MASE_LGB")
    pooled("main", "sarimax", "mase", "MASE_SAR")
    pooled("main", "seasonal_naive", "mase", "MASE_NAIVE")
    for tier_name, token in (
        ("5star", "MAE5"),
        ("4star", "MAE4"),
        ("3star", "MAE3"),
        ("2star", "MAE2"),
    ):
        row = metrics[
            (metrics["run"] == "main")
            & (metrics["model"] == "lightgbm")
            & (metrics["unit"] == tier_name)
        ]
        c.number(token, row.iloc[0]["mae"], 2, met_file, f"lightgbm.{tier_name}.mae")

    def dm(comparison: str, prefix: str) -> None:
        row = tests[
            (tests["run"] == "main")
            & (tests["comparison"] == comparison)
            & (tests["unit"] == "pooled")
        ].iloc[0]
        c.number(
            f"{prefix}_EST", row["mean_loss_difference"], 2, dm_file, f"{comparison}.difference"
        )
        c.number(f"{prefix}_LO", row["ci_low"], 2, dm_file, f"{comparison}.ci_low")
        c.number(f"{prefix}_HI", row["ci_high"], 2, dm_file, f"{comparison}.ci_high")
        c.p_value(f"{prefix}_HOLM", row["p_holm"], dm_file, f"{comparison}.p_holm")

    dm("lightgbm vs seasonal_naive", "DM_N")
    dm("lightgbm vs sarimax", "DM_S")
    pooled("F1_logit", "lightgbm", "mae", "F1_LGB")
    pooled("F1_logit", "sarimax", "mae", "F1_SAR")
    pooled("F2_pandemic_included", "lightgbm", "mae", "F2_LGB")
    pooled("F2_pandemic_included", "sarimax", "mae", "F2_SAR")
    pooled("F2_pandemic_included", "seasonal_naive", "mae", "F2_NAIVE")
    f2 = forecasts[forecasts["run"] == "F2_pandemic_included"]
    c.number(
        "F2_BIAS",
        (f2["lightgbm"] - f2["actual"]).mean(),
        2,
        fc_file,
        "F2 mean forecast minus actual",
    )
    sel_file = "outputs/metrics/forecast_selection.json"
    selection_main = _json("forecast_selection.json")["selection"]["main"]
    sar_5 = selection_main["sarimax"]["5star"]
    other_tiers = ("4star", "3star", "2star")
    other_pairs = {
        (
            tuple(selection_main["sarimax"][t]["order"]),
            tuple(selection_main["sarimax"][t]["seasonal_order"]),
        )
        for t in other_tiers
    }
    if len(other_pairs) != 1:
        raise ValueError("Expected 4star, 3star, and 2star main SARIMAX orders to match")
    sar_other = selection_main["sarimax"]["4star"]
    c.text(
        "SAR_5_ORDER",
        f"({', '.join(str(x) for x in sar_5['order'])})",
        sel_file,
        "selection.main.sarimax.5star.order",
    )
    c.text(
        "SAR_5_SEAS",
        f"({', '.join(str(x) for x in sar_5['seasonal_order'])})",
        sel_file,
        "selection.main.sarimax.5star.seasonal_order",
    )
    c.text(
        "SAR_OTHER_ORDER",
        f"({', '.join(str(x) for x in sar_other['order'])})",
        sel_file,
        "selection.main.sarimax.4star_3star_2star.order",
    )
    c.text(
        "SAR_OTHER_SEAS",
        f"({', '.join(str(x) for x in sar_other['seasonal_order'])})",
        sel_file,
        "selection.main.sarimax.4star_3star_2star.seasonal_order",
    )
    lgb_sel = selection_main["lightgbm"]
    c.number("LGB_LEAVES", lgb_sel["num_leaves"], 0, sel_file, "selection.main.lightgbm.num_leaves")
    c.number(
        "LGB_MIN_LEAF",
        lgb_sel["min_data_in_leaf"],
        0,
        sel_file,
        "selection.main.lightgbm.min_data_in_leaf",
    )
    c.number(
        "LGB_TREES", lgb_sel["n_estimators"], 0, sel_file, "selection.main.lightgbm.n_estimators"
    )
    c.number("N_FITS", len(forecasts), 0, fc_file, "all test fits")
    if not bool(forecasts["sarimax_converged"].all()):
        raise ValueError("A SARIMAX test fit did not converge")
    for index, token in enumerate(("SHAP1", "SHAP2", "SHAP3"), start=0):
        c.number(token, shap.iloc[index]["all_classes"], 2, shap_file, f"row {index} all_classes")
    return c


def slopes_post(tier_name: str, slopes: pd.DataFrame) -> float:
    row = slopes[
        (slopes["regressor"] == "log_visitors_overnight")
        & (slopes["tier"] == tier_name)
        & (slopes["quantity"] == "post_pandemic")
    ].iloc[0]
    return float(row["estimate"])


def render_tex(claims: Claims) -> str:
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    mapping = claims.mapping()
    mapping["TABLE_ONE"] = table_one(claims)
    for token, value in mapping.items():
        needle = f"__{token}__"
        if needle not in text and token != "TABLE_ONE":
            raise KeyError(f"Template is missing {needle}")
        text = text.replace(needle, value)
    leftover = sorted(set(re.findall(r"__[A-Z0-9_]+__", text)))
    if leftover:
        raise ValueError(f"Unreplaced placeholders: {leftover}")
    return text


def render_audit(claims: Claims, main_pages: float) -> str:
    bad = [claim.claim_id for claim in claims.items() if not claim.matches_source()]
    if bad:
        raise ValueError("A cited number does not match its source: " + ", ".join(bad))
    lines = [
        "# Phase A numeric audit",
        "",
        "Numeric audit: PASS.",
        "Every cited number matches its source within rounding tolerance.",
        (
            f"Main text length: {main_pages:.2f} A4 pages "
            "(cover, references, and declarations excluded)."
        ),
        "",
        "## Minimum requirements",
        "",
        "- Research question: H1, H2, and H3, with their nulls and the H3 rule.",
        "- Data: DSEC, State Council notices, and SMG, with the frozen panel counts.",
        "- Analysis: STL, PELT (exploratory only), the regression, and the forecasts.",
        "- Evaluation: Driscoll--Kraay intervals, Holm, HLN-DM, MAE, RMSE, and MASE.",
        "- Results: Figure 1, Table 1, Figure 2, and the three answers.",
        "",
        "Phase B is not estimated in this draft.",
        "",
        "## Claim trace",
        "",
        "| Claim | Reported | Source | Locator |",
        "| --- | --- | --- | --- |",
    ]
    for claim in claims.items():
        lines.append(
            f"| {claim.claim_id} | {claim.reported} | {claim.source_file} | {claim.locator} |"
        )
    lines.append("")
    return "\n".join(lines)


def compile_pdf(tex_path: Path) -> Path:
    env = os.environ.copy()
    env["SOURCE_DATE_EPOCH"] = SOURCE_DATE_EPOCH
    log_path = tex_path.with_suffix(".log")
    for _ in range(2):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if proc.returncode != 0:
            tail = (
                log_path.read_text(encoding="utf-8", errors="replace")[-2500:]
                if log_path.exists()
                else ""
            )
            raise RuntimeError(f"pdflatex failed\n{proc.stdout[-1500:]}\n{tail}")
    produced = tex_path.with_suffix(".pdf")
    final = tex_path.parent / PDF_NAME
    final.write_bytes(produced.read_bytes())
    return final


def _compact(text: str) -> str:
    text = text.replace("−", "-").replace("–", "-")
    text = re.sub(r"\s*([<=])\s*", r"\1", text)
    return re.sub(r"\s+", " ", text)


def pdf_text(pdf_path: Path) -> str:
    proc = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def main_text_pages(pdf_path: Path) -> float:
    """Pages from Introduction through Discussion, with a shared page counted once."""
    reader = PdfReader(str(pdf_path))
    texts = [(page.extract_text() or "") for page in reader.pages]
    start = next(i for i, text in enumerate(texts) if "The question is whether" in text)
    end = next(i for i, text in enumerate(texts) if "Cleveland, R." in text)
    if end <= start:
        raise ValueError("Could not locate the main text in the PDF")
    return float(end - start)


def forbidden_words(tex: str) -> list[str]:
    body = tex.split(r"\section{Introduction}")[1].split(r"\begin{thebibliography}")[0]
    return [word for word in CAUSAL_WORDS if re.search(rf"\b{word}\b", body, re.IGNORECASE)]


def uncited_bibitems(tex: str) -> list[str]:
    bib_keys = re.findall(r"\\bibitem\{([^}]+)\}", tex)
    cited: set[str] = set()
    for match in re.findall(r"\\cite\{([^}]+)\}", tex):
        cited.update(key.strip() for key in match.split(","))
    return [key for key in bib_keys if key not in cited]


def write_report(root: Path = PROJECT_ROOT) -> dict[str, object]:
    claims = build_claims()
    tex = render_tex(claims)
    bad_words = forbidden_words(tex)
    if bad_words:
        raise ValueError("Causal wording in the main text: " + ", ".join(bad_words))
    uncited = uncited_bibitems(tex)
    if uncited:
        raise ValueError("Uncited bibliography entries: " + ", ".join(uncited))
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    tex_path = report_dir / TEX_NAME
    tex_path.write_text(tex, encoding="utf-8")
    pdf_path = compile_pdf(tex_path)
    text = pdf_text(pdf_path)
    for bad_glyph in ("¡", "¿"):
        if bad_glyph in text:
            raise ValueError(f"Unexpected LaTeX encoding glyph {bad_glyph!r} in PDF")
    if "[?]" in text or "??" in text:
        raise ValueError("Unresolved citation or cross-reference found in PDF")
    compact_text = _compact(text)
    missing = [
        claim.reported for claim in claims.items() if _compact(claim.reported) not in compact_text
    ]
    if missing:
        raise ValueError("PDF is missing cited text: " + "; ".join(missing[:8]))
    pages = main_text_pages(pdf_path)
    if not MAIN_PAGE_MIN <= pages <= MAIN_PAGE_MAX:
        raise ValueError(f"Main text is {pages:.2f} pages; the draft must be between 3 and 5")
    for name in (
        "Introduction",
        "Data Collection and Preparation",
        "Methods",
        "Results",
        "Discussion",
    ):
        if name not in text:
            raise ValueError(f"PDF is missing the section {name}")
    audit = render_audit(claims, pages)
    (report_dir / AUDIT_NAME).write_text(audit, encoding="utf-8")
    for suffix in (".aux", ".log", ".out", ".toc"):
        side = tex_path.with_suffix(suffix)
        if side.exists() and side.name != PDF_NAME:
            side.unlink()
    draft_pdf = tex_path.with_suffix(".pdf")
    if draft_pdf.exists() and draft_pdf.resolve() != pdf_path.resolve():
        draft_pdf.unlink()
    return {
        "pdf": str(pdf_path.relative_to(root)),
        "main_pages": pages,
        "claims": len(claims.items()),
    }
