"""Phase A step 3: exploratory description of the frozen panel.

    uv run python scripts/03_eda.py

Reads data/frozen only, after checking it against FREEZE_MANIFEST.json. Writes the STL
decomposition of occupancy, the PELT change points of the visitor-mix shares, the period
summaries, and three figures. Nothing written here is an input to steps 4 and 5.
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from macao_tourism import eda, plots
from macao_tourism.config import (
    OUTPUTS_FIGURES_DIR,
    OUTPUTS_METRICS_DIR,
    OUTPUTS_TABLES_DIR,
    PROJECT_ROOT,
    TIERS,
)
from macao_tourism.freeze import load_frozen
from macao_tourism.tables import markdown_table, write_csv

FIGURE_1 = "fig1_visitor_mix_and_occupancy"
FIGURE_STL = "eda_stl_occupancy"
FIGURE_SCATTER = "eda_scatter_occupancy_overnight"


def render_markdown(summary: dict, tables: dict[str, pd.DataFrame]) -> str:
    stl, pelt = summary["settings"]["stl_occupancy"], summary["settings"]["pelt"]
    sensitivity = tables["runs"][
        ["series", "input", "penalty_multiplier", "n_change_points", "change_points"]
    ]
    changed = "\n".join(f"- {item}" for item in summary["settings"]["changed_after_the_first_run"])
    return "\n\n".join(
        [
            "# Exploratory data analysis (Plan 2, Phase A step 3)",
            "Exploratory description only. Nothing here is a test of H1, H2, or H3, and "
            "nothing here is an input to the regression or the forecasts. The period "
            "boundaries are those of the plan; they were not taken from the change points.",
            f"Frozen panel SHA-256: `{summary['panel_sha256']}`  \n"
            f"Analysis specification SHA-256: `{summary['analysis_spec_sha256']}`",
            "## Settings",
            f"- STL on occupancy: period {stl['period']}, robust, {stl['fits']}; seasonal "
            f"pattern {stl['seasonal']}. Pandemic months are {stl['pandemic_months']}.\n"
            f"- PELT input: {pelt['input']}. Cost {pelt['cost']}, segments of at least "
            f"{pelt['min_segment_months']} months, penalty = {pelt['penalty']}, "
            f"noise_sd = {pelt['noise_sd']}. Main multiplier {pelt['main_multiplier']}.\n"
            f"- {summary['settings']['left_open_by_the_specification']}",
            "### Changed after the first run",
            changed,
            "## Visitor mix by period (means of monthly values)",
            markdown_table(tables["mix"]),
            "## Change points of the visitor-mix shares (main run)",
            "A change point is the first month of a new segment. `mean_share_pct` is the mean "
            "of the published share over the segment.",
            markdown_table(tables["segments"]),
            "### Sensitivity to the penalty and to the seasonal adjustment",
            markdown_table(sensitivity),
            "## STL decomposition of occupancy",
            "Strength = 1 − Var(remainder) / Var(component + remainder). The post-pandemic "
            "block has 45 months, under four yearly cycles, so its seasonal profile rests on "
            "three or four values per calendar month.",
            markdown_table(tables["stl"], decimals=3),
            "### One fit over all months (first run, kept as a check)",
            markdown_table(tables["full_series"]),
            "## Occupancy by star class and period",
            "`pearson_r_log_overnight` describes the scatter plot. No test is attached to it; "
            "it is a same-month correlation of two series that share a seasonal pattern.",
            markdown_table(tables["occupancy"]),
            "## Figures",
            f"- `outputs/figures/{FIGURE_1}.png`: visitor-mix shares with change points, and "
            "occupancy by star class (Figure 1).\n"
            f"- `outputs/figures/{FIGURE_STL}.png`: trend, seasonal profile, and remainder.\n"
            f"- `outputs/figures/{FIGURE_SCATTER}.png`: occupancy against log overnight "
            "visitors by star class and period.",
        ]
    )


def main() -> int:
    frozen = load_frozen()
    spec = frozen.spec["eda"]
    period, robust = int(spec["stl"]["period"]), bool(spec["stl"]["robust"])
    series = list(spec["change_points"]["series"])

    components = eda.stl_components(frozen.panel, period, robust)
    stl = eda.stl_summary(components)
    profiles = eda.seasonal_profiles(components)
    full_series = eda.stl_full_series_check(frozen.panel, period, robust)
    runs = eda.change_point_runs(frozen.months, series, period, robust)
    segments = eda.segments(frozen.months, runs)
    mix = eda.visitor_mix_by_period(frozen.months, series)
    occupancy = eda.occupancy_by_period(frozen.panel)

    write_csv(components.round(eda.DECIMALS), "eda_stl_components.csv")
    write_csv(stl, "eda_stl_summary.csv")
    write_csv(profiles, "eda_stl_seasonal_profiles.csv")
    write_csv(full_series, "eda_stl_full_series_check.csv")
    write_csv(runs, "eda_change_point_runs.csv")
    write_csv(segments, "eda_change_point_segments.csv")
    write_csv(mix, "eda_visitor_mix_by_period.csv")
    write_csv(occupancy, "eda_occupancy_by_period.csv")

    figures = []
    figures += plots.save(
        plots.figure_1(frozen.months, frozen.panel, segments), OUTPUTS_FIGURES_DIR, FIGURE_1
    )
    figures += plots.save(plots.figure_stl(components, profiles), OUTPUTS_FIGURES_DIR, FIGURE_STL)
    figures += plots.save(plots.figure_scatter(frozen.panel), OUTPUTS_FIGURES_DIR, FIGURE_SCATTER)

    tables = {
        "stl": stl,
        "full_series": full_series,
        "runs": runs,
        "segments": segments,
        "mix": mix,
        "occupancy": occupancy,
    }
    main_runs = runs[runs["main"]].set_index("series")
    adjusted_runs = runs[runs["input"] == "seasonally_adjusted"]
    unadjusted = runs[runs["input"] == "unadjusted"].set_index("series")
    summary = {
        "schema_version": "1.0.0",
        "step": "Phase A step 3: exploratory data analysis",
        "role": spec["role"],
        "frozen_at_utc8": frozen.manifest["frozen_at_utc8"],
        "panel_sha256": frozen.panel_sha256,
        "analysis_spec_sha256": frozen.spec_sha256,
        "spec_version": frozen.spec["spec_version"],
        "settings": eda.settings(period, robust),
        "deviations_from_spec": [],
        "change_points_main": {
            name: [m for m in str(main_runs.loc[name, "change_points"]).split(";") if m]
            for name in series
        },
        "change_points_unadjusted_input": {
            name: [m for m in str(unadjusted.loc[name, "change_points"]).split(";") if m]
            for name in series
        },
        "change_point_counts_by_multiplier": {
            name: {
                str(row.penalty_multiplier): int(row.n_change_points)
                for row in adjusted_runs[adjusted_runs["series"] == name].itertuples()
            }
            for name in series
        },
        "visitor_mix_by_period": mix.to_dict("records"),
        "stl_summary": stl.to_dict("records"),
        "stl_full_series_check": full_series.to_dict("records"),
        "occupancy_by_period": occupancy.to_dict("records"),
        "tiers": list(TIERS),
        "figures": [str(path.relative_to(PROJECT_ROOT)) for path in figures],
    }
    OUTPUTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_METRICS_DIR / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUTS_TABLES_DIR / "eda_summary.md").write_text(
        render_markdown(summary, tables) + "\n", encoding="utf-8"
    )

    print(markdown_table(mix))
    print(markdown_table(segments))
    print(markdown_table(runs[["series", "input", "penalty_multiplier", "change_points"]]))
    print(markdown_table(full_series))
    print(markdown_table(stl, decimals=3))
    print(markdown_table(occupancy))
    return 0


if __name__ == "__main__":
    sys.exit(main())
