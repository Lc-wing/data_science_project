from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macao_tourism import eda, plots
from macao_tourism.config import DATA_FROZEN_DIR, OUTPUTS_METRICS_DIR, OUTPUTS_TABLES_DIR, TIERS
from macao_tourism.freeze import Frozen, load_frozen, load_manifest

PERIOD = 12


@pytest.fixture(scope="module")
def frozen() -> Frozen:
    return load_frozen()


@pytest.fixture(scope="module")
def series(frozen: Frozen) -> list[str]:
    return list(frozen.spec["eda"]["change_points"]["series"])


@pytest.fixture(scope="module")
def components(frozen: Frozen) -> pd.DataFrame:
    return eda.stl_components(frozen.panel, PERIOD, robust=True)


@pytest.fixture(scope="module")
def runs(frozen: Frozen, series: list[str]) -> pd.DataFrame:
    return eda.change_point_runs(frozen.months, series, PERIOD, robust=True)


def test_load_frozen_gives_the_frozen_tables_and_spec(frozen: Frozen) -> None:
    assert frozen.panel.shape == (804, 29)
    assert frozen.months.shape == (201, 23)
    assert len(frozen.origins) == 62
    assert frozen.spec["eda"]["stl"] == {
        "series": "occupancy_pct per star class",
        "period": 12,
        "robust": True,
    }
    assert frozen.spec_sha256 == load_manifest()["files"]["data/frozen/analysis_spec.json"]


def test_load_frozen_refuses_a_changed_freeze(tmp_path: Path) -> None:
    copy = tmp_path / "frozen"
    shutil.copytree(DATA_FROZEN_DIR, copy)
    months = copy / "months.csv"
    months.write_text(months.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch: months.csv"):
        load_frozen(copy)


def test_periodic_stl_recovers_a_known_seasonal_pattern() -> None:
    t = np.arange(120)
    values = 80 + 0.05 * t + 5 * np.sin(2 * np.pi * t / 12)
    fit = eda.stl_fit(values, PERIOD, robust=True, periodic=True)
    np.testing.assert_allclose(fit.seasonal[:12], fit.seasonal[12:24], atol=1e-9)
    assert fit.seasonal.max() - fit.seasonal.min() == pytest.approx(10, abs=0.2)
    assert np.abs(fit.resid).max() < 0.2


def test_stl_components_cover_the_two_blocks_and_add_up(components: pd.DataFrame) -> None:
    sizes = components.groupby(["tier", "block"]).size().to_dict()
    assert sizes == {
        (tier, block): months
        for tier in TIERS
        for block, months in (("pre_pandemic", 120), ("post_pandemic", 45))
    }
    assert not components["period"].between("2020-01", "2022-12").any()
    total = components["trend"] + components["seasonal"] + components["remainder"]
    np.testing.assert_allclose(total, components["occupancy_pct"], atol=1e-9)
    spread = components.groupby(["tier", "block", "month"])["seasonal"].agg(np.ptp)
    assert spread.max() < 1e-9


def test_stl_summary_matches_its_definition(components: pd.DataFrame) -> None:
    summary = eda.stl_summary(components)
    assert len(summary) == 8
    assert summary[["trend_strength", "seasonal_strength"]].stack().between(0, 1).all()
    assert summary[["seasonal_peak_month", "seasonal_trough_month"]].stack().between(1, 12).all()
    part = components[(components["tier"] == "5star") & (components["block"] == "pre_pandemic")]
    expected = 1 - part["remainder"].var() / (part["seasonal"] + part["remainder"]).var()
    row = summary[(summary["tier"] == "5star") & (summary["block"] == "pre_pandemic")].iloc[0]
    assert row["seasonal_strength"] == pytest.approx(expected, abs=1e-6)
    profile = part.groupby("month")["seasonal"].mean()
    assert row["seasonal_range_pp"] == pytest.approx(profile.max() - profile.min(), abs=1e-6)
    profiles = eda.seasonal_profiles(components)
    assert len(profiles) == 4 * 2 * 12


def test_full_series_fit_distorts_the_months_after_the_pandemic(frozen: Frozen) -> None:
    check = eda.stl_full_series_check(frozen.panel, PERIOD, robust=True).set_index("tier")
    five = check.loc["5star"]
    assert five["month_of_post_pandemic_max"] == "2023-08"
    assert five["occupancy_pct_that_month"] == 90.6
    assert five["seasonal_that_month_pp"] < -30
    assert five["max_abs_seasonal_post_pandemic_pp"] > 4 * five["max_abs_seasonal_pre_pandemic_pp"]


def test_noise_sd_ignores_a_repeating_pattern_and_recovers_a_known_scale() -> None:
    pattern = np.tile(np.arange(12, dtype=float), 10)
    assert eda.noise_sd(pattern, PERIOD) == 0
    rng = np.random.default_rng(0)
    noisy = pattern + rng.normal(0, 2, 120)
    assert eda.noise_sd(noisy, PERIOD) == pytest.approx(2, rel=0.25)
    assert eda.pelt_penalty(noisy, PERIOD, 2.0) == pytest.approx(
        2 * 2 * eda.noise_sd(noisy, PERIOD) ** 2 * math.log(120)
    )


def test_pelt_finds_a_step_and_nothing_in_noise() -> None:
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, 120)
    step = noise + np.where(np.arange(120) >= 70, 6.0, 0.0)
    assert eda.pelt_breaks(step, eda.pelt_penalty(step, PERIOD, 1.0)) == [70]
    assert eda.pelt_breaks(noise, eda.pelt_penalty(noise, PERIOD, 1.0)) == []


def test_change_point_runs_have_one_main_run_per_series(
    runs: pd.DataFrame, series: list[str], frozen: Frozen
) -> None:
    assert len(runs) == len(series) * (len(eda.PELT_PENALTY_MULTIPLIERS) + 1)
    main = runs[runs["main"]]
    assert main["series"].tolist() == series
    assert (main["input"] == "seasonally_adjusted").all()
    assert (main["penalty_multiplier"] == eda.PELT_MAIN_MULTIPLIER).all()
    months = set(frozen.months["period"])
    for run in runs.itertuples():
        points = [m for m in run.change_points.split(";") if m]
        assert len(points) == run.n_change_points
        assert set(points) <= months
        assert points == sorted(points)
    adjusted = runs[runs["input"] == "seasonally_adjusted"]
    for _, part in adjusted.groupby("series"):
        counts = part.sort_values("penalty_multiplier")["n_change_points"]
        assert counts.is_monotonic_decreasing


def test_segments_partition_the_window(runs: pd.DataFrame, frozen: Frozen) -> None:
    segments = eda.segments(frozen.months, runs)
    for name, part in segments.groupby("series"):
        assert part["months"].sum() == 201
        assert part["months"].min() >= eda.PELT_MIN_SEGMENT_MONTHS
        assert part["first_month"].iloc[0] == "2010-01"
        assert part["last_month"].iloc[-1] == "2026-09"
        assert (part["first_month"].iloc[1:].to_numpy() > part["last_month"].iloc[:-1]).all()
        first = part.iloc[0]
        inside = frozen.months[frozen.months["period"].between("2010-01", first["last_month"])]
        assert first["mean_share_pct"] == pytest.approx(inside[name].mean(), abs=1e-6)


def test_period_summaries_match_direct_computation(frozen: Frozen, series: list[str]) -> None:
    mix = eda.visitor_mix_by_period(frozen.months, series).set_index("period_label")
    assert mix["months"].to_dict() == {"pre_pandemic": 120, "pandemic": 35, "post_pandemic": 45}
    pre = frozen.months[frozen.months["period_label"] == "pre_pandemic"]
    assert mix.loc["pre_pandemic", "mean_same_day_share_pct"] == pytest.approx(
        pre["same_day_share_pct"].mean(), abs=1e-6
    )

    occupancy = eda.occupancy_by_period(frozen.panel)
    assert len(occupancy) == 12
    assert occupancy["pearson_r_log_overnight"].between(-1, 1).all()
    rows = frozen.panel[
        (frozen.panel["tier"] == "3star") & (frozen.panel["period_label"] == "post_pandemic")
    ]
    cell = occupancy[
        (occupancy["tier"] == "3star") & (occupancy["period_label"] == "post_pandemic")
    ].iloc[0]
    expected = np.corrcoef(rows["occupancy_pct"], rows["log_visitors_overnight"])[0, 1]
    assert cell["pearson_r_log_overnight"] == pytest.approx(expected, abs=1e-6)
    assert cell["occupancy_mean"] == pytest.approx(rows["occupancy_pct"].mean(), abs=1e-6)
    assert "transition_2020_01" not in set(occupancy["period_label"])


def _assert_same(computed: pd.DataFrame, name: str) -> None:
    committed = pd.read_csv(OUTPUTS_TABLES_DIR / name, keep_default_na=False)
    computed = computed.reset_index(drop=True)
    assert list(committed.columns) == list(computed.columns), name
    assert len(committed) == len(computed), name
    for column in computed.columns:
        if pd.api.types.is_float_dtype(computed[column]):
            np.testing.assert_allclose(
                committed[column].astype(float), computed[column], atol=1e-4, err_msg=name
            )
        else:
            assert (
                committed[column].astype(str).tolist() == computed[column].astype(str).tolist()
            ), f"{name}: {column}"


def test_committed_tables_are_what_the_frozen_panel_gives(
    frozen: Frozen, series: list[str], components: pd.DataFrame, runs: pd.DataFrame
) -> None:
    _assert_same(components.round(eda.DECIMALS), "eda_stl_components.csv")
    _assert_same(eda.stl_summary(components), "eda_stl_summary.csv")
    _assert_same(eda.seasonal_profiles(components), "eda_stl_seasonal_profiles.csv")
    _assert_same(
        eda.stl_full_series_check(frozen.panel, PERIOD, robust=True),
        "eda_stl_full_series_check.csv",
    )
    _assert_same(runs, "eda_change_point_runs.csv")
    _assert_same(eda.segments(frozen.months, runs), "eda_change_point_segments.csv")
    _assert_same(eda.visitor_mix_by_period(frozen.months, series), "eda_visitor_mix_by_period.csv")
    _assert_same(eda.occupancy_by_period(frozen.panel), "eda_occupancy_by_period.csv")


def test_committed_summary_names_the_freeze_it_was_computed_from(runs: pd.DataFrame) -> None:
    summary = json.loads((OUTPUTS_METRICS_DIR / "eda_summary.json").read_text(encoding="utf-8"))
    manifest = load_manifest()
    assert summary["panel_sha256"] == manifest["files"]["data/frozen/panel_tier_month.csv"]
    assert summary["analysis_spec_sha256"] == manifest["files"]["data/frozen/analysis_spec.json"]
    assert summary["deviations_from_spec"] == []
    assert len(summary["settings"]["changed_after_the_first_run"]) == 2
    main = runs[runs["main"]].set_index("series")["change_points"]
    assert summary["change_points_main"] == {
        name: points.split(";") for name, points in main.items()
    }


def test_figures_are_written(
    frozen: Frozen, components: pd.DataFrame, runs: pd.DataFrame, tmp_path: Path
) -> None:
    segments = eda.segments(frozen.months, runs)
    figures = {
        "one": plots.figure_1(frozen.months, frozen.panel, segments),
        "stl": plots.figure_stl(components, eda.seasonal_profiles(components)),
        "scatter": plots.figure_scatter(frozen.panel),
    }
    assert len(figures["one"].axes) == 4
    assert len(figures["stl"].axes) == 12
    assert len(figures["scatter"].axes) == 4
    for stem, figure in figures.items():
        written = plots.save(figure, tmp_path, stem)
        assert [path.suffix for path in written] == [".png", ".pdf"]
        assert written[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert written[1].read_bytes()[:5] == b"%PDF-"
        assert all(path.stat().st_size > 5000 for path in written)
