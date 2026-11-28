from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats import sandwich_covariance as sw
from statsmodels.stats.multitest import multipletests

from macao_tourism import regression as rg
from macao_tourism.config import OUTPUTS_METRICS_DIR, OUTPUTS_TABLES_DIR, TIERS
from macao_tourism.freeze import Frozen, load_frozen, load_manifest


@pytest.fixture(scope="module")
def frozen() -> Frozen:
    return load_frozen()


@pytest.fixture(scope="module")
def controls(frozen: Frozen) -> list[str]:
    return list(frozen.spec["regression_h1_h2"]["terms"]["controls"])


@pytest.fixture(scope="module")
def design(frozen: Frozen, controls: list[str]) -> rg.Design:
    return rg.build_design(frozen.panel, controls)


def synthetic_outcome(design: rg.Design, beta: np.ndarray, noise_sd: float) -> rg.Design:
    """The real regressors with an invented outcome, to test the machinery only."""
    rng = np.random.default_rng(7)
    y = design.x @ beta + rng.normal(0, noise_sd, len(design.y))
    return rg.Design(y, design.x, design.names, design.rows, design.outcome)


def test_bandwidth_rule_and_month_index() -> None:
    assert [rg.bandwidth_rule(t) for t in (50, 100, 165, 201)] == [3, 4, 4, 4]
    index = rg.month_index(pd.Series(["2010-01", "2010-12", "2019-12", "2023-01", "2026-09"]))
    assert index.tolist() == [0, 11, 119, 156, 200]


def test_main_design_is_the_specified_model(design: rg.Design, controls: list[str]) -> None:
    assert design.x.shape == (660, 34)
    assert design.rows["period"].nunique() == 165
    assert set(design.rows["period_label"]) == {"pre_pandemic", "post_pandemic"}
    expected = (
        [f"a[{t}]" for t in TIERS]
        + [f"a_post[{t}]" for t in TIERS]
        + [f"month[{m}]" for m in range(2, 13)]
        + [f"b[{t}]" for t in TIERS]
        + [f"b_post[{t}]" for t in TIERS]
        + ["c", "c_post"]
        + controls
    )
    assert list(design.names) == expected
    rows = design.rows
    first_post_5star = rows[(rows["tier"] == "5star") & (rows["period"] == "2023-01")].index[0]
    values = dict(zip(design.names, design.x[first_post_5star], strict=True))
    assert values["a[5star]"] == 1 and values["a_post[5star]"] == 1 and values["a[4star]"] == 0
    assert (
        values["b[5star]"]
        == values["b_post[5star]"]
        == rows.loc[first_post_5star, "log_visitors_overnight"]
    )
    assert values["b_post[3star]"] == 0
    assert values["c_post"] == rows.loc[first_post_5star, "same_day_share_pct"]
    assert values["month[2]"] == 0
    np.testing.assert_array_equal(design.y, rows["occupancy_pct"].to_numpy())


def test_sensitivity_designs(frozen: Frozen, controls: list[str]) -> None:
    panel = frozen.panel
    with_pandemic = rg.build_design(panel, controls, include_pandemic=True)
    assert with_pandemic.x.shape == (804, 34 + 4 + 4 + 1)
    flagged = with_pandemic.x[:, with_pandemic.names.index("c_pand")] != 0
    assert set(with_pandemic.rows.loc[flagged, "period"]) == set(
        pd.period_range("2020-01", "2022-12", freq="M").astype(str)
    )
    assert rg.build_design(panel, controls, trends=True).x.shape == (660, 38)
    interacted = rg.build_design(panel, controls, fully_interacted=True)
    assert interacted.x.shape == (660, 34 + 11 + 5)
    logit = rg.build_design(panel, controls, outcome="logit_occupancy")
    occupancy = logit.rows["occupancy_pct"].to_numpy()
    np.testing.assert_allclose(logit.y, np.log(occupancy / (100 - occupancy)))
    with pytest.raises(ValueError, match="Unknown outcome"):
        rg.build_design(panel, controls, outcome="rooms")
    with pytest.raises(ValueError, match="full column rank"):
        rg.build_design(panel, [*controls, "post"])


def test_driscoll_kraay_matches_statsmodels_on_months_without_a_gap() -> None:
    rng = np.random.default_rng(3)
    months, units, k = 60, 4, 5
    time = np.repeat(np.arange(months), units)
    x = np.column_stack([np.ones(months * units), rng.normal(size=(months * units, k - 1))])
    common = np.repeat(rng.normal(size=months), units)
    y = x @ np.arange(1, k + 1) + common + rng.normal(size=months * units)
    resid = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    for bandwidth in (0, 2, 4):
        ours = rg.driscoll_kraay_scores(x, resid, time, bandwidth)
        theirs = sw.S_hac_groupsum(x * resid[:, None], time, nlags=bandwidth)
        np.testing.assert_allclose(ours, theirs, rtol=1e-10)
    reference = sm.OLS(y, x).fit(
        cov_type="hac-groupsum", cov_kwds={"time": time, "maxlags": 4, "use_correction": "hac"}
    )
    rows = pd.DataFrame(
        {"period": pd.period_range("2010-01", periods=months, freq="M").astype(str)}
    )
    rows = rows.loc[rows.index.repeat(units)].reset_index(drop=True)
    result = rg.fit(rg.Design(y, x, tuple(f"x{i}" for i in range(k)), rows, "occupancy_pct"), 4)
    np.testing.assert_allclose(result.beta, reference.params, rtol=1e-9)
    np.testing.assert_allclose(np.sqrt(np.diag(result.cov)), reference.bse, rtol=1e-9)


def test_no_autocovariance_is_taken_across_a_gap() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(80, 3))
    resid = rng.normal(size=80)
    before, after = np.repeat(np.arange(10), 4), np.repeat(np.arange(30, 40), 4)
    time = np.concatenate([before, after])
    whole = rg.driscoll_kraay_scores(x, resid, time, 4)
    parts = rg.driscoll_kraay_scores(x[:40], resid[:40], before, 4) + rg.driscoll_kraay_scores(
        x[40:], resid[40:], after, 4
    )
    np.testing.assert_allclose(whole, parts, rtol=1e-12)
    adjacent = np.concatenate([before, np.repeat(np.arange(10, 20), 4)])
    assert not np.allclose(rg.driscoll_kraay_scores(x, resid, adjacent, 4), parts)
    assert np.linalg.eigvalsh(whole).min() > -1e-9


def test_holm_matches_statsmodels() -> None:
    assert rg.holm([0.01, 0.04]) == pytest.approx([0.02, 0.04])
    assert rg.holm([0.04, 0.03]) == pytest.approx([0.06, 0.06])
    p_values = [0.2, 0.001, 0.03, 0.04, 0.5]
    assert rg.holm(p_values) == pytest.approx(multipletests(p_values, method="holm")[1])


def test_fit_recovers_known_coefficients_and_contrasts(design: rg.Design) -> None:
    rng = np.random.default_rng(11)
    beta = rng.normal(size=len(design.names))
    names = design.names
    result = rg.fit(synthetic_outcome(design, beta, noise_sd=1e-6))
    np.testing.assert_allclose(result.beta, beta, atol=1e-4)
    assert result.bandwidth == 4
    assert (result.n, result.k, result.n_months) == (660, 34, 165)
    assert result.r_squared > 0.999999

    def true(name: str) -> float:
        return beta[names.index(name)]

    h1 = rg.contrast(result, rg.h1_terms())
    expected = true("b[5star]") + true("b_post[5star]") - true("b[3star]") - true("b_post[3star]")
    assert h1["estimate"] == pytest.approx(expected, abs=1e-4)
    assert rg.contrast(result, rg.h2_terms())["estimate"] == pytest.approx(true("c_post"), abs=1e-4)
    slopes = rg.slopes_by_class(result).set_index(["regressor", "tier", "quantity"])["estimate"]
    assert slopes[(rg.FOCAL, "2star", "post_pandemic")] == pytest.approx(
        true("b[2star]") + true("b_post[2star]"), abs=1e-4
    )
    assert slopes[(rg.SHARE, "all", "post_pandemic")] == pytest.approx(
        true("c") + true("c_post"), abs=1e-4
    )


def test_tests_hold_their_level_shape_under_a_true_null(design: rg.Design) -> None:
    beta = np.zeros(len(design.names))
    beta[design.names.index("a[5star]")] = 80
    result = rg.fit(synthetic_outcome(design, beta, noise_sd=3.0))
    tests = rg.confirmatory_tests(result).set_index("hypothesis")
    assert list(tests.index) == ["H1", "H2"]
    for row in tests.itertuples():
        assert row.ci_low < row.estimate < row.ci_high
        assert row.z == pytest.approx(row.estimate / row.se)
        assert row.p_holm >= row.p_value
        assert (row.ci_high - row.ci_low) == pytest.approx(2 * 1.959964 * row.se, rel=1e-6)
    assert tests["p_holm"].min() == pytest.approx(2 * tests["p_value"].min())
    single = rg.wald(result, [rg.h2_terms()])
    assert single["chi2"] == pytest.approx(tests.loc["H2", "z"] ** 2)
    assert single["p_value"] == pytest.approx(tests.loc["H2", "p_value"])
    secondary = rg.secondary_tests(result).set_index("test")
    assert secondary["df"].to_dict() == {
        "equal_post_pandemic_slopes": 3,
        "chow_post_interactions_on_focal_regressors": 5,
    }


def test_wider_bandwidth_changes_only_the_standard_errors(design: rg.Design) -> None:
    beta = np.zeros(len(design.names))
    synthetic = synthetic_outcome(design, beta, noise_sd=2.0)
    narrow, wide = rg.fit(synthetic, 2), rg.fit(synthetic, 8)
    np.testing.assert_array_equal(narrow.beta, wide.beta)
    assert not np.allclose(narrow.cov, wide.cov)
    assert (narrow.bandwidth, wide.bandwidth) == (2, 8)


def test_diagnostics_have_the_reported_shape(design: rg.Design) -> None:
    result = rg.fit(synthetic_outcome(design, np.zeros(len(design.names)), noise_sd=2.0))
    table = rg.diagnostics(result)
    ljung_box = table[table["test"] == "ljung_box"]
    assert len(ljung_box) == 4 * 2 * 2
    assert set(ljung_box["lag"]) == {12, 24}
    assert ljung_box.groupby("block")["n"].first().to_dict() == {
        "pre_pandemic": 120,
        "post_pandemic": 45,
    }
    assert (table["test"] == "breusch_pagan").sum() == 1
    assert table["p_value"].between(0, 1).all()
    residuals = rg.residual_table(result)
    np.testing.assert_allclose(residuals["observed"] - residuals["fitted"], residuals["residual"])


def _assert_same(computed: pd.DataFrame, name: str) -> None:
    committed = pd.read_csv(OUTPUTS_TABLES_DIR / name)
    assert list(committed.columns) == list(computed.columns), name
    assert len(committed) == len(computed), name
    for column in computed.columns:
        if pd.api.types.is_float_dtype(computed[column]):
            np.testing.assert_allclose(
                committed[column], computed[column], rtol=1e-5, atol=1e-5, err_msg=name
            )
        else:
            assert committed[column].tolist() == computed[column].tolist(), f"{name}: {column}"


def test_committed_tables_are_what_the_frozen_panel_gives(
    frozen: Frozen, controls: list[str], design: rg.Design
) -> None:
    result = rg.fit(design)
    _assert_same(rg.coefficient_table(result), "regression_coefficients.csv")
    _assert_same(rg.confirmatory_tests(result), "regression_confirmatory.csv")
    _assert_same(rg.secondary_tests(result), "regression_secondary.csv")
    _assert_same(rg.slopes_by_class(result), "regression_slopes.csv")
    _assert_same(rg.diagnostics(result), "regression_diagnostics.csv")
    _assert_same(rg.residual_table(result).reset_index(drop=True), "regression_residuals.csv")

    committed = pd.read_csv(OUTPUTS_TABLES_DIR / "regression_sensitivity.csv").set_index("run")
    assert list(committed.index) == [
        "main",
        "S1_pandemic_included",
        "S2_logit_outcome",
        "S3_bandwidth_2",
        "S3_bandwidth_8",
        "S4_trends",
        "S5_fully_interacted",
    ]
    variants = {
        "S1_pandemic_included": rg.fit(
            rg.build_design(frozen.panel, controls, include_pandemic=True)
        ),
        "S2_logit_outcome": rg.fit(
            rg.build_design(frozen.panel, controls, outcome="logit_occupancy")
        ),
        "S3_bandwidth_8": rg.fit(design, 8),
        "S5_fully_interacted": rg.fit(
            rg.build_design(frozen.panel, controls, fully_interacted=True)
        ),
    }
    for name, variant in variants.items():
        row = rg.sensitivity_row(name, "", variant)
        for column in ("n", "months", "parameters", "bandwidth"):
            assert committed.loc[name, column] == row[column], (name, column)
        for column in ("H1_estimate", "H1_se", "H1_p_holm", "H2_estimate", "H2_se", "H2_p_holm"):
            assert committed.loc[name, column] == pytest.approx(row[column], rel=1e-5, abs=1e-5), (
                name,
                column,
            )
    main = committed.loc["main"]
    tests = rg.confirmatory_tests(result).set_index("hypothesis")
    assert main["H1_estimate"] == pytest.approx(tests.loc["H1", "estimate"], rel=1e-5)
    assert main["H2_p_holm"] == pytest.approx(tests.loc["H2", "p_holm"], rel=1e-5)


def test_committed_summary_follows_the_frozen_specification(frozen: Frozen) -> None:
    summary = json.loads(
        (OUTPUTS_METRICS_DIR / "regression_summary.json").read_text(encoding="utf-8")
    )
    manifest = load_manifest()
    assert summary["panel_sha256"] == manifest["files"]["data/frozen/panel_tier_month.csv"]
    assert summary["analysis_spec_sha256"] == manifest["files"]["data/frozen/analysis_spec.json"]
    assert summary["specification"] == frozen.spec["regression_h1_h2"]
    assert summary["deviations_from_spec"] == []
    assert summary["main_model"]["bandwidth"] == rg.bandwidth_rule(165) == 4
    assert [test["hypothesis"] for test in summary["confirmatory_tests"]] == ["H1", "H2"]
    spec = frozen.spec["regression_h1_h2"]
    assert spec["sensitivity_analyses"]["S3_bandwidth"].endswith("bandwidth 2 and 8")
    assert rg.SENSITIVITY_BANDWIDTHS == (2, 8)
    assert spec["confirmatory_tests"]["H1"]["null"] == (
        "(b[5star] + b_post[5star]) − (b[3star] + b_post[3star]) = 0"
    )
    assert rg.h1_terms() == {
        "b[5star]": 1.0,
        "b_post[5star]": 1.0,
        "b[3star]": -1.0,
        "b_post[3star]": -1.0,
    }
    assert spec["confirmatory_tests"]["H2"]["null"] == "c_post = 0"
    for name, runs in summary["robustness"].items():
        assert len(runs["rejected"]) + len(runs["not_rejected"]) == 7, name
