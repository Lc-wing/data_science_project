"""M4 contract checks: frozen cells, Table 1 copies, and the numeric audit."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
E2E_DIR = ROOT / "tests" / "e2e"
for _path in (ROOT, E2E_DIR):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from e2e_lib.oracle import sentence_aware_unhedged_causal_hits  # noqa: E402
from e2e_lib.readers import inspect_pdf  # noqa: E402

from macao_equity.freeze_bundle import (  # noqa: E402
    BASE_R9_COLUMNS,
    append_analysis_columns,
    build_manifest_payload,
)
from macao_equity.viz.report import build_prose_claims  # noqa: E402
from macao_equity.viz.tables import (  # noqa: E402
    build_table1_rows,
    parse_reported,
)

ACC_SUFFIXES = (
    "park",
    "health_centre",
    "library",
    "bus_stop",
    "ev_charger",
    "composite",
    "ev_unweighted",
)
REQUIRED_FIGURES = (
    "fig1_accessibility_lisa.png",
    "eda_facilities_pop.png",
    "eda_accessibility_lorenz.png",
    "maup_r8_vs_r9.png",
)


def _decimals(text: str) -> list[float]:
    return [float(match.group(0)) for match in re.finditer(r"-?\d+\.\d+", text)]


def test_append_keeps_m1_values_and_adds_m2_columns() -> None:
    base = gpd.read_file(ROOT / "data/frozen/cells_base_res9.gpkg")
    analysis = gpd.read_file(ROOT / "data/interim/cells_r9_with_2sfca_lisa.gpkg")
    merged = append_analysis_columns(base, analysis, required_base_columns=BASE_R9_COLUMNS)
    assert list(merged["cell_id"]) == list(base["cell_id"])
    for column in BASE_R9_COLUMNS:
        if pd.api.types.is_numeric_dtype(base[column]):
            delta = (merged[column].astype(float) - base[column].astype(float)).abs().max()
            assert delta <= 1e-9
        else:
            assert merged[column].astype(str).tolist() == base[column].astype(str).tolist()
    for suffix in ACC_SUFFIXES:
        assert f"acc10_{suffix}" in merged.columns
        assert f"acc15_{suffix}" in merged.columns
    cluster_cols = [col for col in merged.columns if col.startswith("lisa_cluster_")]
    assert len(cluster_cols) >= 6
    assert set(merged["lisa_cluster_acc15_composite"].astype(str)).issubset(
        {"HH", "LL", "LH", "HL", "NS"}
    )


def test_table1_copies_headline_source_cells() -> None:
    rows, claims = build_table1_rows(ROOT)
    by_id = {claim.claim_id: claim for claim in claims}
    h1 = pd.read_csv(ROOT / "outputs/tables/h1_inequality_lisa.csv")
    h3 = pd.read_csv(ROOT / "outputs/tables/h3_spatial_regression.csv")
    ml = json.loads(
        (ROOT / "outputs/metrics/phase_a_ml_baselines.json").read_text(encoding="utf-8")
    )

    def h1_value(facility: str, column: str) -> float:
        hit = h1[
            (h1["resolution"] == "res9")
            & (h1["threshold_min"].astype(int) == 15)
            & (h1["facility_type"] == facility)
        ]
        return float(hit.iloc[0][column])

    for facility in ("park", "health_centre", "library", "bus_stop", "ev_charger", "composite"):
        claim = by_id[f"h1_15_{facility}_gini"]
        assert abs(parse_reported(claim.reported) - h1_value(facility, "gini")) <= claim.tolerance
        assert abs(parse_reported(claim.reported) - claim.source_value) <= claim.tolerance

    first = h3[
        (h3["threshold_min"].astype(int) == 15)
        & (h3["model_type"] == "OLS_HC3")
        & (h3["term"].str.lower() == "elderly_share")
    ].iloc[0]
    estimates = {row["estimate"] for row in rows}
    first_claim = by_id["h3_first_ols_elderly_share_coef"]
    assert first_claim.reported in estimates
    assert abs(parse_reported(first_claim.reported) - float(first["coef"])) <= first_claim.tolerance

    joined = "\n".join(
        " ".join(row.values()) for row in rows
    ).lower()
    assert "gini" in joined and "theil" in joined
    assert "ols_hc3" in joined and "sem_ml" in joined and "slm_ml" in joined
    assert "oof_roc_auc" in joined and "oof_pr_auc" in joined and "oof_brier" in joined
    rf = by_id["ml_random_forest_oof_roc_auc"]
    lr = by_id["ml_logistic_regression_oof_roc_auc"]
    rf_src = float(ml["models"]["random_forest"]["oof_roc_auc"])
    lr_src = float(ml["models"]["logistic_regression"]["oof_roc_auc"])
    assert abs(rf.source_value - rf_src) <= rf.tolerance
    assert abs(lr.source_value - lr_src) <= lr.tolerance
    assert "nan" not in joined and "none" not in joined and "todo" not in joined


def test_committed_bundle_manifest_and_report() -> None:
    cells = gpd.read_file(ROOT / "data/frozen/cells.gpkg")
    base = gpd.read_file(ROOT / "data/frozen/cells_base_res9.gpkg")
    assert set(BASE_R9_COLUMNS).issubset(cells.columns)
    assert "acc15_composite" in cells.columns
    assert "lisa_cluster_acc15_composite" in cells.columns
    assert set(cells["cell_id"].astype(str)) == set(base["cell_id"].astype(str))

    dictionary = (ROOT / "data/frozen/data_dictionary.md").read_text(encoding="utf-8")
    assert len(dictionary) >= 1000
    lowered = dictionary.lower()
    for token in (
        "cells.gpkg",
        "graph_edges.csv",
        "spatial_folds.csv",
        "cell_poi_texts.csv",
        "32649",
        "acc10_",
        "acc15_",
        "lisa_",
    ):
        assert token in lowered
    for column in cells.columns:
        if column != "geometry":
            assert column in dictionary

    manifest = json.loads((ROOT / "data/frozen/FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    payload = build_manifest_payload(ROOT / "data/frozen", project_root=ROOT)
    assert manifest["files"] == payload["files"]

    for name in REQUIRED_FIGURES:
        path = ROOT / "outputs" / "figures" / name
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0, 0))
            assert dpi[0] >= 299.5 and dpi[1] >= 299.5
            assert image.width > 200 and image.height > 200
        if name.startswith("fig1"):
            with Image.open(path) as image:
                assert image.width >= 2000
                assert image.width > image.height

    table_csv = (ROOT / "outputs/tables/table1_summary.csv").read_text(encoding="utf-8")
    table_md = (ROOT / "outputs/tables/table1_summary.md").read_text(encoding="utf-8")
    table_tex = (ROOT / "outputs/tables/table1_summary.tex").read_text(encoding="utf-8")
    for value in _decimals(table_csv):
        assert min(abs(value - other) for other in _decimals(table_md)) <= 1e-4
        assert min(abs(value - other) for other in _decimals(table_tex)) <= 1e-4

    tex = (ROOT / "report/phase_a_draft.tex").read_text(encoding="utf-8")
    prose = tex.split(r"\begin{document}", 1)[1].split(r"\bibliography", 1)[0]
    prose_l = prose.lower()
    for token in (
        "h1",
        "h2",
        "h3",
        "2sfca",
        "gini",
        "theil",
        "lisa",
        "ols",
        "sem",
        "slm",
        "bootstrap",
        "fdr",
        "holm",
        "maup",
        "brier",
        "figure 1",
        "table 1",
        "limitation",
        "ecological fallacy",
        "high-rise",
        "building-height",
        "capacity metadata",
        "imputed",
        "slope",
        "footbridge",
        "correlational",
        "historical",
        "field verification",
        "candidate",
        "32649",
        "25 m",
        "681,293",
        "82,751",
    ):
        assert token in prose_l, token
    assert "11pt" in tex and "a4paper" in tex and r"\onehalfspacing" in tex
    assert "twocolumn" not in tex
    assert "fig1_accessibility_lisa" in tex
    assert not sentence_aware_unhedged_causal_hits(prose)
    assert "??" not in tex
    bib = (ROOT / "report/references.bib").read_text(encoding="utf-8").lower()
    assert "2609.00065" in bib and "kassis" in bib

    _rows, table_claims = build_table1_rows(ROOT)
    prose_claims = build_prose_claims(ROOT)
    known = {claim.claim_id: claim for claim in (*table_claims, *prose_claims)}
    claim_sources = [claim.source_value for claim in known.values()]
    body_and_table = prose + "\n" + table_tex
    for value in _decimals(body_and_table):
        assert min(abs(value - source) for source in claim_sources) <= 0.005, value

    audit = (ROOT / "report/go_nogo_audit.md").read_text(encoding="utf-8")
    audit_l = audit.lower()
    assert re.search(r"\b(go|pass)\b", audit_l)
    assert not re.search(r"\b(fail|(?<!go/)no-go|tbd_num|todo|fixme)\b", audit_l)
    assert "681,293" in audit
    assert "mismatch" not in audit_l
    for line in audit.splitlines():
        if not line.startswith("| ") or line.startswith("| ---") or line.startswith("| Claim"):
            continue
        cells = [part.strip() for part in line.strip("|").split("|")]
        claim = known[cells[0]]
        assert cells[4] == "PASS"
        assert abs(parse_reported(cells[1]) - claim.source_value) <= claim.tolerance

    pdf = ROOT / "report/CS482_GroupX_MacaoSpatialEquity_PhaseA_Draft.pdf"
    info = inspect_pdf(pdf)
    assert info.total_pages >= 4
    assert 3.50 <= info.main_fractional_pages <= 3.80
    page_match = re.search(r"(\d+\.\d+)\s*(?:a4\s*)?pages", audit, re.I)
    assert page_match is not None
    assert abs(float(page_match.group(1)) - info.main_fractional_pages) <= 0.25
    main = info.main_text.lower()
    assert "figure 1" in main and "table 1" in main
    assert "287" in info.main_text
    ev = next(
        claim.source_value
        for claim in table_claims
        if claim.claim_id == "h1_15_ev_charger_gini"
    )
    rf = next(
        claim.source_value
        for claim in table_claims
        if claim.claim_id == "ml_random_forest_oof_roc_auc"
    )
    main_nums = _decimals(info.main_text)
    assert min(abs(value - ev) for value in main_nums) <= 0.005
    assert min(abs(value - rf) for value in main_nums) <= 0.005
    assert info.cover_pages
    assert info.references_text and info.contribution_text and info.ai_declaration_text


def test_phase_a_cli_advertises_skip_download() -> None:
    script = (ROOT / "scripts/run_phase_a.py").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "--skip-download" in script
    assert "run_phase_a.py" in readme and "--skip-download" in readme
    assert "data/frozen" in readme


@pytest.mark.parametrize(
    "name",
    [
        "src/macao_equity/freeze_bundle.py",
        "src/macao_equity/viz/figures.py",
        "src/macao_equity/viz/tables.py",
        "src/macao_equity/viz/report.py",
        "scripts/04_freeze_and_report.py",
        "scripts/run_phase_a.py",
    ],
)
def test_m4_modules_exist(name: str) -> None:
    assert (ROOT / name).is_file()
