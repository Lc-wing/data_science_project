"""Tier 3 Opaque-Box E2E Suite: Cross-Milestone Numeric, Key & Provenance Consistency."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pypdf import PdfReader

from stock_volatility import config
from stock_volatility.panel.freeze import verify_freeze_manifest
from stock_volatility.report import collect_report_claims


def test_t3_01_freeze_manifest_sha256_matches_disk_and_pdf_prefix(
    project_root: Path,
    freeze_manifest: dict[str, Any],
) -> None:
    """Verify every SHA-256 in FREEZE_MANIFEST matches disk and PDF cites exact prefix."""
    verify_freeze_manifest(frozen_dir=project_root / "data" / "frozen", project_root=project_root)

    panel_sha12 = freeze_manifest["files"]["data/frozen/panel_daily.parquet"]["sha256"][:12]
    tex = (project_root / "report" / "report.tex").read_text(encoding="utf-8")
    assert panel_sha12 in tex, f"Panel SHA-256 prefix {panel_sha12} missing from report.tex"

    reader = PdfReader(str(project_root / "report" / config.REPORT_PDF_NAME))
    cover_txt = reader.pages[0].extract_text() or ""
    assert panel_sha12 in cover_txt, f"Panel SHA-256 prefix {panel_sha12} missing from PDF cover"


def test_t3_02_cross_milestone_row_count_and_key_alignment(
    project_root: Path,
    panel_df: pd.DataFrame,
    events_phase_b_df: pd.DataFrame,
    test_pred_df: pd.DataFrame,
    split_metadata: dict[str, Any],
) -> None:
    """Verify exact 1-to-1 alignment across M1 -> M2 -> M3 -> M4 tables and JSON summaries."""
    day1 = json.loads(
        (project_root / "outputs" / "metrics" / "day1_data_audit.json").read_text(encoding="utf-8")
    )
    eda = json.loads(
        (project_root / "outputs" / "metrics" / "eda_summary.json").read_text(encoding="utf-8")
    )
    h1 = json.loads(
        (project_root / "outputs" / "metrics" / "h1_regression.json").read_text(encoding="utf-8")
    )

    assert day1["ohlcv"]["total_rows"] == len(panel_df) == split_metadata["total_panel_rows"]
    assert eda["modeling_total_rows"] == h1["main_model"]["n_obs"] == 104_040

    # Active event ticker-days match 1-to-1 between panel_daily and events_text_phase_b
    p_ev = set(zip(panel_df.loc[panel_df["has_news"] == 1, "ticker"],
                   panel_df.loc[panel_df["has_news"] == 1, "date"], strict=True))
    b_ev = set(zip(events_phase_b_df.loc[events_phase_b_df["has_news"] == 1, "ticker"],
                   events_phase_b_df.loc[events_phase_b_df["has_news"] == 1, "date"], strict=True))
    assert p_ev == b_ev, "Active event ticker-days differ between panel_daily and Phase B text"

    # H2 2025 test predictions match frozen test modeling-complete rows 1-to-1
    test_pan = panel_df[
        (panel_df["split"] == "test")
        & panel_df["has_full_features"]
        & panel_df["has_full_target"]
    ].sort_values(["ticker", "date"]).reset_index(drop=True)
    test_prd = test_pred_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    assert len(test_pan) == len(test_prd) == split_metadata["splits"]["test"][
        "modeling_complete_rows"
    ]
    assert (test_pan["ticker"].to_numpy() == test_prd["ticker"].to_numpy()).all()
    assert (test_pan["date"].to_numpy() == test_prd["date"].to_numpy()).all()
    np.testing.assert_allclose(
        test_pan["rv_5d_fwd"].to_numpy(), test_prd["rv_5d_fwd"].to_numpy(), rtol=1e-12
    )
    assert (test_pan["y_tail"].astype(int).to_numpy() == test_prd["y_tail"].astype(int).to_numpy()
            ).all()


def test_t3_03_table1_csv_md_tex_numeric_consistency(project_root: Path) -> None:
    """Verify zero numeric drift between M3/M4 sources and Table 1 CSV, Markdown & LaTeX."""
    h1 = json.loads(
        (project_root / "outputs" / "metrics" / "h1_regression.json").read_text(encoding="utf-8")
    )
    h2_metrics = pd.read_csv(project_root / "outputs" / "tables" / "h2_test_metrics.csv")
    t1_csv = pd.read_csv(project_root / "outputs" / "tables" / "table1_summary.csv")
    t1_md = (project_root / "outputs" / "tables" / "table1_summary.md").read_text(encoding="utf-8")
    t1_tex = (project_root / "outputs" / "tables" / "table1_summary.tex").read_text(
        encoding="utf-8"
    )

    assert len(t1_csv) == 9
    expected_tokens: list[str] = [f"{h1['main_model']['within_r2']:.4f}"]
    for item in h1["confirmatory_family_holm"]:
        val = (
            f"{item['estimate']:.2f}"
            if item["test_id"] == "joint_5sector_interaction_wald"
            else f"{item['estimate']:+.4f}"
        )
        expected_tokens.append(val)
    for m_key in ("persistence", "gjr_garch", "har_rv", "lgbm"):
        r = h2_metrics[h2_metrics["model"] == m_key].iloc[0]
        expected_tokens.extend(
            [
                f"{r['qlike']:.4f}",
                f"{r['rmse']:.4f}",
                f"{r['pr_auc']:.4f}",
                f"{r['brier']:.4f}",
                f"{r['ece']:.4f}",
            ]
        )

    csv_blob = t1_csv.to_string()
    for tok in expected_tokens:
        if tok != f"{h1['main_model']['within_r2']:.4f}":
            assert tok in csv_blob, f"Value {tok!r} missing from Table 1 CSV"
            assert tok in t1_md, f"Value {tok!r} missing from Table 1 Markdown"
        assert tok in t1_tex, f"Value {tok!r} missing from Table 1 LaTeX"


def test_t3_04_all_report_claims_trace_to_pdf_and_audit(project_root: Path) -> None:
    """Verify every one of the 61 programmatically collected claims appears in PDF and audit."""
    claims = collect_report_claims(project_root)
    assert len(claims) == 61

    reader = PdfReader(str(project_root / "report" / config.REPORT_PDF_NAME))
    norm_pdf = re.sub(r"\s+", " ", "\n".join(p.extract_text() or "" for p in reader.pages))
    audit_md = (project_root / "report" / "report_audit.md").read_text(encoding="utf-8")

    missing_pdf: list[str] = []
    missing_audit: list[str] = []
    for key, meta in claims.items():
        val = meta["value"]
        variants = {val, val.replace("-", "\u2212"), val.replace(",", "")}
        if not any(v in norm_pdf for v in variants):
            missing_pdf.append(f"{key}={val}")
        if f"`{key}` | `{val}`" not in audit_md:
            missing_audit.append(f"{key}={val}")

    assert not missing_pdf, f"Claims missing from compiled PDF: {missing_pdf}"
    assert not missing_audit, f"Claims missing from report_audit.md: {missing_audit}"
