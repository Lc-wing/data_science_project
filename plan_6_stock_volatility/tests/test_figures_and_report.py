"""Unit, 300-DPI figure, Table 1, A4 CS482 PDF audit, and replay CLI tests for Milestone M5."""

from __future__ import annotations

import re
import subprocess

import pandas as pd
from pypdf import PdfReader

from stock_volatility import config

# ---------------------------------------------------------------------------
# Layer 1: Pure Offline Unit Tests for Table 1, Claims, and Audit Helpers
# ---------------------------------------------------------------------------


def test_table1_summary_builder_matches_h1_and_h2_sources(block_network: None) -> None:
    """Verify build_table1_summary produces consistent H1 and H2 rows from committed M3/M4 files."""
    from stock_volatility.report import build_table1_summary

    t1 = build_table1_summary(config.PROJECT_ROOT)
    df = t1["table_df"]
    assert set(df["panel"]) == {"Panel A: H1 Panel HAR-RV", "Panel B: H2 2025 Holdout"}
    panel_a = df[df["panel"] == "Panel A: H1 Panel HAR-RV"]
    panel_b = df[df["panel"] == "Panel B: H2 2025 Holdout"]
    assert len(panel_a) == 5
    assert len(panel_b) == 4
    assert "Table 1" in t1["markdown"] or "Panel A" in t1["markdown"]
    assert r"\begin{tabular}" in t1["latex"]


def test_audit_text_scanner_catches_forbidden_words_and_artifacts(block_network: None) -> None:
    """Verify report text audit rejects forbidden causal words, ??, TODO, and NaN."""
    from stock_volatility.report import scan_report_text_issues

    clean_text = (
        "This associational study makes no causal claims and does not constitute investment "
        "advice. TreeSHAP explains LightGBM model behavior on 2025 holdout forecasts."
    )
    assert scan_report_text_issues(clean_text) == []

    bad_text = (
        "VIX is the main driver with a large causal impact and elasticity of 0.5 [??] TODO NaN."
    )
    issues = scan_report_text_issues(bad_text)
    assert len(issues) >= 5


# ---------------------------------------------------------------------------
# Layer 2: Committed 300-DPI Figures, Table 1, CS482 PDF & Replay CLI Tests
# ---------------------------------------------------------------------------


def test_committed_m5_figures_exist_and_are_300_dpi(block_network: None) -> None:
    """Verify Figure 1 and Figure 2 exist as non-empty 300-DPI PNG and vector PDF."""
    import matplotlib.image as mpimg

    for stem in ("fig1_volatility_regimes_and_forecasts", "fig2_shap_and_calibration"):
        png_path = config.OUTPUTS_FIGURES_DIR / f"{stem}.png"
        pdf_path = config.OUTPUTS_FIGURES_DIR / f"{stem}.pdf"
        assert png_path.is_file() and png_path.stat().st_size > 40_000, f"Invalid PNG: {png_path}"
        assert pdf_path.is_file() and pdf_path.stat().st_size > 15_000, f"Invalid PDF: {pdf_path}"

        # Check high resolution (>= 2000 px wide at 300 DPI) and PNG pHYs DPI metadata
        img = mpimg.imread(png_path)
        height, width = img.shape[:2]
        assert width >= 2000 and height >= 1000, f"Low-res figure {stem}: {width}x{height}"

        # Inspect PNG pHYs chunk for 300 DPI (~11811 pixels per meter)
        raw = png_path.read_bytes()
        phys_idx = raw.find(b"pHYs")
        assert phys_idx != -1, f"Missing pHYs DPI metadata chunk in {png_path}"
        x_ppm = int.from_bytes(raw[phys_idx + 4 : phys_idx + 8], "big")
        dpi = x_ppm * 0.0254
        assert abs(dpi - 300.0) <= 1.0, f"Expected 300 DPI in {stem}, got {dpi:.2f}"


def test_committed_m5_table1_and_cs482_pdf_audit_pass(block_network: None) -> None:
    """Verify Table 1 files, A4 CS482 PDF layout/page budget, zero forbidden terms & audit PASS."""
    for ext in ("csv", "md", "tex"):
        p = config.OUTPUTS_TABLES_DIR / f"table1_summary.{ext}"
        assert p.is_file() and p.stat().st_size > 200, f"Missing Table 1 file: {p}"

    t1_csv = pd.read_csv(config.OUTPUTS_TABLES_DIR / "table1_summary.csv")
    assert len(t1_csv) == 9  # 5 H1 confirmatory rows + 4 H2 model rows

    tex_path = config.REPORT_DIR / "report.tex"
    bib_path = config.REPORT_DIR / "references.bib"
    pdf_path = config.REPORT_DIR / config.REPORT_PDF_NAME
    audit_path = config.REPORT_DIR / "report_audit.md"
    for p in (tex_path, bib_path, pdf_path, audit_path):
        assert p.is_file() and p.stat().st_size > 500, f"Missing report artifact: {p}"

    # Verify A4 page dimensions via pdfinfo
    info = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    m_size = re.search(r"Page size:\s+([0-9.]+)\s+x\s+([0-9.]+)\s+pts", info.stdout)
    assert m_size is not None, f"Could not parse pdfinfo Page size:\n{info.stdout}"
    w_pts, h_pts = float(m_size.group(1)), float(m_size.group(2))
    assert abs(w_pts - 595.28) < 1.5 and abs(h_pts - 841.89) < 1.5, (
        f"Expected A4 (595.28x841.89 pts), got {w_pts}x{h_pts}"
    )

    # Verify text extraction, required CS482 sections, and zero forbidden wording
    reader = PdfReader(str(pdf_path))
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    for req_phrase in (
        "CS482",
        "not investment",
        "References",
        "Contribution Statement",
        "AI Usage Declaration",
        "2609.00065",
        "0.2306",
        "0.4736",
        "0.1398",
        "0.2819",
    ):
        assert req_phrase.lower() in full_text.lower(), (
            f"Missing required phrase {req_phrase!r} in PDF text"
        )
    assert "??" not in full_text
    assert "NaN" not in full_text

    # Verify automated audit report passed and main text page count is strictly within [3.0, 5.0]
    audit_text = audit_path.read_text(encoding="utf-8")
    assert "Overall Audit Verdict: **PASS**" in audit_text
    m_pages = re.search(r"Main Text Length:\s+`([0-9.]+)`\s+A4 pages", audit_text)
    assert m_pages is not None, "Missing Main Text Length metric in report_audit.md"
    main_pages = float(m_pages.group(1))
    assert 3.0 <= main_pages <= 5.0, f"Main text pages {main_pages} outside CS482 [3.0, 5.0] limit"


def test_run_pipeline_report_only_cli_succeeds_offline(block_network: None) -> None:
    """Verify scripts/run_pipeline.py --report-only executes cleanly offline."""
    script_path = config.PROJECT_ROOT / "scripts" / "run_pipeline.py"
    assert script_path.is_file()
    proc = subprocess.run(
        ["uv", "run", "--no-sync", "python", str(script_path), "--report-only"],
        cwd=config.PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"run_pipeline.py --report-only failed:\n{proc.stdout}\n{proc.stderr}"
    )
    assert '"verdict": "PASS"' in proc.stdout


def test_m5_milestone_delivery_doc_complete_linked_and_not_gitignored() -> None:
    """Verify M5 milestone delivery doc exists, records deliverables + AI log, & is tracked."""
    repo_root = config.PROJECT_ROOT.parent
    milestones_dir = config.PROJECT_ROOT / "docs" / "milestones"
    m5_doc = milestones_dir / "M5_figures_table1_and_cs482_report.md"
    assert m5_doc.is_file(), f"Missing M5 milestone doc: {m5_doc}"
    assert m5_doc.stat().st_size > 2000

    m5_text = m5_doc.read_text(encoding="utf-8")
    for token in (
        "744df5dd8647aaae93e5f4b1049273c781b9bb92",
        "fig1_volatility_regimes_and_forecasts.png",
        "fig2_shap_and_calibration.png",
        "table1_summary.csv",
        "table1_summary.tex",
        "CS482_GroupX_StockVolatilityLLM_PhaseA_Draft.pdf",
        "phase_a_audit.md",
        "06_write_report.py",
        "run_phase_a.py",
        "300",
    ):
        assert token in m5_text, f"Missing {token!r} in M5_figures_table1_and_cs482_report.md"

    ledger_text = (milestones_dir / "README.md").read_text(encoding="utf-8")
    assert "744df5dd8647aaae93e5f4b1049273c781b9bb92" in ledger_text
    assert "M5_figures_table1_and_cs482_report.md" in ledger_text

    plan6_readme = (config.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/milestones/M5_figures_table1_and_cs482_report.md" in plan6_readme

    proc = subprocess.run(
        [
            "git",
            "check-ignore",
            "-v",
            "--no-index",
            "plan_6_stock_volatility/docs/milestones/M5_figures_table1_and_cs482_report.md",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert proc.returncode == 1, f"M5 milestone doc must NOT be gitignored:\n{proc.stdout}"
    assert proc.stdout.strip() == ""
