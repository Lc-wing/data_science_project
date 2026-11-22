"""The Phase A draft copies its numbers from the output files."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from macao_tourism.config import PROJECT_ROOT
from macao_tourism.report import (
    AUDIT_NAME,
    MAIN_PAGE_MAX,
    MAIN_PAGE_MIN,
    PDF_NAME,
    REPORT_DIR,
    TEX_NAME,
    build_claims,
    main_text_pages,
    pdf_text,
    render_tex,
    uncited_bibitems,
    write_report,
)

SECTIONS = (
    "Introduction",
    "Data Collection and Preparation",
    "Methods",
    "Results",
    "Discussion",
)


def test_every_claim_matches_its_source() -> None:
    claims = build_claims()
    assert len(claims.items()) > 40
    for claim in claims.items():
        assert claim.matches_source(), claim.claim_id
    manifest = json.loads(
        (PROJECT_ROOT / "data/frozen/FREEZE_MANIFEST.json").read_text(encoding="utf-8")
    )
    panel_sha = manifest["files"]["data/frozen/panel_tier_month.csv"]
    assert claims.mapping()["PANEL_SHA"] == panel_sha[:12]
    assert claims.mapping()["H1_EST"] == "18.33"
    assert claims.mapping()["SAR_5_ORDER"] == "(1, 0, 1)"
    assert claims.mapping()["LGB_TREES"] == "400"
    assert "H3 is not" not in claims.mapping()["CHANGE_POINTS"]
    assert "2020-03" in claims.mapping()["CHANGE_POINTS"]
    tex = render_tex(claims)
    assert tex == render_tex(build_claims())
    assert uncited_bibitems(tex) == []


def test_draft_covers_the_five_sections_and_audits_clean(tmp_path: Path) -> None:
    (tmp_path / "outputs").symlink_to(PROJECT_ROOT / "outputs")
    result = write_report(root=tmp_path)
    assert MAIN_PAGE_MIN <= float(result["main_pages"]) <= MAIN_PAGE_MAX
    report_dir = tmp_path / "report"
    pdf = report_dir / PDF_NAME
    text = pdf_text(pdf)
    for section in SECTIONS:
        assert section in text
    assert "H3 is not supported" in text
    draft = (report_dir / TEX_NAME).read_text(encoding="utf-8")
    assert r"p<0.001" in draft
    assert "p=<" not in draft
    assert "¡" not in text and "¿" not in text
    assert "[?]" not in text and "??" not in text
    assert "-3.00 points below" not in text
    assert "Figure 1." in text and "Table 1." in text and "Figure 2." in text
    assert "Chronos" not in text and "Qwen" not in text
    audit = (report_dir / AUDIT_NAME).read_text(encoding="utf-8")
    assert "Numeric audit: PASS." in audit
    assert "Phase B is not estimated" in audit
    for claim in build_claims().items():
        assert claim.reported in audit


def test_committed_report_matches_current_template_and_audits_clean() -> None:
    committed_tex = (REPORT_DIR / TEX_NAME).read_text(encoding="utf-8")
    assert committed_tex == render_tex(build_claims())
    committed_pdf = REPORT_DIR / PDF_NAME
    assert committed_pdf.is_file()
    assert MAIN_PAGE_MIN <= main_text_pages(committed_pdf) <= MAIN_PAGE_MAX
    text = pdf_text(committed_pdf)
    assert "¡" not in text and "¿" not in text
    assert "[?]" not in text and "??" not in text
    committed_audit = (REPORT_DIR / AUDIT_NAME).read_text(encoding="utf-8")
    assert "Numeric audit: PASS." in committed_audit


def test_replay_command_replays_without_downloading() -> None:
    script = PROJECT_ROOT / "scripts" / "run_phase_a.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--skip-download" in result.stdout
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "run_phase_a.py --skip-download" in readme
