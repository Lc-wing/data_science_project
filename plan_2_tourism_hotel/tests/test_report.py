"""The Phase A draft copies its numbers from the output files."""

from __future__ import annotations

import json
import subprocess
import sys

from macao_tourism.config import PROJECT_ROOT
from macao_tourism.report import (
    MAIN_PAGE_MAX,
    MAIN_PAGE_MIN,
    PDF_NAME,
    REPORT_DIR,
    build_claims,
    pdf_text,
    render_tex,
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
    assert "H3 is not" not in claims.mapping()["CHANGE_POINTS"]
    assert "2020-03" in claims.mapping()["CHANGE_POINTS"]
    assert render_tex(claims) == render_tex(build_claims())


def test_draft_covers_the_five_sections_and_audits_clean() -> None:
    result = write_report()
    assert MAIN_PAGE_MIN <= float(result["main_pages"]) <= MAIN_PAGE_MAX
    pdf = REPORT_DIR / PDF_NAME
    text = pdf_text(pdf)
    for section in SECTIONS:
        assert section in text
    assert "H3 is not supported" in text
    assert "Figure 1." in text and "Table 1." in text and "Figure 2." in text
    assert "Chronos" not in text and "Qwen" not in text
    audit = (REPORT_DIR / "phase_a_audit.md").read_text(encoding="utf-8")
    assert "Numeric audit: PASS." in audit
    assert "Phase B is not estimated" in audit
    for claim in build_claims().items():
        assert claim.reported in audit


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
