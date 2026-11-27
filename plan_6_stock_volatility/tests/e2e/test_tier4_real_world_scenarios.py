"""Tier 4 Opaque-Box E2E Suite: Real-World Offline CLI Replay, Tamper Defense & M6 Docs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from stock_volatility import config


def test_t4_01_freeze_check_cli_succeeds_offline(project_root: Path) -> None:
    """Scenario 1: `scripts/02_build_panel_and_freeze.py --check` verifies committed bundle."""
    script = config.PROJECT_ROOT / "scripts" / "02_build_panel_and_freeze.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--check"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert proc.returncode == 0, f"--check failed:\n{proc.stdout}\n{proc.stderr}"
    payload = json.loads(proc.stdout)
    assert payload["status"] == "verified"
    assert payload["artifact_count"] == 6


def test_t4_02_isolated_clone_report_only_replay_is_bit_identical(
    project_root: Path,
    tmp_path: Path,
    clone_project: Any,
) -> None:
    """Scenario 2: `run_phase_a.py --report-only --project-root <clone>` is bit-identical."""
    clone_root = clone_project(project_root, tmp_path / "clean_clone")
    # Remove generated M5 outputs inside clone so we prove genuine fresh regeneration
    for rel in (
        "outputs/figures/fig1_volatility_regimes_and_forecasts.png",
        "outputs/figures/fig1_volatility_regimes_and_forecasts.pdf",
        "outputs/figures/fig2_shap_and_calibration.png",
        "outputs/figures/fig2_shap_and_calibration.pdf",
        "outputs/tables/table1_summary.csv",
        "outputs/tables/table1_summary.md",
        "outputs/tables/table1_summary.tex",
        "report/references.bib",
        "report/report.tex",
        f"report/{config.REPORT_PDF_NAME}",
        "report/report_audit.md",
    ):
        target = clone_root / rel
        if target.is_file():
            target.unlink()

    script = config.PROJECT_ROOT / "scripts" / "run_pipeline.py"
    env = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1")
    proc = subprocess.run(
        [sys.executable, str(script), "--report-only", "--project-root", str(clone_root)],
        cwd=config.PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, f"Isolated --report-only failed:\n{proc.stdout}\n{proc.stderr}"
    payload = json.loads(proc.stdout)
    assert payload["verdict"] == "PASS"
    assert payload["claims_verified"] == 61
    assert payload["main_pages_occupied"] == 5

    # Verify bit-for-bit SHA-256 identity across all 11 report & figure artifacts
    for rel in (
        "outputs/figures/fig1_volatility_regimes_and_forecasts.png",
        "outputs/figures/fig1_volatility_regimes_and_forecasts.pdf",
        "outputs/figures/fig2_shap_and_calibration.png",
        "outputs/figures/fig2_shap_and_calibration.pdf",
        "outputs/tables/table1_summary.csv",
        "outputs/tables/table1_summary.md",
        "outputs/tables/table1_summary.tex",
        "report/references.bib",
        "report/report.tex",
        f"report/{config.REPORT_PDF_NAME}",
        "report/report_audit.md",
    ):
        assert config.sha256_file(clone_root / rel) == config.sha256_file(project_root / rel), (
            f"Non-deterministic replay diff detected on {rel}"
        )


def test_t4_03_tampered_frozen_artifact_blocks_replay_cli(
    project_root: Path,
    tmp_path: Path,
    clone_project: Any,
) -> None:
    """Scenario 3: Corrupting 1 byte in frozen data forces `run_pipeline.py` to abort non-zero."""
    clone_root = clone_project(project_root, tmp_path / "tampered_frozen_clone")
    spec_path = clone_root / "data" / "frozen" / "analysis_spec.json"
    spec_path.write_text(spec_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    script = config.PROJECT_ROOT / "scripts" / "run_pipeline.py"
    env = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1")
    proc = subprocess.run(
        [sys.executable, str(script), "--report-only", "--project-root", str(clone_root)],
        cwd=config.PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert proc.returncode != 0, "Replay CLI failed to abort on tampered frozen artifact"
    assert "SHA-256 mismatch" in (proc.stdout + proc.stderr)


def test_t4_04_skip_download_detects_tampered_raw_provenance_in_clone(
    project_root: Path,
    tmp_path: Path,
    clone_project: Any,
) -> None:
    """Scenario 4: `run_skip_download_pipeline(project_root=clone)` rejects tampered raw data."""
    import pytest

    from stock_volatility.ingest.audit import run_skip_download_pipeline

    clone_root = clone_project(project_root, tmp_path / "tampered_raw_clone")
    # Unlink hardlinked raw universe file before writing a tampered copy so repo is untouched
    uni_path = clone_root / "data" / "raw" / "universe_60.csv"
    raw_text = uni_path.read_text(encoding="utf-8")
    uni_path.unlink()
    uni_path.write_text(raw_text + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"SHA-256.*mismatch"):
        run_skip_download_pipeline(project_root=clone_root)


def test_t4_05_m6_milestone_doc_and_test_ready_tracked_in_git() -> None:
    """Scenario 5: Verify M6 delivery doc, TEST_READY.md, ledger & README links are tracked."""
    repo_root = config.PROJECT_ROOT.parent
    milestones_dir = config.PROJECT_ROOT / "docs" / "milestones"
    m6_doc = milestones_dir / "M6_e2e_and_adversarial_hardening.md"
    test_ready = config.PROJECT_ROOT / "TEST_READY.md"

    assert m6_doc.is_file() and m6_doc.stat().st_size > 2000, f"Missing M6 doc: {m6_doc}"
    assert test_ready.is_file() and test_ready.stat().st_size > 1000, (
        f"Missing Plan 6 TEST_READY.md: {test_ready}"
    )

    m6_text = m6_doc.read_text(encoding="utf-8")
    for token in (
        "63dc8ec46dddd910241de6c5ba968f9de308bcc3",
        "test_tier1_feature_coverage.py",
        "test_tier2_boundary_corner.py",
        "test_tier3_cross_feature.py",
        "test_tier4_real_world_scenarios.py",
        "test_tier5_adversarial.py",
        "TEST_READY.md",
        "SOURCE_DATE_EPOCH",
    ):
        assert token in m6_text, f"Missing {token!r} in M6_e2e_and_adversarial_hardening.md"

    ledger_text = (milestones_dir / "README.md").read_text(encoding="utf-8")
    assert "63dc8ec46dddd910241de6c5ba968f9de308bcc3" in ledger_text
    assert "M6_e2e_and_adversarial_hardening.md" in ledger_text

    plan6_readme = (config.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/milestones/M6_e2e_and_adversarial_hardening.md" in plan6_readme

    for rel in (
        "plan_6_stock_volatility/docs/milestones/M6_e2e_and_adversarial_hardening.md",
        "plan_6_stock_volatility/TEST_READY.md",
    ):
        proc = subprocess.run(
            ["git", "check-ignore", "-v", "--no-index", rel],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        assert proc.returncode == 1 and proc.stdout.strip() == "", (
            f"Delivery file {rel} must NOT be gitignored:\n{proc.stdout}"
        )
