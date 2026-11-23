"""Tier 5 White-Box Single-Fault Adversarial Hardening Suite for Plan 6 Phase A.

Injects 8 named single-fault mutations into isolated temporary clones of the real project tree
(never touching the committed workspace), spawns child pytest runs against the targeted E2E
tiers, and requires every named fault to fail cleanly (`returncode == 1`, `errors == 0`) with
the exact expected diagnostic clause. Also unit-tests hardened `audit_compiled_pdf()` and
`scan_report_text_issues()` edge cases directly.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pypdf import PdfReader, PdfWriter

from stock_volatility import config
from stock_volatility.report import (
    audit_compiled_pdf,
    collect_report_claims,
    scan_report_text_issues,
)

if os.environ.get("STOCK_VOL_E2E_IN_SELF_TEST") == "1":
    pytest.skip(
        "Skipping test_tier5_adversarial.py inside child fault-injection subprocess",
        allow_module_level=True,
    )

_E2E_DIR = Path(__file__).resolve().parent
_TIER1 = _E2E_DIR / "test_tier1_feature_coverage.py"
_TIER2 = _E2E_DIR / "test_tier2_boundary_corner.py"
_TIER3 = _E2E_DIR / "test_tier3_cross_feature.py"


def _parse_pytest_counts(output: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for key in ("passed", "failed", "error", "errors", "skipped"):
        matches = re.findall(rf"(\d+)\s+{key}\b", output)
        if matches:
            target_k = "errors" if key.startswith("error") else key
            counts[target_k] = max(counts[target_k], int(matches[-1]))
    return counts


def _apply_named_fault(mutated_root: Path, fault_id: str) -> None:
    """Inject one exact white-box fault into an isolated project clone."""
    if fault_id == "freeze_manifest_sha_tamper":
        p = mutated_root / "data" / "frozen" / "analysis_spec.json"
        p.write_text(p.read_text(encoding="utf-8") + " ", encoding="utf-8")

    elif fault_id == "event_cap_violation":
        p = mutated_root / "data" / "frozen" / "events_text_phase_b.parquet"
        df = pd.read_parquet(p)
        df.loc[df.index[0], "news_count"] = 6
        df.to_parquet(p, index=False, compression="zstd")

    elif fault_id == "split_embargo_leak":
        p = mutated_root / "data" / "frozen" / "panel_daily.parquet"
        df = pd.read_parquet(p)
        # Relabel an embargo_1 date as 'val' to simulate embargo breach
        emb_mask = df["split"] == "embargo_1"
        first_emb_idx = df.index[emb_mask][0]
        df.loc[first_emb_idx, "split"] = "val"
        df.to_parquet(p, index=False, compression="zstd")

    elif fault_id == "test_tail_target_leak":
        p = mutated_root / "data" / "frozen" / "panel_daily.parquet"
        df = pd.read_parquet(p)
        last_mask = df["date"] == config.TEST_END
        df.loc[last_mask, "has_full_target"] = True
        df.to_parquet(p, index=False, compression="zstd")

    elif fault_id == "table1_numeric_drift":
        p = mutated_root / "outputs" / "tables" / "table1_summary.tex"
        txt = p.read_text(encoding="utf-8")
        assert "0.2306" in txt
        p.write_text(txt.replace("0.2306", "0.1999", 1), encoding="utf-8")

    elif fault_id == "pdf_page2_us_letter":
        p = mutated_root / "report" / config.REPORT_PDF_NAME
        reader = PdfReader(str(p))
        writer = PdfWriter()
        for idx, page in enumerate(reader.pages):
            if idx == 1:
                # Sneak a US Letter page (612 x 792 pts) on Page 2 while Page 1 stays A4
                page.mediabox.upper_right = (612.0, 792.0)
            writer.add_page(page)
        with p.open("wb") as fh:
            writer.write(fh)

    elif fault_id == "figure_low_dpi_150":
        p = mutated_root / "outputs" / "figures" / "fig1_volatility_regimes_and_forecasts.png"
        arr = mpimg.imread(p)
        fig = plt.figure(figsize=(8, 4), dpi=150)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(arr)
        ax.axis("off")
        fig.savefig(p, dpi=150)
        plt.close(fig)

    elif fault_id == "claim_dropped_from_audit":
        p = mutated_root / "report" / "phase_a_audit.md"
        txt = p.read_text(encoding="utf-8")
        p.write_text(txt.replace("`LGBM_QLIKE` | `0.2306`", "`LGBM_QLIKE` | `0.9999`"), "utf-8")

    else:
        raise ValueError(f"Unknown fault_id: {fault_id}")


def _run_child_pytest(
    mutated_root: Path,
    target_files: tuple[Path, ...],
    *,
    k_expr: str,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["STOCK_VOL_E2E_IN_SELF_TEST"] = "1"
    env["STOCK_VOL_PROJECT_ROOT"] = str(mutated_root)
    env["MPLBACKEND"] = "Agg"
    env["OMP_NUM_THREADS"] = "1"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        f"--project-root={mutated_root}",
        *[str(path) for path in target_files],
        "-k",
        k_expr,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, check=False, timeout=90)


_TIER5_SPECS: tuple[tuple[str, tuple[Path, ...], str, tuple[str, ...], int], ...] = (
    (
        "freeze_manifest_sha_tamper",
        (_TIER1, _TIER3),
        "test_t1_03_m2_frozen_bundle_manifest_and_schemas or "
        "test_t3_01_freeze_manifest_sha256_matches_disk_and_pdf_prefix",
        ("SHA-256 mismatch",),
        2,
    ),
    (
        "event_cap_violation",
        (_TIER2,),
        "test_t2_01_event_alignment_dst_early_close_and_cap_invariants",
        ("Event cap violated",),
        1,
    ),
    (
        "split_embargo_leak",
        (_TIER2,),
        "test_t2_02_purged_embargo_split_boundaries_and_target_masking",
        ("Val boundary breach",),
        1,
    ),
    (
        "test_tail_target_leak",
        (_TIER2,),
        "test_t2_02_purged_embargo_split_boundaries_and_target_masking",
        ("Incomplete forward target leakage",),
        1,
    ),
    (
        "table1_numeric_drift",
        (_TIER3,),
        "test_t3_03_table1_csv_md_tex_numeric_consistency",
        ("missing from Table 1 LaTeX",),
        1,
    ),
    (
        "pdf_page2_us_letter",
        (_TIER2,),
        "test_t2_07_pdf_every_page_a4_budget_and_zero_forbidden_words",
        ("page 2 is not A4",),
        1,
    ),
    (
        "figure_low_dpi_150",
        (_TIER2,),
        "test_t2_06_figures_300_dpi_high_res_and_non_blank",
        ("DPI below 300",),
        1,
    ),
    (
        "claim_dropped_from_audit",
        (_TIER3,),
        "test_t3_04_all_51_report_claims_trace_to_pdf_and_audit",
        ("Claims missing from phase_a_audit.md", "LGBM_QLIKE"),
        1,
    ),
)


@pytest.mark.parametrize(
    ("fault_id", "target_files", "k_expr", "required_fragments", "min_failed"),
    _TIER5_SPECS,
    ids=[spec[0] for spec in _TIER5_SPECS],
)
def test_t5_named_single_fault_is_detected(
    project_root: Path,
    tmp_path: Path,
    clone_project: Any,
    fault_id: str,
    target_files: tuple[Path, ...],
    k_expr: str,
    required_fragments: tuple[str, ...],
    min_failed: int,
) -> None:
    """Each single-fault adversarial mutation must fail with exit 1, 0 errors & exact clause."""
    mutated_root = clone_project(project_root, tmp_path / f"tier5_{fault_id}")
    _apply_named_fault(mutated_root, fault_id)

    proc = _run_child_pytest(mutated_root, target_files, k_expr=k_expr)
    combined = proc.stdout + "\n" + proc.stderr
    counts = _parse_pytest_counts(combined)

    assert proc.returncode == 1, (
        f"Fault {fault_id!r} did not exit 1 (got {proc.returncode}):\n{combined}"
    )
    assert counts["failed"] >= min_failed and counts["errors"] == 0, (
        f"Fault {fault_id!r} unexpected counts {counts} (min_failed={min_failed}):\n{combined}"
    )
    missing = [frag for frag in required_fragments if not re.search(frag, combined)]
    assert not missing, f"Fault {fault_id!r} missed required clauses {missing}:\n{combined}"


def test_t5_hardened_pdf_auditor_catches_page2_us_letter_and_tbd_num(
    project_root: Path,
    tmp_path: Path,
) -> None:
    """White-box hardening unit check: `audit_compiled_pdf` catches Page-2 US Letter & TBD_NUM."""
    # 1. Placeholder `TBD_NUM` must be flagged by scan_report_text_issues
    issues = scan_report_text_issues("Observed test QLIKE is TBD_NUM on holdout.")
    assert any("TBD" in msg for msg in issues), f"TBD_NUM slipped past scanner: {issues}"

    # 2. Page-2 US Letter sneaked past Page-1 A4 must fail `audit_compiled_pdf()`
    mutated_pdf = tmp_path / "sneaky_page2_letter.pdf"
    reader = PdfReader(str(project_root / "report" / config.REPORT_PDF_NAME))
    writer = PdfWriter()
    for idx, page in enumerate(reader.pages):
        if idx == 1:
            page.mediabox.upper_right = (612.0, 792.0)
        writer.add_page(page)
    with mutated_pdf.open("wb") as fh:
        writer.write(fh)

    claims = collect_report_claims(project_root)
    audit = audit_compiled_pdf(
        pdf_path=mutated_pdf,
        claims=claims,
        generated_at_utc="2026-11-22T14:04:28+00:00",
    )
    assert audit["verdict"] == "FAIL", "audit_compiled_pdf failed to catch Page-2 US Letter!"
    assert audit["checks"]["a4_dimensions"] is False
    assert 2 in audit["non_a4_pages"]
