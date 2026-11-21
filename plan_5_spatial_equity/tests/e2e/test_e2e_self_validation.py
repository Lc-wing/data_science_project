"""Design C Double-Recursion-Guarded Self-Validation for the Opaque-Box 4-Tier E2E Suite."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Guard 1: Module-level recursion guard for child pytest invocations
if os.environ.get("MACAO_E2E_IN_SELF_TEST") == "1":
    pytest.skip(
        "Skipping test_e2e_self_validation.py inside child self-test subprocess",
        allow_module_level=True,
    )

from e2e_lib.synthetic import apply_fault, build_valid_synthetic_project  # noqa: E402

_THIS_FILE = Path(__file__).resolve()
_E2E_DIR = _THIS_FILE.parent
_TIER_FILES = [
    str(_E2E_DIR / "test_tier1_feature_coverage.py"),
    str(_E2E_DIR / "test_tier2_boundary_corner.py"),
    str(_E2E_DIR / "test_tier3_cross_feature.py"),
    str(_E2E_DIR / "test_tier4_real_world_scenarios.py"),
]


def _parse_pytest_counts(output: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for key in ("passed", "failed", "error", "errors", "skipped"):
        matches = re.findall(rf"(\d+)\s+{key}\b", output)
        if matches:
            target_k = "errors" if key.startswith("error") else key
            counts[target_k] = max(counts[target_k], int(matches[-1]))
    return counts


def _run_child_pytest(
    project_root: Path,
    target_files: list[str],
    *,
    k_expr: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    # Guard 2: Environment variable + explicit --ignore CLI flag
    env["MACAO_E2E_IN_SELF_TEST"] = "1"
    env["MACAO_EQUITY_ROOT"] = str(project_root)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        f"--project-root={project_root}",
        f"--ignore={_THIS_FILE}",
        *target_files,
    ]
    if k_expr:
        cmd.extend(["-k", k_expr])
    return subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)


@pytest.fixture(scope="module")
def base_valid_synthetic_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the valid Macao-scale synthetic project once per module for fast fault cloning."""
    base_dir = tmp_path_factory.mktemp("valid_macao_synth_base")
    return build_valid_synthetic_project(base_dir)


def test_self_validation_valid_synthetic_project_passes_all_tiers(
    base_valid_synthetic_root: Path,
) -> None:
    """Valid synthetic Macao project MUST pass 100% of Tiers 1-4 (183 tests, 0 failed/skipped)."""
    proc = _run_child_pytest(base_valid_synthetic_root, _TIER_FILES)
    combined = proc.stdout + "\n" + proc.stderr
    counts = _parse_pytest_counts(combined)
    assert proc.returncode == 0, (
        f"Valid synthetic project failed E2E suite (returncode={proc.returncode}):\n{combined}"
    )
    assert counts["passed"] == 183, (
        f"Expected exact 183 passed tests across Tiers 1-4 (91+71+12+9), got {counts}: {combined}"
    )
    assert counts["failed"] == 0, f"Expected 0 failed, got {counts}"
    assert counts["errors"] == 0, f"Expected 0 errors, got {counts}"
    assert counts["skipped"] == 0, f"Expected 0 skipped, got {counts}"


_FAULT_SPECS: tuple[tuple[str, list[str], str, str, int], ...] = (
    (
        "wrong_crs_epsg4326",
        [str(_E2E_DIR / "test_tier2_boundary_corner.py")],
        "test_t2_crs_strict_epsg32649_all_gpkg",
        r"CRS violation|epsg=4326",
        1,
    ),
    (
        "crs_datum_315m_shift",
        [str(_E2E_DIR / "test_tier2_boundary_corner.py")],
        "test_t2_datum_shift_315m_cross_layer_alignment",
        r"315m EPSG:8433 datum-shift detected",
        1,
    ),
    (
        "cem_latlon_swapped",
        [str(_E2E_DIR / "test_tier2_boundary_corner.py")],
        "test_t2_macao_utm49n_envelope_bounds",
        r"out of Macao UTM49N bounds",
        1,
    ),
    (
        "maritime_z26_included",
        [str(_E2E_DIR / "test_tier2_boundary_corner.py")],
        "test_t2_zones_23_exact_count_and_no_maritime_z26",
        r"Expected 23 terrestrial zones|Maritime zone",
        1,
    ),
    (
        "zero_pop_and_pop_drift",
        [str(_E2E_DIR / "test_tier2_boundary_corner.py")],
        (
            "test_t2_zero_pop_exclusion_all_spatial_units "
            "or test_t2_terrestrial_census_conservation_681293_82751"
        ),
        r"Zero/negative population|Population conservation drift",
        2,
    ),
    (
        "ev_capacity_3x_double_count",
        [str(_E2E_DIR / "test_tier2_boundary_corner.py")],
        "test_t2_cem_ev_capacity_anti_double_count",
        r"EV capacity double-count",
        1,
    ),
    (
        "ev_ml_feature_leak",
        [str(_E2E_DIR / "test_tier2_boundary_corner.py")],
        "test_t2_anti_leakage_feature_manifest_and_ml_json",
        r"overlaps excluded_target_and_audit|Target/EV leak",
        1,
    ),
    (
        "poi_text_ev_leak",
        [str(_E2E_DIR / "test_tier2_boundary_corner.py")],
        "test_t2_anti_leakage_poi_texts_ev_regex_zero_hits",
        r"POI text EV leak",
        1,
    ),
    (
        "graph_edge_and_island_drop",
        [str(_E2E_DIR / "test_tier2_boundary_corner.py")],
        (
            "test_t2_graph_edges_walk_time_strictly_positive_le_10 "
            "or test_t2_graph_dual_landmass_and_no_cross_sea_teleport"
        ),
        r"snap singularity|Taipa-Cotai-Coloane pedestrian edges missing",
        2,
    ),
    (
        "fold_empty_positive_class",
        [str(_E2E_DIR / "test_tier2_boundary_corner.py")],
        "test_t2_spatial_blocks_5_folds_both_ev_classes_geocoherent",
        r"missing class balance",
        1,
    ),
    (
        "theil_additivity_broken",
        [str(_E2E_DIR / "test_tier2_boundary_corner.py")],
        "test_t2_h1_theil_t_exact_additivity_with_zero_acc",
        r"Theil's T additivity broken",
        1,
    ),
    (
        "h2_diff_and_holm_broken",
        [str(_E2E_DIR / "test_tier2_boundary_corner.py")],
        "test_t2_h2_gini_diff_ci_and_holm_ordering",
        r"H2 gini_diff mismatch|Holm p-value ordering",
        1,
    ),
    (
        "h3_pretest_switching",
        [str(_E2E_DIR / "test_tier2_boundary_corner.py")],
        "test_t2_h3_no_pretest_switching_all_3_models",
        r"H3 pretest switching detected",
        1,
    ),
    (
        "freeze_sha256_tampered",
        [str(_E2E_DIR / "test_tier3_cross_feature.py")],
        "test_t3_09_freeze_manifest_sha256_cryptographic_match",
        r"Cryptographic SHA-256 mismatch",
        1,
    ),
    (
        "table1_and_pdf_report_violations",
        [
            str(_E2E_DIR / "test_tier2_boundary_corner.py"),
            str(_E2E_DIR / "test_tier3_cross_feature.py"),
        ],
        (
            "test_t2_figures_exist_valid_png_300dpi_nonblank "
            "or test_t2_pdf_a4_dimensions_every_page "
            "or test_t2_pdf_main_text_page_budget_3_0_to_3_8 "
            "or test_t3_10_table1_csv_md_tex_numeric_consistency"
        ),
        r"DPI below 300|is not A4|outside Phase A budget|missing from Table 1",
        4,
    ),
)


@pytest.mark.parametrize(
    ("fault_id", "target_files", "k_expr", "expected_regex", "min_failed"),
    _FAULT_SPECS,
    ids=[spec[0] for spec in _FAULT_SPECS],
)
def test_self_validation_single_fault_mutation_detected(
    base_valid_synthetic_root: Path,
    tmp_path: Path,
    fault_id: str,
    target_files: list[str],
    k_expr: str,
    expected_regex: str,
    min_failed: int,
) -> None:
    """Every injected single-fault mutation MUST fail with returncode == 1 and diagnostic output."""
    mutated_root = tmp_path / f"mut_{fault_id}"
    shutil.copytree(base_valid_synthetic_root, mutated_root)
    apply_fault(mutated_root, fault_id)

    proc = _run_child_pytest(mutated_root, target_files, k_expr=k_expr)
    combined = proc.stdout + "\n" + proc.stderr
    counts = _parse_pytest_counts(combined)

    # Strictly assert returncode == 1 (pytest ExitCode.TESTS_FAILED), NEVER != 0!
    assert proc.returncode == 1, (
        f"Fault {fault_id!r} did not exit with returncode == 1 (got {proc.returncode}):\n{combined}"
    )
    assert counts["failed"] >= min_failed and counts["errors"] == 0, (
        f"Fault {fault_id!r} unexpected failure counts {counts} "
        f"(min_failed={min_failed}):\n{combined}"
    )
    assert re.search(expected_regex, combined, re.I), (
        f"Fault {fault_id!r} output did not match expected pattern {expected_regex!r}:\n{combined}"
    )
