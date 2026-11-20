"""Tier 5 white-box single-fault adversarial hardening.

TEST_INFRA.md section 5 names five mutations that the Tier 1-4 self-validation
file only injects inside a bundle (`graph_edge_and_island_drop` or
`table1_and_pdf_report_violations`). Each case below injects one of those
mutators and requires every diagnostic that mutator is written to trip.
A match on only one clause of a multi-clause fault is a failure of this tier.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

if os.environ.get("MACAO_E2E_IN_SELF_TEST") == "1":
    pytest.skip(
        "Skipping test_tier5_adversarial.py inside a child fault-injection subprocess",
        allow_module_level=True,
    )

from e2e_lib.synthetic import apply_fault, build_valid_synthetic_project  # noqa: E402

_THIS_FILE = Path(__file__).resolve()
_E2E_DIR = _THIS_FILE.parent
_SELF_VALIDATION = _E2E_DIR / "test_e2e_self_validation.py"
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


def _run_child_pytest(
    project_root: Path,
    target_files: list[Path],
    *,
    k_expr: str,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
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
        f"--ignore={_SELF_VALIDATION}",
        *[str(path) for path in target_files],
        "-k",
        k_expr,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)


@pytest.fixture(scope="module")
def base_valid_synthetic_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One contract-valid synthetic root, cloned per fault so faults cannot leak."""
    base_dir = tmp_path_factory.mktemp("tier5_valid_macao")
    return build_valid_synthetic_project(base_dir)


# fault_id, target files, -k expression, fragments that must all appear, min failed tests
_TIER5_SPECS: tuple[tuple[str, tuple[Path, ...], str, tuple[str, ...], int], ...] = (
    (
        "graph_edge_zero_or_gt10",
        (_TIER2,),
        (
            "test_t2_graph_edges_walk_time_strictly_positive_le_10 "
            "or test_t2_graph_edges_undirected_symmetry_unique"
        ),
        ("snap singularity", "exceeds 10.0", "Asymmetric walk_time_min"),
        2,
    ),
    (
        "graph_dropped_islands",
        (_TIER2,),
        "test_t2_graph_dual_landmass_and_no_cross_sea_teleport",
        ("Taipa-Cotai-Coloane pedestrian edges missing",),
        1,
    ),
    (
        "table1_numeric_drift",
        (_TIER3,),
        "test_t3_10_table1_csv_md_tex_numeric_consistency",
        ("missing from Table 1",),
        1,
    ),
    (
        "pdf_out_of_budget_and_us_letter",
        (_TIER2,),
        ("test_t2_pdf_a4_dimensions_every_page or test_t2_pdf_main_text_page_budget_3_0_to_3_8"),
        ("is not A4", "outside Phase A budget"),
        2,
    ),
    (
        "figure_low_dpi_150",
        (_TIER2,),
        "test_t2_figures_exist_valid_png_300dpi_nonblank",
        ("DPI below 300",),
        1,
    ),
)


@pytest.mark.parametrize(
    ("fault_id", "target_files", "k_expr", "required_fragments", "min_failed"),
    _TIER5_SPECS,
    ids=[spec[0] for spec in _TIER5_SPECS],
)
def test_tier5_named_single_fault_is_detected(
    base_valid_synthetic_root: Path,
    tmp_path: Path,
    fault_id: str,
    target_files: tuple[Path, ...],
    k_expr: str,
    required_fragments: tuple[str, ...],
    min_failed: int,
) -> None:
    """One named single-fault mutation must fail with returncode 1 and every clause."""
    mutated_root = tmp_path / f"tier5_{fault_id}"
    shutil.copytree(base_valid_synthetic_root, mutated_root)
    apply_fault(mutated_root, fault_id)

    proc = _run_child_pytest(mutated_root, list(target_files), k_expr=k_expr)
    combined = proc.stdout + "\n" + proc.stderr
    counts = _parse_pytest_counts(combined)

    assert proc.returncode == 1, (
        f"Fault {fault_id!r} did not exit with returncode == 1 (got {proc.returncode}):\n{combined}"
    )
    assert counts["failed"] >= min_failed and counts["errors"] == 0, (
        f"Fault {fault_id!r} unexpected failure counts {counts} "
        f"(min_failed={min_failed}):\n{combined}"
    )
    missing = [fragment for fragment in required_fragments if not re.search(fragment, combined)]
    assert not missing, f"Fault {fault_id!r} did not trip every named clause {missing}:\n{combined}"
