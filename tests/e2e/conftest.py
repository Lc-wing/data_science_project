"""Pytest configuration and cached opaque-box artifact fixtures for tests/e2e/."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

_E2E_DIR = str(Path(__file__).resolve().parent)
if _E2E_DIR not in sys.path:
    sys.path.insert(0, _E2E_DIR)

from e2e_lib.contracts import (  # noqa: E402
    DEFAULT_REPO_ROOT,
    REPORT_AUDIT_REL,
    REPORT_BIB_REL,
    REPORT_PDF_REL,
    REPORT_TEX_REL,
    resolve_provenance_paths,
    resolve_zones_with_2sfca_path,
)
from e2e_lib.readers import (  # noqa: E402
    GpkgLayer,
    GraphMLSummary,
    PdfInfo,
    PngInfo,
    inspect_pdf,
    inspect_png,
    read_csv_strict,
    read_gpkg,
    read_graphml_summary,
    read_json_strict,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register --project-root CLI option for targeting live or synthetic project trees."""
    parser.addoption(
        "--project-root",
        action="store",
        default=None,
        help="Path to the target MacaoSpatialEquity project root directory.",
    )


@pytest.fixture(scope="session")
def project_root(request: pytest.FixtureRequest) -> Path:
    """Resolve target root: --project-root > MACAO_EQUITY_ROOT > canonical repo path."""
    cli_val = request.config.getoption("--project-root")
    env_val = os.environ.get("MACAO_EQUITY_ROOT")
    raw = cli_val or (env_val if env_val and env_val.strip() else None)
    root = Path(raw).resolve() if raw else DEFAULT_REPO_ROOT.resolve()
    if not root.exists() or not root.is_dir():
        pytest.fail(f"Target --project-root does not exist or is not a directory: {root}")
    return root


@pytest.fixture(scope="session")
def load_gpkg(project_root: Path) -> Callable[..., GpkgLayer]:
    """Session-cached strictly read-only (?mode=ro) GeoPackage loader."""
    cache: dict[tuple[Path, str | None, bool], GpkgLayer] = {}

    def _loader(
        rel_or_abs: str | Path,
        *,
        layer: str | None = None,
        allow_multi_layer: bool = False,
    ) -> GpkgLayer:
        path = Path(rel_or_abs)
        if not path.is_absolute():
            path = project_root / path
        key = (path.resolve(), layer, allow_multi_layer)
        if key not in cache:
            cache[key] = read_gpkg(path, layer=layer, allow_multi_layer=allow_multi_layer)
        return cache[key]

    return _loader


@pytest.fixture(scope="session")
def load_csv(project_root: Path) -> Callable[[str | Path], tuple[list[str], list[dict[str, str]]]]:
    """Session-cached strict CSV loader."""
    cache: dict[Path, tuple[list[str], list[dict[str, str]]]] = {}

    def _loader(rel_or_abs: str | Path) -> tuple[list[str], list[dict[str, str]]]:
        path = Path(rel_or_abs)
        if not path.is_absolute():
            path = project_root / path
        resolved = path.resolve()
        if resolved not in cache:
            cache[resolved] = read_csv_strict(resolved)
        return cache[resolved]

    return _loader


@pytest.fixture(scope="session")
def load_json(project_root: Path) -> Callable[[str | Path], Any]:
    """Session-cached strict JSON loader (rejecting bare NaN/Infinity)."""
    cache: dict[Path, Any] = {}

    def _loader(rel_or_abs: str | Path) -> Any:
        path = Path(rel_or_abs)
        if not path.is_absolute():
            path = project_root / path
        resolved = path.resolve()
        if resolved not in cache:
            cache[resolved] = read_json_strict(resolved)
        return cache[resolved]

    return _loader


@pytest.fixture(scope="session")
def load_png(project_root: Path) -> Callable[[str | Path], PngInfo]:
    """Session-cached PNG inspector."""
    cache: dict[Path, PngInfo] = {}

    def _loader(rel_or_abs: str | Path) -> PngInfo:
        path = Path(rel_or_abs)
        if not path.is_absolute():
            path = project_root / path
        resolved = path.resolve()
        if resolved not in cache:
            cache[resolved] = inspect_png(resolved)
        return cache[resolved]

    return _loader


@pytest.fixture(scope="session")
def cells_r9(load_gpkg: Callable[..., GpkgLayer]) -> GpkgLayer:
    return load_gpkg("data/frozen/cells.gpkg")


@pytest.fixture(scope="session")
def cells_r8(load_gpkg: Callable[..., GpkgLayer]) -> GpkgLayer:
    return load_gpkg("data/frozen/cells_res8.gpkg")


@pytest.fixture(scope="session")
def zones_23(load_gpkg: Callable[..., GpkgLayer]) -> GpkgLayer:
    return load_gpkg("data/frozen/zones_23.gpkg")


@pytest.fixture(scope="session")
def zones_23_with_2sfca(
    project_root: Path,
    load_gpkg: Callable[..., GpkgLayer],
) -> GpkgLayer:
    return load_gpkg(resolve_zones_with_2sfca_path(project_root))


@pytest.fixture(scope="session")
def parishes_7(load_gpkg: Callable[..., GpkgLayer]) -> GpkgLayer:
    return load_gpkg("data/frozen/parishes_7.gpkg")


@pytest.fixture(scope="session")
def facilities(load_gpkg: Callable[..., GpkgLayer]) -> GpkgLayer:
    return load_gpkg("data/frozen/facilities_clean.gpkg", allow_multi_layer=True)


@pytest.fixture(scope="session")
def lrt_stations(load_gpkg: Callable[..., GpkgLayer]) -> GpkgLayer:
    return load_gpkg("data/frozen/lrt_stations.gpkg")


@pytest.fixture(scope="session")
def graph_edges(
    load_csv: Callable[[str | Path], tuple[list[str], list[dict[str, str]]]],
) -> tuple[list[str], list[dict[str, str]]]:
    return load_csv("data/frozen/graph_edges.csv")


@pytest.fixture(scope="session")
def spatial_folds(
    load_csv: Callable[[str | Path], tuple[list[str], list[dict[str, str]]]],
) -> tuple[list[str], list[dict[str, str]]]:
    return load_csv("data/frozen/spatial_folds.csv")


@pytest.fixture(scope="session")
def poi_texts(
    load_csv: Callable[[str | Path], tuple[list[str], list[dict[str, str]]]],
) -> tuple[list[str], list[dict[str, str]]]:
    return load_csv("data/frozen/cell_poi_texts.csv")


@pytest.fixture(scope="session")
def feature_manifest(load_json: Callable[[str | Path], Any]) -> dict[str, Any]:
    return load_json("data/frozen/feature_manifest.json")


@pytest.fixture(scope="session")
def split_metadata(load_json: Callable[[str | Path], Any]) -> dict[str, Any]:
    return load_json("data/frozen/split_metadata.json")


@pytest.fixture(scope="session")
def freeze_manifest(load_json: Callable[[str | Path], Any]) -> dict[str, Any]:
    return load_json("data/frozen/FREEZE_MANIFEST.json")


@pytest.fixture(scope="session")
def provenance_info(
    project_root: Path,
    load_json: Callable[[str | Path], Any],
) -> tuple[list[Path], Any]:
    paths = resolve_provenance_paths(project_root)
    if not paths:
        pytest.fail(
            "Neither outputs/metrics/data_provenance.json nor "
            f"data/raw/data_provenance.json exists under {project_root}"
        )
    return paths, load_json(paths[0])


@pytest.fixture(scope="session")
def walk_graph_summary(project_root: Path) -> GraphMLSummary:
    return read_graphml_summary(project_root / "data/raw/osm_walk_graph.graphml")


@pytest.fixture(scope="session")
def h1_table(
    load_csv: Callable[[str | Path], tuple[list[str], list[dict[str, str]]]],
) -> tuple[list[str], list[dict[str, str]]]:
    return load_csv("outputs/tables/h1_inequality_lisa.csv")


@pytest.fixture(scope="session")
def h2_table(
    load_csv: Callable[[str | Path], tuple[list[str], list[dict[str, str]]]],
) -> tuple[list[str], list[dict[str, str]]]:
    return load_csv("outputs/tables/h2_gini_diff_bootstrap.csv")


@pytest.fixture(scope="session")
def h3_table(
    load_csv: Callable[[str | Path], tuple[list[str], list[dict[str, str]]]],
) -> tuple[list[str], list[dict[str, str]]]:
    return load_csv("outputs/tables/h3_spatial_regression.csv")


@pytest.fixture(scope="session")
def h3_diagnostics(load_json: Callable[[str | Path], Any]) -> dict[str, Any]:
    return load_json("outputs/tables/h3_regression_diagnostics.json")


@pytest.fixture(scope="session")
def maup_table(
    load_csv: Callable[[str | Path], tuple[list[str], list[dict[str, str]]]],
) -> tuple[list[str], list[dict[str, str]]]:
    return load_csv("outputs/tables/maup_r8_vs_r9.csv")


@pytest.fixture(scope="session")
def eda_table(
    load_csv: Callable[[str | Path], tuple[list[str], list[dict[str, str]]]],
) -> tuple[list[str], list[dict[str, str]]]:
    return load_csv("outputs/tables/eda_summary.csv")


@pytest.fixture(scope="session")
def lorenz_table(
    load_csv: Callable[[str | Path], tuple[list[str], list[dict[str, str]]]],
) -> tuple[list[str], list[dict[str, str]]]:
    return load_csv("outputs/tables/lorenz_curves.csv")


@pytest.fixture(scope="session")
def ml_baselines(load_json: Callable[[str | Path], Any]) -> dict[str, Any]:
    return load_json("outputs/metrics/phase_a_ml_baselines.json")


@pytest.fixture(scope="session")
def ml_cv_table(
    load_csv: Callable[[str | Path], tuple[list[str], list[dict[str, str]]]],
) -> tuple[list[str], list[dict[str, str]]]:
    return load_csv("outputs/tables/ml_cv_results.csv")


@pytest.fixture(scope="session")
def ml_oof_table(
    load_csv: Callable[[str | Path], tuple[list[str], list[dict[str, str]]]],
) -> tuple[list[str], list[dict[str, str]]]:
    return load_csv("outputs/tables/ml_oof_predictions.csv")


@pytest.fixture(scope="session")
def report_pdf_info(project_root: Path) -> PdfInfo:
    return inspect_pdf(project_root / REPORT_PDF_REL)


@pytest.fixture(scope="session")
def report_tex_text(project_root: Path) -> str:
    return (project_root / REPORT_TEX_REL).read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def report_bib_text(project_root: Path) -> str:
    return (project_root / REPORT_BIB_REL).read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def go_nogo_audit_text(project_root: Path) -> str:
    return (project_root / REPORT_AUDIT_REL).read_text(encoding="utf-8")
