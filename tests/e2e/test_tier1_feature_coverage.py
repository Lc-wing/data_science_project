"""Tier 1 E2E Tests: Isolated requirement coverage across all Phase A features (A01-A16)."""

from __future__ import annotations

import ast
import importlib.util
import math
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from e2e_lib.contracts import (
    A4_HEIGHT_PT,
    A4_TOL_PT,
    A4_WIDTH_PT,
    ACC_INDIVIDUAL_COLS,
    ALL_ACC_COLS,
    ANALYTICAL_PLACEHOLDER_RE,
    CELLS_BASE_R8_REQUIRED_COLS,
    CELLS_BASE_R9_REQUIRED_COLS,
    DEFAULT_REPO_ROOT,
    EV_CAPACITY_MAX,
    EV_CAPACITY_MIN,
    EV_COL_LEAK_RE,
    EV_TEXT_LEAK_RE,
    FACILITIES_REQUIRED_COLS,
    FACILITY_TYPES,
    FORBIDDEN_HYPE_PHRASES,
    FROZEN_CANONICAL_FILES,
    FROZEN_CORE_HASHED_FILES,
    GLOBAL_SEED,
    GRAPH_EDGES_REQUIRED_COLS,
    H1_REQUIRED_COLS,
    H2_REQUIRED_COLS,
    H3_MODEL_TYPES,
    H3_R8_COUNT_MIN,
    H3_R8_RE,
    H3_R9_COUNT_BOUNDS,
    H3_R9_RE,
    H3_REQUIRED_COLS,
    LISA_CLUSTERS,
    LRT_COUNT_BOUNDS,
    MACAO_POLY_E_MAX,
    MACAO_POLY_E_MIN,
    MACAO_POLY_N_MAX,
    MACAO_POLY_N_MIN,
    MACAO_PT_E_MAX,
    MACAO_PT_E_MIN,
    MACAO_PT_N_MAX,
    MACAO_PT_N_MIN,
    MARITIME_ZONE_RE,
    MAX_GRAPH_WALK_MIN,
    ML_AGGREGATE_KEYS,
    ML_FEATURES_REQUIRED,
    ML_MODEL_NAMES,
    NON_PUBLIC_LIBRARY_RE,
    PARISHES_7_REQUIRED_COLS,
    PDF_MAIN_PAGES_MAX,
    PDF_MAIN_PAGES_MIN,
    PNG_MIN_BYTES,
    PNG_MIN_DPI,
    PNG_MIN_PIXEL_STD,
    POI_TEXTS_REQUIRED_COLS,
    POP_TOL_H3_65PLUS,
    POP_TOL_H3_TOTAL,
    POP_TOL_ZONES_PARISHES,
    REPORT_AUDIT_REL,
    REPORT_BIB_REL,
    REPORT_PDF_REL,
    REPORT_TEX_REL,
    REQUIRED_FIGURES,
    REQUIRED_PYPROJECT_PACKAGES,
    SHA256_RE,
    SPATIAL_FOLDS_REQUIRED_COLS,
    TABLE1_FILES,
    TABLE_PLACEHOLDER_RE,
    TAIPA_PARISH_RE,
    TARGET_AND_AUDIT_COLS,
    TARGET_EPSG,
    TERRESTRIAL_POP_65PLUS,
    TERRESTRIAL_POP_TOTAL,
    TRADITIONAL_FACILITY_TYPES,
    ZONES_23_REQUIRED_COLS,
    normalize_resolution_label,
    normalize_weights_label,
)
from e2e_lib.oracle import (
    average_precision_score,
    brier_score,
    holm_bonferroni,
    ols_hc3_and_vif,
    point_distance_to_polygons,
    roc_auc_score,
    sentence_aware_unhedged_causal_hits,
    spearman_corr,
    top20_hit_rates,
    weighted_gini,
    weighted_theil_t,
)
from e2e_lib.readers import (
    GpkgLayer,
    GraphMLSummary,
    PdfInfo,
    PngInfo,
    as_bool,
    as_float,
    file_sha256,
    nearest_node_snap_fraction,
    read_csv_strict,
    read_toml_strict,
)

# ---------------------------------------------------------------------------
# Internal opaque-box normalization helpers (zero src/ coupling)
# ---------------------------------------------------------------------------


def _extract_provenance_entries(prov_data: Any) -> list[dict[str, Any]]:
    if isinstance(prov_data, list):
        return [e for e in prov_data if isinstance(e, dict)]
    if isinstance(prov_data, dict):
        for key in ("sources", "datasets", "files", "entries"):
            val = prov_data.get(key)
            if isinstance(val, list):
                return [e for e in val if isinstance(e, dict)]
            if isinstance(val, dict):
                return [
                    {"name": k, **v} if isinstance(v, dict) else {"name": k, "value": v}
                    for k, v in val.items()
                ]
        dict_vals = [{"name": k, **v} for k, v in prov_data.items() if isinstance(v, dict)]
        if dict_vals:
            return dict_vals
    return []


def _extract_freeze_file_map(manifest: dict[str, Any]) -> dict[str, str]:
    raw_map: dict[str, Any] = {}
    for key in ("files", "checksums", "sha256", "artifacts"):
        val = manifest.get(key)
        if isinstance(val, dict):
            raw_map = val
            break
    if not raw_map:
        raw_map = {
            k: v
            for k, v in manifest.items()
            if isinstance(v, str) and SHA256_RE.match(v.strip().lower())
        }
    out: dict[str, str] = {}
    for rel, val in raw_map.items():
        if isinstance(val, dict):
            digest = str(val.get("sha256", "")).strip().lower()
        else:
            digest = str(val).strip().lower()
        norm_rel = rel if rel.startswith("data/") else f"data/frozen/{rel}"
        out[norm_rel] = digest
    return out


def _detect_lisa_metrics(columns: list[str]) -> list[str]:
    metrics: list[str] = []
    for col in columns:
        if col.startswith("lisa_cluster_"):
            metrics.append(col.removeprefix("lisa_cluster_"))
    return sorted(metrics)


def _match_h2_facility(comparison: str) -> str | None:
    low = comparison.lower()
    if "park" in low:
        return "park"
    if "health" in low:
        return "health_centre"
    if "librar" in low:
        return "library"
    if "bus" in low:
        return "bus_stop"
    return None


def _find_lorenz_cols(headers: list[str]) -> tuple[str, str]:
    pop_col = next(
        (h for h in headers if h.lower() in {"cum_pop_share", "pop_share", "cum_pop", "p", "x"}),
        None,
    )
    acc_col = next(
        (h for h in headers if h.lower() in {"cum_acc_share", "acc_share", "cum_acc", "q", "y"}),
        None,
    )
    assert pop_col and acc_col, f"Could not identify Lorenz p/q columns in {headers}"
    return pop_col, acc_col


def _find_oof_cols(headers: list[str]) -> tuple[str, dict[str, str]]:
    y_col = next((h for h in headers if h.lower() in {"has_ev_charger", "y_true", "target"}), None)
    assert y_col, f"Missing target column in ml_oof_predictions.csv: {headers}"
    model_cols: dict[str, str] = {}
    for m_name in ML_MODEL_NAMES:
        candidates = [
            f"prob_{m_name}",
            f"pred_{m_name}",
            f"{m_name}_prob",
            m_name,
        ]
        found = next((h for h in headers if h.lower() in candidates), None)
        if found is None:
            token = m_name.split("_")[0]
            found = next((h for h in headers if token in h.lower() and h != y_col), None)
        assert found, f"Missing OOF probability column for {m_name} in {headers}"
        model_cols[m_name] = found
    return y_col, model_cols


def _assert_png_high_dpi_nonblank(png: PngInfo) -> None:
    assert png.crc_valid, f"Corrupt PNG CRC32 in {png.path}"
    assert png.size_bytes >= PNG_MIN_BYTES, (
        f"PNG {png.path.name} too small ({png.size_bytes} bytes < {PNG_MIN_BYTES})"
    )
    has_dpi = (
        png.dpi_x is not None
        and png.dpi_y is not None
        and png.dpi_x >= PNG_MIN_DPI
        and png.dpi_y >= PNG_MIN_DPI
    )
    assert has_dpi, f"PNG {png.path.name} DPI below 300: ({png.dpi_x}, {png.dpi_y})"
    assert png.pixel_std > PNG_MIN_PIXEL_STD, (
        f"PNG {png.path.name} appears blank (pixel_std={png.pixel_std:.2f})"
    )


def _extract_numeric_tokens(text: str) -> list[float]:
    return [float(m.group(0)) for m in re.finditer(r"-?\d+\.\d+", text)]


# ---------------------------------------------------------------------------
# Opaque-Box Purity Guard (AST Verification)
# ---------------------------------------------------------------------------


def _assert_opaque_box_zero_src_imports() -> None:
    """AST-verify that ZERO files under tests/e2e/ import macao_equity or src."""
    e2e_root = Path(__file__).resolve().parent
    py_files = sorted(e2e_root.rglob("*.py"))
    assert len(py_files) >= 6, f"Expected E2E python files under {e2e_root}"
    violations: list[str] = []
    for py_path in py_files:
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in {"macao_equity", "src"} or "macao_equity" in alias.name:
                        violations.append(f"{py_path.name}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                top = mod.split(".")[0]
                if top in {"macao_equity", "src"} or "macao_equity" in mod:
                    violations.append(f"{py_path.name}:{node.lineno}: from {mod} import ...")
    assert not violations, f"Opaque-box violation: internal imports found: {violations}"


# ---------------------------------------------------------------------------
# Group I: Environment, Provenance & Spatial Data Foundation (A01-A04)
# ---------------------------------------------------------------------------


class TestA01Environment:
    """A01: Reproducible Geospatial Python 3.11+ Environment (5 isolated tests)."""

    def test_a01_pyproject_metadata_and_dependencies(self, project_root: Path) -> None:
        toml_data = read_toml_strict(project_root / "pyproject.toml")
        project = toml_data.get("project", {})
        req_py = str(project.get("requires-python", ""))
        assert re.search(r">=\s*3\.1[1-9]", req_py), f"requires-python must be >=3.11: {req_py!r}"

        deps: list[str] = list(project.get("dependencies", []))
        for opt_deps in project.get("optional-dependencies", {}).values():
            deps.extend(opt_deps)
        for grp_deps in toml_data.get("dependency-groups", {}).values():
            if isinstance(grp_deps, list):
                deps.extend(str(d) for d in grp_deps if isinstance(d, str))
        dep_names = {re.split(r"[<>=!~;\[\s]", d.strip().lower())[0] for d in deps}
        missing = [pkg for pkg in REQUIRED_PYPROJECT_PACKAGES if pkg not in dep_names]
        assert not missing, f"Missing required dependencies in pyproject.toml: {missing}"

    def test_a01_ruff_and_pytest_config(self, project_root: Path) -> None:
        _assert_opaque_box_zero_src_imports()
        toml_data = read_toml_strict(project_root / "pyproject.toml")
        tool = toml_data.get("tool", {})
        ruff = tool.get("ruff", {})
        assert ruff.get("line-length") == 100, f"Expected ruff line-length=100, got {ruff}"
        target_v = str(ruff.get("target-version", ""))
        assert target_v in {"py311", "py312", "py313"}, (
            f"Unexpected ruff target-version: {target_v!r}"
        )
        assert "pytest" in tool, "Missing [tool.pytest.*] section in pyproject.toml"

    def test_a01_lockfile_and_gitignore_hygiene(self, project_root: Path) -> None:
        uv_lock = project_root / "uv.lock"
        assert uv_lock.is_file() and uv_lock.stat().st_size > 100, "uv.lock missing or empty"
        gitignore = (project_root / ".gitignore").read_text(encoding="utf-8")
        for token in (".venv", "__pycache__", ".pytest_cache", ".env"):
            assert token in gitignore, f".gitignore missing required entry: {token}"

    def test_a01_geospatial_stack_imports_and_api_versions(self, project_root: Path) -> None:
        venv_py = project_root / ".venv" / "bin" / "python"
        fallback_venv_py = DEFAULT_REPO_ROOT / ".venv" / "bin" / "python"
        is_real_repo = project_root.resolve() == DEFAULT_REPO_ROOT.resolve()
        if not venv_py.is_file() and is_real_repo:
            raise AssertionError(f"Real repository requires .venv/bin/python at {venv_py}")

        probe_py: str | None = None
        if venv_py.is_file():
            probe_py = str(venv_py)
        elif fallback_venv_py.is_file():
            probe_py = str(fallback_venv_py)
        elif importlib.util.find_spec("geopandas") is not None:
            probe_py = sys.executable

        if probe_py is not None:
            snippet = (
                "import geopandas, shapely, pyproj, h3, osmnx, networkx, "
                "libpysal, esda, spreg, statsmodels, sklearn, matplotlib; "
                "assert hasattr(h3, 'latlng_to_cell'), 'h3 v4 required'; "
                "assert int(osmnx.__version__.split('.')[0]) >= 2, 'osmnx v2+ required'; "
                "assert pyproj.CRS.from_epsg(8433) is not None; "
                "assert pyproj.CRS.from_epsg(32649) is not None; "
                "print('STACK_OK')"
            )
            proc = subprocess.run(
                [probe_py, "-c", snippet], capture_output=True, text=True, check=False
            )
            assert proc.returncode == 0 and "STACK_OK" in proc.stdout, (
                f"Geospatial stack probe failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
            )
        else:
            assert sys.version_info >= (3, 11), "Python >= 3.11 required"
            lock_txt = (project_root / "uv.lock").read_text(encoding="utf-8")
            for pkg in ("geopandas", "h3", "osmnx", "spreg"):
                assert pkg in lock_txt, f"uv.lock missing core stack package {pkg!r}"

    def test_a01_headless_matplotlib_and_pysal_pandas_compat(self, project_root: Path) -> None:
        venv_py = project_root / ".venv" / "bin" / "python"
        fallback_venv_py = DEFAULT_REPO_ROOT / ".venv" / "bin" / "python"
        is_real_repo = project_root.resolve() == DEFAULT_REPO_ROOT.resolve()
        if not venv_py.is_file() and is_real_repo:
            raise AssertionError(f"Real repository requires .venv/bin/python at {venv_py}")

        probe_py: str | None = None
        if venv_py.is_file():
            probe_py = str(venv_py)
        elif fallback_venv_py.is_file():
            probe_py = str(fallback_venv_py)
        elif importlib.util.find_spec("matplotlib") and importlib.util.find_spec("spreg"):
            probe_py = sys.executable

        if probe_py is not None:
            snippet = (
                "import io, numpy as np, pandas as pd, matplotlib; "
                "matplotlib.use('Agg'); import matplotlib.pyplot as plt; "
                "fig, ax = plt.subplots(); ax.plot([0, 1], [0, 1]); "
                "buf = io.BytesIO(); fig.savefig(buf, format='png'); plt.close(fig); "
                "assert len(buf.getvalue()) > 100; "
                "import libpysal, esda, spreg; "
                "pts = np.array([[0,0],[1,0],[0,1],[1,1],[2,0],[2,1],[0,2],[1,2]], dtype=float); "
                "w = libpysal.weights.KNN.from_array(pts, k=2); w.transform = 'r'; "
                "df = pd.DataFrame({'y': [1,2.1,1.3,2.5,3,3.2,1.8,2.9], 'x': [0,1,0,1,2,2,0,1]}); "
                "m = esda.Moran(df['y'].to_numpy(), w, permutations=19); "
                "ols = spreg.OLS(df[['y']].to_numpy(), df[['x']].to_numpy()); "
                "assert np.isfinite(m.I) and np.isfinite(ols.r2); "
                "print('COMPAT_OK')"
            )
            env = dict(os.environ)
            env["MPLBACKEND"] = "Agg"
            proc = subprocess.run(
                [probe_py, "-c", snippet], capture_output=True, text=True, env=env, check=False
            )
            assert proc.returncode == 0 and "COMPAT_OK" in proc.stdout, (
                f"Headless matplotlib / PySAL-pandas check failed:\n{proc.stderr}"
            )
        else:
            readme_or_toml = (
                (project_root / "pyproject.toml").read_text(encoding="utf-8")
                + "\n"
                + (project_root / "README.md").read_text(encoding="utf-8")
            )
            assert "MPLBACKEND" in readme_or_toml and "Agg" in readme_or_toml, (
                "MPLBACKEND=Agg headless configuration not documented"
            )
            assert "pandas" in (project_root / "pyproject.toml").read_text(encoding="utf-8")


class TestA02Provenance:
    """A02: Day-1 Live Data Acquisition & Provenance (5 isolated tests)."""

    def test_a02_provenance_manifest_exists_and_valid_json(
        self,
        provenance_info: tuple[list[Path], Any],
        load_json: Callable[[str | Path], Any],
    ) -> None:
        paths, primary = provenance_info
        entries = _extract_provenance_entries(primary)
        assert len(entries) >= 4, f"Expected >= 4 provenance source entries, got {len(entries)}"
        if len(paths) == 2:
            secondary = load_json(paths[1])
            sec_entries = _extract_provenance_entries(secondary)
            assert len(sec_entries) == len(entries), (
                "Dual data_provenance.json paths disagree in entry count"
            )

    def test_a02_provenance_covers_all_required_sources(
        self, provenance_info: tuple[list[Path], Any]
    ) -> None:
        _, primary = provenance_info
        blob = str(primary).lower()
        assert "dsec" in blob and ("building" in blob or "census" in blob), "Missing DSEC source"
        assert "cem" in blob and "ev" in blob, "Missing CEM EV source"
        assert "osm" in blob or "geofabrik" in blob or "overpass" in blob, "Missing OSM source"
        assert any(tok in blob for tok in ("ssm", "library", "mlm", "iam", "registry")), (
            "Missing official Macao registry cross-check source"
        )

    def test_a02_provenance_records_urls_timestamps_and_sha256(
        self,
        project_root: Path,
        provenance_info: tuple[list[Path], Any],
    ) -> None:
        _, primary = provenance_info
        entries = _extract_provenance_entries(primary)
        assert entries, "No provenance entries found"
        for entry in entries:
            url = str(entry.get("url") or entry.get("source_url") or entry.get("dataset_id") or "")
            assert "http://" in url or "https://" in url or len(url) >= 6, (
                f"Provenance entry missing URL/dataset ID: {entry}"
            )
            ts = str(
                entry.get("extracted_at")
                or entry.get("downloaded_at")
                or entry.get("timestamp")
                or entry.get("date")
                or ""
            )
            assert re.search(r"202\d-\d{2}-\d{2}", ts), f"Invalid/missing ISO date in {entry}"
            sha = entry.get("sha256")
            if sha:
                assert SHA256_RE.match(str(sha).strip().lower()), f"Malformed SHA-256 in {entry}"
                local_rel = entry.get("local_path") or entry.get("file") or entry.get("path")
                if local_rel:
                    lp = project_root / str(local_rel)
                    assert lp.is_file(), f"Provenance local snapshot missing: {lp}"
                    assert file_sha256(lp) == str(sha).strip().lower(), (
                        f"Snapshot hash mismatch for {local_rel}"
                    )

    def test_a02_raw_snapshots_present_and_nonempty(self, project_root: Path) -> None:
        raw_dir = project_root / "data" / "raw"
        assert raw_dir.is_dir(), f"Missing data/raw directory under {project_root}"
        files = [p for p in raw_dir.rglob("*") if p.is_file() and p.stat().st_size > 0]
        assert len(files) >= 3, (
            f"Expected non-empty raw snapshots under data/raw, found {len(files)}"
        )
        graphml = raw_dir / "osm_walk_graph.graphml"
        assert graphml.is_file() and graphml.stat().st_size > 0, "Missing osm_walk_graph.graphml"

    def test_a02_datagovmo_fallback_documented(
        self,
        project_root: Path,
        provenance_info: tuple[list[Path], Any],
    ) -> None:
        _, primary = provenance_info
        dd_path = project_root / "data/frozen/data_dictionary.md"
        combined = (str(primary) + "\n" + dd_path.read_text(encoding="utf-8")).lower()
        assert "data.gov.mo" in combined, "data.gov.mo auth/login gate not documented"
        assert any(tok in combined for tok in ("login", "auth", "token", "gate", "fallback")), (
            "Missing authentication gate / fallback rationale for data.gov.mo"
        )


class TestA03CrsAndFacilities:
    """A03: CRS Standardization, Datum Alignment & Cross-Source 25m Dedup (6 isolated tests)."""

    def test_a03_all_frozen_geopackages_strict_epsg32649(
        self,
        project_root: Path,
        load_gpkg: Callable[..., GpkgLayer],
    ) -> None:
        gpkg_rels = [
            "data/frozen/cells.gpkg",
            "data/frozen/cells_res8.gpkg",
            "data/frozen/zones_23.gpkg",
            "data/frozen/parishes_7.gpkg",
            "data/frozen/facilities_clean.gpkg",
            "data/frozen/lrt_stations.gpkg",
        ]
        for opt in ("data/frozen/cells_base_res9.gpkg", "data/frozen/cells_base_res8.gpkg"):
            if (project_root / opt).is_file():
                gpkg_rels.append(opt)
        for rel in gpkg_rels:
            layer = load_gpkg(rel, allow_multi_layer=(rel.endswith("facilities_clean.gpkg")))
            assert layer.epsg == TARGET_EPSG, f"{rel} has EPSG={layer.epsg}, expected {TARGET_EPSG}"
            assert len(layer.geoms) > 0, f"{rel} contains 0 rows"
            assert all(not g.is_empty and g.is_valid for g in layer.geoms), (
                f"{rel} contains empty or invalid geometries"
            )

    def test_a03_macao_projected_extent_and_no_datum_shift(
        self,
        zones_23: GpkgLayer,
        cells_r9: GpkgLayer,
        facilities: GpkgLayer,
        lrt_stations: GpkgLayer,
    ) -> None:
        for name, layer in (("zones_23", zones_23), ("cells", cells_r9)):
            for g in layer.geoms:
                assert g.bbox is not None
                minx, miny, maxx, maxy = g.bbox
                assert MACAO_POLY_E_MIN <= minx and maxx <= MACAO_POLY_E_MAX, (
                    f"{name} easting out of Macao UTM49N bounds: {g.bbox}"
                )
                assert MACAO_POLY_N_MIN <= miny and maxy <= MACAO_POLY_N_MAX, (
                    f"{name} northing out of Macao UTM49N bounds: {g.bbox}"
                )
        for name, layer in (("facilities", facilities), ("lrt_stations", lrt_stations)):
            for g in layer.geoms:
                assert g.point_xy is not None, f"{name} contains non-Point geometry {g.geom_type}"
                x, y = g.point_xy
                assert MACAO_PT_E_MIN <= x <= MACAO_PT_E_MAX, (
                    f"{name} point X out of Macao UTM49N bounds: ({x}, {y})"
                )
                assert MACAO_PT_N_MIN <= y <= MACAO_PT_N_MAX, (
                    f"{name} point Y out of Macao UTM49N bounds: ({x}, {y})"
                )

        # Cross-layer alignment (< 50 m for >= 90% of non-Zona-A EV stations vs zones_23)
        zone_polys = [poly for g in zones_23.geoms for poly in g.polygons]
        ev_dists: list[float] = []
        for row, geom in zip(facilities.rows, facilities.geoms, strict=True):
            if row["facility_type"] == "ev_charger" and not as_bool(row["in_zona_a"]):
                assert geom.point_xy is not None
                ev_dists.append(point_distance_to_polygons(*geom.point_xy, zone_polys))
        assert ev_dists, "No non-Zona-A EV stations found"
        frac_aligned = sum(1 for d in ev_dists if d <= 50.0) / len(ev_dists)
        assert frac_aligned >= 0.90, (
            f"Datum-shift alignment failure: only {frac_aligned:.1%} non-Zona-A EV stations "
            f"within 50 m of zones_23 (median={statistics.median(ev_dists):.1f} m)"
        )

    def test_a03_facilities_schema_and_five_required_types(self, facilities: GpkgLayer) -> None:
        missing_cols = [c for c in FACILITIES_REQUIRED_COLS if c not in facilities.columns]
        assert not missing_cols, f"facilities_clean.gpkg missing columns: {missing_cols}"
        ftypes = {str(r["facility_type"]) for r in facilities.rows}
        assert ftypes == set(FACILITY_TYPES), f"Expected 5 facility types, got {ftypes}"
        ids = [str(r["facility_id"]) for r in facilities.rows]
        assert len(ids) == len(set(ids)), "Duplicate facility_id values in facilities_clean.gpkg"
        for row, geom in zip(facilities.rows, facilities.geoms, strict=True):
            assert geom.point_xy is not None, f"Facility {row['facility_id']} is not a Point"
            cap = as_float(row["capacity"])
            assert cap > 0.0, f"Non-positive capacity for {row['facility_id']}: {cap}"

    def test_a03_cem_ev_station_counts_and_unique_charger_capacity(
        self, facilities: GpkgLayer
    ) -> None:
        ev_rows = [r for r in facilities.rows if r["facility_type"] == "ev_charger"]
        assert len(ev_rows) >= 80, f"Expected >= 80 CEM EV stations, got {len(ev_rows)}"
        total_cap = sum(as_float(r["capacity"]) for r in ev_rows)
        assert EV_CAPACITY_MIN <= total_cap <= EV_CAPACITY_MAX, (
            f"Total EV unique capacity {total_cap:.1f} outside "
            f"[{EV_CAPACITY_MIN}, {EV_CAPACITY_MAX}]"
        )
        pilenums = [as_float(r["pilenum"]) for r in ev_rows if r.get("pilenum") is not None]
        total_kws = [as_float(r["total_kw"]) for r in ev_rows if r.get("total_kw") is not None]
        assert len(pilenums) >= 0.9 * len(ev_rows), "pilenum missing on CEM EV stations"
        assert len(total_kws) >= 0.9 * len(ev_rows), "total_kw missing on CEM EV stations"
        assert any(as_bool(r["in_zona_a"]) for r in ev_rows), "Expected >= 1 Zona A EV station"

    def test_a03_official_facility_whitelisting_health_library_lrt(
        self,
        facilities: GpkgLayer,
        lrt_stations: GpkgLayer,
    ) -> None:
        health_rows = [r for r in facilities.rows if r["facility_type"] == "health_centre"]
        lib_rows = [r for r in facilities.rows if r["facility_type"] == "library"]
        assert 8 <= len(health_rows) <= 15, (
            f"Official SSM health_centre count out of bounds: {len(health_rows)}"
        )
        assert 10 <= len(lib_rows) <= 25, (
            f"Official IC public library count out of bounds: {len(lib_rows)}"
        )
        for r in lib_rows:
            name = str(r.get("name", ""))
            assert not NON_PUBLIC_LIBRARY_RE.search(name), (
                f"Non-public university/museum library leaked into whitelist: {name!r}"
            )
        lo_lrt, hi_lrt = LRT_COUNT_BOUNDS
        assert lo_lrt <= len(lrt_stations.rows) <= hi_lrt, (
            f"Operational MLM LRT station count out of bounds: {len(lrt_stations.rows)}"
        )
        pts = [g.point_xy for g in lrt_stations.geoms if g.point_xy is not None]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                assert d >= 25.0, f"Duplicate LRT stations within 25 m (Lotus dupe?): {d:.1f} m"

    def test_a03_cross_source_25m_dedup_without_collapsing_official_bus_poles(
        self, facilities: GpkgLayer
    ) -> None:
        by_type: dict[str, list[tuple[str, float, float, str]]] = defaultdict(list)
        for row, geom in zip(facilities.rows, facilities.geoms, strict=True):
            assert geom.point_xy is not None
            by_type[str(row["facility_type"])].append(
                (str(row["source"]), geom.point_xy[0], geom.point_xy[1], str(row["facility_id"]))
            )
        assert len(by_type["bus_stop"]) >= 250, (
            f"Too few bus stops ({len(by_type['bus_stop'])}); opposite poles may be over-merged"
        )
        for ftype, items in by_type.items():
            for i in range(len(items)):
                s1, x1, y1, id1 = items[i]
                for j in range(i + 1, len(items)):
                    s2, x2, y2, id2 = items[j]
                    if s1 == s2:
                        continue
                    dist = math.hypot(x1 - x2, y1 - y2)
                    assert dist >= 25.0, (
                        f"Cross-source duplicate < 25 m in {ftype}: "
                        f"{id1}({s1}) vs {id2}({s2}) = {dist:.2f} m"
                    )


class TestA04SpatialUnitsAndCensus:
    """A04: Spatial Units & Exact Building Census Allocation (6 isolated tests)."""

    def test_a04_zones_23_schema_count_and_no_maritime(self, zones_23: GpkgLayer) -> None:
        missing = [c for c in ZONES_23_REQUIRED_COLS if c not in zones_23.columns]
        assert not missing, f"zones_23.gpkg missing columns: {missing}"
        assert len(zones_23.rows) == 23, (
            f"Expected exact 23 terrestrial zones, got {len(zones_23.rows)}"
        )
        zids = [str(r["zone_id"]) for r in zones_23.rows]
        assert len(set(zids)) == 23, "Duplicate zone_id in zones_23.gpkg"
        for r in zones_23.rows:
            assert not MARITIME_ZONE_RE.search(str(r["zone_id"])), (
                f"Maritime zone included: {r['zone_id']}"
            )
            assert not MARITIME_ZONE_RE.search(str(r["zone_name"])), (
                f"Maritime zone included: {r['zone_name']}"
            )
            pop = as_float(r["pop_total"])
            area = as_float(r["area_km2"])
            eld = as_float(r["elderly_share"])
            dens = as_float(r["pop_density"])
            lpd = as_float(r["log_pop_density"])
            assert pop > 0.0 and area > 0.0 and 0.0 <= eld <= 1.0
            assert min(abs(lpd - math.log(dens)), abs(lpd - math.log10(dens))) < 0.02

    def test_a04_parishes_7_exact_partition_and_taipa_present(
        self,
        parishes_7: GpkgLayer,
        zones_23: GpkgLayer,
    ) -> None:
        missing = [c for c in PARISHES_7_REQUIRED_COLS if c not in parishes_7.columns]
        assert not missing, f"parishes_7.gpkg missing columns: {missing}"
        assert len(parishes_7.rows) == 7, f"Expected exact 7 parishes, got {len(parishes_7.rows)}"
        all_names = " ".join(f"{r['parish_name']} {r['parish_name_zh']}" for r in parishes_7.rows)
        assert TAIPA_PARISH_RE.search(all_names), (
            "Taipa parish (嘉模/Carmo) missing from parishes_7"
        )
        p_sum = sum(as_float(r["pop_total"]) for r in parishes_7.rows)
        z_sum = sum(as_float(r["pop_total"]) for r in zones_23.rows)
        assert abs(p_sum - z_sum) / z_sum <= POP_TOL_ZONES_PARISHES

    def test_a04_h3_res9_populated_grid_bounds_and_ids(self, cells_r9: GpkgLayer) -> None:
        lo, hi = H3_R9_COUNT_BOUNDS
        assert lo <= len(cells_r9.rows) <= hi, (
            f"H3 r9 cell count {len(cells_r9.rows)} outside [{lo}, {hi}]"
        )
        ids = [str(r["cell_id"]) for r in cells_r9.rows]
        assert len(ids) == len(set(ids)), "Duplicate cell_id in cells.gpkg"
        for r in cells_r9.rows:
            cid = str(r["cell_id"]).lower()
            assert H3_R9_RE.match(cid), f"Invalid H3 res-9 ID format: {cid!r}"
            assert as_float(r["pop_total"]) > 0.0, f"Unpopulated cell in cells.gpkg: {cid}"

    def test_a04_h3_res8_populated_grid_for_maup(
        self,
        cells_r8: GpkgLayer,
        cells_r9: GpkgLayer,
    ) -> None:
        missing = [c for c in CELLS_BASE_R8_REQUIRED_COLS if c not in cells_r8.columns]
        assert not missing, f"cells_res8.gpkg missing columns: {missing}"
        assert H3_R8_COUNT_MIN <= len(cells_r8.rows) < len(cells_r9.rows), (
            f"Unexpected H3 r8 count {len(cells_r8.rows)} vs r9 {len(cells_r9.rows)}"
        )
        ids = [str(r["cell_id"]) for r in cells_r8.rows]
        assert len(ids) == len(set(ids)), "Duplicate cell_id in cells_res8.gpkg"
        for r in cells_r8.rows:
            cid = str(r["cell_id"]).lower()
            assert H3_R8_RE.match(cid), f"Invalid H3 res-8 ID format: {cid!r}"
            assert as_float(r["pop_total"]) > 0.0, f"Unpopulated cell in cells_res8.gpkg: {cid}"

    def test_a04_exact_terrestrial_census_population_conservation(
        self,
        zones_23: GpkgLayer,
        parishes_7: GpkgLayer,
        cells_r9: GpkgLayer,
        cells_r8: GpkgLayer,
    ) -> None:
        checks = [
            ("zones_23", zones_23, POP_TOL_ZONES_PARISHES, POP_TOL_ZONES_PARISHES),
            ("parishes_7", parishes_7, POP_TOL_ZONES_PARISHES, POP_TOL_ZONES_PARISHES),
            ("cells_r9", cells_r9, POP_TOL_H3_TOTAL, POP_TOL_H3_65PLUS),
            ("cells_r8", cells_r8, POP_TOL_H3_TOTAL, POP_TOL_H3_65PLUS),
        ]
        for label, layer, tol_tot, tol_65 in checks:
            tot = sum(as_float(r["pop_total"]) for r in layer.rows)
            p65 = sum(as_float(r["pop_65plus"]) for r in layer.rows)
            assert abs(tot - TERRESTRIAL_POP_TOTAL) / TERRESTRIAL_POP_TOTAL <= tol_tot, (
                f"{label} pop_total={tot:.1f} drifted from {TERRESTRIAL_POP_TOTAL}"
            )
            assert abs(p65 - TERRESTRIAL_POP_65PLUS) / TERRESTRIAL_POP_65PLUS <= tol_65, (
                f"{label} pop_65plus={p65:.1f} drifted from {TERRESTRIAL_POP_65PLUS}"
            )

    def test_a04_exclave_and_reclamation_handling_documented(
        self,
        project_root: Path,
        provenance_info: tuple[list[Path], Any],
    ) -> None:
        _, prov = provenance_info
        dd_txt = (project_root / "data/frozen/data_dictionary.md").read_text(encoding="utf-8")
        combined = (dd_txt + "\n" + str(prov)).lower()
        assert "z26" in combined or "maritime" in combined or "水域" in combined
        assert "zona a" in combined or "zone a" in combined
        assert "hengqin" in combined or "橫琴" in combined


# ---------------------------------------------------------------------------
# Group II: Walking Network, Anti-Leakage Features, Graph & Blocks (A05-A06)
# ---------------------------------------------------------------------------


class TestA05WalkGraphAndAntiLeakage:
    """A05: Walkable Network, Leak-Free Features & EV-Stripped POI Text (6 isolated tests)."""

    def test_a05_walk_graph_exists_and_retains_peninsula_and_islands(
        self, walk_graph_summary: GraphMLSummary
    ) -> None:
        assert walk_graph_summary.node_count > 50, "Walk graph has too few nodes"
        assert walk_graph_summary.edge_count > 50, "Walk graph has too few edges"
        assert walk_graph_summary.invalid_edge_lengths == 0, (
            f"Walk graph has {walk_graph_summary.invalid_edge_lengths} invalid edge lengths"
        )
        assert walk_graph_summary.peninsula_nodes >= 20, "Walk graph missing Macao Peninsula nodes"
        assert walk_graph_summary.island_nodes >= 10, (
            "Walk graph missing Taipa-Cotai-Coloane nodes (retain_all=False bug?)"
        )

    def test_a05_walk_graph_cell_snap_coverage(
        self,
        walk_graph_summary: GraphMLSummary,
        cells_r9: GpkgLayer,
    ) -> None:
        centroids = [(as_float(r["centroid_x"]), as_float(r["centroid_y"])) for r in cells_r9.rows]
        frac, max_d = nearest_node_snap_fraction(
            centroids, walk_graph_summary.nodes_utm49n, max_dist_m=300.0
        )
        assert frac >= 0.95, (
            f"Only {frac:.1%} of populated cells snap within 300 m of walk graph "
            f"(max={max_d:.1f} m)"
        )

    def test_a05_feature_manifest_schema_and_strict_non_ev_whitelist(
        self, feature_manifest: dict[str, Any]
    ) -> None:
        for key in (
            "ml_features",
            "excluded_target_and_audit",
            "analysis_prefix_excluded",
            "random_seed",
        ):
            assert key in feature_manifest, f"feature_manifest.json missing key {key!r}"
        ml_feats = list(feature_manifest["ml_features"])
        excl = set(feature_manifest["excluded_target_and_audit"])
        assert feature_manifest["random_seed"] == GLOBAL_SEED
        missing_req = [f for f in ML_FEATURES_REQUIRED if f not in ml_feats]
        assert not missing_req, f"ml_features missing required M1 covariates: {missing_req}"
        assert set(ml_feats).isdisjoint(excl), "ml_features overlaps excluded_target_and_audit"
        leaks = [f for f in ml_feats if EV_COL_LEAK_RE.search(f)]
        assert not leaks, f"EV/analysis leakage columns found in ml_features: {leaks}"

    def test_a05_ml_features_complete_finite_and_in_domain(
        self,
        cells_r9: GpkgLayer,
        feature_manifest: dict[str, Any],
    ) -> None:
        for col in feature_manifest["ml_features"]:
            assert col in cells_r9.columns, f"ML feature {col!r} missing from cells.gpkg"
            vals = [as_float(r[col]) for r in cells_r9.rows]
            assert min(vals) < max(vals), f"ML feature {col!r} has zero variance"
            if col in {"elderly_share", "res_bldg_share", "com_bldg_share"}:
                assert all(0.0 <= v <= 1.0 for v in vals), f"{col} outside [0, 1]"
            else:
                assert all(v >= 0.0 for v in vals), f"{col} contains negative values"

    def test_a05_cell_poi_texts_complete_and_zero_ev_leakage(
        self,
        poi_texts: tuple[list[str], list[dict[str, str]]],
        cells_r9: GpkgLayer,
    ) -> None:
        headers, rows = poi_texts
        assert list(POI_TEXTS_REQUIRED_COLS) == headers[: len(POI_TEXTS_REQUIRED_COLS)] or all(
            c in headers for c in POI_TEXTS_REQUIRED_COLS
        )
        cell_ids = {str(r["cell_id"]) for r in cells_r9.rows}
        poi_ids = {r["cell_id"] for r in rows}
        assert poi_ids == cell_ids and len(rows) == len(cell_ids), (
            "cell_poi_texts.csv cell_id set does not match cells.gpkg 1-to-1"
        )
        for r in rows:
            raw_c = int(float(r["poi_count_raw"]))
            filt_c = int(float(r["poi_count_filtered"]))
            assert 0 <= filt_c <= raw_c, f"poi_count_filtered > raw in {r['cell_id']}"
            hit = EV_TEXT_LEAK_RE.search(r["poi_text"])
            assert hit is None, (
                f"EV token leakage in cell_poi_texts.csv cell={r['cell_id']}: {hit.group(0)!r}"
            )

    def test_a05_cell_poi_texts_nontrivial_coverage(
        self, poi_texts: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        _, rows = poi_texts
        covered = sum(
            1 for r in rows if r["poi_text"].strip() and int(float(r["poi_count_filtered"])) > 0
        )
        frac = covered / len(rows)
        assert frac >= 0.50, f"POI text coverage too low ({frac:.1%} < 50%)"


class TestA06GraphAndSpatialBlocks:
    """A06: 10-Min Pedestrian Graph & Frozen 5 Spatial Blocks (6 isolated tests)."""

    def test_a06_graph_edges_schema_and_valid_cell_ids(
        self,
        graph_edges: tuple[list[str], list[dict[str, str]]],
        cells_r9: GpkgLayer,
    ) -> None:
        headers, rows = graph_edges
        missing = [c for c in GRAPH_EDGES_REQUIRED_COLS if c not in headers]
        assert not missing, f"graph_edges.csv missing columns: {missing}"
        assert len(rows) > 0, "graph_edges.csv is empty"
        valid_ids = {str(r["cell_id"]) for r in cells_r9.rows}
        for r in rows:
            assert r["src_cell_id"] in valid_ids and r["dst_cell_id"] in valid_ids, (
                f"Edge references unknown cell_id: {r['src_cell_id']} -> {r['dst_cell_id']}"
            )

    def test_a06_graph_edges_no_self_loops_and_strict_10min_bounds(
        self, graph_edges: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        _, rows = graph_edges
        for r in rows:
            assert r["src_cell_id"] != r["dst_cell_id"], f"Self-loop detected: {r['src_cell_id']}"
            t = as_float(r["walk_time_min"])
            w = as_float(r["weight"])
            assert 0.0 < t <= MAX_GRAPH_WALK_MIN + 1e-9, f"walk_time_min out of (0, 10] bounds: {t}"
            assert abs(w - 1.0 / t) / max(w, 1e-12) < 1e-5, (
                f"Edge weight {w} does not match 1/walk_time_min ({1.0 / t})"
            )

    def test_a06_graph_edges_undirected_symmetry_and_no_duplicates(
        self, graph_edges: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        _, rows = graph_edges
        edge_map: dict[tuple[str, str], tuple[float, bool]] = {}
        for r in rows:
            key = (r["src_cell_id"], r["dst_cell_id"])
            assert key not in edge_map, f"Duplicate directed edge: {key}"
            edge_map[key] = (as_float(r["walk_time_min"]), as_bool(r["same_block"]))
        for (u, v), (t_uv, sb_uv) in edge_map.items():
            assert (v, u) in edge_map, f"Asymmetric graph edge missing reverse ({v}, {u})"
            t_vu, sb_vu = edge_map[(v, u)]
            assert abs(t_uv - t_vu) < 1e-5, f"Asymmetric walk_time_min on ({u}, {v})"
            assert sb_uv == sb_vu, f"Asymmetric same_block on ({u}, {v})"

    def test_a06_spatial_folds_schema_and_exact_5_blocks(
        self,
        spatial_folds: tuple[list[str], list[dict[str, str]]],
        cells_r9: GpkgLayer,
    ) -> None:
        headers, rows = spatial_folds
        assert all(c in headers for c in SPATIAL_FOLDS_REQUIRED_COLS)
        fold_map = {r["cell_id"]: int(float(r["spatial_block"])) for r in rows}
        cell_map = {str(r["cell_id"]): int(r["spatial_block"]) for r in cells_r9.rows}
        assert len(rows) == len(cell_map) and fold_map == cell_map, (
            "spatial_folds.csv does not match cells.gpkg spatial_block 1-to-1"
        )
        assert set(fold_map.values()) == {0, 1, 2, 3, 4}, (
            f"Unexpected blocks: {set(fold_map.values())}"
        )

    def test_a06_split_metadata_all_blocks_have_both_ev_classes(
        self,
        split_metadata: dict[str, Any],
        cells_r9: GpkgLayer,
    ) -> None:
        seed_val = split_metadata.get("seed", split_metadata.get("random_seed"))
        assert seed_val == GLOBAL_SEED, f"Expected split_metadata seed=42, got {seed_val}"
        assert any(k in split_metadata for k in ("method", "clustering_method", "algorithm"))
        for blk in range(5):
            b_rows = [r for r in cells_r9.rows if int(r["spatial_block"]) == blk]
            n_pos = sum(int(r["has_ev_charger"]) for r in b_rows)
            n_neg = len(b_rows) - n_pos
            assert n_pos >= 1, f"Spatial block {blk} has 0 positive EV cells (n_pos=0)"
            assert n_neg >= 1, f"Spatial block {blk} has 0 negative EV cells (n_neg=0)"

    def test_a06_spatial_blocks_geographically_clustered_and_same_block_exact(
        self,
        cells_r9: GpkgLayer,
        graph_edges: tuple[list[str], list[dict[str, str]]],
    ) -> None:
        pts = [
            (
                str(r["cell_id"]),
                as_float(r["centroid_x"]),
                as_float(r["centroid_y"]),
                int(r["spatial_block"]),
            )
            for r in cells_r9.rows
        ]
        same_d: list[float] = []
        diff_d: list[float] = []
        for i in range(len(pts)):
            _, x1, y1, b1 = pts[i]
            for j in range(i + 1, len(pts)):
                _, x2, y2, b2 = pts[j]
                d = math.hypot(x1 - x2, y1 - y2)
                if b1 == b2:
                    same_d.append(d)
                else:
                    diff_d.append(d)
        assert statistics.mean(same_d) < statistics.mean(diff_d), (
            "Spatial blocks are not geographically clustered "
            "(within-block dist >= between-block dist)"
        )
        blk_map = {cid: b for cid, _, _, b in pts}
        _, edges = graph_edges
        for r in edges:
            expected_sb = blk_map[r["src_cell_id"]] == blk_map[r["dst_cell_id"]]
            assert as_bool(r["same_block"]) == expected_sb, (
                f"same_block mismatch on edge {r['src_cell_id']} -> {r['dst_cell_id']}"
            )


# ---------------------------------------------------------------------------
# Group III: Gaussian 2SFCA Accessibility & EDA (A07-A08)
# ---------------------------------------------------------------------------


class TestA07Gaussian2SFCA:
    """A07: Gaussian 2SFCA Accessibility at 10 & 15 min (6 isolated tests)."""

    def test_a07_cells_r9_all_2sfca_columns_present_and_finite(self, cells_r9: GpkgLayer) -> None:
        missing = [c for c in ALL_ACC_COLS if c not in cells_r9.columns]
        assert not missing, f"cells.gpkg missing 2SFCA columns: {missing}"
        for col in ALL_ACC_COLS:
            vals = [as_float(r[col]) for r in cells_r9.rows]
            if col in ACC_INDIVIDUAL_COLS:
                assert all(v >= 0.0 for v in vals), f"Negative 2SFCA score in {col}"

    def test_a07_cells_r8_and_zones_23_2sfca_coverage(
        self,
        cells_r8: GpkgLayer,
        zones_23_with_2sfca: GpkgLayer,
    ) -> None:
        for label, layer in (
            ("cells_res8", cells_r8),
            ("zones_23_with_2sfca", zones_23_with_2sfca),
        ):
            missing = [c for c in ALL_ACC_COLS if c not in layer.columns]
            assert not missing, f"{label} missing 2SFCA columns: {missing}"
            for col in ALL_ACC_COLS:
                vals = [as_float(r[col]) for r in layer.rows]
                if col in ACC_INDIVIDUAL_COLS:
                    assert all(v >= 0.0 for v in vals), f"Negative 2SFCA in {label}.{col}"

    def test_a07_2sfca_supply_conservation_property(
        self,
        cells_r9: GpkgLayer,
        facilities: GpkgLayer,
    ) -> None:
        pops = [as_float(r["pop_total"]) for r in cells_r9.rows]
        cap_by_type: dict[str, float] = defaultdict(float)
        for r in facilities.rows:
            cap_by_type[str(r["facility_type"])] += as_float(r["capacity"])

        for cutoff in (10, 15):
            for ftype in FACILITY_TYPES:
                col = f"acc{cutoff}_{ftype}"
                weighted_sum = sum(
                    p * as_float(r[col]) for p, r in zip(pops, cells_r9.rows, strict=True)
                )
                supply = cap_by_type[ftype]
                raw_ratio = weighted_sum / supply
                # Detect per-capita scale K in {1, 1e3, 1e4, 1e5}
                scale_k = min(
                    (1.0, 1e3, 1e4, 1e5), key=lambda k: abs(math.log10(max(raw_ratio, 1e-12) / k))
                )
                norm_ratio = raw_ratio / scale_k
                min_reach = 0.50 if ftype == "ev_charger" else 0.70
                assert min_reach <= norm_ratio <= 1.001, (
                    f"2SFCA supply conservation violation for {col}: "
                    f"sum(P*A)/(K*sum(S))={norm_ratio:.4f} (K={scale_k:g})"
                )

    def test_a07_15min_coverage_dominates_10min_coverage(self, cells_r9: GpkgLayer) -> None:
        for ftype in FACILITY_TYPES:
            a10 = [as_float(r[f"acc10_{ftype}"]) for r in cells_r9.rows]
            a15 = [as_float(r[f"acc15_{ftype}"]) for r in cells_r9.rows]
            for idx, (v10, v15) in enumerate(zip(a10, a15, strict=True)):
                if v10 > 1e-12:
                    assert v15 > 0.0, f"Support monotonicity violated for {ftype} at row {idx}"
            frac10 = sum(1 for v in a10 if v > 1e-12) / len(a10)
            frac15 = sum(1 for v in a15 if v > 1e-12) / len(a15)
            assert frac15 >= frac10 - 1e-12

    def test_a07_ev_weighted_vs_unweighted_sensitivity_distinct(self, cells_r9: GpkgLayer) -> None:
        wt = [as_float(r["acc15_ev_charger"]) for r in cells_r9.rows]
        unw = [as_float(r["acc15_ev_unweighted"]) for r in cells_r9.rows]
        assert max(wt) > 0.0 and max(unw) > 0.0
        rho = spearman_corr(wt, unw)
        assert rho > 0.50, f"Weighted vs unweighted EV Spearman r too low: {rho:.3f}"
        max_diff = max(abs(a - b) for a, b in zip(wt, unw, strict=True))
        assert max_diff > 1e-9, (
            "Weighted and unweighted EV 2SFCA are identical (no capacity effect)"
        )

    def test_a07_composite_index_positively_tracks_public_facilities(
        self, cells_r9: GpkgLayer
    ) -> None:
        for cutoff in (10, 15):
            comp = [as_float(r[f"acc{cutoff}_composite"]) for r in cells_r9.rows]
            assert max(comp) > min(comp), f"acc{cutoff}_composite is degenerate"
            corrs = [
                spearman_corr(comp, [as_float(r[f"acc{cutoff}_{ft}"]) for r in cells_r9.rows])
                for ft in TRADITIONAL_FACILITY_TYPES
            ]
            assert statistics.mean(corrs) > 0.25, (
                f"acc{cutoff}_composite fails to track traditional facilities: {corrs}"
            )


class TestA08EdaAndLorenz:
    """A08: Exploratory Data Analysis & Lorenz Curves (5 isolated tests)."""

    def test_a08_eda_summary_csv_complete(
        self, eda_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        headers, rows = eda_table
        assert len(rows) >= 10, f"eda_summary.csv too short: {len(rows)} rows"
        text = " ".join(" ".join(r.values()) for r in rows).lower()
        for ft in FACILITY_TYPES:
            assert ft in text, f"Facility type {ft!r} missing from eda_summary.csv"
        assert "pop" in text, "Population variable missing from eda_summary.csv"
        for col in headers:
            if col.lower() in {"min", "median", "mean", "max", "std"}:
                for r in rows:
                    as_float(r[col])

    def test_a08_lorenz_curves_csv_mathematical_validity(
        self, lorenz_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        headers, rows = lorenz_table
        p_col, q_col = _find_lorenz_cols(headers)
        grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
        for r in rows:
            thr = str(r.get("threshold_min", "15")).strip()
            ft = str(r["facility_type"]).strip()
            grouped[(thr, ft)].append((as_float(r[p_col]), as_float(r[q_col])))

        ftypes_seen = {ft for _, ft in grouped}
        assert set(FACILITY_TYPES).issubset(ftypes_seen), f"Lorenz missing types: {ftypes_seen}"
        for key, pts in grouped.items():
            assert len(pts) >= 5, f"Too few points in Lorenz curve {key}"
            assert pts[0][0] <= 0.05 and pts[0][1] <= 0.05, (
                f"Lorenz {key} does not start near (0,0)"
            )
            assert abs(pts[-1][0] - 1.0) <= 1e-3 and abs(pts[-1][1] - 1.0) <= 1e-3, (
                f"Lorenz {key} does not end at (1,1): {pts[-1]}"
            )
            for idx, (p, q) in enumerate(pts):
                assert 0.0 - 1e-6 <= p <= 1.0 + 1e-6 and 0.0 - 1e-6 <= q <= 1.0 + 1e-6
                assert q <= p + 1e-4, f"Lorenz {key} rises above 45-deg line: q={q} > p={p}"
                if idx > 0:
                    assert p >= pts[idx - 1][0] - 1e-9 and q >= pts[idx - 1][1] - 1e-9

    def test_a08_lorenz_area_consistent_with_gini(
        self,
        lorenz_table: tuple[list[str], list[dict[str, str]]],
        h1_table: tuple[list[str], list[dict[str, str]]],
    ) -> None:
        headers, rows = lorenz_table
        p_col, q_col = _find_lorenz_cols(headers)
        grouped: dict[tuple[int, str], list[tuple[float, float]]] = defaultdict(list)
        for r in rows:
            thr = int(float(r.get("threshold_min", "15")))
            grouped[(thr, r["facility_type"].strip())].append(
                (as_float(r[p_col]), as_float(r[q_col]))
            )

        _, h1_rows = h1_table
        h1_r9 = {
            (int(float(r["threshold_min"])), r["facility_type"].strip()): as_float(r["gini"])
            for r in h1_rows
            if normalize_resolution_label(r["resolution"]) == "res9"
        }
        matched = 0
        for key, pts in grouped.items():
            if key not in h1_r9:
                continue
            area = sum(
                0.5 * (pts[i][0] - pts[i - 1][0]) * (pts[i][1] + pts[i - 1][1])
                for i in range(1, len(pts))
            )
            gini_lorenz = 1.0 - 2.0 * area
            assert abs(gini_lorenz - h1_r9[key]) <= 0.02, (
                f"Lorenz trapezoid Gini {gini_lorenz:.4f} vs H1 {h1_r9[key]:.4f} for {key}"
            )
            matched += 1
        assert matched >= 5, f"Expected >= 5 matched Lorenz-to-H1 curves, got {matched}"

    def test_a08_eda_facilities_pop_figure_valid_high_dpi(
        self, load_png: Callable[[str | Path], PngInfo]
    ) -> None:
        _assert_png_high_dpi_nonblank(load_png("outputs/figures/eda_facilities_pop.png"))

    def test_a08_eda_accessibility_lorenz_figure_valid_high_dpi(
        self, load_png: Callable[[str | Path], PngInfo]
    ) -> None:
        _assert_png_high_dpi_nonblank(load_png("outputs/figures/eda_accessibility_lorenz.png"))


# ---------------------------------------------------------------------------
# Group IV: Confirmatory Spatial Statistics, Econometrics & MAUP (A09-A12)
# ---------------------------------------------------------------------------


class TestA09H1InequalityAndLisa:
    """A09: H1 Confirmatory Inequality & LISA Inference (6 isolated tests)."""

    def test_a09_h1_table_schema_and_full_coverage(
        self, h1_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        headers, rows = h1_table
        missing = [c for c in H1_REQUIRED_COLS if c not in headers]
        assert not missing, f"h1_inequality_lisa.csv missing columns: {missing}"
        r9_keys = {
            (int(float(r["threshold_min"])), r["facility_type"].strip())
            for r in rows
            if normalize_resolution_label(r["resolution"]) == "res9"
        }
        for thr in (10, 15):
            for ft in FACILITY_TYPES:
                assert (thr, ft) in r9_keys, f"Missing H1 res9 row for ({thr}m, {ft})"

    def test_a09_gini_bounds_and_bootstrap_ci_bracketing(
        self, h1_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        _, rows = h1_table
        for r in rows:
            g = as_float(r["gini"])
            lo = as_float(r["gini_ci_low"])
            hi = as_float(r["gini_ci_high"])
            assert 0.0 <= g <= 1.0 and 0.0 <= lo <= hi <= 1.0
            assert hi > lo, f"Zero-width Gini CI in {r}"
            assert lo - 0.015 <= g <= hi + 0.015, f"Gini {g} outside CI [{lo}, {hi}]"

    def test_a09_theil_t_nonnegativity_and_exact_parish_additivity(
        self, h1_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        _, rows = h1_table
        for r in rows:
            tt = as_float(r["theil_total"])
            tw = as_float(r["theil_within"])
            tb = as_float(r["theil_between"])
            bs = as_float(r["theil_between_share"])
            assert tt >= -1e-9 and tw >= -1e-9 and tb >= -1e-9
            assert abs(tt - (tw + tb)) < 1e-6, (
                f"Theil additivity broken: total={tt} != within({tw}) + between({tb})"
            )
            assert 0.0 - 1e-6 <= bs <= 1.0 + 1e-6
            if tt > 1e-6:
                assert abs(bs - tb / tt) < 1e-4

    def test_a09_global_moran_i_valid_bounds_and_perm_resolution(
        self, h1_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        _, rows = h1_table
        for r in rows:
            mi = as_float(r["moran_i"])
            mp = as_float(r["moran_p"])
            n_ll = int(float(r["n_ll_fdr"]))
            n_hh = int(float(r["n_hh_fdr"]))
            assert -1.5 <= mi <= 1.5
            assert 0.001 - 1e-9 <= mp <= 1.0, f"Invalid 999-perm moran_p={mp}"
            assert n_ll >= 0 and n_hh >= 0

    def test_a09_lisa_columns_frozen_in_cells_with_valid_labels_and_fdr(
        self, cells_r9: GpkgLayer
    ) -> None:
        metrics = _detect_lisa_metrics(cells_r9.columns)
        assert len(metrics) >= 6, f"Expected >= 6 LISA metrics in cells.gpkg, got {metrics}"
        for metric in metrics:
            i_col = f"lisa_i_{metric}"
            p_col = f"lisa_p_{metric}"
            q_col = f"lisa_q_fdr_{metric}"
            c_col = f"lisa_cluster_{metric}"
            for col in (i_col, p_col, q_col, c_col):
                assert col in cells_r9.columns, f"Missing paired LISA column {col}"
            for row in cells_r9.rows:
                as_float(row[i_col])
                p = as_float(row[p_col])
                q = as_float(row[q_col])
                cl = str(row[c_col])
                assert cl in LISA_CLUSTERS, f"Invalid LISA cluster {cl!r} in {c_col}"
                assert 0.0 <= p <= 1.0 and 0.0 <= q <= 1.0
                assert q >= p - 1e-9, f"BH-FDR q={q} < p={p} in {metric}"
                if cl != "NS":
                    assert q <= 0.05 + 1e-9, f"Significant cluster {cl} has q={q} > 0.05"
                else:
                    assert q >= 0.05 - 1e-9, f"NS cluster has q={q} < 0.05"

    def test_a09_independent_recomputation_of_gini_and_theil_from_frozen_cells(
        self,
        cells_r9: GpkgLayer,
        h1_table: tuple[list[str], list[dict[str, str]]],
    ) -> None:
        pops = [as_float(r["pop_total"]) for r in cells_r9.rows]
        parishes = [str(r["parish_id"]) for r in cells_r9.rows]
        _, h1_rows = h1_table
        checked = 0
        for r in h1_rows:
            if normalize_resolution_label(r["resolution"]) != "res9":
                continue
            thr = int(float(r["threshold_min"]))
            ft = r["facility_type"].strip()
            col = f"acc{thr}_{ft}"
            if ft not in FACILITY_TYPES or col not in cells_r9.columns:
                continue
            vals = [as_float(row[col]) for row in cells_r9.rows]
            g_exp = weighted_gini(vals, pops)
            t_exp = weighted_theil_t(vals, pops, parishes)
            assert abs(as_float(r["gini"]) - g_exp) < 1e-3, (
                f"H1 Gini mismatch for {col}: table={r['gini']} vs oracle={g_exp:.6f}"
            )
            assert abs(as_float(r["theil_total"]) - t_exp["total"]) < 1e-4, (
                f"H1 Theil total mismatch for {col}: "
                f"table={r['theil_total']} vs oracle={t_exp['total']:.6f}"
            )
            checked += 1
        assert checked >= 10


class TestA10H2PairedBootstrap:
    """A10: H2 Confirmatory Paired Bootstrap Gini Diffs (5 isolated tests)."""

    def test_a10_h2_table_schema_and_all_four_facility_pairs(
        self, h2_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        headers, rows = h2_table
        missing = [c for c in H2_REQUIRED_COLS if c not in headers]
        assert not missing, f"h2_gini_diff_bootstrap.csv missing columns: {missing}"
        pairs = {
            (int(float(r["threshold_min"])), _match_h2_facility(r["comparison"])) for r in rows
        }
        for thr in (10, 15):
            for ft in TRADITIONAL_FACILITY_TYPES:
                assert (thr, ft) in pairs, f"Missing H2 paired comparison ({thr}m, EV vs {ft})"

    def test_a10_gini_diff_exact_arithmetic_and_ci_validity(
        self, h2_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        _, rows = h2_table
        for r in rows:
            g_ev = as_float(r["gini_ev"])
            g_oth = as_float(r["gini_other"])
            diff = as_float(r["gini_diff"])
            lo = as_float(r["ci_low"])
            hi = as_float(r["ci_high"])
            assert abs(diff - (g_ev - g_oth)) < 1e-6, (
                f"H2 gini_diff arithmetic error: {diff} != {g_ev} - {g_oth}"
            )
            assert -1.0 <= lo < hi <= 1.0
            assert lo - 0.015 <= diff <= hi + 0.015

    def test_a10_bootstrap_replicates_and_fixed_seed(
        self,
        h2_table: tuple[list[str], list[dict[str, str]]],
        feature_manifest: dict[str, Any],
    ) -> None:
        _, rows = h2_table
        for r in rows:
            assert int(float(r["n_boot"])) >= 2000, f"n_boot < 2000: {r['n_boot']}"
            assert int(float(r["seed"])) == feature_manifest["random_seed"] == GLOBAL_SEED

    def test_a10_holm_adjusted_pvalues_valid_and_ordered(
        self, h2_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        _, rows = h2_table
        by_thr: dict[int, list[dict[str, str]]] = defaultdict(list)
        for r in rows:
            p_raw = as_float(r["p_boot_raw"])
            p_holm = as_float(r["p_holm"])
            assert 0.0 <= p_raw <= p_holm <= 1.0 + 1e-12, (
                f"Holm p-value ordering violated: p_holm={p_holm} < p_raw={p_raw}"
            )
            by_thr[int(float(r["threshold_min"]))].append(r)

        # Check Holm matches either per-threshold family or global family
        per_thr_ok = True
        for grp in by_thr.values():
            raws = [as_float(r["p_boot_raw"]) for r in grp]
            obs = [as_float(r["p_holm"]) for r in grp]
            exp = holm_bonferroni(raws)
            if not all(abs(a - b) <= 1e-4 for a, b in zip(obs, exp, strict=True)):
                per_thr_ok = False
        global_raws = [as_float(r["p_boot_raw"]) for r in rows]
        global_obs = [as_float(r["p_holm"]) for r in rows]
        global_exp = holm_bonferroni(global_raws)
        global_ok = all(abs(a - b) <= 1e-4 for a, b in zip(global_obs, global_exp, strict=True))
        assert per_thr_ok or global_ok, (
            "p_holm does not match Holm-Bonferroni step-down calculation"
        )

    def test_a10_h2_gini_values_match_h1_table(
        self,
        h2_table: tuple[list[str], list[dict[str, str]]],
        h1_table: tuple[list[str], list[dict[str, str]]],
    ) -> None:
        _, h1_rows = h1_table
        h1_map = {
            (int(float(r["threshold_min"])), r["facility_type"].strip()): as_float(r["gini"])
            for r in h1_rows
            if normalize_resolution_label(r["resolution"]) == "res9"
        }
        _, h2_rows = h2_table
        for r in h2_rows:
            thr = int(float(r["threshold_min"]))
            ft = _match_h2_facility(r["comparison"])
            assert ft is not None
            assert abs(as_float(r["gini_ev"]) - h1_map[(thr, "ev_charger")]) < 1e-5
            assert abs(as_float(r["gini_other"]) - h1_map[(thr, ft)]) < 1e-5


class TestA11H3SpatialRegression:
    """A11: H3 Confirmatory Zone Spatial Regression at N=23 (6 isolated tests)."""

    def test_a11_h3_table_schema_and_n_obs_23(
        self, h3_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        headers, rows = h3_table
        missing = [c for c in H3_REQUIRED_COLS if c not in headers]
        assert not missing, f"h3_spatial_regression.csv missing columns: {missing}"
        assert len(rows) > 0
        for r in rows:
            assert int(float(r["n_obs"])) == 23, f"Expected n_obs=23 zones, got {r['n_obs']}"

    def test_a11_side_by_side_three_models_no_pretest_switching(
        self, h3_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        _, rows = h3_table
        groups: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for r in rows:
            key = (r["threshold_min"].strip(), r["outcome"].strip(), r["weights_spec"].strip())
            groups[key].add(r["model_type"].strip())
        assert groups, "No regression groups found in h3_spatial_regression.csv"
        for key, models in groups.items():
            assert models == H3_MODEL_TYPES, (
                f"Pretest switching violation for {key}: expected {H3_MODEL_TYPES}, got {models}"
            )

    def test_a11_required_predictors_and_weights_sensitivity(
        self, h3_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        _, rows = h3_table
        w_specs = {normalize_weights_label(r["weights_spec"]) for r in rows}
        assert "knn4" in w_specs and "bridge_queen" in w_specs, (
            f"Missing required W sensitivity specs (knn4 & bridge_queen): {w_specs}"
        )
        by_block: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
        for r in rows:
            key = (r["threshold_min"], r["outcome"], r["weights_spec"], r["model_type"])
            by_block[key].add(r["term"].strip().lower())
        for key, terms in by_block.items():
            assert "elderly_share" in terms and "log_pop_density" in terms, (
                f"Missing required predictors in H3 block {key}: {terms}"
            )

    def test_a11_coefficients_se_ci_pvalues_and_spatial_params_valid(
        self, h3_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        _, rows = h3_table
        sem_terms: set[str] = set()
        slm_terms: set[str] = set()
        for r in rows:
            coef = as_float(r["coef"])
            se = as_float(r["se"])
            lo = as_float(r["ci_low"])
            hi = as_float(r["ci_high"])
            pv = as_float(r["p_value"])
            as_float(r["r2_or_pseudo_r2"])
            as_float(r["aic"])
            as_float(r["resid_moran_i"])
            rmp = as_float(r["resid_moran_p"])
            assert se > 0.0, f"Non-positive SE in {r}"
            assert lo < coef < hi, f"CI [{lo}, {hi}] does not bracket coef {coef}"
            assert 0.0 <= pv <= 1.0 and 0.0 < rmp <= 1.0
            if r["model_type"] == "SEM_ML":
                sem_terms.add(r["term"].strip().lower())
            elif r["model_type"] == "SLM_ML":
                slm_terms.add(r["term"].strip().lower())
        assert any("lambda" in t or "lam" in t for t in sem_terms), (
            f"SEM_ML missing spatial error parameter lambda: {sem_terms}"
        )
        assert any("rho" in t or "w_" in t for t in slm_terms), (
            f"SLM_ML missing spatial lag parameter rho/W_y: {slm_terms}"
        )

    def test_a11_diagnostics_json_contains_vif_and_weights_connectivity(
        self, h3_diagnostics: dict[str, Any]
    ) -> None:
        text = str(h3_diagnostics).lower()
        assert "vif" in text, "h3_regression_diagnostics.json missing VIF"
        assert "elderly_share" in text and "log_pop_density" in text
        vif_block = h3_diagnostics.get("vif", {})
        if isinstance(vif_block, dict):
            for k in ("elderly_share", "log_pop_density"):
                if k in vif_block:
                    assert as_float(vif_block[k]) >= 1.0
        # Verify zero disconnected islands reported for W
        weights = h3_diagnostics.get("weights", {})
        if isinstance(weights, dict):
            for w_name, w_info in weights.items():
                if isinstance(w_info, dict) and "n_islands" in w_info:
                    assert int(w_info["n_islands"]) == 0, f"Disconnected islands in W={w_name}"

    def test_a11_independent_ols_hc3_verification_against_zones_23(
        self,
        zones_23_with_2sfca: GpkgLayer,
        h3_table: tuple[list[str], list[dict[str, str]]],
    ) -> None:
        _, rows = h3_table
        ols_15 = [
            r
            for r in rows
            if int(float(r["threshold_min"])) == 15
            and r["model_type"].strip() == "OLS_HC3"
            and r["outcome"].strip() in zones_23_with_2sfca.columns
        ]
        assert ols_15, "No 15-min OLS_HC3 outcome matches zones_23_with_2sfca columns"
        outcome_col = ols_15[0]["outcome"].strip()
        first_w = ols_15[0]["weights_spec"].strip()
        block = [
            r
            for r in ols_15
            if r["outcome"].strip() == outcome_col and r["weights_spec"].strip() == first_w
        ]

        y = [as_float(z[outcome_col]) for z in zones_23_with_2sfca.rows]
        x_eld = [as_float(z["elderly_share"]) for z in zones_23_with_2sfca.rows]
        x_lpd = [as_float(z["log_pop_density"]) for z in zones_23_with_2sfca.rows]
        est = ols_hc3_and_vif(y, [x_eld, x_lpd], ["elderly_share", "log_pop_density"])

        for r in block:
            term = r["term"].strip().lower()
            if term in {"elderly_share", "log_pop_density"}:
                assert abs(as_float(r["coef"]) - est["coefs"][term]) < 1e-3, (
                    f"OLS_HC3 coef mismatch for {term}: {r['coef']} vs {est['coefs'][term]}"
                )
                assert abs(as_float(r["se"]) - est["se_hc3"][term]) < 1e-3, (
                    f"OLS_HC3 SE mismatch for {term}: {r['se']} vs {est['se_hc3'][term]}"
                )


class TestA12MaupSensitivity:
    """A12: MAUP Sensitivity Analysis H3 r8 vs r9 (5 isolated tests)."""

    def test_a12_maup_csv_exists_and_compares_r8_vs_r9(
        self, maup_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        headers, rows = maup_table
        assert len(rows) >= 5
        joined_h = " ".join(headers).lower()
        assert "r8" in joined_h and "r9" in joined_h, f"MAUP columns missing r8/r9: {headers}"
        ftypes = {r["facility_type"].strip() for r in rows if "facility_type" in r}
        assert set(FACILITY_TYPES).issubset(ftypes)

    def test_a12_maup_values_finite_and_in_valid_ranges(
        self, maup_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        headers, rows = maup_table
        for r in rows:
            for col in headers:
                low = col.lower()
                if any(tok in low for tok in ("gini", "theil", "moran", "delta")):
                    val = as_float(r[col])
                    if "gini" in low and "delta" not in low:
                        assert 0.0 <= val <= 1.0
                    if "theil" in low and "delta" not in low:
                        assert val >= -1e-9

    def test_a12_maup_r9_entries_consistent_with_h1_table(
        self,
        maup_table: tuple[list[str], list[dict[str, str]]],
        h1_table: tuple[list[str], list[dict[str, str]]],
    ) -> None:
        _, h1_rows = h1_table
        h1_r9 = {
            (int(float(r["threshold_min"])), r["facility_type"].strip()): r
            for r in h1_rows
            if normalize_resolution_label(r["resolution"]) == "res9"
        }
        _, m_rows = maup_table
        checked = 0
        for r in m_rows:
            thr = int(float(r.get("threshold_min", "15")))
            ft = r["facility_type"].strip()
            if (thr, ft) not in h1_r9:
                continue
            ref = h1_r9[(thr, ft)]
            if "gini_r9" in r:
                assert abs(as_float(r["gini_r9"]) - as_float(ref["gini"])) < 1e-4
                checked += 1
        assert checked >= 5

    def test_a12_maup_independent_r8_gini_verification(
        self,
        cells_r8: GpkgLayer,
        maup_table: tuple[list[str], list[dict[str, str]]],
    ) -> None:
        pops = [as_float(r["pop_total"]) for r in cells_r8.rows]
        _, m_rows = maup_table
        checked = 0
        for r in m_rows:
            thr = int(float(r.get("threshold_min", "15")))
            ft = r["facility_type"].strip()
            col = f"acc{thr}_{ft}"
            if ft in FACILITY_TYPES and col in cells_r8.columns and "gini_r8" in r:
                g_exp = weighted_gini([as_float(row[col]) for row in cells_r8.rows], pops)
                assert abs(as_float(r["gini_r8"]) - g_exp) < 1e-3, (
                    f"MAUP r8 Gini mismatch for {col}: table={r['gini_r8']} vs oracle={g_exp:.6f}"
                )
                checked += 1
        assert checked >= 5

    def test_a12_maup_figure_valid_high_dpi(
        self, load_png: Callable[[str | Path], PngInfo]
    ) -> None:
        _assert_png_high_dpi_nonblank(load_png("outputs/figures/maup_r8_vs_r9.png"))


# ---------------------------------------------------------------------------
# Group V: Leak-Free ML Baselines (A13)
# ---------------------------------------------------------------------------


class TestA13MlBaselines:
    """A13: Phase A Leak-Free ML Baselines on Frozen 5 Spatial Blocks (6 isolated tests)."""

    def test_a13_ml_baselines_json_contract_and_three_models(
        self,
        ml_baselines: dict[str, Any],
        cells_r9: GpkgLayer,
    ) -> None:
        assert ml_baselines["seed"] == GLOBAL_SEED
        n_cells = len(cells_r9.rows)
        n_pos = sum(int(r["has_ev_charger"]) for r in cells_r9.rows)
        assert int(ml_baselines["n_cells"]) == n_cells
        assert int(ml_baselines["n_pos"]) == n_pos
        assert abs(as_float(ml_baselines["prevalence"]) - n_pos / n_cells) < 1e-6
        assert set(ML_MODEL_NAMES).issubset(set(ml_baselines["models"].keys()))

    def test_a13_ml_metrics_complete_finite_and_in_bounds(
        self, ml_baselines: dict[str, Any]
    ) -> None:
        for m_name in ML_MODEL_NAMES:
            m = ml_baselines["models"][m_name]
            folds = m["fold_metrics"]
            assert len(folds) == 5, f"{m_name} must have 5 spatial folds"
            for key in ML_AGGREGATE_KEYS:
                val = as_float(m[key])
                assert 0.0 <= val <= 1.0, f"{m_name}.{key}={val} out of [0, 1]"

    def test_a13_ml_strict_zero_ev_feature_leakage(
        self,
        ml_baselines: dict[str, Any],
        feature_manifest: dict[str, Any],
    ) -> None:
        fcols = list(ml_baselines["feature_columns"])
        allowed = set(feature_manifest["ml_features"])
        excl = set(feature_manifest["excluded_target_and_audit"])
        assert fcols, "feature_columns is empty"
        assert set(fcols).issubset(allowed), f"Unwhitelisted ML features: {set(fcols) - allowed}"
        assert set(fcols).isdisjoint(excl)
        leaks = [c for c in fcols if EV_COL_LEAK_RE.search(c)]
        assert not leaks, f"EV leakage columns in phase_a_ml_baselines.json: {leaks}"

    def test_a13_naive_baseline_sanity_bounds(self, ml_baselines: dict[str, Any]) -> None:
        naive = ml_baselines["models"]["naive_prior"]
        for fm in naive["fold_metrics"]:
            assert abs(as_float(fm["roc_auc"]) - 0.5) < 1e-6, (
                f"naive_prior per-fold ROC-AUC must be 0.5, got {fm['roc_auc']}"
            )
        assert abs(as_float(naive["mean_roc_auc"]) - 0.5) < 1e-6
        prev = as_float(ml_baselines["prevalence"])
        assert abs(as_float(naive["oof_brier"]) - prev * (1.0 - prev)) < 0.05

    def test_a13_oof_predictions_complete_valid_probabilities_and_metric_match(
        self,
        ml_oof_table: tuple[list[str], list[dict[str, str]]],
        ml_baselines: dict[str, Any],
        cells_r9: GpkgLayer,
    ) -> None:
        headers, rows = ml_oof_table
        y_col, model_cols = _find_oof_cols(headers)
        assert len(rows) == len(cells_r9.rows)
        y_true = [int(float(r[y_col])) for r in rows]
        for m_name in ML_MODEL_NAMES:
            probs = [as_float(r[model_cols[m_name]]) for r in rows]
            assert all(0.0 <= p <= 1.0 for p in probs)
            m = ml_baselines["models"][m_name]
            assert abs(roc_auc_score(y_true, probs) - as_float(m["oof_roc_auc"])) < 1e-4
            assert abs(average_precision_score(y_true, probs) - as_float(m["oof_pr_auc"])) < 1e-4
            assert abs(brier_score(y_true, probs) - as_float(m["oof_brier"])) < 1e-4
            if m_name != "naive_prior":
                rec20, prec20 = top20_hit_rates(y_true, probs)
                rep20 = as_float(m["oof_top20_hit_rate"])
                assert min(abs(rep20 - rec20), abs(rep20 - prec20)) < 0.02

    def test_a13_ml_cv_results_csv_consistent_with_json(
        self,
        ml_cv_table: tuple[list[str], list[dict[str, str]]],
        ml_baselines: dict[str, Any],
    ) -> None:
        headers, rows = ml_cv_table
        m_col = next((h for h in headers if "model" in h.lower()), headers[0])
        by_model = {r[m_col].strip(): r for r in rows}
        for m_name in ML_MODEL_NAMES:
            assert m_name in by_model, f"Model {m_name!r} missing from ml_cv_results.csv"
            row = by_model[m_name]
            ref = ml_baselines["models"][m_name]
            for key in ("mean_roc_auc", "oof_roc_auc", "oof_pr_auc", "oof_brier"):
                if key in row:
                    assert abs(as_float(row[key]) - as_float(ref[key])) < 1e-4


# ---------------------------------------------------------------------------
# Group VI: Canonical Frozen Bundle, Publication Visuals & Report (A14-A16)
# ---------------------------------------------------------------------------


class TestA14FrozenBundleAndCli:
    """A14: Canonical Phase A->B Frozen Bundle & Replay CLI (6 isolated tests)."""

    def test_a14_all_required_frozen_bundle_files_exist(self, project_root: Path) -> None:
        for rel in FROZEN_CANONICAL_FILES:
            p = project_root / rel
            assert p.is_file() and p.stat().st_size > 0, f"Missing or empty frozen file: {rel}"

    def test_a14_freeze_manifest_sha256_integrity(
        self,
        project_root: Path,
        freeze_manifest: dict[str, Any],
    ) -> None:
        fmap = _extract_freeze_file_map(freeze_manifest)
        for rel in FROZEN_CORE_HASHED_FILES:
            assert rel in fmap, f"FREEZE_MANIFEST.json missing {rel}"
        for rel, expected_sha in fmap.items():
            assert SHA256_RE.match(expected_sha), (
                f"Invalid SHA-256 format for {rel}: {expected_sha}"
            )
            fp = project_root / rel
            assert fp.is_file(), f"Manifest file missing on disk: {rel}"
            live_sha = file_sha256(fp)
            assert live_sha == expected_sha, (
                f"SHA-256 tamper/mismatch for {rel}: live={live_sha} != manifest={expected_sha}"
            )

    def test_a14_canonical_cells_gpkg_complete_merged_schema(
        self,
        project_root: Path,
        cells_r9: GpkgLayer,
        load_gpkg: Callable[..., GpkgLayer],
    ) -> None:
        required = set(CELLS_BASE_R9_REQUIRED_COLS) | set(ALL_ACC_COLS) | set(TARGET_AND_AUDIT_COLS)
        missing = sorted(required - set(cells_r9.columns))
        assert not missing, f"Canonical cells.gpkg missing columns: {missing}"
        assert len(_detect_lisa_metrics(cells_r9.columns)) >= 6
        base_path = project_root / "data/frozen/cells_base_res9.gpkg"
        if base_path.is_file():
            base = load_gpkg(base_path)
            assert {str(r["cell_id"]) for r in base.rows} == {
                str(r["cell_id"]) for r in cells_r9.rows
            }

    def test_a14_data_dictionary_documents_schema_and_anti_leakage(
        self, project_root: Path
    ) -> None:
        txt = (project_root / "data/frozen/data_dictionary.md").read_text(encoding="utf-8")
        assert len(txt) >= 1000, "data_dictionary.md is too short"
        low = txt.lower()
        for tok in (
            "cells.gpkg",
            "graph_edges.csv",
            "spatial_folds.csv",
            "cell_poi_texts.csv",
            "32649",
            "acc10_",
            "acc15_",
            "lisa_",
        ):
            assert tok in low, f"data_dictionary.md missing documentation for {tok!r}"

    def test_a14_unified_cli_scripts_exist_and_expose_skip_download(
        self, project_root: Path
    ) -> None:
        cli = project_root / "scripts" / "run_phase_a.py"
        assert cli.is_file(), f"Missing unified CLI: {cli}"
        for stage in (
            "scripts/01_ingest_and_prepare.py",
            "scripts/02_run_spatial_analysis.py",
            "scripts/03_run_ml_baselines.py",
            "scripts/04_freeze_and_report.py",
        ):
            assert (project_root / stage).is_file(), f"Missing stage CLI: {stage}"
        venv_py = project_root / ".venv" / "bin" / "python"
        py_bin = str(venv_py) if venv_py.is_file() else sys.executable
        proc = subprocess.run(
            [py_bin, str(cli), "--help"], capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, f"run_phase_a.py --help failed: {proc.stderr}"
        assert "--skip-download" in (proc.stdout + proc.stderr), (
            "run_phase_a.py --help does not advertise --skip-download"
        )

    def test_a14_readme_documents_reproducible_phase_a_cli(self, project_root: Path) -> None:
        txt = (project_root / "README.md").read_text(encoding="utf-8")
        assert len(txt) >= 400, "README.md is too short"
        low = txt.lower()
        assert "run_phase_a.py" in low and "--skip-download" in low
        assert "data/frozen" in low


class TestA15FiguresAndTable1:
    """A15: Programmatic Figure 1 & Table 1 Generation (5 isolated tests)."""

    def test_a15_fig1_dual_panel_exists_high_dpi_and_wide_aspect(
        self, load_png: Callable[[str | Path], PngInfo]
    ) -> None:
        png = load_png("outputs/figures/fig1_accessibility_lisa.png")
        _assert_png_high_dpi_nonblank(png)
        assert png.width > png.height, (
            f"Fig 1 must be dual-panel landscape: {png.width}x{png.height}"
        )
        assert png.width >= 2000, (
            f"Fig 1 pixel width too narrow for dual-panel 300 DPI: {png.width}"
        )

    def test_a15_all_four_required_figures_present_and_nonzero(
        self, load_png: Callable[[str | Path], PngInfo]
    ) -> None:
        for rel in REQUIRED_FIGURES:
            _assert_png_high_dpi_nonblank(load_png(rel))

    def test_a15_table1_summary_all_three_formats_exist_and_nonempty(
        self, project_root: Path
    ) -> None:
        for rel in TABLE1_FILES:
            p = project_root / rel
            assert p.is_file() and p.stat().st_size > 100, f"Missing/empty {rel}"
            txt = p.read_text(encoding="utf-8")
            assert not TABLE_PLACEHOLDER_RE.search(txt), f"Placeholder/NaN found in {rel}"
        headers, rows = read_csv_strict(project_root / "outputs/tables/table1_summary.csv")
        assert len(headers) >= 3 and len(rows) >= 8
        md_txt = (project_root / "outputs/tables/table1_summary.md").read_text(encoding="utf-8")
        tex_txt = (project_root / "outputs/tables/table1_summary.tex").read_text(encoding="utf-8")
        assert "|" in md_txt
        assert r"\begin{tabular}" in tex_txt

    def test_a15_table1_covers_all_three_required_panels(self, project_root: Path) -> None:
        combined = "\n".join(
            (project_root / rel).read_text(encoding="utf-8").lower() for rel in TABLE1_FILES
        )
        assert "gini" in combined and "theil" in combined
        assert "ols" in combined and ("sem" in combined or "slm" in combined)
        assert "roc" in combined and "pr" in combined and "brier" in combined

    def test_a15_table1_numbers_match_verified_source_csv_json(
        self,
        project_root: Path,
        h1_table: tuple[list[str], list[dict[str, str]]],
        h3_table: tuple[list[str], list[dict[str, str]]],
        ml_baselines: dict[str, Any],
    ) -> None:
        t1_csv_txt = (project_root / "outputs/tables/table1_summary.csv").read_text(
            encoding="utf-8"
        )
        t1_nums = _extract_numeric_tokens(t1_csv_txt)
        assert t1_nums, "No numeric entries found in table1_summary.csv"

        _, h1_rows = h1_table
        ev_r9_15 = next(
            as_float(r["gini"])
            for r in h1_rows
            if normalize_resolution_label(r["resolution"]) == "res9"
            and int(float(r["threshold_min"])) == 15
            and r["facility_type"].strip() == "ev_charger"
        )
        park_r9_15 = next(
            as_float(r["gini"])
            for r in h1_rows
            if normalize_resolution_label(r["resolution"]) == "res9"
            and int(float(r["threshold_min"])) == 15
            and r["facility_type"].strip() == "park"
        )
        rf_auc = as_float(ml_baselines["models"]["random_forest"]["oof_roc_auc"])
        lr_auc = as_float(ml_baselines["models"]["logistic_regression"]["oof_roc_auc"])

        _, h3_rows = h3_table
        h3_eld = next(
            as_float(r["coef"])
            for r in h3_rows
            if int(float(r["threshold_min"])) == 15
            and r["model_type"].strip() == "OLS_HC3"
            and r["term"].strip().lower() == "elderly_share"
        )

        for label, target in (
            ("15m EV Gini", ev_r9_15),
            ("15m Park Gini", park_r9_15),
            ("H3 OLS elderly_share", h3_eld),
            ("LR OOF ROC-AUC", lr_auc),
            ("RF OOF ROC-AUC", rf_auc),
        ):
            best_err = min(abs(v - target) for v in t1_nums)
            assert best_err <= 0.005, (
                f"Table 1 numeric drift on {label}: expected {target:.4f}, min_err={best_err:.4f}"
            )


class TestA16ReportAndGoNoGo:
    """A16: CS482 Phase A Report Draft & 12/03 Go/No-Go Audit (7 isolated tests)."""

    def test_a16_report_artifacts_exist_and_pdf_valid(
        self,
        project_root: Path,
        report_pdf_info: PdfInfo,
    ) -> None:
        for rel in (REPORT_TEX_REL, REPORT_BIB_REL, REPORT_PDF_REL, REPORT_AUDIT_REL):
            p = project_root / rel
            assert p.is_file() and p.stat().st_size > 100, f"Missing/empty report file: {rel}"
        assert not report_pdf_info.encrypted, "Submitted PDF must not be encrypted"
        assert report_pdf_info.total_pages >= 4, f"PDF too short: {report_pdf_info.total_pages}p"

    def test_a16_pdf_a4_single_column_11pt_15spacing_format(
        self,
        report_pdf_info: PdfInfo,
        report_tex_text: str,
    ) -> None:
        for p_idx, (w, h) in enumerate(report_pdf_info.page_sizes_pt, start=1):
            assert abs(w - A4_WIDTH_PT) <= A4_TOL_PT and abs(h - A4_HEIGHT_PT) <= A4_TOL_PT, (
                f"PDF page {p_idx} is not A4 ({w}x{h} pt; expected {A4_WIDTH_PT}x{A4_HEIGHT_PT})"
            )
        assert report_pdf_info.single_column_ratio >= 0.65, (
            f"PDF does not appear single-column (ratio={report_pdf_info.single_column_ratio:.2f})"
        )
        assert 14.0 <= report_pdf_info.median_line_pitch_pt <= 23.0, (
            f"Unexpected 1.5-spaced line pitch: {report_pdf_info.median_line_pitch_pt:.2f} pt"
        )
        assert "11pt" in report_tex_text, "LaTeX source missing 11pt option"
        assert "twocolumn" not in report_tex_text, "LaTeX source must be single-column"
        assert re.search(
            r"\\onehalfspacing|\\setstretch\{1\.[2-5]|\\linespread\{1\.[2-5]",
            report_tex_text,
        ), "LaTeX source missing 1.5 line spacing directive"

    def test_a16_pdf_main_text_page_count_30_to_38_pages(self, report_pdf_info: PdfInfo) -> None:
        mp = report_pdf_info.main_fractional_pages
        assert PDF_MAIN_PAGES_MIN <= mp <= PDF_MAIN_PAGES_MAX, (
            f"PDF main-text fractional pages={mp:.2f} outside Phase A budget "
            f"[{PDF_MAIN_PAGES_MIN}, {PDF_MAIN_PAGES_MAX}]"
        )

    def test_a16_report_satisfies_all_five_minimum_requirements(
        self, report_pdf_info: PdfInfo
    ) -> None:
        txt = report_pdf_info.main_text.lower()
        # MR1: Research question, H1-H3, beneficiaries
        assert all(tok in txt for tok in ("h1", "h2", "h3"))
        assert any(tok in txt for tok in ("beneficiar", "planner", "resident", "dsscu", "dsat"))
        # MR2: Data collection & prep
        assert "dsec" in txt and "cem" in txt and ("osm" in txt or "openstreetmap" in txt)
        assert "32649" in txt and ("25 m" in txt or "25m" in txt or "dedup" in txt)
        # MR3: Analysis
        assert "2sfca" in txt and "gini" in txt and "theil" in txt and "lisa" in txt
        assert "ols" in txt and "sem" in txt and "slm" in txt
        # MR4: Evaluation
        assert "bootstrap" in txt and "fdr" in txt and "holm" in txt and "maup" in txt
        assert "roc" in txt and "brier" in txt
        # MR5: Results & discussion
        assert "figure 1" in txt and "table 1" in txt and "limitation" in txt

    def test_a16_report_excluded_mandatory_sections_and_kdense_citation(
        self,
        report_pdf_info: PdfInfo,
        report_bib_text: str,
    ) -> None:
        assert report_pdf_info.cover_pages, "Cover page missing from PDF"
        assert (
            "student id" in report_pdf_info.cover_text.lower()
            or "group" in report_pdf_info.cover_text.lower()
        )
        assert report_pdf_info.references_text, "References section missing from PDF"
        assert report_pdf_info.contribution_text, "Contribution Statement missing from PDF"
        assert report_pdf_info.ai_declaration_text, "AI Usage Declaration missing from PDF"
        ref_and_bib = (report_pdf_info.references_text + "\n" + report_bib_text).lower()
        assert "2609.00065" in ref_and_bib or "kassis" in ref_and_bib, (
            "K-Dense citation (Kassis et al., arXiv:2609.00065) missing"
        )
        ai_low = report_pdf_info.ai_declaration_text.lower()
        assert any(tok in ai_low for tok in ("tool", "assistant", "llm", "model"))
        assert any(tok in ai_low for tok in ("plan", "ideation", "coding", "writing"))
        assert any(tok in ai_low for tok in ("verif", "check", "audit", "test"))

    def test_a16_report_correlational_framing_four_limitations_and_zero_hype(
        self, report_pdf_info: PdfInfo
    ) -> None:
        main_low = report_pdf_info.main_text.lower()
        # Correlational framing & historical allocation caveat
        assert any(
            tok in main_low
            for tok in ("correlational", "associational", "associated with", "cross-sectional")
        )
        assert "historical" in main_low and (
            "field verification" in main_low or "candidate" in main_low
        )
        unhedged = sentence_aware_unhedged_causal_hits(report_pdf_info.main_text)
        assert not unhedged, f"Unhedged causal claims in main text: {unhedged[:3]}"
        # 4 mandatory limitations
        assert any(tok in main_low for tok in ("high-rise", "building-height", "dasymetric"))
        assert any(tok in main_low for tok in ("capacity metadata", "9488798", "imputed"))
        assert any(tok in main_low for tok in ("slope", "topograph", "footbridge", "overpass"))
        assert "ecological fallacy" in main_low
        # Zero hype & zero analytical placeholders
        full_low = report_pdf_info.full_text.lower()
        for phrase in FORBIDDEN_HYPE_PHRASES:
            assert phrase not in full_low, f"Forbidden hype phrase in PDF: {phrase!r}"
        check_scope = report_pdf_info.main_text + "\n" + report_pdf_info.references_text
        bad = ANALYTICAL_PLACEHOLDER_RE.search(check_scope)
        assert bad is None, f"Unresolved analytical placeholder in PDF main/refs: {bad.group(0)!r}"

    def test_a16_go_nogo_audit_passes_and_traces_numbers(self, go_nogo_audit_text: str) -> None:
        low = go_nogo_audit_text.lower()
        assert len(go_nogo_audit_text) >= 300
        assert re.search(r"\b(go|pass)\b", low), "go_nogo_audit.md missing explicit GO/PASS verdict"
        assert not re.search(r"\b(fail|(?<!go/)no-go|tbd_num|todo|fixme)\b", low), (
            "go_nogo_audit.md contains FAIL/NO-GO or unresolved placeholder"
        )
        assert "681,293" in go_nogo_audit_text or "681293" in go_nogo_audit_text
        assert "23" in go_nogo_audit_text and "7" in go_nogo_audit_text
