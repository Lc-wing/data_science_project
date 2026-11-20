"""Tier 2 E2E Tests: 71 boundary, corner, anti-leakage, and negative tests across FA-1..FA-10."""

from __future__ import annotations

import math
import re
import statistics
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
    EV_CAPACITY_MAX,
    EV_CAPACITY_MIN,
    EV_COL_LEAK_RE,
    EV_TEXT_LEAK_RE,
    FACILITIES_REQUIRED_COLS,
    FACILITY_COUNT_BOUNDS,
    FACILITY_TYPES,
    FORBIDDEN_HYPE_PHRASES,
    FROZEN_CORE_HASHED_FILES,
    GLOBAL_SEED,
    H3_MODEL_TYPES,
    H3_R8_COUNT_MIN,
    H3_R8_RE,
    H3_R9_COUNT_BOUNDS,
    H3_R9_RE,
    ISLAND_PARISH_IDS,
    ISLANDS_MAX_N,
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
    ML_METRICS,
    ML_MODEL_NAMES,
    PDF_MAIN_PAGES_MAX,
    PDF_MAIN_PAGES_MIN,
    PENINSULA_MIN_N,
    PENINSULA_PARISH_IDS,
    PNG_MIN_BYTES,
    PNG_MIN_DPI,
    PNG_MIN_PIXEL_STD,
    POP_TOL_H3_65PLUS,
    POP_TOL_H3_TOTAL,
    POP_TOL_ZONES_PARISHES,
    RAW_SOURCE_GPKG_NAMES,
    REQUIRED_FIGURES,
    REQUIRED_PYPROJECT_PACKAGES,
    SECRET_TOKEN_RE,
    SHA256_RE,
    TABLE1_FILES,
    TABLE_PLACEHOLDER_RE,
    TAIPA_PARISH_RE,
    TARGET_EPSG,
    TERRESTRIAL_POP_65PLUS,
    TERRESTRIAL_POP_TOTAL,
    TRADITIONAL_FACILITY_TYPES,
    UNIT_CAPACITY_FACILITIES,
    ZONA_A_MIN_E,
    facility_source_family,
    normalize_resolution_label,
    normalize_weights_label,
    provenance_count,
    provenance_timestamp,
)
from e2e_lib.oracle import (
    holm_bonferroni,
    point_distance_to_polygons,
    sentence_aware_unhedged_causal_hits,
)
from e2e_lib.readers import (
    GpkgLayer,
    GraphMLSummary,
    PdfInfo,
    PngInfo,
    as_bool,
    as_float,
    read_toml_strict,
)


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
    return sorted(
        col.removeprefix("lisa_cluster_") for col in columns if col.startswith("lisa_cluster_")
    )


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
        candidates = [f"prob_{m_name}", f"pred_{m_name}", f"{m_name}_prob", m_name]
        found = next((h for h in headers if h.lower() in candidates), None)
        if found is None:
            token = m_name.split("_")[0]
            found = next((h for h in headers if token in h.lower() and h != y_col), None)
        assert found, f"Missing OOF probability column for {m_name} in {headers}"
        model_cols[m_name] = found
    return y_col, model_cols


# ---------------------------------------------------------------------------
# FA-1 (A01, A02): Environment, Config & Provenance Boundaries (5 tests)
# ---------------------------------------------------------------------------


class TestFA01EnvAndProvenanceBoundaries:
    """FA-1: Environment, Config & Provenance Boundaries (5 tests)."""

    def test_t2_env_pyproject_python_ruff_pytest(self, project_root: Path) -> None:
        toml_data = read_toml_strict(project_root / "pyproject.toml")
        proj = toml_data.get("project", {})
        req_py = str(proj.get("requires-python", ""))
        assert re.search(r">=\s*3\.1[1-9]", req_py), f"Invalid requires-python: {req_py!r}"
        ruff = toml_data.get("tool", {}).get("ruff", {})
        assert ruff.get("line-length") == 100
        assert "pytest" in toml_data.get("tool", {})
        deps: list[str] = list(proj.get("dependencies", []))
        for opt in proj.get("optional-dependencies", {}).values():
            deps.extend(opt)
        for grp in toml_data.get("dependency-groups", {}).values():
            if isinstance(grp, list):
                deps.extend(str(d) for d in grp if isinstance(d, str))
        dep_names = {re.split(r"[<>=!~;\[\s]", d.strip().lower())[0] for d in deps}
        for pkg in REQUIRED_PYPROJECT_PACKAGES:
            assert pkg in dep_names, f"Missing dependency {pkg!r} in pyproject.toml"
        uv_lock = project_root / "uv.lock"
        assert uv_lock.is_file() and uv_lock.stat().st_size > 50

    def test_t2_env_gitignore_protects_venv_and_secrets(self, project_root: Path) -> None:
        gitignore_txt = (project_root / ".gitignore").read_text(encoding="utf-8")
        for pattern in (".venv", "__pycache__", ".pytest_cache", ".env"):
            assert pattern in gitignore_txt, f".gitignore fails to exclude {pattern!r}"

    def test_t2_provenance_schema_urls_dates_hashes(
        self, provenance_info: tuple[list[Path], Any]
    ) -> None:
        _, prov = provenance_info
        entries = _extract_provenance_entries(prov)
        assert len(entries) >= 4
        for entry in entries:
            url = str(entry.get("url") or entry.get("source_url") or entry.get("dataset_id") or "")
            assert url.startswith("http://") or url.startswith("https://"), (
                f"Provenance entry lacks http(s):// URL: {entry}"
            )
            ts = provenance_timestamp(entry)
            assert re.search(r"^\d{4}-\d{2}-\d{2}", ts.strip()), f"Invalid ISO date: {ts!r}"
            cnt = provenance_count(entry)
            assert cnt is not None and cnt > 0, f"Non-positive record count in {entry}"
            sha = entry.get("sha256")
            if sha is not None:
                assert SHA256_RE.match(str(sha).strip().lower()), f"Invalid SHA-256 hex: {sha!r}"

    def test_t2_provenance_documents_datagovmo_auth_gate(
        self, provenance_info: tuple[list[Path], Any]
    ) -> None:
        paths, prov = provenance_info
        raw_text = paths[0].read_text(encoding="utf-8")
        assert "data.gov.mo" in raw_text.lower(), "data.gov.mo not documented in provenance"
        assert any(w in raw_text.lower() for w in ("login", "auth", "token", "gate", "fallback")), (
            "Missing login/auth gate description in provenance"
        )
        assert not SECRET_TOKEN_RE.search(raw_text), "Leaked secret/API token in provenance JSON!"
        assert prov is not None

    def test_t2_global_seed_determinism_consistency(
        self,
        feature_manifest: dict[str, Any],
        split_metadata: dict[str, Any],
        h2_table: tuple[list[str], list[dict[str, str]]],
        ml_baselines: dict[str, Any],
    ) -> None:
        s_fm = int(feature_manifest.get("random_seed", feature_manifest.get("seed", -1)))
        s_sm = int(split_metadata.get("seed", split_metadata.get("random_seed", -2)))
        s_ml = int(ml_baselines.get("seed", ml_baselines.get("random_seed", -3)))
        _, h2_rows = h2_table
        h2_seeds = {int(float(r["seed"])) for r in h2_rows}
        assert s_fm == s_sm == s_ml == GLOBAL_SEED, (
            f"Inconsistent global seeds: manifest={s_fm}, split={s_sm}, ml={s_ml}"
        )
        assert h2_seeds == {GLOBAL_SEED}, f"Inconsistent H2 seeds: {h2_seeds}"


# ---------------------------------------------------------------------------
# FA-2 (A03): CRS, Geometry & Macao Geographic Alignment Traps (7 tests)
# ---------------------------------------------------------------------------


class TestFA02CrsGeometryAndAlignment:
    """FA-2: CRS, Geometry & Macao Geographic Alignment Traps (7 tests)."""

    def test_t2_crs_strict_epsg32649_all_gpkg(
        self,
        project_root: Path,
        load_gpkg: Callable[..., GpkgLayer],
    ) -> None:
        gpkg_paths: list[Path] = []
        for sub in ("data/frozen", "data/interim"):
            d = project_root / sub
            if d.is_dir():
                gpkg_paths.extend(
                    p
                    for p in sorted(d.glob("*.gpkg"))
                    if p.name not in RAW_SOURCE_GPKG_NAMES
                )
        assert len(gpkg_paths) >= 6, f"Expected >= 6 .gpkg files, found {len(gpkg_paths)}"
        for p in gpkg_paths:
            layer = load_gpkg(p, allow_multi_layer=(p.name == "facilities_clean.gpkg"))
            assert layer.epsg == TARGET_EPSG, (
                f"CRS violation in {p.name}: epsg={layer.epsg} != {TARGET_EPSG}"
            )
            assert all(g.srs_id == TARGET_EPSG for g in layer.geoms), (
                f"Geometry blob srs_id mismatch in {p.name}"
            )

    def test_t2_geometries_valid_nonempty_correct_type(
        self,
        zones_23: GpkgLayer,
        parishes_7: GpkgLayer,
        cells_m1_r9: GpkgLayer,
        cells_m1_r8: GpkgLayer,
        facilities: GpkgLayer,
        lrt_stations: GpkgLayer,
    ) -> None:
        cells_r9 = cells_m1_r9
        cells_r8 = cells_m1_r8
        poly_layers = (
            ("zones_23", zones_23),
            ("parishes_7", parishes_7),
            ("cells_r9", cells_r9),
            ("cells_r8", cells_r8),
        )
        for name, layer in poly_layers:
            for idx, g in enumerate(layer.geoms):
                assert not g.is_empty and g.is_valid, f"Invalid/empty polygon at {name}[{idx}]"
                assert g.geom_type in {"Polygon", "MultiPolygon"}, (
                    f"Expected Polygon/MultiPolygon in {name}[{idx}], got {g.geom_type}"
                )
                assert g.area > 0.0, f"Non-positive polygon area in {name}[{idx}]"

        for name, layer in (("facilities", facilities), ("lrt_stations", lrt_stations)):
            for idx, g in enumerate(layer.geoms):
                assert not g.is_empty and g.is_valid, f"Invalid/empty point at {name}[{idx}]"
                assert g.geom_type in {"Point", "MultiPoint"} and g.point_xy is not None, (
                    f"Expected Point in {name}[{idx}], got {g.geom_type}"
                )

    def test_t2_macao_utm49n_envelope_bounds(
        self,
        zones_23: GpkgLayer,
        parishes_7: GpkgLayer,
        cells_m1_r9: GpkgLayer,
        cells_m1_r8: GpkgLayer,
        facilities: GpkgLayer,
        lrt_stations: GpkgLayer,
    ) -> None:
        cells_r9 = cells_m1_r9
        cells_r8 = cells_m1_r8
        for name, layer in (
            ("zones_23", zones_23),
            ("parishes_7", parishes_7),
            ("cells_r9", cells_r9),
            ("cells_r8", cells_r8),
        ):
            for g in layer.geoms:
                assert g.bbox is not None
                minx, miny, maxx, maxy = g.bbox
                assert MACAO_POLY_E_MIN <= minx <= maxx <= MACAO_POLY_E_MAX, (
                    f"{name} polygon easting out of Macao UTM49N envelope: {g.bbox}"
                )
                assert MACAO_POLY_N_MIN <= miny <= maxy <= MACAO_POLY_N_MAX, (
                    f"{name} polygon northing out of Macao UTM49N envelope: {g.bbox}"
                )

        for name, layer in (("facilities", facilities), ("lrt_stations", lrt_stations)):
            for g in layer.geoms:
                assert g.point_xy is not None
                x, y = g.point_xy
                assert MACAO_PT_E_MIN <= x <= MACAO_PT_E_MAX, (
                    f"{name} point X={x} out of Macao UTM49N bounds "
                    f"[{MACAO_PT_E_MIN}, {MACAO_PT_E_MAX}]"
                )
                assert MACAO_PT_N_MIN <= y <= MACAO_PT_N_MAX, (
                    f"{name} point Y={y} out of Macao UTM49N bounds "
                    f"[{MACAO_PT_N_MIN}, {MACAO_PT_N_MAX}]"
                )

    def test_t2_datum_shift_315m_cross_layer_alignment(
        self,
        zones_23: GpkgLayer,
        facilities: GpkgLayer,
    ) -> None:
        zone_polys = [poly for g in zones_23.geoms for poly in g.polygons]
        dists: list[float] = []
        for row, geom in zip(facilities.rows, facilities.geoms, strict=True):
            ft = str(row["facility_type"])
            if ft in {"ev_charger", "health_centre", "library", "bus_stop"}:
                if ft == "ev_charger" and as_bool(row["in_zona_a"]):
                    continue
                assert geom.point_xy is not None
                dists.append(point_distance_to_polygons(*geom.point_xy, zone_polys))
        assert len(dists) >= 50
        within_50m = sum(1 for d in dists if d <= 50.0) / len(dists)
        med_dist = statistics.median(dists)
        assert within_50m >= 0.90 and med_dist <= 5.0, (
            f"315m EPSG:8433 datum-shift detected: within_50m={within_50m:.1%}, "
            f"median_dist={med_dist:.2f} m"
        )

    def test_t2_esri_multiring_zone_area_sanity(self, zones_23: GpkgLayer) -> None:
        total_area = 0.0
        for r, g in zip(zones_23.rows, zones_23.geoms, strict=True):
            area_col = as_float(r["area_km2"])
            geom_area_km2 = g.area / 1e6
            assert 0.05 <= area_col <= 20.0, f"Zone {r['zone_id']} area out of range: {area_col}"
            assert abs(area_col - geom_area_km2) / area_col < 1e-2, (
                f"Zone {r['zone_id']} area_km2={area_col} != geom.area={geom_area_km2}"
            )
            total_area += area_col
        assert 20.0 <= total_area <= 45.0, f"Total Macao terrestrial area {total_area} km2 invalid"

    def test_t2_dual_landmass_spatial_coverage(
        self,
        zones_23: GpkgLayer,
        parishes_7: GpkgLayer,
        cells_m1_r9: GpkgLayer,
        facilities: GpkgLayer,
    ) -> None:
        cells_r9 = cells_m1_r9
        for name, layer in (
            ("zones_23", zones_23),
            ("parishes_7", parishes_7),
            ("cells_r9", cells_r9),
            ("facilities", facilities),
        ):
            ys = [g.centroid[1] for g in layer.geoms if g.centroid is not None]
            has_pen = any(y > PENINSULA_MIN_N for y in ys)
            has_isl = any(y < ISLANDS_MAX_N for y in ys)
            assert has_pen and has_isl, (
                f"{name} missing dual-landmass coverage (peninsula={has_pen}, islands={has_isl})"
            )

    def test_t2_cell_centroids_and_area_arithmetic(
        self,
        project_root: Path,
        cells_m1_r9: GpkgLayer,
        load_gpkg: Callable[..., GpkgLayer],
    ) -> None:
        cells_r9 = cells_m1_r9
        layers = [cells_r9]
        base_r9 = project_root / "data/frozen/cells_base_res9.gpkg"
        if base_r9.is_file():
            layers.append(load_gpkg(base_r9))
        for layer in layers:
            for r, g in zip(layer.rows, layer.geoms, strict=True):
                assert g.centroid is not None
                cx, cy = as_float(r["centroid_x"]), as_float(r["centroid_y"])
                area_col = as_float(r["area_km2"])
                geom_area_km2 = g.area / 1e6
                assert abs(cx - g.centroid[0]) < 1.0 and abs(cy - g.centroid[1]) < 1.0, (
                    f"Centroid mismatch in cell {r['cell_id']}"
                )
                assert abs(area_col - geom_area_km2) / area_col < 1e-3, (
                    f"Area mismatch in cell {r['cell_id']}: {area_col} vs {geom_area_km2}"
                )


# ---------------------------------------------------------------------------
# FA-3 (A04): Census Conservation & Spatial Units Traps (7 tests)
# ---------------------------------------------------------------------------


class TestFA03CensusAndSpatialUnits:
    """FA-3: Census Conservation & Spatial Units Traps (7 tests)."""

    def test_t2_zones_23_exact_count_and_no_maritime_z26(self, zones_23: GpkgLayer) -> None:
        assert len(zones_23.rows) == 23, f"Expected 23 terrestrial zones, got {len(zones_23.rows)}"
        zids = [str(r["zone_id"]) for r in zones_23.rows]
        assert len(set(zids)) == 23
        for r in zones_23.rows:
            assert not MARITIME_ZONE_RE.search(str(r["zone_id"])), (
                f"Maritime zone Z26 must be excluded: {r['zone_id']}"
            )
            assert not MARITIME_ZONE_RE.search(str(r["zone_name"])), (
                f"Maritime zone 水域 must be excluded: {r['zone_name']}"
            )

    def test_t2_parishes_7_exact_count_and_taipa_present(self, parishes_7: GpkgLayer) -> None:
        assert len(parishes_7.rows) == 7, f"Expected 7 parishes, got {len(parishes_7.rows)}"
        pids = [str(r["parish_id"]) for r in parishes_7.rows]
        assert len(set(pids)) == 7
        for r in parishes_7.rows:
            assert str(r["parish_name"]).strip() and str(r["parish_name_zh"]).strip()
        joined = " ".join(f"{r['parish_name']} {r['parish_name_zh']}" for r in parishes_7.rows)
        assert TAIPA_PARISH_RE.search(joined), "Taipa (嘉模/Carmo) missing from parishes_7.gpkg"

    def test_t2_zero_pop_exclusion_all_spatial_units(
        self,
        project_root: Path,
        zones_23: GpkgLayer,
        parishes_7: GpkgLayer,
        cells_m1_r9: GpkgLayer,
        cells_m1_r8: GpkgLayer,
        load_gpkg: Callable[..., GpkgLayer],
    ) -> None:
        cells_r9 = cells_m1_r9
        cells_r8 = cells_m1_r8
        layers = [
            ("zones_23", zones_23),
            ("parishes_7", parishes_7),
            ("cells", cells_r9),
            ("cells_res8", cells_r8),
        ]
        for opt_rel in ("data/frozen/cells_base_res9.gpkg", "data/frozen/cells_base_res8.gpkg"):
            if (project_root / opt_rel).is_file():
                layers.append((opt_rel, load_gpkg(opt_rel)))
        for label, layer in layers:
            for r in layer.rows:
                pop = as_float(r["pop_total"])
                assert pop > 0.0, f"Zero/negative population unit found in {label}: pop_total={pop}"

    def test_t2_terrestrial_census_conservation_681293_82751(
        self,
        project_root: Path,
        zones_23: GpkgLayer,
        parishes_7: GpkgLayer,
        cells_m1_r9: GpkgLayer,
        cells_m1_r8: GpkgLayer,
        load_gpkg: Callable[..., GpkgLayer],
    ) -> None:
        cells_r9 = cells_m1_r9
        cells_r8 = cells_m1_r8
        layers = [
            ("zones_23", zones_23, POP_TOL_ZONES_PARISHES, POP_TOL_ZONES_PARISHES),
            ("parishes_7", parishes_7, POP_TOL_ZONES_PARISHES, POP_TOL_ZONES_PARISHES),
            ("cells", cells_r9, POP_TOL_H3_TOTAL, POP_TOL_H3_65PLUS),
            ("cells_res8", cells_r8, POP_TOL_H3_TOTAL, POP_TOL_H3_65PLUS),
        ]
        for opt_rel in ("data/frozen/cells_base_res9.gpkg", "data/frozen/cells_base_res8.gpkg"):
            if (project_root / opt_rel).is_file():
                layers.append((opt_rel, load_gpkg(opt_rel), POP_TOL_H3_TOTAL, POP_TOL_H3_65PLUS))
        for label, layer, tol_tot, tol_65 in layers:
            tot = sum(as_float(r["pop_total"]) for r in layer.rows)
            p65 = sum(as_float(r["pop_65plus"]) for r in layer.rows)
            assert abs(tot - TERRESTRIAL_POP_TOTAL) / TERRESTRIAL_POP_TOTAL <= tol_tot, (
                f"Population conservation drift in {label}: {tot:.1f} vs {TERRESTRIAL_POP_TOTAL}"
            )
            assert abs(p65 - TERRESTRIAL_POP_65PLUS) / TERRESTRIAL_POP_65PLUS <= tol_65, (
                f"Elderly population drift in {label}: {p65:.1f} vs {TERRESTRIAL_POP_65PLUS}"
            )

    def test_t2_demographic_ratios_and_log_density(
        self,
        zones_23: GpkgLayer,
        cells_m1_r9: GpkgLayer,
        cells_m1_r8: GpkgLayer,
    ) -> None:
        cells_r9 = cells_m1_r9
        cells_r8 = cells_m1_r8
        for label, layer in (
            ("zones_23", zones_23),
            ("cells_r9", cells_r9),
            ("cells_r8", cells_r8),
        ):
            for r in layer.rows:
                ptot = as_float(r["pop_total"])
                p65 = as_float(r["pop_65plus"])
                eld = as_float(r["elderly_share"])
                area = as_float(r["area_km2"])
                dens = as_float(r["pop_density"])
                assert 0.0 <= p65 <= ptot + 1e-6, f"{label}: pop_65plus > pop_total"
                assert 0.0 <= eld <= 1.0, f"{label}: elderly_share={eld} outside [0, 1]"
                assert abs(eld - p65 / ptot) < 1e-4, (
                    f"{label}: elderly_share != pop_65plus/pop_total"
                )
                assert abs(dens - ptot / area) / dens < 1e-3, (
                    f"{label}: pop_density != pop_total/area_km2"
                )

        for r in zones_23.rows:
            dens = as_float(r["pop_density"])
            lpd = as_float(r["log_pop_density"])
            err_ln = abs(lpd - math.log(dens)) / max(abs(math.log(dens)), 1.0)
            err_l10 = abs(lpd - math.log10(dens)) / max(abs(math.log10(dens)), 1.0)
            assert min(err_ln, err_l10) <= 5e-3, f"log_pop_density mismatch in {r['zone_id']}"

    def test_t2_h3_v4_cell_id_format_and_counts(
        self,
        cells_m1_r9: GpkgLayer,
        cells_m1_r8: GpkgLayer,
    ) -> None:
        cells_r9 = cells_m1_r9
        cells_r8 = cells_m1_r8
        lo9, hi9 = H3_R9_COUNT_BOUNDS
        assert lo9 <= len(cells_r9.rows) <= hi9
        assert H3_R8_COUNT_MIN <= len(cells_r8.rows) < len(cells_r9.rows)
        r9_ids = [str(r["cell_id"]).lower() for r in cells_r9.rows]
        r8_ids = [str(r["cell_id"]).lower() for r in cells_r8.rows]
        assert len(r9_ids) == len(set(r9_ids))
        assert len(r8_ids) == len(set(r8_ids))
        assert all(H3_R9_RE.match(cid) for cid in r9_ids)
        assert all(H3_R8_RE.match(cid) for cid in r8_ids)

    def test_t2_cell_covariates_domain_and_no_nans(self, cells_m1_r9: GpkgLayer) -> None:
        cells_r9 = cells_m1_r9
        for r in cells_r9.rows:
            for feat in ML_FEATURES_REQUIRED:
                val = as_float(r[feat])
                if feat in {"elderly_share", "res_bldg_share", "com_bldg_share"}:
                    assert 0.0 <= val <= 1.0, f"{feat}={val} out of [0, 1]"
                else:
                    assert val >= 0.0, f"{feat}={val} < 0"
            has_ev = int(r["has_ev_charger"])
            ev_cnt = int(r["ev_station_count"])
            ev_cap = int(r["ev_charger_capacity"])
            assert has_ev in {0, 1}
            assert (has_ev == 1) == (ev_cnt > 0) == (ev_cap > 0), (
                f"Inconsistent EV target/audit columns in cell {r['cell_id']}: "
                f"has_ev={has_ev}, cnt={ev_cnt}, cap={ev_cap}"
            )
            assert ev_cap >= ev_cnt >= 0


# ---------------------------------------------------------------------------
# FA-4 (A03 Facilities): Whitelisting, CEM EV Capacity & 25m Dedup (7 tests)
# ---------------------------------------------------------------------------


class TestFA04FacilitiesAndDedup:
    """FA-4: Facility Whitelisting, CEM EV Capacity & 25 m Dedup Traps (7 tests)."""

    def test_t2_facilities_exact_5_types_and_pk(self, facilities: GpkgLayer) -> None:
        for col in FACILITIES_REQUIRED_COLS:
            assert col in facilities.columns, f"Missing {col} in facilities_clean.gpkg"
        types_seen = {str(r["facility_type"]) for r in facilities.rows}
        assert types_seen == set(FACILITY_TYPES), f"Facility types mismatch: {types_seen}"
        fids = [str(r["facility_id"]).strip() for r in facilities.rows]
        assert all(fids) and len(fids) == len(set(fids)), "Non-unique or blank facility_id"
        for r in facilities.rows:
            assert (
                str(r["name"]).strip()
                and str(r["source"]).strip()
                and str(r["capacity_unit"]).strip()
            )
            as_bool(r["capacity_imputed"])
            as_bool(r["in_zona_a"])

    def test_t2_facilities_realistic_category_counts(self, facilities: GpkgLayer) -> None:
        counts: dict[str, int] = defaultdict(int)
        for r in facilities.rows:
            counts[str(r["facility_type"])] += 1
        for ftype, (lo, hi) in FACILITY_COUNT_BOUNDS.items():
            c = counts[ftype]
            assert lo <= c <= hi, (
                f"Facility {ftype} count {c} outside realistic bounds [{lo}, {hi}]"
            )

    def test_t2_cem_ev_capacity_anti_double_count(self, facilities: GpkgLayer) -> None:
        ev_rows = [r for r in facilities.rows if r["facility_type"] == "ev_charger"]
        caps = [as_float(r["capacity"]) for r in ev_rows]
        assert all(c >= 1.0 for c in caps), "EV station capacity must be >= 1.0"
        tot_cap = sum(caps)
        assert EV_CAPACITY_MIN <= tot_cap <= EV_CAPACITY_MAX, (
            f"EV capacity double-count or unit-weight trap: sum(capacity)={tot_cap:.1f} "
            f"outside [{EV_CAPACITY_MIN}, {EV_CAPACITY_MAX}]"
        )
        with_pile = [r for r in ev_rows if r.get("pilenum") is not None]
        assert with_pile, "pilenum column empty on EV stations"
        assert (
            sum(as_float(r["capacity"]) for r in with_pile)
            <= sum(as_float(r["pilenum"]) for r in with_pile) + 1e-6
        )

    def test_t2_facilities_positive_capacity_and_park_area(self, facilities: GpkgLayer) -> None:
        park_caps: list[float] = []
        for r in facilities.rows:
            ft = str(r["facility_type"])
            cap = as_float(r["capacity"])
            assert cap > 0.0
            if ft in UNIT_CAPACITY_FACILITIES:
                assert abs(cap - 1.0) < 1e-6, f"Expected unit capacity 1.0 for {ft}, got {cap}"
            elif ft == "park":
                park_caps.append(cap)
        assert statistics.median(park_caps) > 100.0, "Park median area (m2) unexpectedly small"

    def test_t2_cross_source_25m_dedup_no_false_bus_collapse(self, facilities: GpkgLayer) -> None:
        by_type: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
        for r, g in zip(facilities.rows, facilities.geoms, strict=True):
            assert g.point_xy is not None
            by_type[str(r["facility_type"])].append(
                (str(r["source"]), g.point_xy[0], g.point_xy[1])
            )
        for ftype, pts in by_type.items():
            for i in range(len(pts)):
                s1, x1, y1 = pts[i]
                for j in range(i + 1, len(pts)):
                    s2, x2, y2 = pts[j]
                    if facility_source_family(s1) != facility_source_family(s2):
                        d = math.hypot(x1 - x2, y1 - y2)
                        assert d >= 25.0, f"Cross-source pair < 25m in {ftype}: {d:.2f} m"

    def test_t2_zona_a_ev_flag_integrity(self, facilities: GpkgLayer) -> None:
        zona_a_xs: list[float] = []
        for r, g in zip(facilities.rows, facilities.geoms, strict=True):
            flag = as_bool(r["in_zona_a"])
            if r["facility_type"] == "ev_charger" and flag:
                assert g.point_xy is not None
                zona_a_xs.append(g.point_xy[0])
        assert zona_a_xs, "Expected >= 1 EV station flagged with in_zona_a == True"
        assert all(x > ZONA_A_MIN_E for x in zona_a_xs), (
            f"Zona A EV station easting <= {ZONA_A_MIN_E}: {zona_a_xs}"
        )

    def test_t2_lrt_stations_count_bounds_and_no_lotus_dupe(self, lrt_stations: GpkgLayer) -> None:
        lo, hi = LRT_COUNT_BOUNDS
        assert lo <= len(lrt_stations.rows) <= hi
        pts = [g.point_xy for g in lrt_stations.geoms if g.point_xy is not None]
        assert len(pts) == len(lrt_stations.rows)
        for x, y in pts:
            assert MACAO_PT_E_MIN <= x <= MACAO_PT_E_MAX
            assert MACAO_PT_N_MIN <= y <= MACAO_PT_N_MAX
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dist = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                assert dist >= 25.0, f"Duplicate LRT stations < 25m apart: {dist:.2f} m"


# ---------------------------------------------------------------------------
# FA-5 (A05, A06): Walk Graph, Spatial Blocks & Anti-Leakage Traps (8 tests)
# ---------------------------------------------------------------------------


class TestFA05WalkGraphBlocksAndAntiLeakage:
    """FA-5: Walk Graph, Spatial Blocks & Anti-Leakage Traps (8 tests)."""

    def test_t2_graph_edges_walk_time_strictly_positive_le_10(
        self, graph_edges: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        _, rows = graph_edges
        assert len(rows) > 0, "graph_edges.csv is empty"
        for idx, r in enumerate(rows):
            t = as_float(r["walk_time_min"])
            assert t > 0.0, f"Zero/negative walk_time_min (snap singularity) at row {idx}: {t}"
            assert t <= MAX_GRAPH_WALK_MIN + 1e-9, (
                f"walk_time_min exceeds 10.0 min cutoff at row {idx}: {t}"
            )

    def test_t2_graph_edges_no_self_loops_reciprocal_weight(
        self, graph_edges: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        _, rows = graph_edges
        for r in rows:
            assert r["src_cell_id"] != r["dst_cell_id"], f"Self-loop on {r['src_cell_id']}"
            t = as_float(r["walk_time_min"])
            w = as_float(r["weight"])
            assert w > 0.0
            assert abs(w - 1.0 / t) <= max(1e-6, 1e-5 * (1.0 / t)), (
                f"weight={w} != 1/walk_time_min ({1.0 / t})"
            )

    def test_t2_graph_edges_undirected_symmetry_unique(
        self, graph_edges: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        _, rows = graph_edges
        seen: dict[tuple[str, str], tuple[float, float, bool]] = {}
        for r in rows:
            pair = (r["src_cell_id"], r["dst_cell_id"])
            assert pair not in seen, f"Duplicate graph edge: {pair}"
            seen[pair] = (
                as_float(r["walk_time_min"]),
                as_float(r["weight"]),
                as_bool(r["same_block"]),
            )
        for (u, v), (t1, w1, sb1) in seen.items():
            assert (v, u) in seen, f"Missing symmetric reverse edge ({v}, {u})"
            t2, w2, sb2 = seen[(v, u)]
            assert abs(t1 - t2) < 1e-6 and abs(w1 - w2) < 1e-5 and sb1 == sb2

    def test_t2_graph_dual_landmass_and_no_cross_sea_teleport(
        self,
        graph_edges: tuple[list[str], list[dict[str, str]]],
        cells_m1_r9: GpkgLayer,
        walk_graph_summary: GraphMLSummary,
    ) -> None:
        cells_r9 = cells_m1_r9
        assert walk_graph_summary.peninsula_nodes >= 20
        assert walk_graph_summary.island_nodes >= 10, (
            "osm_walk_graph.graphml missing Taipa-Cotai-Coloane nodes"
        )
        parish_by_cell = {str(r["cell_id"]): str(r["parish_id"]) for r in cells_r9.rows}
        _, rows = graph_edges
        pen_edges = 0
        isl_edges = 0
        cross_sea = 0
        for r in rows:
            p_u = parish_by_cell[r["src_cell_id"]]
            p_v = parish_by_cell[r["dst_cell_id"]]
            u_pen = p_u in PENINSULA_PARISH_IDS
            v_pen = p_v in PENINSULA_PARISH_IDS
            u_isl = p_u in ISLAND_PARISH_IDS
            v_isl = p_v in ISLAND_PARISH_IDS
            if u_pen and v_pen:
                pen_edges += 1
            elif u_isl and v_isl:
                isl_edges += 1
            elif (u_pen and v_isl) or (u_isl and v_pen):
                cross_sea += 1
        assert pen_edges >= 10, f"Too few Peninsula pedestrian edges: {pen_edges}"
        assert isl_edges >= 5, (
            f"Taipa-Cotai-Coloane pedestrian edges missing/dropped ({isl_edges} < 5)!"
        )
        assert cross_sea == 0, f"Impossible <=10 min cross-harbour teleport edges: {cross_sea}"

    def test_t2_spatial_blocks_5_folds_both_ev_classes_geocoherent(
        self,
        spatial_folds: tuple[list[str], list[dict[str, str]]],
        cells_m1_r9: GpkgLayer,
        graph_edges: tuple[list[str], list[dict[str, str]]],
    ) -> None:
        cells_r9 = cells_m1_r9
        _, f_rows = spatial_folds
        blocks = {int(float(r["spatial_block"])) for r in f_rows}
        assert blocks == {0, 1, 2, 3, 4}, f"Spatial blocks must be {{0..4}}, got {blocks}"
        for blk in range(5):
            b_cells = [r for r in cells_r9.rows if int(r["spatial_block"]) == blk]
            n_pos = sum(int(r["has_ev_charger"]) for r in b_cells)
            n_neg = len(b_cells) - n_pos
            assert n_pos >= 1 and n_neg >= 1, (
                f"Spatial block {blk} missing class balance: n_pos={n_pos}, n_neg={n_neg}"
            )
        _, e_rows = graph_edges
        same_block_rate = sum(1 for r in e_rows if as_bool(r["same_block"])) / len(e_rows)
        assert same_block_rate >= 0.50, (
            f"Spatial blocks lack geographic coherence "
            f"(same_block_rate={same_block_rate:.2f} < 0.50)"
        )

    def test_t2_split_metadata_json_schema_and_counts(
        self,
        split_metadata: dict[str, Any],
        cells_m1_r9: GpkgLayer,
    ) -> None:
        cells_r9 = cells_m1_r9
        assert split_metadata.get("seed", split_metadata.get("random_seed")) == GLOBAL_SEED
        blocks_container = (
            split_metadata.get("blocks")
            or split_metadata.get("per_block")
            or split_metadata.get("folds")
        )
        assert blocks_container is not None, "split_metadata.json missing per-block counts"
        total_meta_cells = 0
        for blk in range(5):
            b_info = (
                blocks_container[str(blk)]
                if isinstance(blocks_container, dict) and str(blk) in blocks_container
                else (
                    blocks_container[blk]
                    if isinstance(blocks_container, (list, dict)) and blk in blocks_container
                    else blocks_container[blk]
                )
            )
            b_cells = [r for r in cells_r9.rows if int(r["spatial_block"]) == blk]
            exp_n = len(b_cells)
            exp_pos = sum(int(r["has_ev_charger"]) for r in b_cells)
            exp_neg = exp_n - exp_pos
            assert int(b_info["n_cells"]) == exp_n
            assert int(b_info["n_pos_ev"]) == exp_pos and exp_pos >= 1
            assert int(b_info["n_neg_ev"]) == exp_neg and exp_neg >= 1
            total_meta_cells += exp_n
        assert total_meta_cells == len(cells_r9.rows)

    def test_t2_anti_leakage_feature_manifest_and_ml_json(
        self,
        feature_manifest: dict[str, Any],
        ml_baselines: dict[str, Any],
        cells_r9: GpkgLayer,
    ) -> None:
        ml_feats = list(feature_manifest["ml_features"])
        excl = set(feature_manifest["excluded_target_and_audit"])
        prefixes = tuple(feature_manifest["analysis_prefix_excluded"])
        assert ml_feats, "ml_features cannot be empty"
        assert set(ml_feats).isdisjoint(excl), "ml_features overlaps excluded_target_and_audit"
        for col in ml_feats:
            assert not col.startswith(prefixes), f"Analysis column leaked into ml_features: {col}"
            assert not EV_COL_LEAK_RE.search(col), f"Target/EV leak in ml_features: {col}"
            for r in cells_r9.rows:
                as_float(r[col])
        ml_cols = list(ml_baselines["feature_columns"])
        assert ml_cols == ml_feats, (
            f"phase_a_ml_baselines.json feature_columns {ml_cols} != feature_manifest {ml_feats}"
        )

    def test_t2_anti_leakage_poi_texts_ev_regex_zero_hits(
        self,
        poi_texts: tuple[list[str], list[dict[str, str]]],
        cells_m1_r9: GpkgLayer,
    ) -> None:
        cells_r9 = cells_m1_r9
        _, rows = poi_texts
        assert len(rows) == len(cells_r9.rows)
        for r in rows:
            raw_c = int(float(r["poi_count_raw"]))
            filt_c = int(float(r["poi_count_filtered"]))
            assert 0 <= filt_c <= raw_c
            hit = EV_TEXT_LEAK_RE.search(r["poi_text"])
            assert hit is None, f"POI text EV leak in cell {r['cell_id']}: matched {hit.group(0)!r}"


# ---------------------------------------------------------------------------
# FA-6 (A07, A08): Gaussian 2SFCA Mathematical Invariants & EDA/Lorenz (7 tests)
# ---------------------------------------------------------------------------


class TestFA06Gaussian2SFCAMathAndEda:
    """FA-6: Gaussian 2SFCA Mathematical Invariants & EDA/Lorenz (7 tests)."""

    def test_t2_2sfca_finite_nonnegative_all_columns(
        self,
        cells_r9: GpkgLayer,
        cells_r8: GpkgLayer,
        zones_23_with_2sfca: GpkgLayer,
    ) -> None:
        for name, layer in (
            ("cells", cells_r9),
            ("cells_res8", cells_r8),
            ("zones_23_with_2sfca", zones_23_with_2sfca),
        ):
            for col in ALL_ACC_COLS:
                assert col in layer.columns, f"{name} missing {col}"
                vals = [as_float(r[col]) for r in layer.rows]
                if col in ACC_INDIVIDUAL_COLS:
                    assert all(v >= 0.0 for v in vals), f"{name}.{col} has negative score"

    def test_t2_2sfca_non_degenerate_distribution(self, cells_r9: GpkgLayer) -> None:
        for col in ALL_ACC_COLS:
            vals = [as_float(r[col]) for r in cells_r9.rows]
            assert max(vals) > 0.0, f"{col} is all-zero"
            assert statistics.pstdev(vals) > 0.0, f"{col} has zero variance"

    def test_t2_2sfca_weighted_supply_conservation(
        self,
        cells_r9: GpkgLayer,
        cells_r8: GpkgLayer,
        facilities: GpkgLayer,
    ) -> None:
        cap_by_type: dict[str, float] = defaultdict(float)
        n_ev_stations = 0
        for r in facilities.rows:
            ft = str(r["facility_type"])
            cap_by_type[ft] += as_float(r["capacity"])
            if ft == "ev_charger":
                n_ev_stations += 1

        for label, layer in (("cells_r9", cells_r9), ("cells_r8", cells_r8)):
            pops = [as_float(r["pop_total"]) for r in layer.rows]
            for cutoff in (10, 15):
                for ft in FACILITY_TYPES:
                    col = f"acc{cutoff}_{ft}"
                    pa_sum = sum(
                        p * as_float(r[col]) for p, r in zip(pops, layer.rows, strict=True)
                    )
                    supply = cap_by_type[ft]
                    raw_ratio = pa_sum / supply
                    scale_k = min(
                        (1.0, 1e3, 1e4, 1e5),
                        key=lambda k: abs(math.log10(max(raw_ratio, 1e-12) / k)),
                    )
                    assert 0.0 < pa_sum / scale_k <= supply * (1.0 + 1e-4), (
                        f"Supply conservation exceeded in {label}.{col}: "
                        f"sum(P*A)/K={pa_sum / scale_k:.2f} > sum(S)={supply:.2f}"
                    )
                unw_col = f"acc{cutoff}_ev_unweighted"
                pa_unw = sum(
                    p * as_float(r[unw_col]) for p, r in zip(pops, layer.rows, strict=True)
                )
                raw_unw = pa_unw / n_ev_stations
                scale_u = min(
                    (1.0, 1e3, 1e4, 1e5),
                    key=lambda k: abs(math.log10(max(raw_unw, 1e-12) / k)),
                )
                assert 0.0 < pa_unw / scale_u <= n_ev_stations * (1.0 + 1e-4)

    def test_t2_2sfca_support_and_total_supply_monotonicity_10_to_15(
        self, cells_r9: GpkgLayer
    ) -> None:
        pops = [as_float(r["pop_total"]) for r in cells_r9.rows]
        for ft in (*FACILITY_TYPES, "ev_unweighted"):
            a10 = [as_float(r[f"acc10_{ft}"]) for r in cells_r9.rows]
            a15 = [as_float(r[f"acc15_{ft}"]) for r in cells_r9.rows]
            for i, (v10, v15) in enumerate(zip(a10, a15, strict=True)):
                if v10 > 1e-12:
                    assert v15 > 0.0, f"10->15m support monotonicity broken for {ft} at cell {i}"
            z10 = 100.0 * sum(1 for v in a10 if v <= 1e-12) / len(a10)
            z15 = 100.0 * sum(1 for v in a15 if v <= 1e-12) / len(a15)
            assert z15 <= z10 + 1e-9, f"zero_acc_pct(15)={z15} > zero_acc_pct(10)={z10} for {ft}"
            tot10 = sum(p * v for p, v in zip(pops, a10, strict=True))
            tot15 = sum(p * v for p, v in zip(pops, a15, strict=True))
            assert tot15 >= tot10 - 1e-5, f"Total reached supply decreased 10->15m for {ft}"

    def test_t2_2sfca_ev_weighted_vs_unweighted_theorem(self, cells_r9: GpkgLayer) -> None:
        for cutoff in (10, 15):
            wt = [as_float(r[f"acc{cutoff}_ev_charger"]) for r in cells_r9.rows]
            unw = [as_float(r[f"acc{cutoff}_ev_unweighted"]) for r in cells_r9.rows]
            for idx, (w_val, u_val) in enumerate(zip(wt, unw, strict=True)):
                assert (w_val > 1e-12) == (u_val > 1e-12), (
                    f"Support mismatch between weighted and unweighted EV at row {idx}"
                )
                assert w_val >= u_val - 1e-9, (
                    f"Pointwise dominance violated at row {idx}: "
                    f"weighted={w_val} < unweighted={u_val}"
                )

    def test_t2_eda_summary_order_statistics(
        self, eda_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        headers, rows = eda_table
        assert rows
        h_map = {h.lower(): h for h in headers}
        for req in ("min", "median", "mean", "max", "std"):
            assert req in h_map, f"eda_summary.csv missing {req} column"
        for r in rows:
            vmin = as_float(r[h_map["min"]])
            vmed = as_float(r[h_map["median"]])
            vmean = as_float(r[h_map["mean"]])
            vmax = as_float(r[h_map["max"]])
            vstd = as_float(r[h_map["std"]])
            assert vmin <= vmed + 1e-9 <= vmax + 2e-9
            assert vmin <= vmean + 1e-9 <= vmax + 2e-9
            assert vstd >= 0.0

    def test_t2_lorenz_curves_monotonic_convex_subdiagonal(
        self, lorenz_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        headers, rows = lorenz_table
        p_col, q_col = _find_lorenz_cols(headers)
        curves: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
        for r in rows:
            curves[(str(r.get("threshold_min", "15")), r["facility_type"].strip())].append(
                (as_float(r[p_col]), as_float(r[q_col]))
            )
        assert len(curves) >= 5
        for key, pts in curves.items():
            assert abs(pts[0][0]) <= 1e-3 and abs(pts[0][1]) <= 1e-3
            assert abs(pts[-1][0] - 1.0) <= 1e-3 and abs(pts[-1][1] - 1.0) <= 1e-3
            for i, (p, q) in enumerate(pts):
                assert -1e-6 <= q <= p + 1e-5 <= 1.0 + 2e-5, f"Subdiagonal violated in {key}"
                if i > 0:
                    assert p >= pts[i - 1][0] - 1e-9 and q >= pts[i - 1][1] - 1e-9


# ---------------------------------------------------------------------------
# FA-7 (A09, A10, A12): H1 Inequality/LISA, H2 Paired Bootstrap & MAUP (8 tests)
# ---------------------------------------------------------------------------


class TestFA07InequalityLisaBootstrapAndMaup:
    """FA-7: H1 Inequality/LISA, H2 Paired Bootstrap & MAUP (8 tests)."""

    def test_t2_h1_gini_bounds_ci_and_coverage(
        self, h1_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        _, rows = h1_table
        seen_keys: set[tuple[str, int, str]] = set()
        for r in rows:
            res = normalize_resolution_label(r["resolution"])
            thr = int(float(r["threshold_min"]))
            ft = r["facility_type"].strip()
            key = (res, thr, ft)
            assert key not in seen_keys, f"Duplicate H1 row key: {key}"
            seen_keys.add(key)
            g = as_float(r["gini"])
            lo = as_float(r["gini_ci_low"])
            hi = as_float(r["gini_ci_high"])
            zpct = as_float(r["zero_acc_pct"])
            assert 0.0 <= g <= 1.0 and 0.0 <= lo < hi <= 1.0
            assert lo - 0.015 <= g <= hi + 0.015
            assert 0.0 <= zpct < 100.0
        for res in ("res9", "res8"):
            for thr in (10, 15):
                for ft in (*FACILITY_TYPES, "composite"):
                    assert (res, thr, ft) in seen_keys, f"Missing H1 key {(res, thr, ft)}"

    def test_t2_h1_theil_t_exact_additivity_with_zero_acc(
        self, h1_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        _, rows = h1_table
        has_zero_acc_row = False
        for r in rows:
            tt = as_float(r["theil_total"])
            tw = as_float(r["theil_within"])
            tb = as_float(r["theil_between"])
            lo = as_float(r["theil_total_ci_low"])
            hi = as_float(r["theil_total_ci_high"])
            bs = as_float(r["theil_between_share"])
            if as_float(r["zero_acc_pct"]) > 0.0:
                has_zero_acc_row = True
            assert tt >= -1e-9 and tw >= -1e-9 and tb >= -1e-9
            assert abs(tt - (tw + tb)) < 1e-6, f"Theil's T additivity broken: {tt} != {tw} + {tb}"
            assert 0.0 <= lo <= hi and lo - 0.02 <= tt <= hi + 0.02
            assert -1e-6 <= bs <= 1.0 + 1e-6
            if tt > 1e-6:
                assert abs(bs - tb / tt) < 1e-4
        assert has_zero_acc_row or len(rows) >= 12

    def test_t2_h1_global_moran_and_lisa_enum_bounds(
        self,
        h1_table: tuple[list[str], list[dict[str, str]]],
        cells_r9: GpkgLayer,
    ) -> None:
        _, rows = h1_table
        for r in rows:
            mi = as_float(r["moran_i"])
            mp = as_float(r["moran_p"])
            assert -1.5 <= mi <= 1.5 and 0.0 < mp <= 1.0
        for metric in _detect_lisa_metrics(cells_r9.columns):
            for r in cells_r9.rows:
                as_float(r[f"lisa_i_{metric}"])
                lp = as_float(r[f"lisa_p_{metric}"])
                lq = as_float(r[f"lisa_q_fdr_{metric}"])
                cl = str(r[f"lisa_cluster_{metric}"])
                assert 0.0 <= lp <= 1.0 and 0.0 <= lq <= 1.0
                assert cl in LISA_CLUSTERS

    def test_t2_h1_lisa_fdr_monotonicity_and_quadrant_signs(self, cells_r9: GpkgLayer) -> None:
        for metric in _detect_lisa_metrics(cells_r9.columns):
            for r in cells_r9.rows:
                li = as_float(r[f"lisa_i_{metric}"])
                lp = as_float(r[f"lisa_p_{metric}"])
                lq = as_float(r[f"lisa_q_fdr_{metric}"])
                cl = str(r[f"lisa_cluster_{metric}"])
                assert lq >= lp - 1e-9
                if cl != "NS":
                    assert lq <= 0.05 + 1e-9
                if lq > 0.05 + 1e-9:
                    assert cl == "NS"
                if cl in {"HH", "LL"}:
                    assert li >= -1e-9, f"Quadrant sign violation: {cl} has negative lisa_i={li}"
                elif cl in {"HL", "LH"}:
                    assert li <= 1e-9, f"Quadrant sign violation: {cl} has positive lisa_i={li}"

    def test_t2_h2_all_4_comparisons_both_thresholds_nboot_2000(
        self, h2_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        _, rows = h2_table
        assert len(rows) >= 8
        seen = set()
        for r in rows:
            thr = int(float(r["threshold_min"]))
            ft = _match_h2_facility(r["comparison"])
            assert ft in TRADITIONAL_FACILITY_TYPES
            seen.add((thr, ft))
            assert int(float(r["n_boot"])) == 2000
            assert int(float(r["seed"])) == GLOBAL_SEED
        for thr in (10, 15):
            for ft in TRADITIONAL_FACILITY_TYPES:
                assert (thr, ft) in seen

    def test_t2_h2_gini_diff_ci_and_holm_ordering(
        self, h2_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        _, rows = h2_table
        for r in rows:
            gev = as_float(r["gini_ev"])
            goth = as_float(r["gini_other"])
            diff = as_float(r["gini_diff"])
            lo = as_float(r["ci_low"])
            hi = as_float(r["ci_high"])
            praw = as_float(r["p_boot_raw"])
            pholm = as_float(r["p_holm"])
            assert abs(diff - (gev - goth)) < 1e-6, (
                f"H2 gini_diff mismatch: {diff} != {gev} - {goth}"
            )
            assert -1.0 <= lo < hi <= 1.0
            assert lo - 0.015 <= diff <= hi + 0.015
            assert 0.0 <= praw <= pholm <= 1.0 + 1e-12, (
                f"H2 Holm p-value ordering broken: p_holm={pholm} < p_boot_raw={praw}"
            )

    def test_t2_h2_holm_bonferroni_exact_stepdown(
        self, h2_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        _, rows = h2_table
        by_thr: dict[int, list[dict[str, str]]] = defaultdict(list)
        for r in rows:
            by_thr[int(float(r["threshold_min"]))].append(r)
        per_thr_match = all(
            all(
                abs(as_float(r["p_holm"]) - exp) <= 1e-4
                for r, exp in zip(
                    grp,
                    holm_bonferroni([as_float(x["p_boot_raw"]) for x in grp]),
                    strict=True,
                )
            )
            for grp in by_thr.values()
        )
        global_exp = holm_bonferroni([as_float(r["p_boot_raw"]) for r in rows])
        global_match = all(
            abs(as_float(r["p_holm"]) - exp) <= 1e-4
            for r, exp in zip(rows, global_exp, strict=True)
        )
        assert per_thr_match or global_match, "H2 p_holm fails Holm-Bonferroni step-down check"

    def test_t2_maup_r8_vs_r9_completeness_and_finiteness(
        self, maup_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        headers, rows = maup_table
        assert len(rows) >= 5
        ftypes = {r["facility_type"].strip() for r in rows}
        assert set(FACILITY_TYPES).issubset(ftypes)
        num_cols = [
            h for h in headers if any(k in h.lower() for k in ("gini", "theil", "moran", "delta"))
        ]
        assert len(num_cols) >= 6
        for r in rows:
            for col in num_cols:
                as_float(r[col])


# ---------------------------------------------------------------------------
# FA-8 (A11): H3 Statistical-Zone Spatial Regression Boundaries (6 tests)
# ---------------------------------------------------------------------------


class TestFA08H3SpatialRegressionBoundaries:
    """FA-8: H3 Statistical-Zone Spatial Regression Boundaries at N=23 (6 tests)."""

    def test_t2_h3_no_pretest_switching_all_3_models(
        self, h3_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        _, rows = h3_table
        by_spec: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for r in rows:
            key = (r["threshold_min"].strip(), r["outcome"].strip(), r["weights_spec"].strip())
            by_spec[key].add(r["model_type"].strip())
        assert by_spec
        for key, models in by_spec.items():
            assert models == H3_MODEL_TYPES, (
                f"H3 pretest switching detected for {key}: models={models} != {H3_MODEL_TYPES}"
            )

    def test_t2_h3_exact_n_obs_23_zones(
        self, h3_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        _, rows = h3_table
        for r in rows:
            assert int(float(r["n_obs"])) == 23, f"Invalid H3 n_obs={r['n_obs']} (must be 23)"

    def test_t2_h3_required_predictors_and_weights_specs(
        self, h3_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        _, rows = h3_table
        w_types = {normalize_weights_label(r["weights_spec"]) for r in rows}
        assert {"knn4", "bridge_queen"}.issubset(w_types)
        blocks: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
        for r in rows:
            blocks[(r["threshold_min"], r["outcome"], r["weights_spec"], r["model_type"])].add(
                r["term"].strip().lower()
            )
        for key, terms in blocks.items():
            assert "elderly_share" in terms and "log_pop_density" in terms, (
                f"H3 model {key} missing elderly_share / log_pop_density: {terms}"
            )

    def test_t2_h3_coef_se_ci_p_geometry(
        self, h3_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        _, rows = h3_table
        for r in rows:
            coef = as_float(r["coef"])
            se = as_float(r["se"])
            lo = as_float(r["ci_low"])
            hi = as_float(r["ci_high"])
            pv = as_float(r["p_value"])
            assert se > 0.0
            assert lo < coef < hi
            z_mult = (hi - lo) / (2.0 * se)
            assert 1.85 <= z_mult <= 2.35, f"Unexpected CI multiplier {z_mult:.3f} in {r}"
            assert 0.0 <= pv <= 1.0

    def test_t2_h3_fit_stats_and_resid_moran_i(
        self, h3_table: tuple[list[str], list[dict[str, str]]]
    ) -> None:
        _, rows = h3_table
        for r in rows:
            r2 = as_float(r["r2_or_pseudo_r2"])
            as_float(r["aic"])
            rmi = as_float(r["resid_moran_i"])
            rmp = as_float(r["resid_moran_p"])
            assert -1.0 <= r2 <= 1.0
            assert -1.5 <= rmi <= 1.5
            assert 0.0 <= rmp <= 1.0

    def test_t2_h3_diagnostics_json_finite_vif_and_connected_w(
        self, h3_diagnostics: dict[str, Any]
    ) -> None:
        vif = h3_diagnostics.get("vif", {})
        assert isinstance(vif, dict) and vif
        for pred in ("elderly_share", "log_pop_density"):
            assert pred in vif, f"Missing VIF for {pred}"
            assert 1.0 <= as_float(vif[pred]) < 100.0
        weights = h3_diagnostics.get("weights", {})
        assert isinstance(weights, dict) and weights
        for w_spec, info in weights.items():
            if isinstance(info, dict) and "n_islands" in info:
                assert int(info["n_islands"]) == 0, f"W spec {w_spec} has disconnected islands"


# ---------------------------------------------------------------------------
# FA-9 (A13): Leak-Free ML Baselines & Spatial-Block CV Boundaries (6 tests)
# ---------------------------------------------------------------------------


class TestFA09MlBaselinesBoundaries:
    """FA-9: Leak-Free ML Baselines & Spatial-Block CV Boundaries (6 tests)."""

    def test_t2_ml_json_schema_all_3_models(self, ml_baselines: dict[str, Any]) -> None:
        for key in ("seed", "n_cells", "n_pos", "prevalence", "feature_columns", "models"):
            assert key in ml_baselines, f"phase_a_ml_baselines.json missing {key}"
        assert int(ml_baselines["seed"]) == GLOBAL_SEED
        assert set(ML_MODEL_NAMES).issubset(set(ml_baselines["models"].keys()))

    def test_t2_ml_prevalence_exact_arithmetic(self, ml_baselines: dict[str, Any]) -> None:
        n_cells = int(ml_baselines["n_cells"])
        n_pos = int(ml_baselines["n_pos"])
        prev = as_float(ml_baselines["prevalence"])
        assert 0 < n_pos < n_cells
        assert abs(prev - (n_pos / n_cells)) < 1e-6

    def test_t2_ml_fold_metrics_length_5_and_unit_interval(
        self, ml_baselines: dict[str, Any]
    ) -> None:
        for m_name in ML_MODEL_NAMES:
            m = ml_baselines["models"][m_name]
            folds = m["fold_metrics"]
            assert len(folds) == 5
            for fm in folds:
                for met in ML_METRICS:
                    val = as_float(fm[met])
                    assert 0.0 <= val <= 1.0
            for k in ML_AGGREGATE_KEYS:
                val = as_float(m[k])
                assert 0.0 <= val <= 1.0

    def test_t2_ml_mean_std_match_fold_metrics(self, ml_baselines: dict[str, Any]) -> None:
        for m_name in ML_MODEL_NAMES:
            m = ml_baselines["models"][m_name]
            folds = m["fold_metrics"]
            for met in ML_METRICS:
                vals = [as_float(fm[met]) for fm in folds]
                mean_obs = as_float(m[f"mean_{met}"])
                std_obs = as_float(m[f"std_{met}"])
                assert abs(mean_obs - statistics.mean(vals)) < 1e-5
                std_pop = statistics.pstdev(vals)
                std_samp = statistics.stdev(vals)
                assert (
                    std_obs >= 0.0 and min(abs(std_obs - std_pop), abs(std_obs - std_samp)) < 1e-5
                )

    def test_t2_ml_naive_prior_fold_roc_auc_half(self, ml_baselines: dict[str, Any]) -> None:
        naive = ml_baselines["models"]["naive_prior"]
        for fm in naive["fold_metrics"]:
            assert abs(as_float(fm["roc_auc"]) - 0.5) < 1e-6
        assert abs(as_float(naive["mean_roc_auc"]) - 0.5) < 1e-6

    def test_t2_ml_oof_predictions_probabilities_unit_interval(
        self,
        ml_oof_table: tuple[list[str], list[dict[str, str]]],
        ml_baselines: dict[str, Any],
    ) -> None:
        headers, rows = ml_oof_table
        y_col, model_cols = _find_oof_cols(headers)
        assert len(rows) == int(ml_baselines["n_cells"])
        cids = [r["cell_id"] for r in rows]
        assert len(cids) == len(set(cids))
        for r in rows:
            assert int(float(r[y_col])) in {0, 1}
            for m_name in ML_MODEL_NAMES:
                p = as_float(r[model_cols[m_name]])
                assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# FA-10 (A14, A15, A16): Frozen Bundle, Figures, Table 1 & Report PDF (10 tests)
# ---------------------------------------------------------------------------


class TestFA10FrozenBundleVisualsAndReport:
    """FA-10: Frozen Bundle, Figures, Table 1 & Report PDF Boundaries (10 tests)."""

    def test_t2_freeze_manifest_completeness_and_sha256_format(
        self, freeze_manifest: dict[str, Any]
    ) -> None:
        fmap = _extract_freeze_file_map(freeze_manifest)
        for rel in FROZEN_CORE_HASHED_FILES:
            assert rel in fmap, f"FREEZE_MANIFEST.json missing {rel}"
        for rel, sha in fmap.items():
            assert SHA256_RE.match(sha), f"Invalid 64-hex SHA-256 for {rel}: {sha!r}"

    def test_t2_figures_exist_valid_png_300dpi_nonblank(
        self, load_png: Callable[[str | Path], PngInfo]
    ) -> None:
        for rel in REQUIRED_FIGURES:
            png = load_png(rel)
            assert png.crc_valid and png.size_bytes >= PNG_MIN_BYTES
            assert (
                png.dpi_x is not None
                and png.dpi_y is not None
                and png.dpi_x >= PNG_MIN_DPI
                and png.dpi_y >= PNG_MIN_DPI
            ), f"{rel} DPI below 300: ({png.dpi_x}, {png.dpi_y})"
            assert png.pixel_std > PNG_MIN_PIXEL_STD

    def test_t2_table1_three_formats_zero_nan_or_placeholders(self, project_root: Path) -> None:
        for rel in TABLE1_FILES:
            p = project_root / rel
            assert p.is_file() and p.stat().st_size > 100
            txt = p.read_text(encoding="utf-8")
            hit = TABLE_PLACEHOLDER_RE.search(txt)
            assert hit is None, f"Table 1 file {rel} contains placeholder/NaN: {hit.group(0)!r}"
            low = txt.lower()
            assert "gini" in low and "ols" in low and "roc" in low

    def test_t2_pdf_a4_dimensions_every_page(self, report_pdf_info: PdfInfo) -> None:
        assert report_pdf_info.total_pages >= 4
        assert len(report_pdf_info.page_sizes_pt) == report_pdf_info.total_pages
        for idx, (w, h) in enumerate(report_pdf_info.page_sizes_pt, start=1):
            assert abs(w - A4_WIDTH_PT) <= A4_TOL_PT and abs(h - A4_HEIGHT_PT) <= A4_TOL_PT, (
                f"Page {idx} is not A4 (got {w}x{h} pt; US Letter 612x792 trap?)"
            )

    def test_t2_pdf_main_text_page_budget_3_0_to_3_8(self, report_pdf_info: PdfInfo) -> None:
        mp = report_pdf_info.main_fractional_pages
        assert PDF_MAIN_PAGES_MIN <= mp <= PDF_MAIN_PAGES_MAX, (
            f"PDF main text is {mp:.2f} pages, outside Phase A budget "
            f"[{PDF_MAIN_PAGES_MIN}, {PDF_MAIN_PAGES_MAX}]"
        )

    def test_t2_latex_format_a4_11pt_singlecol_1_5_spacing(self, report_tex_text: str) -> None:
        assert "a4paper" in report_tex_text
        assert "11pt" in report_tex_text
        assert "twocolumn" not in report_tex_text
        assert re.search(
            r"\\onehalfspacing|\\setstretch\{1\.[2-5]|\\linespread\{1\.[2-5]",
            report_tex_text,
        )
        assert "fig1_accessibility_lisa" in report_tex_text

    def test_t2_report_zero_nan_todo_tbdnum_or_broken_refs(
        self,
        report_pdf_info: PdfInfo,
        report_tex_text: str,
    ) -> None:
        scope = (
            report_pdf_info.main_text
            + "\n"
            + report_pdf_info.references_text
            + "\n"
            + report_tex_text
        )
        bad = ANALYTICAL_PLACEHOLDER_RE.search(scope)
        assert bad is None, f"Analytical placeholder / broken ref found in report: {bad.group(0)!r}"

    def test_t2_report_5_minimum_reqs_4_limitations_correlational_framing(
        self, report_pdf_info: PdfInfo
    ) -> None:
        low = report_pdf_info.main_text.lower()
        for tok in (
            "h1",
            "h2",
            "h3",
            "2sfca",
            "gini",
            "theil",
            "lisa",
            "ols",
            "sem",
            "slm",
            "maup",
        ):
            assert tok in low, f"Report main text missing required concept {tok!r}"
        assert any(tok in low for tok in ("high-rise", "building-height", "dasymetric"))
        assert any(tok in low for tok in ("capacity metadata", "9488798", "imputed"))
        assert any(
            tok in low for tok in ("slope", "topograph", "footbridge", "cross-sea", "harbour")
        )
        assert "ecological fallacy" in low
        assert any(tok in low for tok in ("correlational", "associational", "associated with"))
        assert not sentence_aware_unhedged_causal_hits(report_pdf_info.main_text)
        for hype in FORBIDDEN_HYPE_PHRASES:
            assert hype not in report_pdf_info.full_text.lower()

    def test_t2_report_excluded_sections_and_kdense_citation(
        self,
        report_pdf_info: PdfInfo,
        report_bib_text: str,
    ) -> None:
        assert report_pdf_info.cover_pages
        assert report_pdf_info.references_text
        assert report_pdf_info.contribution_text
        assert report_pdf_info.ai_declaration_text
        combined_bib = (report_pdf_info.references_text + "\n" + report_bib_text).lower()
        assert "2609.00065" in combined_bib or "kassis" in combined_bib

    def test_t2_go_nogo_audit_md_complete_and_pass(self, go_nogo_audit_text: str) -> None:
        low = go_nogo_audit_text.lower()
        assert re.search(r"\b(go|pass)\b", low)
        assert not re.search(r"\b(fail|(?<!go/)no-go|tbd_num|todo|fixme)\b", low)
        for mr in ("mr1", "mr2", "mr3", "mr4", "mr5"):
            assert mr in low or "minimum requirement" in low
