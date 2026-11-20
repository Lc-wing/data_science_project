"""Milestone M4: canonical Phase A to Phase B cell bundle and SHA-256 manifest.

M1 attribute columns are copied from the frozen base grids. M2 accessibility and
LISA columns are appended from the interim layers. Nothing in this module
recomputes a Gini, a Theil index, or a model coefficient.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from macao_equity.config import (
    ANALYSIS_CRS,
    DATA_FROZEN_DIR,
    RANDOM_SEED,
    sha256_file,
    utc8_now_iso,
)

CANONICAL_R9_NAME = "cells.gpkg"
CANONICAL_R8_NAME = "cells_res8.gpkg"
MANIFEST_NAME = "FREEZE_MANIFEST.json"

# Attribute columns the M4 reader must keep identical to the M1 base grids.
BASE_R9_COLUMNS: tuple[str, ...] = (
    "cell_id",
    "centroid_x",
    "centroid_y",
    "area_km2",
    "zone_id",
    "zone_name",
    "parish_id",
    "parish_name",
    "pop_total",
    "pop_65plus",
    "pop_density",
    "elderly_share",
    "intersection_density",
    "road_density_km_per_km2",
    "res_bldg_share",
    "com_bldg_share",
    "bldg_coverage_ratio",
    "dist_nearest_lrt_m",
    "dist_nearest_bus_m",
    "has_ev_charger",
    "ev_station_count",
    "ev_charger_capacity",
    "spatial_block",
)
BASE_R8_COLUMNS: tuple[str, ...] = (
    "cell_id",
    "area_km2",
    "zone_id",
    "parish_id",
    "parish_name",
    "pop_total",
    "pop_65plus",
    "pop_density",
    "elderly_share",
    "has_ev_charger",
)
INTEGER_COLUMNS: frozenset[str] = frozenset(
    {
        "has_ev_charger",
        "ev_station_count",
        "ev_charger_capacity",
        "spatial_block",
    }
)


def append_analysis_columns(
    base: gpd.GeoDataFrame,
    analysis: gpd.GeoDataFrame,
    *,
    required_base_columns: tuple[str, ...],
) -> gpd.GeoDataFrame:
    """Return ``base`` geometries and attributes plus analysis-only columns.

    Row order and M1 values follow ``base``. Analysis columns are aligned on
    ``cell_id``. Geometry is not replaced.
    """
    missing = [col for col in required_base_columns if col not in base.columns]
    if missing:
        raise ValueError(f"Base grid is missing M1 columns: {missing}")
    if "cell_id" not in analysis.columns:
        raise ValueError("Analysis grid has no cell_id column")

    base_ids = base["cell_id"].astype(str)
    if base_ids.duplicated().any():
        raise ValueError("Base grid cell_id values are not unique")
    analysis_ids = analysis["cell_id"].astype(str)
    if analysis_ids.duplicated().any():
        raise ValueError("Analysis grid cell_id values are not unique")
    missing_ids = sorted(set(base_ids) - set(analysis_ids))
    if missing_ids:
        raise ValueError(f"Analysis grid is missing {len(missing_ids)} base cell ids")

    extra = [col for col in analysis.columns if col not in base.columns and col != "geometry"]
    keyed = analysis.copy()
    keyed.index = analysis_ids.to_numpy()
    out = base.copy()
    ordered = keyed.loc[base_ids.to_list(), extra]
    for col in extra:
        values = ordered[col]
        if col in INTEGER_COLUMNS:
            out[col] = values.to_numpy()
        elif pd.api.types.is_numeric_dtype(values):
            out[col] = values.to_numpy(dtype="float64")
        else:
            out[col] = values.astype(str).to_numpy()
    for col in INTEGER_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("int64")
    if out.crs is None or out.crs.to_epsg() != 32649:
        out = out.set_crs(ANALYSIS_CRS, allow_override=True)
    return out


def write_cells_gpkg(frame: gpd.GeoDataFrame, path: Path) -> None:
    """Write one EPSG:32649 GeoPackage layer, replacing any previous file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    frame.to_file(path, driver="GPKG", layer="cells")


def frozen_files_to_hash(frozen_dir: Path) -> list[Path]:
    """Every frozen file except the manifest, which would otherwise hash itself."""
    files = [path for path in frozen_dir.iterdir() if path.is_file() and path.name != MANIFEST_NAME]
    return sorted(files, key=lambda path: path.name)


def build_manifest_payload(frozen_dir: Path, *, project_root: Path) -> dict[str, object]:
    """SHA-256 map for the current frozen directory, excluding the manifest."""
    files: dict[str, str] = {}
    for path in frozen_files_to_hash(frozen_dir):
        rel = path.resolve().relative_to(project_root.resolve()).as_posix()
        files[rel] = sha256_file(path)
    return {
        "schema_version": "1.0.0",
        "generated_at_utc8": utc8_now_iso(),
        "seed": RANDOM_SEED,
        "crs": ANALYSIS_CRS,
        "phase": "A_to_B",
        "note": (
            "SHA-256 of every file in data/frozen except FREEZE_MANIFEST.json. "
            "cells.gpkg and cells_res8.gpkg keep M1 columns and append M2 access and LISA columns."
        ),
        "files": files,
    }


def write_freeze_manifest(frozen_dir: Path, *, project_root: Path) -> Path:
    payload = build_manifest_payload(frozen_dir, project_root=project_root)
    path = frozen_dir / MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def render_data_dictionary(r9_columns: list[str], r8_columns: list[str]) -> str:
    """Schema note for the frozen bundle. Column names are copied from the grids."""
    r9_lines = "\n".join(f"- `{col}`" for col in r9_columns if col != "geometry")
    r8_lines = "\n".join(f"- `{col}`" for col in r8_columns if col != "geometry")
    return f"""# MacaoSpatialEquity Phase A to Phase B data dictionary

This note is the schema contract for the frozen bundle that Phase B reads.
Analysis stays in EPSG:32649. Statistical estimates are not redefined here.
They live in `outputs/tables/` and `outputs/metrics/phase_a_ml_baselines.json`.

## Coordinate reference system

- Every analysis GeoPackage uses **EPSG:32649** (WGS 84 / UTM zone 49N).
- DSEC census geometries are read in native EPSG:8433 (`always_xy=True`) and
  transformed locally. The broken server-side `outSR=4326` path is not used.
- Maritime zone Z26 is excluded. Terrestrial 2021 census totals used as the
  population control are 681,293 residents and 82,751 residents aged 65 or older.
- H3 cell population is an area-weighted sum of DSEC building-group counts.
  It is not a uniform building-height dasymetric surface. Grouped high-rise
  footprints can still spread people evenly inside one polygon.

## Frozen files

- `cells.gpkg`: canonical populated H3 resolution-9 cells. M1 columns are kept
  and M2 `acc10_`, `acc15_`, and `lisa_` columns are appended.
- `cells_res8.gpkg`: canonical populated H3 resolution-8 cells for the MAUP
  check, with the same class of appended columns when the interim layer has them.
- `cells_base_res9.gpkg` and `cells_base_res8.gpkg`: M1 base grids. Phase B
  should prefer `cells.gpkg` / `cells_res8.gpkg`.
- `zones_23.gpkg`: 23 terrestrial DSEC zones (Z01–Z23), EPSG:32649.
- `parishes_7.gpkg`: 7 official parishes dissolved from those zones, including
  Taipa (Nossa Senhora do Carmo / 嘉模堂區).
- `facilities_clean.gpkg`: cleaned park, health centre, library, bus stop, and
  EV charger points after cross-source 25 m deduplication. Same-source
  neighbours, including opposite bus poles, are not collapsed.
- `lrt_stations.gpkg`: operational MLM stations inside the SAR.
- `graph_edges.csv`: 10-minute pedestrian edges with columns `src_cell_id`,
  `dst_cell_id`, `walk_time_min`, `weight`, `same_block`.
- `spatial_folds.csv`: frozen geographic blocks with columns `cell_id`,
  `spatial_block` (0–4, seed 42).
- `split_metadata.json`: block sizes and positive-cell counts.
- `cell_poi_texts.csv`: columns `cell_id`, `poi_count_raw`, `poi_count_filtered`,
  `poi_text`. POI text has EV and charging tokens removed for a later text model.
- `feature_manifest.json`: leak-free `ml_features` whitelist.
- `FREEZE_MANIFEST.json`: SHA-256 of every other file in this directory.

## Anti-leakage

Machine-learning and later graph models may use only `ml_features` from
`feature_manifest.json`. The target and audit columns are `has_ev_charger`,
`ev_station_count`, and `ev_charger_capacity`. Columns that start with
`acc10_`, `acc15_`, or `lisa_` are spatial-analysis outputs. They are not
features for predicting whether a cell has a charger. `cell_poi_texts.csv`
is stripped of EV, charger, and charging-station strings before any text model
sees it. Imputation and scaling for the Phase A baselines are fit inside each
training block, not on the full grid.

`spatial_block` is the frozen cross-validation label. It is not a covariate.

## `cells.gpkg` columns

{r9_lines}

## `cells_res8.gpkg` columns

{r8_lines}

## Reading rule for Phase B

Join on `cell_id`. Do not recompute the published Gini, Theil, regression, or
cross-validation numbers from scratch if the goal is to quote Phase A. Those
numbers are the committed tables. This bundle freezes the cell attributes those
tables were built from, plus the accessibility and LISA fields needed for maps
and for a later graph model that must not treat them as charger-prediction features.
"""


def write_data_dictionary(
    path: Path,
    *,
    r9_columns: list[str],
    r8_columns: list[str],
) -> Path:
    path.write_text(
        render_data_dictionary(r9_columns, r8_columns),
        encoding="utf-8",
    )
    return path


def build_canonical_bundles(
    *,
    base_r9: Path,
    analysis_r9: Path,
    base_r8: Path,
    analysis_r8: Path,
    out_r9: Path,
    out_r8: Path,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Merge M1 bases with M2 interim columns and write both canonical grids."""
    r9 = append_analysis_columns(
        gpd.read_file(base_r9),
        gpd.read_file(analysis_r9),
        required_base_columns=BASE_R9_COLUMNS,
    )
    r8 = append_analysis_columns(
        gpd.read_file(base_r8),
        gpd.read_file(analysis_r8),
        required_base_columns=BASE_R8_COLUMNS,
    )
    write_cells_gpkg(r9, out_r9)
    write_cells_gpkg(r8, out_r8)
    return r9, r8


def default_bundle_paths(root: Path | None = None) -> dict[str, Path]:
    frozen = DATA_FROZEN_DIR if root is None else root / "data" / "frozen"
    interim = frozen.parent / "interim"
    return {
        "base_r9": frozen / "cells_base_res9.gpkg",
        "base_r8": frozen / "cells_base_res8.gpkg",
        "analysis_r9": interim / "cells_r9_with_2sfca_lisa.gpkg",
        "analysis_r8": interim / "cells_r8_with_2sfca.gpkg",
        "out_r9": frozen / CANONICAL_R9_NAME,
        "out_r8": frozen / CANONICAL_R8_NAME,
        "dictionary": frozen / "data_dictionary.md",
        "manifest": frozen / MANIFEST_NAME,
        "frozen_dir": frozen,
    }
