"""Shared fixtures and lightweight clone helper for Plan 6 E2E Tiers 1-5."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def freeze_manifest(project_root: Path) -> dict[str, Any]:
    """Load `data/frozen/FREEZE_MANIFEST.json` from the active project root."""
    path = project_root / "data" / "frozen" / "FREEZE_MANIFEST.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def split_metadata(project_root: Path) -> dict[str, Any]:
    """Load `data/frozen/split_metadata.json` from the active project root."""
    path = project_root / "data" / "frozen" / "split_metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def feature_manifest(project_root: Path) -> dict[str, Any]:
    """Load `data/frozen/feature_manifest.json` from the active project root."""
    path = project_root / "data" / "frozen" / "feature_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def analysis_spec(project_root: Path) -> dict[str, Any]:
    """Load `data/frozen/analysis_spec.json` from the active project root."""
    path = project_root / "data" / "frozen" / "analysis_spec.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def panel_df(project_root: Path) -> pd.DataFrame:
    """Load frozen `data/frozen/panel_daily.parquet` once per E2E session."""
    return pd.read_parquet(project_root / "data" / "frozen" / "panel_daily.parquet")


@pytest.fixture(scope="session")
def events_phase_b_df(project_root: Path) -> pd.DataFrame:
    """Load frozen `data/frozen/events_text_phase_b.parquet` once per E2E session."""
    return pd.read_parquet(project_root / "data" / "frozen" / "events_text_phase_b.parquet")


@pytest.fixture(scope="session")
def test_pred_df(project_root: Path) -> pd.DataFrame:
    """Load `outputs/tables/h2_test_predictions.parquet` once per E2E session."""
    return pd.read_parquet(project_root / "outputs" / "tables" / "h2_test_predictions.parquet")


def clone_project_for_mutation(src_root: Path, dst_root: Path) -> Path:
    """Create a fast, isolated project clone copying mutable files and linking raw archives.

    Copies `data/frozen`, `data/interim`, `outputs`, `report`, and `docs` cleanly into `dst_root`
    so adversarial mutations never touch the real repository workspace. `data/raw` files are
    hardlinked (fallback copy) unless mutated individually.
    """
    dst_root.mkdir(parents=True, exist_ok=True)
    for rel_dir in (
        "data/raw",
        "data/interim",
        "data/frozen",
        "outputs/tables",
        "outputs/metrics",
        "outputs/figures",
        "report",
        "docs/milestones",
    ):
        src_dir = src_root / rel_dir
        if not src_dir.is_dir():
            continue
        dst_dir = dst_root / rel_dir
        if rel_dir == "data/raw":
            def _fast_link_or_copy(src: str, dst: str) -> str:
                try:
                    os.link(src, dst)
                    return dst
                except OSError:
                    return shutil.copy2(src, dst)

            shutil.copytree(src_dir, dst_dir, copy_function=_fast_link_or_copy)
        else:
            shutil.copytree(src_dir, dst_dir)

    for rel_file in ("README.md", "pyproject.toml", ".gitignore"):
        src_f = src_root / rel_file
        if src_f.is_file():
            shutil.copy2(src_f, dst_root / rel_file)

    return dst_root


@pytest.fixture()
def clone_project() -> Any:
    """Return the isolated clone helper for Tier 4/Tier 5 mutation and replay tests."""
    return clone_project_for_mutation
