"""Deterministic M2 panel builder, SHA-256 freeze manifest writer, and tamper verifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from stock_volatility import config
from stock_volatility.ingest import ProvenanceLog
from stock_volatility.ingest.universe import load_universe_csv
from stock_volatility.panel.events import (
    align_events_to_trading_days,
    build_daily_event_features_and_phase_b_text,
    dedup_and_cap_events,
)
from stock_volatility.panel.features import build_panel_features_and_targets
from stock_volatility.panel.spec import (
    build_analysis_spec,
    build_feature_manifest,
    render_data_dictionary_markdown,
)
from stock_volatility.panel.splits import (
    assign_purged_splits,
    validate_purged_embargo_splits,
)

FREEZE_ARTIFACT_NAMES: tuple[str, ...] = (
    "panel_daily.parquet",
    "events_text_phase_b.parquet",
    "analysis_spec.json",
    "split_metadata.json",
    "feature_manifest.json",
    "data_dictionary.md",
)


def _write_json_deterministic(path: Path, payload: dict[str, Any]) -> None:
    """Write formatted, UTF-8 JSON deterministically with trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_freeze_manifest(
    *,
    frozen_dir: Path | None = None,
    project_root: Path | None = None,
    generated_at_utc: str,
    provenance_timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Compute SHA-256 digests and byte sizes for all 6 frozen artifacts and write manifest."""
    root = (Path(project_root) if project_root is not None else config.PROJECT_ROOT).resolve()
    f_dir = (Path(frozen_dir) if frozen_dir is not None else (root / "data" / "frozen")).resolve()
    f_dir.mkdir(parents=True, exist_ok=True)

    files_section: dict[str, dict[str, Any]] = {}
    for name in FREEZE_ARTIFACT_NAMES:
        artifact = f_dir / name
        if not artifact.is_file():
            raise FileNotFoundError(f"Cannot freeze missing artifact: {artifact}")
        rel_path = artifact.relative_to(root).as_posix()
        files_section[rel_path] = {
            "filename": name,
            "byte_size": int(artifact.stat().st_size),
            "sha256": config.sha256_file(artifact),
        }

    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "project": "plan_6_stock_volatility",
        "milestone": "M2",
        "generated_at_utc": generated_at_utc,
        "source_provenance_utc": provenance_timestamp_utc or generated_at_utc,
        "random_seed": config.RANDOM_SEED,
        "artifact_count": len(files_section),
        "files": files_section,
    }
    manifest_path = f_dir / "FREEZE_MANIFEST.json"
    _write_json_deterministic(manifest_path, manifest)
    return manifest


def verify_freeze_manifest(
    frozen_dir: Path | None = None,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Verify all frozen Phase A -> Phase B artifacts exist and match their SHA-256 digests.

    Raises `ValueError` on missing manifest, missing artifact, byte-size mismatch, or SHA-256
    mismatch. Called at startup by all downstream scripts (`03`..`06` and `run_phase_a.py`).
    """
    root = (Path(project_root) if project_root is not None else config.PROJECT_ROOT).resolve()
    f_dir = (Path(frozen_dir) if frozen_dir is not None else (root / "data" / "frozen")).resolve()
    manifest_path = f_dir / "FREEZE_MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing FREEZE_MANIFEST.json at {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files_map = manifest.get("files") or {}
    expected_filenames = set(FREEZE_ARTIFACT_NAMES)
    recorded_filenames = {meta.get("filename") for meta in files_map.values()}
    missing_names = expected_filenames - recorded_filenames
    if missing_names:
        raise ValueError(
            f"FREEZE_MANIFEST.json missing required artifacts: {sorted(missing_names)}"
        )

    for rel_path, meta in files_map.items():
        artifact = root / rel_path
        if not artifact.is_file():
            raise ValueError(f"Frozen artifact missing on disk: {rel_path}")
        actual_sha = config.sha256_file(artifact)
        if actual_sha != str(meta["sha256"]):
            raise ValueError(
                f"Frozen artifact SHA-256 mismatch for {rel_path}: "
                f"expected {meta['sha256']}, got {actual_sha}"
            )
        actual_size = int(artifact.stat().st_size)
        if actual_size != int(meta["byte_size"]):
            raise ValueError(
                f"Frozen artifact byte size mismatch for {rel_path}: "
                f"expected {meta['byte_size']}, got {actual_size}"
            )

    return {
        "status": "verified",
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "generated_at_utc": manifest.get("generated_at_utc"),
        "artifact_count": len(files_map),
        "files": files_map,
    }


def _compute_target_prevalence_by_split(panel_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Compute target summary statistics for Train, Val, and Test modeling-complete slices."""
    out: dict[str, dict[str, Any]] = {}
    for name in ("train", "val", "test"):
        sub = panel_df[
            (panel_df["split"] == name)
            & panel_df["has_full_features"]
            & panel_df["has_full_target"]
        ]
        n_rows = int(len(sub))
        y_tail_pos = int(sub["y_tail"].astype(int).sum())
        y_rv90_pos = int(sub["y_tail_rv90"].astype(int).sum())
        out[name] = {
            "n_rows": n_rows,
            "n_dates": int(sub["date"].nunique()),
            "rv_5d_fwd_median": float(sub["rv_5d_fwd"].median()),
            "rv_5d_fwd_mean": float(sub["rv_5d_fwd"].mean()),
            "rv_5d_fwd_max": float(sub["rv_5d_fwd"].max()),
            "y_tail_pos": y_tail_pos,
            "y_tail_rate": float(y_tail_pos / n_rows) if n_rows else 0.0,
            "y_tail_rv90_pos": y_rv90_pos,
            "y_tail_rv90_rate": float(y_rv90_pos / n_rows) if n_rows else 0.0,
        }
    return out


def build_and_freeze_panel(
    *,
    project_root: Path | None = None,
    refreeze: bool = False,
) -> dict[str, Any]:
    """Build M2 aligned events, features/targets, splits, specs, and frozen SHA-256 manifest.

    Strictly offline: verifies M1 `data/raw/data_provenance.json` SHA-256 hashes first and uses
    the provenance timestamp so repeated `--refreeze` executions are bit-for-bit identical.
    """
    root = Path(project_root) if project_root is not None else config.PROJECT_ROOT
    raw_dir = root / "data" / "raw"
    interim_dir = root / "data" / "interim"
    frozen_dir = root / "data" / "frozen"
    manifest_path = frozen_dir / "FREEZE_MANIFEST.json"

    if manifest_path.is_file() and not refreeze:
        raise FileExistsError(
            f"{manifest_path} already exists. Pass --check to verify or --refreeze to rebuild."
        )

    # 1. Verify M1 raw data integrity offline before building M2
    prov_path = raw_dir / "data_provenance.json"
    prov = ProvenanceLog.load(prov_path, project_root=root)
    prov.verify_files()
    generated_at_utc = str(prov.generated_at_utc)

    # 2. Load committed M1 inputs
    universe_df = load_universe_csv(raw_dir / "universe_60.csv")
    ohlcv_df = pd.read_parquet(raw_dir / "prices" / "ohlcv_daily.parquet")
    macro_df = pd.read_parquet(interim_dir / "macro_daily.parquet")
    events_raw_df = pd.read_parquet(interim_dir / "events_raw.parquet")

    trading_dates = sorted(macro_df["date"].astype(str).unique().tolist())
    split_df = assign_purged_splits(trading_dates)

    # 3. Strict 16:00 ET (13:00 ET early-close) alignment, dedup & <= 5/day capping
    events_aligned_df, event_audit = align_events_to_trading_days(events_raw_df, trading_dates)
    events_capped_df, dedup_audit = dedup_and_cap_events(
        events_aligned_df, max_events_per_day=config.MAX_EVENTS_PER_DAY
    )

    interim_dir.mkdir(parents=True, exist_ok=True)
    events_aligned_path = interim_dir / "events_aligned.parquet"
    events_capped_df.to_parquet(events_aligned_path, index=False, compression="zstd")

    # 4. Aggregate capped events onto full (ticker, date, split) grid for Phase A & Phase B
    panel_keys_df = (
        ohlcv_df[["ticker", "date"]]
        .drop_duplicates()
        .merge(split_df, on="date", how="left", validate="many_to_one")
        .sort_values(["ticker", "date"], kind="mergesort")
        .reset_index(drop=True)
    )
    daily_events_df, phase_b_text_df = build_daily_event_features_and_phase_b_text(
        events_capped_df, panel_keys_df
    )

    # 5. Build zero-leakage panel features, targets, and Train-only preprocessing parameters
    panel_df, train_params = build_panel_features_and_targets(
        ohlcv_df=ohlcv_df,
        universe_df=universe_df,
        macro_df=macro_df,
        daily_events_df=daily_events_df,
        split_df=split_df,
        horizon_days=config.HORIZON_DAYS,
    )

    # 6. Formally validate purged+embargoed splits
    split_metadata = validate_purged_embargo_splits(
        panel_df, trading_dates, horizon_days=config.HORIZON_DAYS
    )
    split_metadata["generated_at_utc"] = generated_at_utc
    split_metadata["event_alignment_audit"] = event_audit
    split_metadata["event_dedup_and_cap_audit"] = dedup_audit
    target_prevalence = _compute_target_prevalence_by_split(panel_df)
    split_metadata["target_prevalence"] = target_prevalence

    # 7. Build pre-registered specs & data dictionary
    analysis_spec = build_analysis_spec(
        generated_at_utc=generated_at_utc,
        split_metadata=split_metadata,
        train_params=train_params,
    )
    feature_manifest = build_feature_manifest(
        generated_at_utc=generated_at_utc,
        train_params=train_params,
        event_audit=event_audit,
        dedup_audit=dedup_audit,
    )
    data_dict_md = render_data_dictionary_markdown(
        generated_at_utc=generated_at_utc,
        split_metadata=split_metadata,
        train_params=train_params,
        event_audit=event_audit,
        dedup_audit=dedup_audit,
        target_prevalence=target_prevalence,
    )

    # 8. Write all 6 frozen artifacts deterministically
    frozen_dir.mkdir(parents=True, exist_ok=True)
    panel_df.to_parquet(frozen_dir / "panel_daily.parquet", index=False, compression="zstd")
    phase_b_text_df.to_parquet(
        frozen_dir / "events_text_phase_b.parquet", index=False, compression="zstd"
    )
    _write_json_deterministic(frozen_dir / "analysis_spec.json", analysis_spec)
    _write_json_deterministic(frozen_dir / "split_metadata.json", split_metadata)
    _write_json_deterministic(frozen_dir / "feature_manifest.json", feature_manifest)
    (frozen_dir / "data_dictionary.md").write_text(data_dict_md, encoding="utf-8")

    # 9. Write & self-verify FREEZE_MANIFEST.json
    manifest = write_freeze_manifest(
        frozen_dir=frozen_dir,
        project_root=root,
        generated_at_utc=generated_at_utc,
        provenance_timestamp_utc=generated_at_utc,
    )
    verify_freeze_manifest(frozen_dir=frozen_dir, project_root=root)

    return {
        "mode": "refreeze" if refreeze else "freeze",
        "generated_at_utc": generated_at_utc,
        "panel_rows": int(len(panel_df)),
        "events_aligned_rows": int(len(events_capped_df)),
        "active_event_ticker_days": int(dedup_audit["active_ticker_days"]),
        "early_close_shifted_events": int(event_audit["shifted_due_to_early_close"]),
        "train_modeling_rows": int(split_metadata["splits"]["train"]["modeling_complete_rows"]),
        "val_modeling_rows": int(split_metadata["splits"]["val"]["modeling_complete_rows"]),
        "test_scorable_rows": int(split_metadata["splits"]["test"]["modeling_complete_rows"]),
        "rv90_train_threshold": round(float(train_params["rv90_threshold"]), 6),
        "frozen_artifact_count": int(manifest["artifact_count"]),
    }
