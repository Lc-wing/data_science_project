"""Milestone M2: Event alignment, zero-leakage features/targets, purged splits, & SHA-256 freeze."""

from __future__ import annotations

from stock_volatility.panel.events import (
    EARLY_CLOSE_HOUR,
    EVENTS_ALIGNED_COLUMNS,
    EVENTS_PHASE_B_COLUMNS,
    NYSE_EARLY_CLOSE_DATES,
    align_events_to_trading_days,
    build_daily_event_features_and_phase_b_text,
    dedup_and_cap_events,
)
from stock_volatility.panel.features import (
    CONTINUOUS_FEATURE_COLUMNS,
    EVENT_FEATURE_COLUMNS,
    PANEL_DAILY_COLUMNS,
    SECTOR_DUMMY_COLUMNS,
    TARGET_COLUMNS,
    apply_train_winsor_and_scale,
    build_panel_features_and_targets,
    fit_train_preprocessing_params,
)
from stock_volatility.panel.freeze import (
    FREEZE_ARTIFACT_NAMES,
    build_and_freeze_panel,
    verify_freeze_manifest,
    write_freeze_manifest,
)
from stock_volatility.panel.spec import (
    build_analysis_spec,
    build_feature_manifest,
    render_data_dictionary_markdown,
)
from stock_volatility.panel.splits import (
    SPLIT_ORDER,
    assign_purged_splits,
    assign_split_label,
    validate_purged_embargo_splits,
)

__all__ = [
    "CONTINUOUS_FEATURE_COLUMNS",
    "EARLY_CLOSE_HOUR",
    "EVENTS_ALIGNED_COLUMNS",
    "EVENTS_PHASE_B_COLUMNS",
    "EVENT_FEATURE_COLUMNS",
    "FREEZE_ARTIFACT_NAMES",
    "NYSE_EARLY_CLOSE_DATES",
    "PANEL_DAILY_COLUMNS",
    "SECTOR_DUMMY_COLUMNS",
    "SPLIT_ORDER",
    "TARGET_COLUMNS",
    "align_events_to_trading_days",
    "apply_train_winsor_and_scale",
    "assign_purged_splits",
    "assign_split_label",
    "build_analysis_spec",
    "build_and_freeze_panel",
    "build_daily_event_features_and_phase_b_text",
    "build_feature_manifest",
    "build_panel_features_and_targets",
    "dedup_and_cap_events",
    "fit_train_preprocessing_params",
    "render_data_dictionary_markdown",
    "validate_purged_embargo_splits",
    "verify_freeze_manifest",
    "write_freeze_manifest",
]
