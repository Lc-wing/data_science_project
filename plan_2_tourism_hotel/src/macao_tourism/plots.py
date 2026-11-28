"""Figures of the exploratory step."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from macao_tourism.config import PERIODS, TIERS

TIER_LABELS = {"5star": "5-star", "4star": "4-star", "3star": "3-star", "2star": "2-star"}
TIER_COLORS = {"5star": "#1b4f72", "4star": "#2e86c1", "3star": "#d68910", "2star": "#922b21"}
PERIOD_NAMES = {
    "pre_pandemic": "Pre-pandemic",
    "transition_2020_01": "2020-01 (transition)",
    "pandemic": "Pandemic",
    "post_pandemic": "Post-pandemic",
}
PERIOD_LABELS = {
    key: name if key == "transition_2020_01" else f"{name} ({' to '.join(PERIODS[key])})"
    for key, name in PERIOD_NAMES.items()
}
PERIOD_COLORS = {
    "pre_pandemic": "#5d6d7e",
    "transition_2020_01": "#111111",
    "pandemic": "#c0392b",
    "post_pandemic": "#1e8449",
}
SHARE_LABELS = {
    "same_day_share_pct": "Same-day visitors",
    "international_share_pct": "International visitors",
    "package_tour_share_pct": "Package-tour visitors",
}
SHARE_COLOR = "#34495e"
SEGMENT_COLOR = "#c0392b"
FORMATS: tuple[str, ...] = ("png", "pdf")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)


def month_start(months: pd.Series) -> pd.Series:
    return pd.to_datetime(months + "-01")


def _shade_pandemic(ax: Axes) -> None:
    first, last = PERIODS["pandemic"]
    end = pd.Timestamp(last + "-01") + pd.offsets.MonthBegin(1)
    ax.axvspan(pd.Timestamp(first + "-01"), end, color="#bdc3c7", alpha=0.35, linewidth=0)


def _year_axis(ax: Axes, step: int = 2) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.margins(x=0.01)


def save(fig: Figure, directory: Path, stem: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in FORMATS:
        path = directory / f"{stem}.{suffix}"
        metadata = {"CreationDate": None} if suffix == "pdf" else {"Software": None}
        fig.savefig(path, dpi=300, bbox_inches="tight", metadata=metadata)
        written.append(path)
    plt.close(fig)
    return written


def figure_1(months: pd.DataFrame, panel: pd.DataFrame, segments: pd.DataFrame) -> Figure:
    """Visitor-mix timeline with PELT change points, alongside occupancy by star class."""
    series = list(SHARE_LABELS)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.1), sharex=True)
    flat = axes.ravel()
    ordered = months.sort_values("period")
    dates = month_start(ordered["period"])
    for ax, name in zip(flat[:3], series, strict=True):
        _shade_pandemic(ax)
        ax.plot(dates, ordered[name], color=SHARE_COLOR, linewidth=0.95)
        parts = segments[segments["series"] == name]
        for part in parts.itertuples():
            start = pd.Timestamp(part.first_month + "-01")
            end = pd.Timestamp(part.last_month + "-01") + pd.offsets.MonthBegin(1)
            ax.hlines(part.mean_share_pct, start, end, color=SEGMENT_COLOR, linewidth=1.4)
            if part.segment > 1:
                ax.axvline(start, color=SEGMENT_COLOR, linewidth=0.8, linestyle="--")
        ax.set_ylabel("% of visitors")
        ax.set_title(SHARE_LABELS[name], loc="left", fontsize=8.5, pad=3)
        _year_axis(ax, step=4)
    occupancy = flat[3]
    _shade_pandemic(occupancy)
    for tier in TIERS:
        rows = panel[panel["tier"] == tier].sort_values("period")
        occupancy.plot(
            month_start(rows["period"]),
            rows["occupancy_pct"],
            color=TIER_COLORS[tier],
            linewidth=0.95,
            label=TIER_LABELS[tier],
        )
    occupancy.set_ylabel("Occupancy rate (%)")
    occupancy.set_title("Hotel occupancy by star class", loc="left", fontsize=8.5, pad=3)
    occupancy.set_ylim(0, 100)
    occupancy.legend(ncols=2, loc="lower left", fontsize=7, columnspacing=0.9, handlelength=1.4)
    _year_axis(occupancy, step=4)
    fig.subplots_adjust(hspace=0.26, wspace=0.22)
    return fig


BLOCK_STYLES = {"pre_pandemic": "-", "post_pandemic": "--"}


def figure_stl(components: pd.DataFrame, profiles: pd.DataFrame) -> Figure:
    """Occupancy with its trend, the seasonal profile, and the remainder, by star class."""
    fig, axes = plt.subplots(3, len(TIERS), figsize=(7.2, 5.4), sharey="row")
    for column, tier in enumerate(TIERS):
        color = TIER_COLORS[tier]
        level, seasonal, remainder = axes[:, column]
        _shade_pandemic(level)
        _shade_pandemic(remainder)
        remainder.axhline(0, color="#7f8c8d", linewidth=0.5)
        seasonal.axhline(0, color="#7f8c8d", linewidth=0.5)
        for block, style in BLOCK_STYLES.items():
            rows = components[(components["tier"] == tier) & (components["block"] == block)]
            rows = rows.sort_values("period")
            dates = month_start(rows["period"])
            level.plot(dates, rows["occupancy_pct"], color=color, linewidth=0.5, alpha=0.45)
            level.plot(dates, rows["trend"], color=color, linewidth=1.2)
            remainder.plot(dates, rows["remainder"], color=color, linewidth=0.7)
            profile = profiles[(profiles["tier"] == tier) & (profiles["block"] == block)]
            seasonal.plot(
                profile["month"],
                profile["seasonal_pp"],
                color=color,
                linewidth=1.0,
                linestyle=style,
                marker="o",
                markersize=2,
                label=PERIOD_NAMES[block],
            )
        level.set_title(TIER_LABELS[tier], fontsize=8, pad=3)
        seasonal.set_xticks([1, 4, 7, 10])
        seasonal.set_xticklabels(["Jan", "Apr", "Jul", "Oct"])
        for ax in (level, remainder):
            ax.xaxis.set_major_locator(mdates.YearLocator(6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0, 0].set_ylabel("Occupancy and trend (%)")
    axes[1, 0].set_ylabel("Seasonal (pp)")
    axes[2, 0].set_ylabel("Remainder (pp)")
    axes[1, 0].legend(fontsize=6, loc="lower left")
    fig.subplots_adjust(hspace=0.28, wspace=0.08)
    return fig


def figure_scatter(panel: pd.DataFrame) -> Figure:
    """Occupancy against log overnight visitors, one panel per star class, coloured by period."""
    fig, axes = plt.subplots(1, len(TIERS), figsize=(7.2, 2.6), sharex=True, sharey=True)
    for ax, tier in zip(axes, TIERS, strict=False):
        rows = panel[panel["tier"] == tier]
        for label in PERIOD_LABELS:
            inside = rows[rows["period_label"] == label]
            ax.scatter(
                inside["log_visitors_overnight"],
                inside["occupancy_pct"],
                s=20 if label == "transition_2020_01" else 7,
                marker="x" if label == "transition_2020_01" else "o",
                color=PERIOD_COLORS[label],
                alpha=0.75,
                linewidths=0.6 if label == "transition_2020_01" else 0,
                label=PERIOD_LABELS[label],
            )
        ax.set_title(TIER_LABELS[tier], fontsize=8, pad=3)
        ax.set_xlabel("log overnight visitors")
    axes[0].set_ylabel("Occupancy rate (%)")
    axes[0].set_ylim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=2, loc="lower center", bbox_to_anchor=(0.5, -0.2), fontsize=7)
    fig.subplots_adjust(wspace=0.08)
    return fig


QUANTITY_STYLES = {
    "pre_pandemic": ("#5d6d7e", "o", -0.12),
    "post_pandemic": ("#1e8449", "s", 0.12),
}


def figure_slopes(slopes: pd.DataFrame, focal: str, share: str) -> Figure:
    """Regression slopes before and after the pandemic with 95% confidence intervals."""
    fig, (by_class, common) = plt.subplots(
        1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [4, 1.3], "wspace": 0.35}
    )
    for quantity, (color, marker, offset) in QUANTITY_STYLES.items():
        rows = slopes[(slopes["regressor"] == focal) & (slopes["quantity"] == quantity)]
        rows = rows.set_index("tier").loc[list(TIERS)]
        positions = [index + offset for index in range(len(TIERS))]
        by_class.errorbar(
            positions,
            rows["estimate"],
            yerr=[rows["estimate"] - rows["ci_low"], rows["ci_high"] - rows["estimate"]],
            fmt=marker,
            color=color,
            markersize=4,
            capsize=2,
            linewidth=1,
            label=PERIOD_NAMES[quantity],
        )
        row = slopes[(slopes["regressor"] == share) & (slopes["quantity"] == quantity)].iloc[0]
        common.errorbar(
            [offset],
            [row["estimate"]],
            yerr=[[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]],
            fmt=marker,
            color=color,
            markersize=4,
            capsize=2,
            linewidth=1,
        )
    for ax in (by_class, common):
        ax.axhline(0, color="#7f8c8d", linewidth=0.6)
        ax.grid(axis="x", visible=False)
    by_class.set_xticks(range(len(TIERS)))
    by_class.set_xticklabels([TIER_LABELS[tier] for tier in TIERS])
    by_class.set_ylabel("Occupancy (pp) per unit of\nlog overnight visitors")
    by_class.set_title("Log overnight visitors, by star class", loc="left", fontsize=8, pad=3)
    by_class.legend(fontsize=7, loc="best")
    common.set_xticks([0])
    common.set_xticklabels(["All classes"])
    common.set_xlim(-0.5, 0.5)
    common.set_ylabel("Occupancy (pp) per pp of\nsame-day share")
    common.set_title("Same-day share", loc="left", fontsize=8, pad=3)
    return fig


MODEL_STYLES = {
    "seasonal_naive": ("Seasonal naive", "#7f8c8d", ":"),
    "sarimax": ("SARIMAX", "#d68910", "--"),
    "lightgbm": ("LightGBM", "#1e8449", "-"),
}


def figure_test_window(forecasts: pd.DataFrame) -> Figure:
    """Occupancy and the three one-month-ahead forecasts over the test window, by star class."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True)
    for ax, tier in zip(axes.ravel(), TIERS, strict=True):
        rows = forecasts[forecasts["tier"] == tier].sort_values("target")
        dates = month_start(rows["target"])
        ax.plot(dates, rows["actual"], color="#111111", linewidth=1.3, label="Observed")
        for model, (label, color, style) in MODEL_STYLES.items():
            ax.plot(dates, rows[model], color=color, linewidth=1.0, linestyle=style, label=label)
        ax.set_title(TIER_LABELS[tier], loc="left", fontsize=8, pad=3)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for ax in axes[:, 0]:
        ax.set_ylabel("Occupancy rate (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, ncols=4, loc="lower center", bbox_to_anchor=(0.5, -0.03), fontsize=7
    )
    fig.subplots_adjust(hspace=0.22, wspace=0.16)
    return fig


def figure_shap_importance(importance: pd.DataFrame, top: int = 10) -> Figure:
    """Mean absolute SHAP value of the most used features, by star class."""
    rows = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    height = 0.8 / len(TIERS)
    for position, tier in enumerate(TIERS):
        offsets = [index + (len(TIERS) / 2 - position - 0.5) * height for index in range(len(rows))]
        ax.barh(
            offsets, rows[tier], height=height, color=TIER_COLORS[tier], label=TIER_LABELS[tier]
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows["feature"])
    ax.set_xlabel("Mean absolute SHAP value (percentage points of the forecast)")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, loc="lower right")
    return fig
