"""300 DPI maps and charts drawn from frozen cells and committed M2 tables.

These functions plot stored columns and CSV rows. They do not recompute
accessibility, Gini, or LISA labels.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

FIGURE_DPI = 300
FACILITY_COLORS = {
    "park": "#1b9e77",
    "health_centre": "#d95f02",
    "library": "#7570b3",
    "bus_stop": "#66a61e",
    "ev_charger": "#e7298a",
}
LISA_COLORS = {
    "HH": "#d73027",
    "LL": "#4575b4",
    "LH": "#fee090",
    "HL": "#fc8d59",
    "NS": "#f0f0f0",
}
LISA_DRAW_ORDER = ("NS", "LH", "HL", "HH", "LL")
FACILITY_LABELS = {
    "park": "Park",
    "health_centre": "Health centre",
    "library": "Library",
    "bus_stop": "Bus stop",
    "ev_charger": "EV charger",
    "composite": "Composite",
}


def _prepare_axes(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.set_axis_off()


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)


def write_fig1_accessibility_lisa(
    cells: gpd.GeoDataFrame,
    parishes: gpd.GeoDataFrame,
    path: Path,
) -> None:
    """Dual panel: 15-minute composite choropleth and FDR LISA clusters."""
    cluster_col = "lisa_cluster_acc15_composite"
    value_col = "acc15_composite"
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6))
    vals = cells[value_col].to_numpy(dtype=float)
    vmax = float(vals.max()) if vals.size else 1.0
    p98 = float(pd.Series(vals).quantile(0.98)) if vals.size else 1.0
    use_cap = p98 > 0.0 and p98 < vmax * 0.99
    cells.plot(
        column=value_col,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=p98 if use_cap else vmax,
        linewidth=0.15,
        edgecolor="#f7f7f7",
        legend=True,
        legend_kwds={
            "label": "15-min log-scaled composite 2SFCA",
            "shrink": 0.72,
            "extend": "max" if use_cap else "neither",
        },
        ax=axes[0],
    )
    parishes.boundary.plot(ax=axes[0], color="#222222", linewidth=0.45)
    axes[0].set_title("Composite accessibility (capped at 98th pct)")
    _prepare_axes(axes[0])

    for label in LISA_DRAW_ORDER:
        part = cells[cells[cluster_col].astype(str) == label]
        if part.empty:
            continue
        part.plot(
            ax=axes[1],
            color=LISA_COLORS[label],
            linewidth=0.15,
            edgecolor="#d9d9d9",
        )
    parishes.boundary.plot(ax=axes[1], color="#222222", linewidth=0.45)
    axes[1].set_title("FDR LISA clusters")
    axes[1].legend(
        handles=[
            Patch(facecolor=LISA_COLORS[name], edgecolor="#666666", label=name)
            for name in ("HH", "LL", "LH", "HL", "NS")
        ],
        loc="lower left",
        frameon=True,
        fontsize=8,
    )
    _prepare_axes(axes[1])
    fig.suptitle(
        "Figure 1. 15-minute composite walking access and FDR local clusters",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, path)


def write_eda_facilities_pop(
    cells: gpd.GeoDataFrame,
    facilities: gpd.GeoDataFrame,
    parishes: gpd.GeoDataFrame,
    path: Path,
) -> None:
    """Population density and cleaned facility locations on the same grid."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6))
    cells.plot(
        column="pop_density",
        cmap="YlOrRd",
        norm=LogNorm(vmin=max(float(cells["pop_density"].min()), 1e-3)),
        linewidth=0.12,
        edgecolor="#ffffff",
        legend=True,
        legend_kwds={"label": "Residents per km²", "shrink": 0.72},
        ax=axes[0],
    )
    parishes.boundary.plot(ax=axes[0], color="#222222", linewidth=0.45)
    axes[0].set_title("Population density")
    _prepare_axes(axes[0])

    cells.plot(ax=axes[1], color="#f4f4f4", edgecolor="#d0d0d0", linewidth=0.12)
    parishes.boundary.plot(ax=axes[1], color="#222222", linewidth=0.45)
    handles: list[Line2D] = []
    for facility, color in FACILITY_COLORS.items():
        part = facilities[facilities["facility_type"].astype(str) == facility]
        if part.empty:
            continue
        part.plot(
            ax=axes[1],
            color=color,
            markersize=8 if facility == "bus_stop" else 18,
            alpha=0.75 if facility == "bus_stop" else 0.9,
            linewidth=0,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markersize=6,
                label=FACILITY_LABELS[facility],
            )
        )
    axes[1].legend(handles=handles, loc="lower left", fontsize=8, frameon=True)
    axes[1].set_title("Cleaned facilities")
    _prepare_axes(axes[1])
    fig.suptitle("EDA. Population density and public-facility locations", fontsize=12)
    fig.tight_layout()
    _save(fig, path)


def write_lorenz_figure(lorenz: pd.DataFrame, path: Path) -> None:
    """Lorenz curves of population-weighted accessibility at 10 and 15 minutes."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharey=True)
    colors = {
        "park": "#1b9e77",
        "health_centre": "#d95f02",
        "library": "#7570b3",
        "bus_stop": "#66a61e",
        "ev_charger": "#e7298a",
        "composite": "#000000",
    }
    for ax, threshold in zip(axes, (10, 15), strict=True):
        ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1.0, label="Equality")
        block = lorenz[lorenz["threshold_min"].astype(int) == threshold]
        for facility, color in colors.items():
            curve = block[block["facility_type"].astype(str) == facility].sort_values(
                "cum_pop_share"
            )
            if curve.empty:
                continue
            ax.plot(
                curve["cum_pop_share"],
                curve["cum_acc_share"],
                color=color,
                linewidth=1.6 if facility in {"ev_charger", "composite"} else 1.1,
                label=FACILITY_LABELS[facility],
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Cumulative population share")
        ax.set_title(f"{threshold}-minute threshold")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Cumulative accessibility share")
    axes[1].legend(loc="upper left", fontsize=8, frameon=True)
    fig.suptitle("Lorenz curves of population-weighted 2SFCA accessibility", fontsize=12)
    fig.tight_layout()
    _save(fig, path)


def write_maup_figure(maup: pd.DataFrame, path: Path) -> None:
    """Side-by-side res-8 and res-9 Gini values from the committed MAUP table."""
    block = maup[maup["threshold_min"].astype(int) == 15].sort_values("facility_type")
    labels = [FACILITY_LABELS.get(str(name), str(name)) for name in block["facility_type"]]
    x = range(len(block))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    width = 0.36
    axes[0].bar(
        [i - width / 2 for i in x],
        block["gini_r8"],
        width=width,
        label="Resolution 8",
        color="#80cdc1",
    )
    axes[0].bar(
        [i + width / 2 for i in x],
        block["gini_r9"],
        width=width,
        label="Resolution 9",
        color="#01665e",
    )
    axes[0].set_xticks(list(x), labels, rotation=25, ha="right")
    axes[0].set_ylabel("Population-weighted Gini")
    axes[0].set_title("15-minute Gini")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_ylim(0, 1)

    axes[1].bar(
        [i - width / 2 for i in x],
        block["theil_r8"],
        width=width,
        label="Resolution 8",
        color="#fdb863",
    )
    axes[1].bar(
        [i + width / 2 for i in x],
        block["theil_r9"],
        width=width,
        label="Resolution 9",
        color="#b35806",
    )
    axes[1].set_xticks(list(x), labels, rotation=25, ha="right")
    axes[1].set_ylabel("Theil T")
    axes[1].set_title("15-minute Theil T")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("MAUP. Resolution 8 versus resolution 9 at 15 minutes", fontsize=12)
    fig.tight_layout()
    _save(fig, path)


def write_all_figures(
    *,
    cells_r9: Path,
    facilities: Path,
    parishes: Path,
    lorenz_csv: Path,
    maup_csv: Path,
    figures_dir: Path,
) -> dict[str, Path]:
    cells = gpd.read_file(cells_r9)
    facilities_gdf = gpd.read_file(facilities)
    parishes_gdf = gpd.read_file(parishes)
    lorenz = pd.read_csv(lorenz_csv)
    maup = pd.read_csv(maup_csv)
    outputs = {
        "fig1": figures_dir / "fig1_accessibility_lisa.png",
        "eda": figures_dir / "eda_facilities_pop.png",
        "lorenz": figures_dir / "eda_accessibility_lorenz.png",
        "maup": figures_dir / "maup_r8_vs_r9.png",
    }
    write_fig1_accessibility_lisa(cells, parishes_gdf, outputs["fig1"])
    write_eda_facilities_pop(cells, facilities_gdf, parishes_gdf, outputs["eda"])
    write_lorenz_figure(lorenz, outputs["lorenz"])
    write_maup_figure(maup, outputs["maup"])
    return outputs
