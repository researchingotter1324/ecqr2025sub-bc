import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator, NullFormatter
from datetime import datetime
import pandas as pd
from typing import Optional, Dict, Literal
import time
import os
import logging
import numpy as np
import math
import re
from hpobench.utils import AnalysisPathManager
from hpobench.config.schema import BenchmarkDataSchema
from hpobench.config.tuner_configurations import DEFAULT_NUMBER_OF_PRECONFORMAL_TRIALS
import seaborn as sns
from matplotlib.colors import ListedColormap

matplotlib.use("Agg")  # Use non-GUI backend
logger = logging.getLogger(__name__)

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

PLOT_DPI = 300
PLOT_FORMATS = ["eps", "png", "pdf"]
DEFAULT_COLOR_PALETTE = [
    "#464646",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#E74C3C",
    "#3498DB",
    "#2ECC71",
    "#F39C12",
    "#9B59B6",
    "#1ABC9C",
    "#E67E22",
    "#34495E",
    "#16A085",
    "#27AE60",
    "#2980B9",
    "#8E44AD",
]


def get_label(label: Optional[str], default: Optional[str]) -> Optional[str]:
    """Get formatted label text with fallback to default value.

    Args:
        label: Primary label text to use.
        default: Fallback label text (will be formatted).

    Returns:
        Formatted label string or None if both inputs are None.
    """
    if label is not None:
        return label
    elif default is not None:
        return default.replace("_", " ").title()
    else:
        return None


def search_metric_label(metric_col: str) -> str:
    """Return a human-readable y-axis label for search performance metrics."""
    if metric_col == "normalized_regret":
        label = "Normalized Regret"
    elif metric_col == "rank":
        label = "Rank"
    else:
        label = metric_col
    return label


NORM_REGRET_YTOP = 1.0  # 10^0 — fixed top tick on the normalized-regret axis
NORM_REGRET_LOG_SUBS = (2, 3, 4, 6, 8)  # five sub-decade tick marks per full decade
NORM_REGRET_TICK_LOG_MIN_SEP = 0.12  # min log10 gap between labeled major ticks
PANEL_LABEL_GUTTER_WIDTH = 0.45  # GridSpec width ratio for (a)/(b) label gutters


def _power_of_ten_tick_label(value: float, _pos: int | None = None) -> str:
    """Format tick values as mantissa × 10^integer_exponent (e.g. 0.2 → 2×10^{-1})."""
    if value <= 0 or not np.isfinite(value):
        return ""
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / (10.0 ** exponent)
    if abs(mantissa - 1.0) < 1e-9:
        return f"$10^{{{exponent}}}$"
    if abs(mantissa - 10.0) < 1e-9:
        return f"$10^{{{exponent + 1}}}$"
    mantissa_str = f"{mantissa:.1f}".rstrip("0").rstrip(".")
    return f"${mantissa_str} \\times 10^{{{exponent}}}$"


def _normalized_regret_data_ymin(ax: "plt.Axes") -> float:
    ymin = np.inf
    for line in ax.get_lines():
        y = np.asarray(line.get_ydata(), dtype=float)
        y = y[np.isfinite(y) & (y > 0)]
        if y.size:
            ymin = min(ymin, float(np.min(y)))
    if np.isfinite(ymin):
        return ymin
    y0, _ = ax.get_ylim()
    return max(float(y0), 1e-6)


def _normalized_regret_major_ticks(y_bottom: float, y_top: float) -> list[float]:
    ticks = {y_bottom, y_top}
    exp_top = int(np.floor(np.log10(y_top)))
    exp_bottom = int(np.ceil(np.log10(y_bottom)))
    for exp in range(exp_top, exp_bottom - 1, -1):
        tick = 10.0 ** exp
        if y_bottom <= tick <= y_top:
            ticks.add(tick)
    return sorted(ticks)


def _dedupe_normalized_regret_major_ticks(
    ticks: list[float], y_bottom: float
) -> list[float]:
    """Drop higher major ticks that sit too close to a lower neighbor on a log axis."""
    ticks = sorted(set(ticks))
    if len(ticks) < 2:
        return ticks
    kept: list[float] = []
    for tick in ticks:
        if not kept:
            kept.append(tick)
            continue
        prev = kept[-1]
        if np.log10(tick / prev) < NORM_REGRET_TICK_LOG_MIN_SEP:
            if abs(prev - y_bottom) < 1e-12 or abs(tick - y_bottom) < 1e-12:
                kept[-1] = min(prev, tick)
            elif tick < prev:
                kept[-1] = tick
        else:
            kept.append(tick)
    return kept


def _normalized_regret_minor_ticks(y_bottom: float, y_top: float) -> list[float]:
    """Sub-decade ticks at fixed log positions within each decade, clipped to the visible range."""
    minors: list[float] = []
    exp_top = int(np.floor(np.log10(y_top)))
    exp_bottom = int(np.floor(np.log10(y_bottom)))
    for exp in range(exp_top, exp_bottom - 1, -1):
        decade_high = 10.0 ** exp
        decade_low = 10.0 ** (exp - 1)
        if y_bottom >= decade_high or y_top <= decade_low:
            continue
        for sub in NORM_REGRET_LOG_SUBS:
            tick = decade_low * sub
            if y_bottom < tick < y_top:
                minors.append(tick)
    return sorted(set(minors))


def apply_metric_yscale(
    ax: "plt.Axes",
    metric_col: str,
    y_bottom: Optional[float] = None,
    y_top: Optional[float] = None,
) -> None:
    if metric_col != "normalized_regret":
        return

    y_top = NORM_REGRET_YTOP if y_top is None else float(y_top)
    if y_bottom is None:
        y_bottom = _normalized_regret_data_ymin(ax)
    else:
        y_bottom = float(y_bottom)
    y_bottom = min(y_bottom, y_top)
    if y_bottom <= 0 or not np.isfinite(y_bottom):
        y_bottom = y_top / 10.0
    if y_bottom >= y_top:
        y_bottom = y_top / 10.0

    ax.set_yscale("log")
    ax.set_ylim(y_bottom, y_top)

    major_ticks = _dedupe_normalized_regret_major_ticks(
        _normalized_regret_major_ticks(y_bottom, y_top), y_bottom
    )
    ax.set_yticks(major_ticks)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_power_of_ten_tick_label))

    minor_ticks = _normalized_regret_minor_ticks(y_bottom, y_top)
    if minor_ticks:
        ax.yaxis.set_minor_locator(FixedLocator(minor_ticks))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _prepare_plot_data_with_bounds(
    df: pd.DataFrame, metric_cols: tuple[str, ...] | list[str]
) -> pd.DataFrame:
    """Fill missing confidence-bound values from their metric columns for plotting."""
    plot_data = df.copy()
    for metric_col in metric_cols:
        lower_col = f"{metric_col}_lower"
        upper_col = f"{metric_col}_upper"
        if lower_col in plot_data.columns and metric_col in plot_data.columns:
            plot_data[lower_col] = plot_data[lower_col].fillna(plot_data[metric_col])
        if upper_col in plot_data.columns and metric_col in plot_data.columns:
            plot_data[upper_col] = plot_data[upper_col].fillna(plot_data[metric_col])
    return plot_data


def _apply_shared_search_row_yscale(
    search_axes: list,
    row_data: pd.DataFrame,
    search_metric_col: str,
    y_margin_fraction: float = 0.05,
) -> None:
    """Align y-limits across search panels using metric values and CI bounds."""
    if not search_axes:
        return

    lower_col = f"{search_metric_col}_lower"
    upper_col = f"{search_metric_col}_upper"
    use_ci = search_metric_col != "normalized_regret"
    y_col_lower = lower_col if use_ci and lower_col in row_data.columns else None
    y_col_upper = upper_col if use_ci and upper_col in row_data.columns else None

    if search_metric_col == "normalized_regret":
        shared_ymin = float(row_data[search_metric_col].min())
        apply_metric_yscale(search_axes[0], search_metric_col, y_bottom=shared_ymin)
        return

    y_min, y_max = get_y_bounds(
        row_data, search_metric_col, y_col_lower, y_col_upper
    )
    margin = y_margin_fraction * (y_max - y_min) if y_max > y_min else 0.5
    for ax_search in search_axes:
        ax_search.set_ylim(y_min - margin, y_max + margin)


def _add_panel_label_gutter(fig: "plt.Figure", gs: GridSpec, row: int, col: int, label: str) -> None:
    ax_gutter = fig.add_subplot(gs[row, col])
    ax_gutter.set_axis_off()
    ax_gutter.text(
        0.5,
        0.5,
        label,
        transform=ax_gutter.transAxes,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )


def _draw_search_progression_ax(
    ax: "plt.Axes",
    row_data: pd.DataFrame,
    entity_col: str,
    x_col: str,
    metric_col: str,
    row_title: Optional[str],
    x_label: Optional[str],
    x_axis_start: Optional[float],
    legend_handles: list,
    legend_labels: list,
    add_to_legend: bool,
    show_xlabel: bool,
    add_confidence_intervals: bool = True,
) -> None:
    entity_color_map = build_entity_color_map(row_data[entity_col].unique())
    for entity, entity_data in row_data.groupby(entity_col):
        display_label = display_entity_label(entity)
        color = entity_color_map[entity]
        linestyle = entity_plot_linestyle(entity)
        line = ax.plot(
            entity_data[x_col],
            entity_data[metric_col],
            label=display_label,
            alpha=0.8,
            color=color,
            marker=None,
            markersize=4,
            linestyle=linestyle,
        )[0]

        if add_to_legend and display_label not in legend_labels:
            legend_handles.append(line)
            legend_labels.append(display_label)

        lower_col = f"{metric_col}_lower"
        upper_col = f"{metric_col}_upper"
        if (
            add_confidence_intervals
            and metric_col != "normalized_regret"
            and lower_col in entity_data.columns
            and upper_col in entity_data.columns
        ):
            ax.fill_between(
                entity_data[x_col],
                entity_data[lower_col],
                entity_data[upper_col],
                alpha=0.2,
                color=color,
            )

    if show_xlabel:
        ax.set_xlabel(get_label(x_label, x_col), fontsize=14)
    ax.set_ylabel(search_metric_label(metric_col), fontsize=14, labelpad=10)
    if row_title is not None:
        ax.set_title(row_title, fontsize=14, pad=20)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    apply_metric_yscale(ax, metric_col)

    if x_axis_start is not None:
        current_xlim = ax.get_xlim()
        ax.set_xlim(left=x_axis_start, right=current_xlim[1])

    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_linewidth(1.2)
    ax.tick_params(axis="both", which="major", labelsize=12, length=6, width=1.2)
    ax.tick_params(axis="both", which="minor", labelsize=10, length=3, width=1.0)


def sort_legend_items(handles: list, labels: list) -> tuple[list, list]:
    """Sort legend items: numerically if starts with number, otherwise alphabetically."""
    combined = sorted(zip(handles, labels), key=lambda x: legend_sort_key(x[1]))
    sorted_handles, sorted_labels = zip(*combined)
    return list(sorted_handles), list(sorted_labels)


def compute_legend_ncols(n_items: int, max_cols: int = 6) -> int:
    """Choose the number of legend columns for a balanced, readable layout.

    Rules:
    - Up to ``max_cols`` items may appear in a single row.
    - Prefer fitting everything in at most 2 rows.
    - Never leave exactly 1 item on the last row (looks unbalanced).
    - Among valid column counts, pick the one that minimises unevenness
      (i.e. the number of empty slots on the last row).  When there is a
      tie, prefer the larger column count (fewer rows).
    - If N > 2 * max_cols the layout will require 3+ rows; in that case
      simply use max_cols to keep row count as low as possible.

    Args:
        n_items: Total number of legend entries.
        max_cols: Maximum permitted columns per row (default 6).

    Returns:
        Number of columns to pass to ``fig.legend(ncol=...)``.
    """
    if n_items <= max_cols:
        return n_items

    min_ncols_for_2rows = math.ceil(n_items / 2)
    if min_ncols_for_2rows > max_cols:
        return max_cols

    best_ncols = max_cols
    best_unevenness = float("inf")

    for ncols in range(min_ncols_for_2rows, max_cols + 1):
        last_row = n_items % ncols
        if last_row == 0:
            last_row = ncols
        if last_row == 1:
            continue
        unevenness = ncols - last_row
        if unevenness < best_unevenness or (
            unevenness == best_unevenness and ncols > best_ncols
        ):
            best_unevenness = unevenness
            best_ncols = ncols

    return best_ncols


def calculate_legend_position(
    num_subplot_rows: int, num_legend_rows: int, plot_type: str = "standard"
) -> tuple[float, float]:
    """Calculate legend position and bottom margin based on subplot and legend configuration.

    Args:
        num_subplot_rows: Number of subplot rows
        num_legend_rows: Number of legend rows
        plot_type: Type of plot ("standard", "matrix", "cd")

    Returns:
        Tuple of (legend_anchor_y, legend_bottom_margin)
    """
    base_legend_anchor_y = -0.16
    base_bottom_margin = 0.20

    if plot_type == "matrix":
        base_legend_anchor_y = -0.08
        subplot_row_factor = 0.002
        legend_row_factor = 0.035
    elif plot_type == "cd":
        subplot_row_factor = 0.035
        legend_row_factor = 0.06
    else:
        subplot_row_factor = 0.035
        legend_row_factor = 0.06

    subplot_row_adjustment = (num_subplot_rows - 1) * subplot_row_factor
    legend_row_adjustment = (num_legend_rows - 1) * legend_row_factor

    legend_anchor_y = (
        base_legend_anchor_y + subplot_row_adjustment - legend_row_adjustment
    )
    legend_bottom_margin = base_bottom_margin + legend_row_adjustment

    return legend_anchor_y, legend_bottom_margin


def get_axis_values(data: pd.DataFrame, measure: Optional[str]) -> list:
    """Extract unique values from a DataFrame column for axis configuration.

    Args:
        data: DataFrame containing the data.
        measure: Column name to extract unique values from.

    Returns:
        List of unique values from the column, or [None] if measure is None.
    """
    if measure is None:
        return [None]
    else:
        return list(data[measure].unique())


def get_y_bounds(
    subset: pd.DataFrame,
    y_col: str,
    y_col_lower: Optional[str],
    y_col_upper: Optional[str],
) -> tuple[float, float]:
    y_min = (
        subset[y_col_lower].min()
        if y_col_lower and y_col_lower in subset.columns
        else subset[y_col].min()
    )
    y_max = (
        subset[y_col_upper].max()
        if y_col_upper and y_col_upper in subset.columns
        else subset[y_col].max()
    )
    return y_min, y_max


def legend_sort_key(label: str) -> tuple:
    """Sort key for legend items: numeric labels sort first, then alphabetical."""
    label = str(label)
    if label and label[0].isdigit():
        match = re.match(r"^(\d+(?:\.\d+)?)(.*)", label)
        if match:
            return (0, float(match.group(1)), match.group(2).lower())
    return (1, label.lower())


def identifier_sort_key(identifier: str) -> float:
    """Sort key extracting leading numeric value from an identifier string."""
    match = re.match(r"^(\d+(?:\.\d+)?)", str(identifier))
    return float(match.group(1)) if match else float("inf")


def trim_y_axis(ax: "plt.Axes", col_values: pd.Series, margin: float = 0.10) -> None:
    """Set linear y-axis limits focused on the bulk of the data.

    Uses the IQR (25th–75th percentile) as the anchor for the interesting
    range, then extends outward to the full data min/max, and adds a
    fractional whitespace margin on both sides.

    Args:
        ax: Axes to configure.
        col_values: All plotted values for this panel (across all architectures).
        margin: Fractional whitespace to add above and below the clipped range.
    """
    finite_vals = col_values.dropna()
    if finite_vals.empty:
        return

    q25 = float(np.percentile(finite_vals, 25))
    q75 = float(np.percentile(finite_vals, 75))
    iqr = q75 - q25

    fence_lo = q25 - 1.5 * iqr
    fence_hi = q75 + 1.5 * iqr

    data_min = float(finite_vals.min())
    data_max = float(finite_vals.max())

    y_lo = max(fence_lo, data_min)
    y_hi = min(fence_hi, data_max)

    if y_hi <= y_lo:
        y_lo, y_hi = data_min, data_max

    span = y_hi - y_lo if y_hi > y_lo else abs(y_hi) * 0.1 or 0.01
    ax.set_ylim(y_lo - margin * span, y_hi + margin * span)


def is_non_local(legend_label: str) -> bool:
    """Determine if a variant is non-local based on its label."""
    label = str(legend_label)
    if "NL-" in label:
        return True
    return False


def is_ucb_entity(label: str) -> bool:
    label = str(label)
    return label == "UCB" or label.endswith("-UCB")


def display_entity_label(label: str) -> str:
    """Normalize entity labels for display (e.g. drop NL- prefix for UCB)."""
    label = str(label)
    if is_ucb_entity(label) and label.startswith("NL-"):
        return label[3:]
    return label


def entity_plot_linestyle(label: str) -> str:
    if is_ucb_entity(label):
        return "-"
    return "--" if is_non_local(label) else "-"


def build_entity_color_map(entities) -> dict[str, str]:
    """Map raw entity identifiers to colors via their display labels."""
    display_labels = sorted(
        {display_entity_label(entity) for entity in entities},
        key=legend_sort_key,
    )
    display_to_color = {
        display_label: DEFAULT_COLOR_PALETTE[idx % len(DEFAULT_COLOR_PALETTE)]
        for idx, display_label in enumerate(display_labels)
    }
    return {
        entity: display_to_color[display_entity_label(entity)] for entity in entities
    }


def canonical_architecture_label(arch: str) -> str:
    """Strip the non-local prefix so architectures share one color across samplers."""
    arch = str(arch)
    return arch[3:] if arch.startswith("NL-") else arch


def joint_plot_sampler_label(sampler: str) -> str:
    """Return a short sampler title for joint architecture plots."""
    sampler = str(sampler)
    if sampler.startswith("UCB"):
        return "UCB"
    if sampler.startswith("LBS"):
        return "LBS"
    return sampler


def plot_tuner(
    ax: "plt.Axes",
    tuner_data: pd.DataFrame,
    x_col: str,
    y_col: str,
    color: str,
    add_ci: bool,
    y_col_lower: Optional[str],
    y_col_upper: Optional[str],
    legend_label: str,
    marker: str = "o",
    add_markers: bool = True,
) -> None:
    marker_style = marker if add_markers else "None"
    display_label = display_entity_label(legend_label)
    linestyle = entity_plot_linestyle(legend_label)

    ax.plot(
        tuner_data[x_col],
        tuner_data[y_col],
        label=display_label,
        alpha=0.8,
        color=color,
        marker=marker_style,
        markersize=4,
        linestyle=linestyle,
    )
    if (
        add_ci
        and y_col != "normalized_regret"
        and y_col_lower
        and y_col_upper
    ):
        ax.fill_between(
            tuner_data[x_col],
            tuner_data[y_col_lower],
            tuner_data[y_col_upper],
            alpha=0.2,
            color=color,
        )


def plot_benchmark_data(
    data: pd.DataFrame,
    plot_path: str,
    x_col: str = "runtime",
    y_col: str = "best_performance",
    entity_col: str = "tuner",
    y_col_lower: Optional[str] = None,
    y_col_upper: Optional[str] = None,
    row_measure: Optional[str] = "dataset",
    col_measure: Optional[str] = "model",
    add_confidence_intervals: bool = True,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    col_measure_label: Optional[str] = None,
    row_measure_label: Optional[str] = None,
    share_y_axis: bool = False,
    entity_legend_mapping: Optional[dict] = None,
    add_markers: bool = False,
    hide_col_and_row_labels: bool = True,
    x_axis_start: Optional[float] = None,
) -> None:
    """
    Plots benchmark data in a grid of subplots, with rows and columns determined by specified measures.

    Args:
        data: The benchmark data to plot.
        plot_path: The base path to save the plot.
        x_col: The column to use for the x-axis.
        y_col: The column to use for the y-axis.
        y_col_lower: The column to use for the lower confidence bound. If None, will use "{y_col}_q10" if available.
        y_col_upper: The column to use for the upper confidence bound. If None, will use "{y_col}_q90" if available.
        row_measure: The column to determine subplot rows.
        col_measure: The column to determine subplot columns.
        add_confidence_intervals: Whether to add confidence intervals.
        color_palette: Custom color palette for plotting.
        x_label: Custom label for the x-axis.
        y_label: Custom label for the y-axis.
        col_measure_label: Custom label for the column measure (subplot title).
        row_measure_label: Custom label for the row measure (subplot title).
        add_markers: Whether to add circular markers to the plotted lines.
        hide_col_and_row_labels: Whether to hide the column and row measure labels, using only the axis labels.
        x_axis_start: Starting value for the x-axis. If None, the axis starts at the minimum data value.

    Raises:
        ValueError: If there are duplicate X-axis values for the same combination of row_measure, col_measure, and tuner.
    """
    if row_measure is None and col_measure is None:
        raise ValueError("At least one of row_measure or col_measure must be provided.")

    plt.clf()
    formatted_row_measure = get_label(row_measure_label, row_measure)
    formatted_col_measure = get_label(col_measure_label, col_measure)
    row_values = get_axis_values(data, row_measure)
    col_values = get_axis_values(data, col_measure)

    base_width = 4.0
    base_height = 3.0
    fig_width = base_width * len(col_values)
    fig_height = base_height * len(row_values)
    fig, axes = plt.subplots(
        nrows=len(row_values),
        ncols=len(col_values),
        figsize=(fig_width, fig_height),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    single_row = False
    if len(row_values) == 1 and len(col_values) == 1:
        axes = [[axes]]
        single_row = True
    elif len(row_values) == 1:
        axes = [axes]
        single_row = True
    elif len(col_values) == 1:
        axes = [[ax] for ax in axes]

    if share_y_axis and y_col != "normalized_regret":
        global_y_min, global_y_max = get_y_bounds(
            data, y_col, y_col_lower, y_col_upper
        )
        y_range = global_y_max - global_y_min
        buffer = 0.05 * y_range if y_range > 0 else 0.05
        global_y_min -= buffer
        global_y_max += buffer

    for i, row_value in enumerate(row_values):
        for j, col_value in enumerate(col_values):
            ax = axes[i][j]
            subset = data
            if row_measure is not None:
                subset = subset[subset[row_measure] == row_value]
            if col_measure is not None:
                subset = subset[subset[col_measure] == col_value]
            entity_color_map = build_entity_color_map(subset[entity_col].unique())
            for entity, entity_data in subset.groupby(entity_col):
                if entity_data[x_col].duplicated().any():
                    raise ValueError(
                        f"Duplicate X-axis values found for {x_col} in entity '{entity}' "
                        f"with {row_measure}={row_value} and {col_measure}={col_value}. "
                        "Each X-axis unit must have only one value per line."
                    )
                legend_label = (
                    entity_legend_mapping[entity]
                    if entity_legend_mapping and entity in entity_legend_mapping
                    else entity
                )
                plot_tuner(
                    ax=ax,
                    tuner_data=entity_data,
                    x_col=x_col,
                    y_col=y_col,
                    color=entity_color_map[entity],
                    add_ci=add_confidence_intervals,
                    y_col_lower=y_col_lower,
                    y_col_upper=y_col_upper,
                    legend_label=legend_label,
                    marker="o",
                    add_markers=add_markers,
                )

            if subset.empty:
                ax.set_xticks([])
                ax.set_yticks([])
            elif share_y_axis and y_col != "normalized_regret":
                ax.set_ylim((global_y_min, global_y_max))
            elif y_col != "normalized_regret":
                y_min, y_max = get_y_bounds(subset, y_col, y_col_lower, y_col_upper)
                y_range = y_max - y_min
                buffer = 0.05 * y_range if y_range > 0 else 0.05
                ax.set_ylim((y_min - buffer, y_max + buffer))

            apply_metric_yscale(ax, y_col)

            if x_axis_start is not None:
                current_xlim = ax.get_xlim()
                ax.set_xlim(left=x_axis_start, right=current_xlim[1])

            x_label_to_use = x_label if x_label is not None else get_label(None, x_col)
            y_label_to_use = y_label if y_label is not None else get_label(None, y_col)
            if y_label_to_use is None and y_col is not None:
                y_label_to_use = y_col.replace("_", " ").title()

            if col_measure is not None and i == 0:
                if hide_col_and_row_labels:
                    col_title = f"{col_value}"
                else:
                    col_title = f"{formatted_col_measure}: {col_value}"
                ax.set_title(col_title, fontsize=14)

            if y_label_to_use is not None and j == 0:
                if row_measure is not None:
                    if single_row:
                        row_title = f"{y_label_to_use}"
                    else:
                        if hide_col_and_row_labels:
                            row_title = f"{row_value} \n\n{y_label_to_use}"
                        else:
                            row_title = f"{formatted_row_measure}: {row_value} \n\n{y_label_to_use}"
                else:
                    row_title = f"{y_label_to_use}"
                ax.set_ylabel(row_title, fontsize=14, labelpad=10)

            if i == len(row_values) - 1:
                ax.set_xlabel(x_label_to_use, fontsize=14)

            if not subset.empty:
                ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
                ax.spines["top"].set_linewidth(1.2)
                ax.spines["right"].set_linewidth(1.2)
                ax.spines["bottom"].set_linewidth(1.2)
                ax.spines["left"].set_linewidth(1.2)
                ax.tick_params(
                    axis="both", which="major", labelsize=12, length=6, width=1.2
                )
                ax.tick_params(axis="both", which="minor", labelsize=10, length=3, width=1.0)
            else:
                for spine in ax.spines.values():
                    spine.set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    handles, labels = sort_legend_items(handles, labels)

    num_subplot_rows = len(row_values) if row_measure else 1
    legend_ncols = compute_legend_ncols(len(labels))
    num_legend_rows = math.ceil(len(labels) / legend_ncols) if labels else 1
    legend_anchor_y, legend_bottom_margin = calculate_legend_position(
        num_subplot_rows, num_legend_rows, "standard"
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=legend_ncols,
        fontsize=13,
        bbox_to_anchor=(0.5, legend_anchor_y),
        frameon=False,
    )
    fig.subplots_adjust(
        wspace=0.15,
        hspace=0.22,
        bottom=legend_bottom_margin,
        top=0.93,
        left=0.09,
        right=0.98,
    )

    for file_format in PLOT_FORMATS:
        fig.savefig(
            f"{plot_path}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.{file_format}",
            dpi=PLOT_DPI,
            format=file_format,
            bbox_inches="tight",
            transparent=False,
        )

    plt.close(fig)


def plot_and_save(
    data: pd.DataFrame,
    x_col: str,
    y_cols: list,
    entity_col: str,
    cache_path: str,
    run_start_str: str,
    filename_prefix: str,
    analysis_type: str,
    subfolder: str,
    col_measure: Optional[str],
    row_measure: Optional[str],
    y_cols_lower: Optional[list] = None,
    y_cols_upper: Optional[list] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    col_measure_label: Optional[str] = None,
    row_measure_label: Optional[str] = None,
    share_y_axis: bool = False,
    entity_legend_mapping: Optional[dict] = None,
    add_markers: bool = False,
    hide_col_and_row_labels: bool = True,
    x_axis_start: Optional[float] = None,
):
    """Generate and save plots for multiple y-columns with proper path organization.

    Creates faceted plots for each y-column specified, organizing them by analysis
    type and saving to appropriate cache directories. Supports confidence intervals
    and custom labeling.

    Args:
        data: DataFrame containing the plotting data.
        x_col: Column name for x-axis values.
        y_cols: List of column names for y-axis values (one plot per column).
        entity_col: Column name for grouping/coloring entities.
        cache_path: Base cache directory path.
        run_start_str: Timestamp identifier for the current run.
        filename_prefix: Prefix for generated plot filenames.
        analysis_type: Analysis category for path organization.
        subfolder: Optional subfolder within analysis directory.
        col_measure: Column for subplot columns.
        row_measure: Column for subplot rows.
        y_cols_lower: Optional list of lower confidence bound columns.
        y_cols_upper: Optional list of upper confidence bound columns.
        x_label: Custom x-axis label.
        y_label: Custom y-axis label.
        col_measure_label: Custom column facet label.
        row_measure_label: Custom row facet label.
        share_y_axis: Whether to share y-axis across subplots.
        entity_legend_mapping: Dictionary mapping entity values to display names.
        add_markers: Whether to add markers to lines.
        hide_col_and_row_labels: Whether to hide facet labels.
        x_axis_start: Optional starting value for x-axis.
    """

    path_manager = AnalysisPathManager(cache_path, run_start_str)
    output_path = path_manager.get_analysis_path(analysis_type, "plots", subfolder)
    plot_path = os.path.join(output_path, filename_prefix)

    plot_data = _prepare_plot_data_with_bounds(data, y_cols)

    if y_cols_lower is None:
        y_cols_lower = [
            f"{y_col}_q10" if f"{y_col}_q10" in plot_data.columns else None
            for y_col in y_cols
        ]
    if y_cols_upper is None:
        y_cols_upper = [
            f"{y_col}_q90" if f"{y_col}_q90" in plot_data.columns else None
            for y_col in y_cols
        ]

    for idx, y_col in enumerate(y_cols):
        y_col_lower = (
            y_cols_lower[idx] if y_cols_lower and len(y_cols_lower) > idx else None
        )
        y_col_upper = (
            y_cols_upper[idx] if y_cols_upper and len(y_cols_upper) > idx else None
        )

        plot_benchmark_data(
            data=plot_data,
            plot_path=f"{plot_path}__{y_col}",
            x_col=x_col,
            y_col=y_col,
            entity_col=entity_col,
            y_col_lower=y_col_lower,
            y_col_upper=y_col_upper,
            add_confidence_intervals=True,
            col_measure=col_measure,
            row_measure=row_measure,
            x_label=x_label,
            y_label=y_label,
            col_measure_label=col_measure_label,
            row_measure_label=row_measure_label,
            share_y_axis=share_y_axis,
            entity_legend_mapping=entity_legend_mapping,
            add_markers=add_markers,
            hide_col_and_row_labels=hide_col_and_row_labels,
            x_axis_start=x_axis_start,
        )
        time.sleep(1)
    logger.debug(f"Plots saved in {output_path} with prefix {filename_prefix}")


def plot_critical_difference_diagram(
    ax,
    mean_ranks: Dict[str, float],
    significance_results: pd.DataFrame,
    alpha: float = 0.05,
    title: Optional[str] = None,
    title_fontweight: str = "normal",
    p_value_column: str = "p_value_corrected",
) -> None:
    """Plot a critical difference diagram using scikit-posthocs."""
    try:
        import scikit_posthocs as sp
    except ImportError:
        logger.error("scikit-posthocs is required for critical difference diagrams")
        return

    # Convert to format expected by scikit-posthocs
    raw_to_display = {
        entity: display_entity_label(entity) for entity in mean_ranks.keys()
    }
    display_mean_ranks: dict[str, float] = {}
    for entity, rank in mean_ranks.items():
        display_mean_ranks[raw_to_display[entity]] = rank
    ranks_series = pd.Series(display_mean_ranks)
    algorithms = list(display_mean_ranks.keys())

    # Create significance matrix
    sig_matrix = pd.DataFrame(1.0, index=algorithms, columns=algorithms)
    np.fill_diagonal(sig_matrix.values, 1.0)

    for _, row in significance_results.iterrows():
        alg1 = raw_to_display.get(row["entity1"], row["entity1"])
        alg2 = raw_to_display.get(row["entity2"], row["entity2"])
        if alg1 in algorithms and alg2 in algorithms:
            p_val = row[p_value_column]
            sig_matrix.loc[alg1, alg2] = p_val
            sig_matrix.loc[alg2, alg1] = p_val

    ax.clear()

    sp.critical_difference_diagram(
        ranks=ranks_series,
        sig_matrix=sig_matrix,
        ax=ax,
        label_fmt_left="{label} [{rank:.2f}]  ",
        label_fmt_right="  [{rank:.2f}] {label}",
    )

    apply_cd_formatting(ax)
    ax.set_aspect("auto")

    if title:
        ax.set_title(title, fontsize=13, fontweight=title_fontweight, pad=20)


def apply_cd_formatting(ax):
    """Remove colored elements, circles, and vertical grid lines from critical difference diagram."""
    while ax.collections:
        ax.collections[0].remove()

    for line in ax.get_lines():
        line.set_color("black")
        line.set_linewidth(1)

    for text in ax.findobj(match=matplotlib.text.Text):
        text.set_color("black")
        text.set_fontsize(10)

    while ax.patches:
        ax.patches[0].remove()

    ax.grid(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])


def plot_significance_matrix(
    ax, significance_data: pd.DataFrame, rank_data: pd.DataFrame, entity_col: str
):
    entities = sorted(
        rank_data[entity_col].unique(),
        key=lambda entity: legend_sort_key(display_entity_label(entity)),
    )
    display_entities = [display_entity_label(entity) for entity in entities]
    avg_ranks = dict(zip(rank_data[entity_col], rank_data["rank"]))

    p_matrix = pd.DataFrame(np.nan, index=entities, columns=entities)
    color_matrix = pd.DataFrame(0, index=entities, columns=entities)

    for entity in entities:
        p_matrix.loc[entity, entity] = 1.0
        color_matrix.loc[entity, entity] = 0

    for _, row in significance_data.iterrows():
        entity1 = row["entity1"]
        entity2 = row["entity2"]
        if entity1 in entities and entity2 in entities:
            p_val = row["p_value_corrected"]
            p_matrix.loc[entity1, entity2] = p_val
            p_matrix.loc[entity2, entity1] = p_val

            if p_val <= 0.05:
                color_matrix.loc[entity1, entity2] = 1
                color_matrix.loc[entity2, entity1] = 1

    annot_matrix = p_matrix.copy()
    for i in range(len(entities)):
        for j in range(len(entities)):
            if i == j:
                annot_matrix.iloc[i, j] = ""
            else:
                p_val = p_matrix.iloc[i, j]
                if not pd.isna(p_val):
                    annot_matrix.iloc[i, j] = f"{p_val:.3f}"

    colors = ["white", "#D3D3D3"]
    cmap = ListedColormap(colors)

    sns.heatmap(
        color_matrix,
        annot=annot_matrix,
        fmt="",
        cmap=cmap,
        vmin=0,
        vmax=1,
        square=True,
        cbar=False,
        annot_kws={"size": 7, "color": "black"},
        linewidths=0.5,
        linecolor="black",
        xticklabels=display_entities,
        yticklabels=display_entities,
        ax=ax,
    )

    for i, entity in enumerate(entities):
        rank = avg_ranks[entity]
        ax.text(
            i + 0.5,
            -0.25,
            f"{rank:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="normal",
            color="black",
            transform=ax.transData,
        )

    ax.text(
        -0.1,
        -0.25,
        "Ranks:",
        ha="right",
        va="center",
        fontsize=8,
        fontweight="normal",
        color="black",
        transform=ax.transData,
    )

    ax.set_title(
        "Wilcoxon@100% (Benjamini-Hochberg)",
        fontsize=13,
        fontweight="normal",
        pad=20,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=8, colors="black")

    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_linewidth(2.4)
        ax.spines[spine].set_color("black")


def plot_paired_rank_and_cd(
    data: pd.DataFrame,
    significance_data: pd.DataFrame,
    x_col: str,
    entity_col: str,
    cache_path: str,
    run_start_str: str,
    filename_prefix: str,
    analysis_type: str,
    subfolder: str,
    row_measure: str,
    cd_budget: int = 100,
    alpha: float = 0.05,
    x_label: Optional[str] = None,
    x_axis_start: Optional[float] = None,
    y_col_lower: Optional[str] = None,
    y_col_upper: Optional[str] = None,
    significance_plot_type: Literal["cd", "matrix"] = "cd",
) -> None:
    """Plot paired visualizations: search progression and significance analysis.

    Creates a plot where:
    - Matrix mode: normalized regret progression, rank progression, then significance matrix
    - CD mode: rank evolution over budget on the left; CD diagrams on the right
    - Shared legend at the bottom center

    Args:
        data: Aggregated rank data with budget information
        significance_data: Pairwise significance test results
        x_col: Column for x-axis (budget)
        entity_col: Column for algorithms/entities
        cache_path: Base cache path
        run_start_str: Run identifier
        filename_prefix: Prefix for saved files
        analysis_type: Analysis type for path organization
        subfolder: Subfolder for saving plots
        row_measure: Column for row grouping (e.g., benchmark)
        cd_budget: Budget value to use for analysis
        alpha: Significance level
        x_label: Custom x-axis label
        x_axis_start: Optional starting value for x-axis
        y_col_lower: Optional column name for lower confidence bound
        y_col_upper: Optional column name for upper confidence bound
        significance_plot_type: Type of significance plot ("cd" or "matrix")
    """
    path_manager = AnalysisPathManager(cache_path, run_start_str)
    output_path = path_manager.get_analysis_path(analysis_type, "plots", subfolder)
    plot_path = os.path.join(output_path, f"{filename_prefix}_paired")

    plot_data = _prepare_plot_data_with_bounds(
        data, ("rank", "normalized_regret")
    )

    row_values = plot_data[row_measure].unique()

    base_width = 4.0
    base_height = 3.0

    if significance_plot_type == "matrix":
        fig_width = base_width * 3
        fig_height = base_height * len(row_values) + 1.0

        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = GridSpec(
            nrows=len(row_values),
            ncols=3,
            figure=fig,
            width_ratios=[1, 1, 1],
            height_ratios=[1] * len(row_values),
            wspace=0.25,
            hspace=0.22,
        )
    else:
        fig_width = base_width * 2
        fig_height = base_height * len(row_values)

        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
        gs = GridSpec(nrows=len(row_values) * 2, ncols=2, figure=fig)

    axes = []
    for i in range(len(row_values)):
        if significance_plot_type == "matrix":
            ax_normalized_regret = fig.add_subplot(gs[i, 0])
            ax_rank = fig.add_subplot(gs[i, 1])
            ax_matrix = fig.add_subplot(gs[i, 2])
            axes.append([ax_normalized_regret, ax_rank, ax_matrix])
        else:
            ax_rank = fig.add_subplot(gs[i * 2 : (i + 1) * 2, 0])
            ax_cd_uncorrected = fig.add_subplot(gs[i * 2, 1])
            ax_cd_corrected = fig.add_subplot(gs[i * 2 + 1, 1])
            axes.append([ax_rank, ax_cd_uncorrected, ax_cd_corrected])

    legend_handles = []
    legend_labels = []

    for i, row_value in enumerate(row_values):
        row_data = plot_data[plot_data[row_measure] == row_value]
        row_sig_data = significance_data[significance_data[row_measure] == row_value]
        show_xlabel = i == len(row_values) - 1

        if significance_plot_type == "matrix":
            ax_normalized_regret = axes[i][0]
            ax_rank = axes[i][1]
            _draw_search_progression_ax(
                ax_normalized_regret,
                row_data,
                entity_col,
                x_col,
                "normalized_regret",
                row_value,
                x_label,
                x_axis_start,
                legend_handles,
                legend_labels,
                i == 0,
                show_xlabel,
                add_confidence_intervals=False,
            )
            _draw_search_progression_ax(
                ax_rank,
                row_data,
                entity_col,
                x_col,
                "rank",
                row_value,
                x_label,
                x_axis_start,
                legend_handles,
                legend_labels,
                False,
                show_xlabel,
                add_confidence_intervals=False,
            )

            ax_matrix = axes[i][2]
            cd_data = row_data[row_data[x_col] == cd_budget]

            if not cd_data.empty and not row_sig_data.empty:
                plot_significance_matrix(
                    ax=ax_matrix,
                    significance_data=row_sig_data,
                    rank_data=cd_data,
                    entity_col=entity_col,
                )
            else:
                ax_matrix.set_xticks([])
                ax_matrix.set_yticks([])
                for spine in ax_matrix.spines.values():
                    spine.set_visible(False)
        else:
            ax_rank = axes[i][0]
            _draw_search_progression_ax(
                ax_rank,
                row_data,
                entity_col,
                x_col,
                "rank",
                row_value,
                x_label,
                x_axis_start,
                legend_handles,
                legend_labels,
                i == 0,
                show_xlabel,
            )

            ax_cd_uncorrected = axes[i][1]
            ax_cd_corrected = axes[i][2]
            cd_data = row_data[row_data[x_col] == cd_budget]

            if not cd_data.empty and not row_sig_data.empty:
                mean_ranks = dict(zip(cd_data[entity_col], cd_data["rank"]))
                plot_critical_difference_diagram(
                    ax=ax_cd_uncorrected,
                    mean_ranks=mean_ranks,
                    significance_results=row_sig_data,
                    alpha=alpha,
                    title=f"CD@{cd_budget}% (Raw)",
                    p_value_column="p_value",
                )
                plot_critical_difference_diagram(
                    ax=ax_cd_corrected,
                    mean_ranks=mean_ranks,
                    significance_results=row_sig_data,
                    alpha=alpha,
                    title=f"CD@{cd_budget}% (Benjamini-Hochberg)",
                    p_value_column="p_value_corrected",
                )
            else:
                for ax_cd in [ax_cd_uncorrected, ax_cd_corrected]:
                    ax_cd.set_xticks([])
                    ax_cd.set_yticks([])
                    for spine in ax_cd.spines.values():
                        spine.set_visible(False)

            for ax_cd in [ax_cd_uncorrected, ax_cd_corrected]:
                for spine in ["top", "right", "bottom", "left"]:
                    if spine in ax_cd.spines:
                        ax_cd.spines[spine].set_linewidth(1.2)
                ax_cd.tick_params(
                    axis="both", which="major", labelsize=11, length=6, width=1.2
                )
                ax_cd.tick_params(
                    axis="both", which="minor", labelsize=9, length=3, width=1.0
                )

    handles, labels = sort_legend_items(legend_handles, legend_labels)

    num_subplot_rows = len(row_values)
    legend_ncols = compute_legend_ncols(len(labels)) if labels else 1
    num_legend_rows = math.ceil(len(labels) / legend_ncols) if labels else 1
    plot_type = "matrix" if significance_plot_type == "matrix" else "cd"
    legend_anchor_y, legend_bottom_margin = calculate_legend_position(
        num_subplot_rows, num_legend_rows, plot_type
    )

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=legend_ncols,
            fontsize=13,
            bbox_to_anchor=(0.5, legend_anchor_y),
            frameon=False,
        )

    if significance_plot_type == "matrix":
        fig.subplots_adjust(
            wspace=0.15,
            hspace=0.22,
            bottom=legend_bottom_margin,
            top=0.93,
            left=0.09,
            right=0.98,
        )
    else:
        fig.subplots_adjust(
            wspace=0.15,
            hspace=0.22,
            bottom=legend_bottom_margin,
            top=0.90,
            left=0.09,
            right=0.98,
        )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for fmt in PLOT_FORMATS:
        full_path = f"{plot_path}_{timestamp}.{fmt}"
        fig.savefig(full_path, dpi=PLOT_DPI, bbox_inches="tight", format=fmt)

    plt.close(fig)
    logger.debug(
        f"Paired plots saved in {output_path} with prefix {filename_prefix}_paired"
    )

def plot_joint_architecture_and_static(
    main_processed_df: pd.DataFrame,
    static_processed_df: pd.DataFrame,
    cache_path: str,
    run_start_str: str,
    filename_prefix: str,
    analysis_type: str,
    subfolder: str,
    schema: BenchmarkDataSchema,
    search_x_col: str,
    search_x_col_label: str,
    search_metric_col: str = "rank",
) -> None:
    """Plot joint analysis comparing architecture optimization ranks and estimator errors.

    Produces one row per benchmark.  Within each row there is one search-rank
    panel per unique sampler (ordered alphabetically) followed by a single
    pinball-loss panel on the right.  Each search-rank panel shows one line per
    estimator architecture for that sampler; the pinball-loss panel is shared
    across all samplers and shows lines per estimator architecture over training
    data size.

    Args:
        search_x_col: Column to use as the x-axis for search-rank panels.
        search_x_col_label: X-axis label for search-rank panels.
    """
    path_manager = AnalysisPathManager(cache_path, run_start_str)
    output_path = path_manager.get_analysis_path(analysis_type, "plots", subfolder)
    plot_path = os.path.join(output_path, f"{filename_prefix}__{search_metric_col}")

    plot_data = _prepare_plot_data_with_bounds(
        main_processed_df, (search_metric_col,)
    )

    row_measure = schema.bench_col
    arch_col = schema.estimator_architecture_col
    sampler_col = schema.sampler_col

    row_values = plot_data[row_measure].unique()
    samplers = sorted(plot_data[sampler_col].unique())
    n_sampler_cols = len(samplers)
    search_col_offset = 1
    gutter_b_col = n_sampler_cols + 1
    static_col = n_sampler_cols + 2
    width_ratios = (
        [PANEL_LABEL_GUTTER_WIDTH]
        + [1.0] * n_sampler_cols
        + [PANEL_LABEL_GUTTER_WIDTH]
        + [1.0]
    )
    n_gs_cols = len(width_ratios)

    all_archs = sorted(
        {
            canonical_architecture_label(arch)
            for arch in plot_data[arch_col].unique()
        }
    )
    color_map = {
        arch: DEFAULT_COLOR_PALETTE[idx % len(DEFAULT_COLOR_PALETTE)]
        for idx, arch in enumerate(all_archs)
    }

    base_width = 4.0
    base_height = 3.0
    fig_width = base_width * (n_sampler_cols + 1 + PANEL_LABEL_GUTTER_WIDTH)
    fig_height = base_height * len(row_values)

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(
        nrows=len(row_values),
        ncols=n_gs_cols,
        figure=fig,
        width_ratios=width_ratios,
        wspace=0.28,
        hspace=0.22,
    )

    legend_handles: list = []
    legend_labels: list = []

    for i, row_value in enumerate(row_values):
        _add_panel_label_gutter(fig, gs, i, 0, "(a)")
        _add_panel_label_gutter(fig, gs, i, gutter_b_col, "(b)")

        main_row_data = plot_data[plot_data[row_measure] == row_value]

        search_axes = []
        for j, sampler in enumerate(samplers):
            if j == 0:
                ax_search = fig.add_subplot(gs[i, search_col_offset + j])
            else:
                ax_search = fig.add_subplot(
                    gs[i, search_col_offset + j], sharey=search_axes[0]
                )
            search_axes.append(ax_search)
            sampler_data = main_row_data[main_row_data[sampler_col] == sampler]

            for arch in sorted(sampler_data[arch_col].unique()):
                arch_data = sampler_data[sampler_data[arch_col] == arch]
                if arch_data.empty:
                    continue
                canon_arch = canonical_architecture_label(arch)
                color = color_map[canon_arch]
                line = ax_search.plot(
                    arch_data[search_x_col],
                    arch_data[search_metric_col],
                    label=canon_arch,
                    alpha=0.8,
                    color=color,
                    marker=None,
                    markersize=4,
                )[0]
                if i == 0 and j == 0 and canon_arch not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(canon_arch)
                lower_col = f"{search_metric_col}_lower"
                upper_col = f"{search_metric_col}_upper"
                if (
                    search_metric_col != "normalized_regret"
                    and lower_col in arch_data.columns
                    and upper_col in arch_data.columns
                ):
                    ax_search.fill_between(
                        arch_data[search_x_col],
                        arch_data[lower_col],
                        arch_data[upper_col],
                        alpha=0.2,
                        color=color,
                    )

            ax_search.set_xlabel(search_x_col_label, fontsize=14)
            ax_search.set_ylabel(
                f"Search {search_metric_label(search_metric_col)}",
                fontsize=14,
                labelpad=10,
            )
            ax_search.set_title(
                joint_plot_sampler_label(sampler),
                fontsize=14,
                pad=20,
            )
            ax_search.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            for spine in ["top", "right", "bottom", "left"]:
                ax_search.spines[spine].set_linewidth(1.2)
            ax_search.tick_params(axis="both", which="major", labelsize=12, length=6, width=1.2)
            ax_search.tick_params(axis="both", which="minor", labelsize=10, length=3, width=1.0)

        _apply_shared_search_row_yscale(search_axes, main_row_data, search_metric_col)

        ax_static = fig.add_subplot(gs[i, static_col])
        static_row_data = static_processed_df[static_processed_df[row_measure] == row_value]

        for arch in sorted(static_row_data[arch_col].unique()):
            entity_data = static_row_data[static_row_data[arch_col] == arch]
            canon_arch = canonical_architecture_label(arch)
            color = color_map[canon_arch]
            ax_static.plot(
                entity_data[schema.data_size_col],
                entity_data["rank"],
                label=canon_arch,
                alpha=0.8,
                color=color,
                marker="o",
                markersize=4,
            )

        ax_static.set_xlabel("Training Data Size", fontsize=14)
        ax_static.set_ylabel("Error Rank", fontsize=14, labelpad=10)
        ax_static.set_title(f"Estimation Error Sensitivity", fontsize=14, pad=20)
        ax_static.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        for spine in ["top", "right", "bottom", "left"]:
            ax_static.spines[spine].set_linewidth(1.2)
        ax_static.tick_params(axis="both", which="major", labelsize=12, length=6, width=1.2)
        ax_static.tick_params(axis="both", which="minor", labelsize=10, length=3, width=1.0)

    handles, labels = sort_legend_items(legend_handles, legend_labels)
    legend_ncols = compute_legend_ncols(len(labels)) if labels else 1
    num_legend_rows = math.ceil(len(labels) / legend_ncols) if labels else 1
    legend_anchor_y, legend_bottom_margin = calculate_legend_position(
        len(row_values), num_legend_rows, "standard"
    )

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=legend_ncols,
            fontsize=13,
            bbox_to_anchor=(0.5, legend_anchor_y),
            frameon=False,
        )

    fig.subplots_adjust(
        wspace=0.28,
        hspace=0.22,
        bottom=legend_bottom_margin,
        top=0.90,
        left=0.08,
        right=0.98,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for fmt in PLOT_FORMATS:
        full_path = f"{plot_path}_{timestamp}.{fmt}"
        fig.savefig(full_path, dpi=PLOT_DPI, bbox_inches="tight", format=fmt)

    plt.close(fig)
    logger.debug(f"Joint plots saved in {output_path} with prefix {filename_prefix}")


def plot_ei_architecture_triplot(
    search_performance_df: pd.DataFrame,
    ei_metrics_df: pd.DataFrame,
    cache_path: str,
    run_start_str: str,
    filename_prefix: str,
    analysis_type: str,
    subfolder: str,
    schema: BenchmarkDataSchema,
    search_x_col: str,
    search_x_col_label: str,
    search_metric_col: str = "rank",
    n_pre_conformal_trials: int = DEFAULT_NUMBER_OF_PRECONFORMAL_TRIALS,
) -> None:
    """Three-panel EI architecture figure: search ranks | ei_collapsed rate | perc_zero_ei.

    Each panel has one line per estimator architecture.  The left panel uses a
    linear y-axis (rank); the middle and right panels use a log y-axis so that
    small values and sudden jumps are both readable.  Log-axis ticks are placed at
    every decade *and* at several intermediate sub-decade values, and are labelled
    explicitly to make the scale unambiguous.

    Args:
        search_x_col: Column to use as the x-axis for the search-rank panel.
        search_x_col_label: X-axis label for the search-rank panel.
        n_pre_conformal_trials: Number of non-conformal trials before conformalization;
            a vertical line is drawn at this value plus one on the EI metric panels.
    """
    path_manager = AnalysisPathManager(cache_path, run_start_str)
    output_path = path_manager.get_analysis_path(analysis_type, "plots", subfolder)
    plot_path = os.path.join(output_path, f"{filename_prefix}__{search_metric_col}")

    plot_data = _prepare_plot_data_with_bounds(
        search_performance_df, (search_metric_col,)
    )

    arch_col = schema.estimator_architecture_col
    bench_col = schema.bench_col

    row_values = sorted(
        set(plot_data[bench_col].unique()).union(
            ei_metrics_df[bench_col].unique()
        )
    )

    all_archs = sorted(
        set(plot_data[arch_col].unique()).union(
            ei_metrics_df[arch_col].unique()
        )
    )
    color_map = {
        arch: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
        for i, arch in enumerate(all_archs)
    }

    base_width = 4.0
    base_height = 3.0
    fig_width = base_width * 3
    fig_height = base_height * len(row_values)

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = GridSpec(nrows=len(row_values), ncols=3, figure=fig)

    legend_handles: list = []
    legend_labels: list = []

    for i, row_value in enumerate(row_values):
        ax_search = fig.add_subplot(gs[i, 0])
        ax_collapsed = fig.add_subplot(gs[i, 1])
        ax_zero_ei = fig.add_subplot(gs[i, 2])

        search_row = plot_data[plot_data[bench_col] == row_value]
        ei_row = ei_metrics_df[ei_metrics_df[bench_col] == row_value]

        for arch in all_archs:
            arch_data = search_row[search_row[arch_col] == arch]
            if arch_data.empty:
                continue
            color = color_map[arch]
            line = ax_search.plot(
                arch_data[search_x_col],
                arch_data[search_metric_col],
                label=arch,
                alpha=0.85,
                color=color,
            )[0]
            if i == 0:
                legend_handles.append(line)
                legend_labels.append(arch)
            lower_col = f"{search_metric_col}_lower"
            upper_col = f"{search_metric_col}_upper"
            if (
                search_metric_col != "normalized_regret"
                and lower_col in arch_data.columns
                and upper_col in arch_data.columns
            ):
                ax_search.fill_between(
                    arch_data[search_x_col],
                    arch_data[lower_col],
                    arch_data[upper_col],
                    alpha=0.18,
                    color=color,
                )

        ax_search.set_xlabel(search_x_col_label, fontsize=14)
        ax_search.set_ylabel(search_metric_label(search_metric_col), fontsize=14, labelpad=10)
        ax_search.set_title(f"Search Performance", fontsize=14, pad=20)
        ax_search.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        for spine in ["top", "right", "bottom", "left"]:
            ax_search.spines[spine].set_linewidth(1.2)
        ax_search.tick_params(axis="both", which="major", labelsize=12, length=6, width=1.2)
        ax_search.tick_params(axis="both", which="minor", labelsize=10, length=3, width=1.0)
        apply_metric_yscale(ax_search, search_metric_col)

        collapsed_col = "cumulative_ei_collapsed_rate"
        for arch in all_archs:
            arch_data = ei_row[ei_row[arch_col] == arch].sort_values(
                by=schema.iter_unit
            )
            if arch_data.empty or collapsed_col not in arch_data.columns:
                continue
            color = color_map[arch]
            ax_collapsed.plot(
                arch_data[schema.iter_unit],
                arch_data[collapsed_col],
                label=arch,
                alpha=0.85,
                color=color,
            )
        ax_collapsed.set_xlabel("Iteration", fontsize=14)
        ax_collapsed.set_ylabel("Cumulative Failed Iteration Rate (%)", fontsize=14, labelpad=10)
        ax_collapsed.set_title(
            f"EI Collapse Rate", fontsize=14, pad=20
        )
        ax_collapsed.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
        ax_collapsed.axvline(
            n_pre_conformal_trials + 1,
            color="black",
            linestyle="--",
            linewidth=1.2,
        )
        for spine in ["top", "right", "bottom", "left"]:
            ax_collapsed.spines[spine].set_linewidth(1.2)
        ax_collapsed.tick_params(axis="both", which="major", labelsize=12, length=6, width=1.2)
        ax_collapsed.tick_params(axis="both", which="minor", labelsize=10, length=3, width=1.0)

        zero_ei_col = "perc_zero_ei"
        for arch in all_archs:
            arch_data = ei_row[ei_row[arch_col] == arch].sort_values(
                by=schema.iter_unit
            )
            if arch_data.empty or zero_ei_col not in arch_data.columns:
                continue
            color = color_map[arch]
            ax_zero_ei.plot(
                arch_data[schema.iter_unit],
                arch_data[zero_ei_col],
                label=arch,
                alpha=0.85,
                color=color,
            )
        ax_zero_ei.set_xlabel("Iteration", fontsize=14)
        ax_zero_ei.set_ylabel("Zero EI Rate (%)", fontsize=14, labelpad=10)
        ax_zero_ei.set_title(f"Zero EI Rate", fontsize=14, pad=20)
        ax_zero_ei.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
        ax_zero_ei.axvline(
            n_pre_conformal_trials + 1,
            color="black",
            linestyle="--",
            linewidth=1.2,
        )
        for spine in ["top", "right", "bottom", "left"]:
            ax_zero_ei.spines[spine].set_linewidth(1.2)
        ax_zero_ei.tick_params(axis="both", which="major", labelsize=12, length=6, width=1.2)
        ax_zero_ei.tick_params(axis="both", which="minor", labelsize=10, length=3, width=1.0)

    handles, labels = sort_legend_items(legend_handles, legend_labels)
    legend_ncols = compute_legend_ncols(len(labels)) if labels else 1
    num_legend_rows = math.ceil(len(labels) / legend_ncols) if labels else 1
    legend_anchor_y, legend_bottom_margin = calculate_legend_position(
        len(row_values), num_legend_rows, "standard"
    )

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=legend_ncols,
            fontsize=13,
            bbox_to_anchor=(0.5, legend_anchor_y),
            frameon=False,
        )

    fig.subplots_adjust(
        wspace=0.30,
        hspace=0.22,
        bottom=legend_bottom_margin,
        top=0.90,
        left=0.08,
        right=0.98,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for fmt in PLOT_FORMATS:
        full_path = f"{plot_path}_{timestamp}.{fmt}"
        fig.savefig(full_path, dpi=PLOT_DPI, bbox_inches="tight", format=fmt)

    plt.close(fig)
    logger.debug(
        f"EI architecture tri-plot saved in {output_path} with prefix {filename_prefix}"
    )


def plot_joint_candidates_and_extreme_quantile(
    search_performance_df: pd.DataFrame,
    extreme_quantile_df: pd.DataFrame,
    cache_path: str,
    run_start_str: str,
    filename_prefix: str,
    analysis_type: str,
    subfolder: str,
    schema: BenchmarkDataSchema,
    search_x_col: str,
    search_x_col_label: str,
    search_metric_col: str = "rank",
) -> None:
    """Plot joint analysis of candidate-count search ranks and extreme-quantile usage.

    Produces a two-panel figure (one row per benchmark) for a single estimator
    architecture and sampler, with one line per number-of-candidates value:
    - Left panel: search performance rank over the normalized budget.
    - Right panel: percentage of trials acquired via the lowest (extreme) quantile bound.

    Args:
        search_x_col: Column for the x-axis of the search-rank panel.
        search_x_col_label: X-axis label for the search-rank panel.
    """
    path_manager = AnalysisPathManager(cache_path, run_start_str)
    output_path = path_manager.get_analysis_path(analysis_type, "plots", subfolder)
    plot_path = os.path.join(output_path, f"{filename_prefix}__{search_metric_col}")

    plot_data = _prepare_plot_data_with_bounds(
        search_performance_df, (search_metric_col,)
    )

    row_measure = schema.bench_col
    identifier_col = "plotting_identifier"
    row_values = plot_data[row_measure].unique()

    raw_identifiers = sorted(
        set(plot_data[identifier_col]).union(
            extreme_quantile_df[identifier_col]
        ),
        key=lambda entity: legend_sort_key(display_entity_label(entity)),
    )
    raw_identifier_color_map = build_entity_color_map(raw_identifiers)

    base_width = 4.0
    base_height = 3.0
    fig_width = base_width * 2
    fig_height = base_height * len(row_values)

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = GridSpec(nrows=len(row_values), ncols=2, figure=fig)

    legend_handles = []
    legend_labels = []

    for i, row_value in enumerate(row_values):
        ax_search = fig.add_subplot(gs[i, 0])
        ax_extreme = fig.add_subplot(gs[i, 1])

        search_row_data = plot_data[plot_data[row_measure] == row_value]

        for identifier in raw_identifiers:
            entity_data = search_row_data[
                search_row_data[identifier_col] == identifier
            ]
            if entity_data.empty:
                continue
            color = raw_identifier_color_map[identifier]
            display_label = display_entity_label(identifier)
            line = ax_search.plot(
                entity_data[search_x_col],
                entity_data[search_metric_col],
                label=display_label,
                alpha=0.8,
                color=color,
                marker=None,
                markersize=4,
                linestyle=entity_plot_linestyle(identifier),
            )[0]
            if i == 0 and display_label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(display_label)
            lower_col = f"{search_metric_col}_lower"
            upper_col = f"{search_metric_col}_upper"
            if (
                search_metric_col != "normalized_regret"
                and lower_col in entity_data.columns
                and upper_col in entity_data.columns
            ):
                ax_search.fill_between(
                    entity_data[search_x_col],
                    entity_data[lower_col],
                    entity_data[upper_col],
                    alpha=0.2,
                    color=color,
                )

        ax_search.set_xlabel(search_x_col_label, fontsize=14)
        ax_search.set_ylabel(search_metric_label(search_metric_col), fontsize=14, labelpad=10)
        ax_search.set_title(f"Search Performance", fontsize=14, pad=20)
        ax_search.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        for spine in ["top", "right", "bottom", "left"]:
            ax_search.spines[spine].set_linewidth(1.2)
        ax_search.tick_params(axis="both", which="major", labelsize=12, length=6, width=1.2)
        ax_search.tick_params(axis="both", which="minor", labelsize=10, length=3, width=1.0)
        apply_metric_yscale(ax_search, search_metric_col)

        extreme_row_data = extreme_quantile_df[
            extreme_quantile_df[row_measure] == row_value
        ]

        for identifier in raw_identifiers:
            entity_data = extreme_row_data[
                extreme_row_data[identifier_col] == identifier
            ].sort_values(by=schema.iter_unit)
            if entity_data.empty:
                continue
            color = raw_identifier_color_map[identifier]
            display_label = display_entity_label(identifier)
            ax_extreme.plot(
                entity_data[schema.iter_unit],
                entity_data["cumulative_extreme_quantile_rate"] * 100,
                label=display_label,
                alpha=0.8,
                color=color,
                marker=None,
                markersize=4,
                linestyle=entity_plot_linestyle(identifier),
            )

        ax_extreme.set_xlabel("Iteration", fontsize=14)
        ax_extreme.set_ylabel("Extreme Quantile Usage (%)", fontsize=14, labelpad=10)
        ax_extreme.set_title(f"Quantile Collapse", fontsize=14, pad=20)
        ax_extreme.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        for spine in ["top", "right", "bottom", "left"]:
            ax_extreme.spines[spine].set_linewidth(1.2)
        ax_extreme.tick_params(axis="both", which="major", labelsize=12, length=6, width=1.2)
        ax_extreme.tick_params(axis="both", which="minor", labelsize=10, length=3, width=1.0)

    handles, labels = sort_legend_items(legend_handles, legend_labels)
    legend_ncols = compute_legend_ncols(len(labels)) if labels else 1
    num_legend_rows = math.ceil(len(labels) / legend_ncols) if labels else 1
    legend_anchor_y, legend_bottom_margin = calculate_legend_position(
        len(row_values), num_legend_rows, "standard"
    )

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=legend_ncols,
            fontsize=13,
            bbox_to_anchor=(0.5, legend_anchor_y),
            frameon=False,
        )

    fig.subplots_adjust(wspace=0.25, hspace=0.22, bottom=legend_bottom_margin, top=0.90, left=0.09, right=0.98)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for fmt in PLOT_FORMATS:
        full_path = f"{plot_path}_{timestamp}.{fmt}"
        fig.savefig(full_path, dpi=PLOT_DPI, bbox_inches="tight", format=fmt)

    plt.close(fig)
    logger.debug(f"Joint plots saved in {output_path} with prefix {filename_prefix}")
