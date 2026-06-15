import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
import seaborn as sns
from matplotlib.colors import ListedColormap

matplotlib.use("Agg")  # Use non-GUI backend
logger = logging.getLogger(__name__)

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

PLOT_DPI = 300
PLOT_FORMATS = ["eps", "png"]
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


def sort_legend_items(handles: list, labels: list) -> tuple[list, list]:
    """Sort legend items: numerically if starts with number, otherwise alphabetically."""
    combined = sorted(zip(handles, labels), key=lambda x: legend_sort_key(x[1]))
    sorted_handles, sorted_labels = zip(*combined)
    return list(sorted_handles), list(sorted_labels)


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
    linestyle = "--" if is_non_local(legend_label) else "-"

    ax.plot(
        tuner_data[x_col],
        tuner_data[y_col],
        label=legend_label,
        alpha=0.8,
        color=color,
        marker=marker_style,
        markersize=4,
        linestyle=linestyle,
    )
    if add_ci and y_col_lower and y_col_upper:
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

    if share_y_axis:
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
            for entity_idx, (entity, entity_data) in enumerate(
                subset.groupby(entity_col)
            ):
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
                    color=DEFAULT_COLOR_PALETTE[
                        entity_idx % len(DEFAULT_COLOR_PALETTE)
                    ],
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
            elif share_y_axis:
                ax.set_ylim((global_y_min, global_y_max))
            else:
                y_min, y_max = get_y_bounds(subset, y_col, y_col_lower, y_col_upper)
                y_range = y_max - y_min
                buffer = 0.05 * y_range if y_range > 0 else 0.05
                ax.set_ylim((y_min - buffer, y_max + buffer))

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
                ax.set_title(col_title, fontsize=13)

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
                ax.set_ylabel(row_title, fontsize=13, labelpad=10)

            if i == len(row_values) - 1:
                ax.set_xlabel(x_label_to_use, fontsize=13)

            if not subset.empty:
                ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
                ax.spines["top"].set_linewidth(1.2)
                ax.spines["right"].set_linewidth(1.2)
                ax.spines["bottom"].set_linewidth(1.2)
                ax.spines["left"].set_linewidth(1.2)
                ax.tick_params(
                    axis="both", which="major", labelsize=11, length=6, width=1.2
                )
                ax.tick_params(axis="both", which="minor", labelsize=9, length=3, width=1.0)
            else:
                for spine in ax.spines.values():
                    spine.set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    handles, labels = sort_legend_items(handles, labels)

    num_subplot_rows = len(row_values) if row_measure else 1
    num_legend_rows = math.ceil(len(labels) / 4)

    legend_anchor_y, legend_bottom_margin = calculate_legend_position(
        num_subplot_rows, num_legend_rows, "standard"
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(4, len(labels)),
        fontsize=12,
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

    if y_cols_lower is None:
        y_cols_lower = [
            f"{y_col}_q10" if f"{y_col}_q10" in data.columns else None
            for y_col in y_cols
        ]
    if y_cols_upper is None:
        y_cols_upper = [
            f"{y_col}_q90" if f"{y_col}_q90" in data.columns else None
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
            data=data,
            plot_path=plot_path,
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
    ranks_series = pd.Series(mean_ranks)
    algorithms = list(mean_ranks.keys())

    # Create significance matrix
    sig_matrix = pd.DataFrame(1.0, index=algorithms, columns=algorithms)
    np.fill_diagonal(sig_matrix.values, 1.0)

    for _, row in significance_results.iterrows():
        alg1, alg2 = row["entity1"], row["entity2"]
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
    entities = sorted(rank_data[entity_col].unique())
    avg_ranks = dict(zip(rank_data[entity_col], rank_data["rank"]))

    p_matrix = pd.DataFrame(np.nan, index=entities, columns=entities)
    color_matrix = pd.DataFrame(0, index=entities, columns=entities)

    for entity in entities:
        p_matrix.loc[entity, entity] = 1.0
        color_matrix.loc[entity, entity] = 0

    for _, row in significance_data.iterrows():
        entity1, entity2 = row["entity1"], row["entity2"]
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
        annot_kws={"size": 10, "color": "black"},
        linewidths=0.5,
        linecolor="black",
        xticklabels=entities,
        yticklabels=entities,
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
            fontsize=10,
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
        fontsize=10,
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
    ax.tick_params(axis="both", labelsize=10, colors="black")

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
    """Plot paired visualizations: rank evolution and significance analysis.

    Creates a plot where:
    - Left column: Rank evolution over budget
    - Right column: Either CD diagrams ("cd") or significance matrix ("matrix")
      - "cd": Two CD diagrams stacked vertically (uncorrected/corrected p-values)
      - "matrix": Single significance matrix with corrected p-values
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

    # Get unique row values
    row_values = data[row_measure].unique()

    # Create figure with layout based on significance plot type
    base_width = 4.0
    base_height = 4.5

    if significance_plot_type == "matrix":
        # Square layout for matrix - ensure both charts are square and same size
        square_size = base_width  # Use base width for square dimensions
        fig_width = square_size * 2.2  # Two squares plus some spacing
        fig_height = square_size * len(row_values) + 1.0  # Account for legend space

        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = GridSpec(
            nrows=len(row_values),
            ncols=2,
            figure=fig,
            width_ratios=[1, 1],
            height_ratios=[1] * len(row_values),
            wspace=0.25,
            hspace=0.25,
        )
    else:
        fig_width = base_width * 2
        fig_height = base_height * len(row_values)

        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
        gs = GridSpec(nrows=len(row_values) * 2, ncols=2, figure=fig)

    axes = []
    for i in range(len(row_values)):
        if significance_plot_type == "matrix":
            ax_rank = fig.add_subplot(gs[i, 0])
            ax_matrix = fig.add_subplot(gs[i, 1])
            axes.append([ax_rank, ax_matrix])
        else:
            ax_rank = fig.add_subplot(gs[i * 2 : (i + 1) * 2, 0])
            ax_cd_uncorrected = fig.add_subplot(gs[i * 2, 1])
            ax_cd_corrected = fig.add_subplot(gs[i * 2 + 1, 1])
            axes.append([ax_rank, ax_cd_uncorrected, ax_cd_corrected])

    legend_handles = []
    legend_labels = []

    for i, row_value in enumerate(row_values):
        row_data = data[data[row_measure] == row_value]
        row_sig_data = significance_data[significance_data[row_measure] == row_value]
        ax_rank = axes[i][0]

        for entity_idx, (entity, entity_data) in enumerate(
            row_data.groupby(entity_col)
        ):
            color = DEFAULT_COLOR_PALETTE[entity_idx % len(DEFAULT_COLOR_PALETTE)]
            linestyle = "--" if is_non_local(entity) else "-"
            line = ax_rank.plot(
                entity_data[x_col],
                entity_data["rank"],
                label=entity,
                alpha=0.8,
                color=color,
                marker=None,
                markersize=4,
                linestyle=linestyle,
            )[0]

            if i == 0:
                legend_handles.append(line)
                legend_labels.append(entity)

            if (
                y_col_lower is not None
                and y_col_upper is not None
                and y_col_lower in entity_data.columns
                and y_col_upper in entity_data.columns
            ):
                ax_rank.fill_between(
                    entity_data[x_col],
                    entity_data[y_col_lower],
                    entity_data[y_col_upper],
                    alpha=0.2,
                    color=color,
                )

        # Format left plot consistent with plot_benchmark_data
        ax_rank.set_xlabel(get_label(x_label, x_col), fontsize=13)
        ax_rank.set_ylabel("Rank", fontsize=13, labelpad=10)
        ax_rank.set_title(
            f"{row_value}",
            fontsize=13,
            pad=20,
        )
        ax_rank.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

        if significance_plot_type == "matrix":
            ax_rank.set_aspect("auto")

        if x_axis_start is not None:
            current_xlim = ax_rank.get_xlim()
            ax_rank.set_xlim(left=x_axis_start, right=current_xlim[1])

        for spine in ["top", "right", "bottom", "left"]:
            ax_rank.spines[spine].set_linewidth(1.2)

        ax_rank.tick_params(
            axis="both", which="major", labelsize=11, length=6, width=1.2
        )
        ax_rank.tick_params(
            axis="both", which="minor", labelsize=9, length=3, width=1.0
        )

        if significance_plot_type == "matrix":
            ax_matrix = axes[i][1]
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
                for ax_cd, _ in [
                    (ax_cd_uncorrected, None),
                    (ax_cd_corrected, None),
                ]:
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
    num_legend_rows = math.ceil(len(labels) / 4) if labels else 1
    plot_type = "matrix" if significance_plot_type == "matrix" else "cd"
    legend_anchor_y, legend_bottom_margin = calculate_legend_position(
        num_subplot_rows, num_legend_rows, plot_type
    )

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, len(labels)),
            fontsize=12,
            bbox_to_anchor=(0.5, legend_anchor_y),
            frameon=False,
        )

    if significance_plot_type == "matrix":
        fig.subplots_adjust(
            wspace=0.25,
            hspace=0.25,
            bottom=legend_bottom_margin,
            top=0.88,
            left=0.08,
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
) -> None:
    """Plot joint analysis comparing architecture optimization ranks and estimator errors.

    Produces one row per benchmark.  Within each row there is one search-rank
    panel per unique sampler (ordered alphabetically) followed by a single
    pinball-loss panel on the right.  Each search-rank panel shows one line per
    estimator architecture for that sampler; the pinball-loss panel is shared
    across all samplers and shows lines per estimator architecture over training
    data size.
    """
    path_manager = AnalysisPathManager(cache_path, run_start_str)
    output_path = path_manager.get_analysis_path(analysis_type, "plots", subfolder)
    plot_path = os.path.join(output_path, filename_prefix)

    row_measure = schema.bench_col
    arch_col = schema.estimator_architecture_col
    sampler_col = schema.sampler_col

    row_values = main_processed_df[row_measure].unique()
    samplers = sorted(main_processed_df[sampler_col].unique())
    n_sampler_cols = len(samplers)
    n_cols = n_sampler_cols + 1  # sampler columns + pinball-loss column

    all_archs = sorted(main_processed_df[arch_col].unique())
    color_map = {
        arch: DEFAULT_COLOR_PALETTE[idx % len(DEFAULT_COLOR_PALETTE)]
        for idx, arch in enumerate(all_archs)
    }

    base_width = 4.0
    base_height = 4.5
    fig_width = base_width * n_cols
    fig_height = base_height * len(row_values)

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = GridSpec(nrows=len(row_values), ncols=n_cols, figure=fig)

    legend_handles: list = []
    legend_labels: list = []

    for i, row_value in enumerate(row_values):
        main_row_data = main_processed_df[main_processed_df[row_measure] == row_value]

        for j, sampler in enumerate(samplers):
            ax_search = fig.add_subplot(gs[i, j])
            sampler_data = main_row_data[main_row_data[sampler_col] == sampler]

            for arch in all_archs:
                arch_data = sampler_data[sampler_data[arch_col] == arch]
                if arch_data.empty:
                    continue
                color = color_map[arch]
                line = ax_search.plot(
                    arch_data[schema.norm_iter_unit],
                    arch_data["rank"],
                    label=arch,
                    alpha=0.8,
                    color=color,
                    marker=None,
                    markersize=4,
                )[0]
                if i == 0 and j == 0:
                    legend_handles.append(line)
                    legend_labels.append(arch)
                if "rank_lower" in arch_data.columns and "rank_upper" in arch_data.columns:
                    ax_search.fill_between(
                        arch_data[schema.norm_iter_unit],
                        arch_data["rank_lower"],
                        arch_data["rank_upper"],
                        alpha=0.2,
                        color=color,
                    )

            ax_search.set_xlabel("Normalized Iteration Budget", fontsize=13)
            ax_search.set_ylabel("Rank", fontsize=13, labelpad=10)
            ax_search.set_title(
                f"Optimization Performance ({sampler}): {row_value}",
                fontsize=11,
                pad=20,
            )
            ax_search.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            for spine in ["top", "right", "bottom", "left"]:
                ax_search.spines[spine].set_linewidth(1.2)
            ax_search.tick_params(axis="both", which="major", labelsize=11, length=6, width=1.2)
            ax_search.tick_params(axis="both", which="minor", labelsize=9, length=3, width=1.0)

        ax_static = fig.add_subplot(gs[i, n_sampler_cols])
        static_row_data = static_processed_df[static_processed_df[row_measure] == row_value]

        for entity_idx, (entity, entity_data) in enumerate(
            static_row_data.groupby(arch_col)
        ):
            color = DEFAULT_COLOR_PALETTE[entity_idx % len(DEFAULT_COLOR_PALETTE)]
            ax_static.plot(
                entity_data[schema.data_size_col],
                entity_data["rank"],
                label=entity,
                alpha=0.8,
                color=color,
                marker="o",
                markersize=4,
            )

        ax_static.set_xlabel("Training Data Size", fontsize=13)
        ax_static.set_ylabel("Rank (Pinball Loss)", fontsize=13, labelpad=10)
        ax_static.set_title(f"Estimator Error: {row_value}", fontsize=13, pad=20)
        ax_static.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        for spine in ["top", "right", "bottom", "left"]:
            ax_static.spines[spine].set_linewidth(1.2)
        ax_static.tick_params(axis="both", which="major", labelsize=11, length=6, width=1.2)
        ax_static.tick_params(axis="both", which="minor", labelsize=9, length=3, width=1.0)

    handles, labels = sort_legend_items(legend_handles, legend_labels)
    num_legend_rows = math.ceil(len(labels) / 4) if labels else 1
    legend_anchor_y, legend_bottom_margin = calculate_legend_position(
        len(row_values), num_legend_rows, "standard"
    )

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, len(labels)),
            fontsize=12,
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


def plot_ei_architecture_triplot(
    search_performance_df: pd.DataFrame,
    ei_metrics_df: pd.DataFrame,
    cache_path: str,
    run_start_str: str,
    filename_prefix: str,
    analysis_type: str,
    subfolder: str,
    schema: BenchmarkDataSchema,
) -> None:
    """Three-panel EI architecture figure: search ranks | ei_collapsed rate | perc_zero_ei.

    Each panel has one line per estimator architecture.  The left panel uses a
    linear y-axis (rank); the middle and right panels use a log y-axis so that
    small values and sudden jumps are both readable.  Log-axis ticks are placed at
    every decade *and* at several intermediate sub-decade values, and are labelled
    explicitly to make the scale unambiguous.
    """
    path_manager = AnalysisPathManager(cache_path, run_start_str)
    output_path = path_manager.get_analysis_path(analysis_type, "plots", subfolder)
    plot_path = os.path.join(output_path, filename_prefix)

    arch_col = schema.estimator_architecture_col
    bench_col = schema.bench_col

    row_values = sorted(
        set(search_performance_df[bench_col].unique()).union(
            ei_metrics_df[bench_col].unique()
        )
    )

    all_archs = sorted(
        set(search_performance_df[arch_col].unique()).union(
            ei_metrics_df[arch_col].unique()
        )
    )
    color_map = {
        arch: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
        for i, arch in enumerate(all_archs)
    }

    base_width = 4.0
    base_height = 4.5
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

        search_row = search_performance_df[
            search_performance_df[bench_col] == row_value
        ]
        ei_row = ei_metrics_df[ei_metrics_df[bench_col] == row_value]

        for arch in all_archs:
            arch_data = search_row[search_row[arch_col] == arch]
            if arch_data.empty:
                continue
            color = color_map[arch]
            line = ax_search.plot(
                arch_data[schema.norm_iter_unit],
                arch_data["rank"],
                label=arch,
                alpha=0.85,
                color=color,
            )[0]
            if i == 0:
                legend_handles.append(line)
                legend_labels.append(arch)
            if "rank_lower" in arch_data.columns and "rank_upper" in arch_data.columns:
                ax_search.fill_between(
                    arch_data[schema.norm_iter_unit],
                    arch_data["rank_lower"],
                    arch_data["rank_upper"],
                    alpha=0.18,
                    color=color,
                )

        ax_search.set_xlabel("Normalized Iteration Budget", fontsize=12)
        ax_search.set_ylabel("Rank", fontsize=12, labelpad=8)
        ax_search.set_title(f"Search Performance: {row_value}", fontsize=12, pad=16)
        ax_search.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        for spine in ["top", "right", "bottom", "left"]:
            ax_search.spines[spine].set_linewidth(1.2)
        ax_search.tick_params(axis="both", which="major", labelsize=10, length=5)

        collapsed_col = "cumulative_ei_collapsed_rate"
        all_collapsed_vals = pd.Series(dtype=float)
        for arch in all_archs:
            arch_data = ei_row[ei_row[arch_col] == arch].sort_values(
                by=schema.iter_unit
            )
            if arch_data.empty or collapsed_col not in arch_data.columns:
                continue
            vals = arch_data[collapsed_col].dropna()
            all_collapsed_vals = pd.concat([all_collapsed_vals, vals])
            color = color_map[arch]
            ax_collapsed.plot(
                arch_data[schema.iter_unit],
                arch_data[collapsed_col],
                label=arch,
                alpha=0.85,
                color=color,
            )

        trim_y_axis(ax_collapsed, all_collapsed_vals)
        ax_collapsed.set_xlabel("Iteration (Trial)", fontsize=12)
        ax_collapsed.set_ylabel("Cumulative EI Collapsed Rate", fontsize=11, labelpad=8)
        ax_collapsed.set_title(
            f"EI Collapsed Rate: {row_value}", fontsize=12, pad=16
        )
        ax_collapsed.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
        for spine in ["top", "right", "bottom", "left"]:
            ax_collapsed.spines[spine].set_linewidth(1.2)
        ax_collapsed.tick_params(axis="both", which="major", labelsize=10, length=5)

        zero_ei_col = "perc_zero_ei"
        all_zero_ei_vals = pd.Series(dtype=float)
        for arch in all_archs:
            arch_data = ei_row[ei_row[arch_col] == arch].sort_values(
                by=schema.iter_unit
            )
            if arch_data.empty or zero_ei_col not in arch_data.columns:
                continue
            vals = arch_data[zero_ei_col].dropna()
            all_zero_ei_vals = pd.concat([all_zero_ei_vals, vals])
            color = color_map[arch]
            ax_zero_ei.plot(
                arch_data[schema.iter_unit],
                arch_data[zero_ei_col],
                label=arch,
                alpha=0.85,
                color=color,
            )

        trim_y_axis(ax_zero_ei, all_zero_ei_vals)
        ax_zero_ei.set_xlabel("Iteration (Trial)", fontsize=12)
        ax_zero_ei.set_ylabel("% Zero EI", fontsize=11, labelpad=8)
        ax_zero_ei.set_title(f"Zero-EI Rate: {row_value}", fontsize=12, pad=16)
        ax_zero_ei.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
        for spine in ["top", "right", "bottom", "left"]:
            ax_zero_ei.spines[spine].set_linewidth(1.2)
        ax_zero_ei.tick_params(axis="both", which="major", labelsize=10, length=5)

    handles, labels = sort_legend_items(legend_handles, legend_labels)
    num_legend_rows = math.ceil(len(labels) / 4) if labels else 1
    legend_anchor_y, legend_bottom_margin = calculate_legend_position(
        len(row_values), num_legend_rows, "standard"
    )

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, len(labels)),
            fontsize=12,
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
) -> None:
    """Plot joint analysis of candidate-count search ranks and extreme-quantile usage.

    Produces a two-panel figure (one row per benchmark) for a single estimator
    architecture and sampler, with one line per number-of-candidates value:
    - Left panel: search performance rank over the normalized iteration budget.
    - Right panel: percentage of trials acquired via the lowest (extreme) quantile bound.
    """
    path_manager = AnalysisPathManager(cache_path, run_start_str)
    output_path = path_manager.get_analysis_path(analysis_type, "plots", subfolder)
    plot_path = os.path.join(output_path, filename_prefix)

    row_measure = schema.bench_col
    identifier_col = "plotting_identifier"
    row_values = search_performance_df[row_measure].unique()

    all_identifiers = sorted(
        set(search_performance_df[identifier_col]).union(
            extreme_quantile_df[identifier_col]
        ),
        key=identifier_sort_key,
    )
    color_map = {
        identifier: DEFAULT_COLOR_PALETTE[idx % len(DEFAULT_COLOR_PALETTE)]
        for idx, identifier in enumerate(all_identifiers)
    }

    base_width = 4.0
    base_height = 4.5
    fig_width = base_width * 2
    fig_height = base_height * len(row_values)

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = GridSpec(nrows=len(row_values), ncols=2, figure=fig)

    legend_handles = []
    legend_labels = []

    for i, row_value in enumerate(row_values):
        ax_search = fig.add_subplot(gs[i, 0])
        ax_extreme = fig.add_subplot(gs[i, 1])

        search_row_data = search_performance_df[
            search_performance_df[row_measure] == row_value
        ]

        for identifier in all_identifiers:
            entity_data = search_row_data[
                search_row_data[identifier_col] == identifier
            ]
            if entity_data.empty:
                continue
            color = color_map[identifier]
            line = ax_search.plot(
                entity_data[schema.norm_iter_unit],
                entity_data["rank"],
                label=identifier,
                alpha=0.8,
                color=color,
                marker=None,
                markersize=4,
            )[0]
            if i == 0:
                legend_handles.append(line)
                legend_labels.append(identifier)
            if (
                "rank_lower" in entity_data.columns
                and "rank_upper" in entity_data.columns
            ):
                ax_search.fill_between(
                    entity_data[schema.norm_iter_unit],
                    entity_data["rank_lower"],
                    entity_data["rank_upper"],
                    alpha=0.2,
                    color=color,
                )

        ax_search.set_xlabel("Normalized Iteration Budget", fontsize=13)
        ax_search.set_ylabel("Rank", fontsize=13, labelpad=10)
        ax_search.set_title(f"Optimization Performance: {row_value}", fontsize=13, pad=20)
        ax_search.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        for spine in ["top", "right", "bottom", "left"]:
            ax_search.spines[spine].set_linewidth(1.2)
        ax_search.tick_params(axis="both", which="major", labelsize=11, length=6, width=1.2)
        ax_search.tick_params(axis="both", which="minor", labelsize=9, length=3, width=1.0)

        extreme_row_data = extreme_quantile_df[
            extreme_quantile_df[row_measure] == row_value
        ]

        for identifier in all_identifiers:
            entity_data = extreme_row_data[
                extreme_row_data[identifier_col] == identifier
            ].sort_values(by=schema.iter_unit)
            if entity_data.empty:
                continue
            color = color_map[identifier]
            ax_extreme.plot(
                entity_data[schema.iter_unit],
                entity_data["cumulative_extreme_quantile_rate"] * 100,
                label=identifier,
                alpha=0.8,
                color=color,
                marker=None,
                markersize=4,
            )

        ax_extreme.set_xlabel("Iteration (Trial)", fontsize=13)
        ax_extreme.set_ylabel("Extreme Quantile Used (%)", fontsize=13, labelpad=10)
        ax_extreme.set_title(f"Extreme Quantile Usage: {row_value}", fontsize=13, pad=20)
        ax_extreme.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        for spine in ["top", "right", "bottom", "left"]:
            ax_extreme.spines[spine].set_linewidth(1.2)
        ax_extreme.tick_params(axis="both", which="major", labelsize=11, length=6, width=1.2)
        ax_extreme.tick_params(axis="both", which="minor", labelsize=9, length=3, width=1.0)

    handles, labels = sort_legend_items(legend_handles, legend_labels)
    num_legend_rows = math.ceil(len(labels) / 4) if labels else 1
    legend_anchor_y, legend_bottom_margin = calculate_legend_position(
        len(row_values), num_legend_rows, "standard"
    )

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, len(labels)),
            fontsize=12,
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
