"""
Plotting utilities for stratification diagnostics.

Style follows hpobench/plot.py conventions:
  - Font: STIXGeneral (mathtext: stix)
  - DPI: 300, formats: eps + png
  - Subplot size: 4 × 3 per cell
  - Grid: dashed, 0.5 linewidth, 0.7 alpha
  - Spines: all four sides, 1.2 linewidth
  - Tick params: major labelsize 11, minor labelsize 9
  - Legend: below figure, frameon=False, ncol ≤ 4, fontsize 12
  - constrained_layout=True
  - Colourblind-safe two-colour palette:
      selected = #E69F00  (warm orange, same as DEFAULT_COLOR_PALETTE[1])
      rest     = #464646  (dark grey,   same as DEFAULT_COLOR_PALETTE[0])

Two primary outputs per stratification type (called once across all benchmarks):

1. group_score_boxplot(results_by_benchmark, score_label, out_path)
   One figure, columns = benchmarks (lcbench | rbv2_aknn).
   Each column: box + strip plot of score values, Selected vs Rest.
   MWU p-value and Cliff's delta annotated on each panel.

2. target_kde_per_task(results_by_benchmark, out_path)
   One figure, columns = benchmarks.
   Each column: KDE of the performance metric for every task,
   selected tasks drawn in orange (thick), rest in dark grey (thin).
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

PLOT_DPI = 300
PLOT_FORMATS = ["eps", "png"]

COLOR_SELECTED = "#E69F00"
COLOR_REST = "#464646"
CELL_WIDTH = 4.0
CELL_HEIGHT = 3.0


# ── Internal helpers ──────────────────────────────────────────────────────────

def _style_ax(ax):
    """Apply hpobench spine, tick, and grid style to an axis."""
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.tick_params(axis="both", which="major", labelsize=11, length=6, width=1.2)
    ax.tick_params(axis="both", which="minor", labelsize=9,  length=3, width=1.0)


def _add_figure_legend(fig, handles, labels):
    """Place a shared legend below the figure, hpobench style."""
    num_legend_rows = math.ceil(len(labels) / 4)
    anchor_y = -0.10 - (num_legend_rows - 1) * 0.06
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(4, len(labels)),
        fontsize=12,
        bbox_to_anchor=(0.5, anchor_y),
        frameon=False,
    )


def _save(fig, path_no_ext):
    """Save figure in all PLOT_FORMATS."""
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
    for fmt in PLOT_FORMATS:
        fig.savefig(
            f"{path_no_ext}.{fmt}",
            dpi=PLOT_DPI,
            format=fmt,
            bbox_inches="tight",
            transparent=False,
        )
    plt.close(fig)


def _significance_stars(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


# ── Public plotting functions ─────────────────────────────────────────────────

def group_score_boxplot(
    results_by_benchmark: dict,
    score_label: str,
    out_path: str,
):
    """
    Box + strip plot comparing score distributions of selected vs rest tasks,
    one column per benchmark.  MWU p-value and Cliff's delta annotated on each
    panel.

    Parameters
    ----------
    results_by_benchmark : dict
        Keys are benchmark names.  Each value is a dict with:
          "selected_scores" : list[float]  — score for each selected task
          "rest_scores"     : list[float]  — score for each non-selected task
          "mwu_p_value"     : float
          "cliffs_delta"    : float
    score_label : str
        Y-axis label (e.g. "|Moors skewness score|").
    out_path : str
        Full path without extension; saved as .eps and .png.
    """
    benchmark_names = list(results_by_benchmark.keys())
    num_cols = len(benchmark_names)
    fig, axes = plt.subplots(
        1, num_cols,
        figsize=(CELL_WIDTH * num_cols, CELL_HEIGHT),
        constrained_layout=True,
    )
    if num_cols == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []

    for col, benchmark in enumerate(benchmark_names):
        ax = axes[col]
        info = results_by_benchmark[benchmark]

        selected_scores = info["selected_scores"]
        rest_scores = info["rest_scores"]

        rows = (
            [{"score": v, "group": "Selected"} for v in selected_scores]
            + [{"score": v, "group": "Rest"} for v in rest_scores]
        )
        plot_data = pd.DataFrame(rows)

        palette = {"Selected": COLOR_SELECTED, "Rest": COLOR_REST}
        order = ["Selected", "Rest"]

        box = sns.boxplot(
            data=plot_data, x="group", y="score", ax=ax,
            palette=palette, order=order,
            width=0.45, linewidth=1.2, fliersize=3,
        )
        sns.stripplot(
            data=plot_data, x="group", y="score", ax=ax,
            palette=palette, order=order,
            size=4, jitter=True, alpha=0.65,
        )

        p_val = info.get("mwu_p_value", float("nan"))
        delta = info.get("cliffs_delta", float("nan"))
        stars = _significance_stars(p_val)
        annotation = f"MWU {stars}\np = {p_val:.3e}\n$\\delta$ = {delta:.2f}"
        ax.text(
            0.97, 0.97, annotation,
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85),
        )

        ax.set_title(benchmark, fontsize=13)
        ax.set_xlabel("")
        if col == 0:
            ax.set_ylabel(score_label, fontsize=13, labelpad=10)
        else:
            ax.set_ylabel("")
        _style_ax(ax)

        if col == 0:
            from matplotlib.patches import Patch
            legend_handles = [
                Patch(facecolor=COLOR_SELECTED, label=f"Selected (n={len(selected_scores)})"),
                Patch(facecolor=COLOR_REST,     label=f"Rest (n={len(rest_scores)})"),
            ]
            legend_labels = [h.get_label() for h in legend_handles]

    _add_figure_legend(fig, legend_handles, legend_labels)
    _save(fig, out_path)


def target_kde_per_task(
    results_by_benchmark: dict,
    out_path: str,
):
    """
    KDE of the performance metric for every task, one column per benchmark.
    Selected tasks: orange thick line.  Rest: dark grey thin line.

    Parameters
    ----------
    results_by_benchmark : dict
        Keys are benchmark names.  Each value is a dict with:
          "selected_task_ids" : list[str]
          "task_performance"  : dict[task_id -> np.ndarray of y values]
    out_path : str
        Full path without extension; saved as .eps and .png.
    """
    benchmark_names = list(results_by_benchmark.keys())
    num_cols = len(benchmark_names)
    fig, axes = plt.subplots(
        1, num_cols,
        figsize=(CELL_WIDTH * num_cols, CELL_HEIGHT),
        constrained_layout=True,
    )
    if num_cols == 1:
        axes = [axes]

    for col, benchmark in enumerate(benchmark_names):
        ax = axes[col]
        info = results_by_benchmark[benchmark]
        selected = set(info["selected_task_ids"])
        task_perf = info["task_performance"]

        for task_id, perf_values in task_perf.items():
            if len(perf_values) < 2:
                continue
            if task_id in selected:
                sns.kdeplot(
                    perf_values, ax=ax,
                    color=COLOR_SELECTED, linewidth=2.0, alpha=0.9,
                )
            else:
                sns.kdeplot(
                    perf_values, ax=ax,
                    color=COLOR_REST, linewidth=0.7, alpha=0.25,
                )

        ax.set_title(benchmark, fontsize=13)
        ax.set_xlabel("Performance", fontsize=13)
        if col == 0:
            ax.set_ylabel("Density", fontsize=13, labelpad=10)
        else:
            ax.set_ylabel("")
        _style_ax(ax)

    from matplotlib.lines import Line2D
    n_sel_total = sum(
        len(info["selected_task_ids"]) for info in results_by_benchmark.values()
    )
    n_rest_total = sum(
        len(info["task_performance"]) - len(info["selected_task_ids"])
        for info in results_by_benchmark.values()
    )
    legend_handles = [
        Line2D([0], [0], color=COLOR_SELECTED, linewidth=2,
               label=f"Selected tasks (n={n_sel_total} total)"),
        Line2D([0], [0], color=COLOR_REST, linewidth=1, alpha=0.5,
               label=f"Rest (n={n_rest_total} total)"),
    ]
    _add_figure_legend(fig, legend_handles, [h.get_label() for h in legend_handles])
    _save(fig, out_path)
