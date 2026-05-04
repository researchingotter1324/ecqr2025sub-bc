import json
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

BENCHMARKS = ["rbv2_aknn", "lcbench"]

SUMMARY_DIR = os.path.join("cache", "snippets_outputs", "summary")
OUT_PATH = os.path.join("cache", "snippets_outputs", "stratification_boxplots")

PLOT_DPI = 300
PLOT_FORMATS = ["eps", "png"]

COLOR_SELECTED = "#E69F00"
COLOR_REST = "#464646"
CELL_WIDTH = 4.0
CELL_HEIGHT = 3.0

COLUMN_LABELS = ["Heteroscedastic Stratification", "Asymmetric Stratification"]
SCORE_LABELS = [
    "Breusch-Pagan R²",
    "Groeneveld-Meeden Asymmetry",
]
DATA_FILENAME_TEMPLATES = [
    "heteroscedasticity_boxplot_{benchmark}.json",
    "asymmetry_boxplot_{benchmark}.json",
]


def load_boxplot_data(benchmark: str, filename_template: str) -> dict | None:
    """Load the selected/rest score arrays written by a stratification script.

    Args:
        benchmark: Benchmark name to substitute into the filename template.
        filename_template: Filename template containing ``{benchmark}``.

    Returns:
        Dict with keys ``selected_scores`` and ``rest_scores`` (lists of
        floats), or None if the file does not exist.
    """
    path = os.path.join(SUMMARY_DIR, filename_template.format(benchmark=benchmark))
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def style_ax(ax):
    """Apply shared spine, tick, and grid style to an axis.

    Args:
        ax: Matplotlib Axes instance to style.
    """
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.tick_params(axis="both", which="major", labelsize=11, length=6, width=1.2)
    ax.tick_params(axis="both", which="minor", labelsize=9, length=3, width=1.0)


def save_figure(fig, path_no_ext: str):
    """Save a figure in all configured formats.

    Args:
        fig: Matplotlib Figure to save.
        path_no_ext: Output path without file extension.
    """
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


def draw_boxplot_panel(ax, selected_scores: list, rest_scores: list):
    """Draw a box + strip plot of selected vs rest scores on one axis.

    Args:
        ax: Matplotlib Axes to draw on.
        selected_scores: Score values for the selected tasks.
        rest_scores: Score values for the non-selected tasks.
    """
    rows = (
        [{"score": v, "group": "Selected"} for v in selected_scores]
        + [{"score": v, "group": "Rest"} for v in rest_scores]
    )
    plot_data = pd.DataFrame(rows)
    palette = {"Selected": COLOR_SELECTED, "Rest": COLOR_REST}
    order = ["Selected", "Rest"]

    sns.boxplot(
        data=plot_data, x="group", y="score", ax=ax,
        palette=palette, order=order,
        width=0.45, linewidth=1.2,
        flierprops=dict(marker=""),
    )
    np.random.seed(42)
    sns.stripplot(
        data=plot_data, x="group", y="score", ax=ax,
        palette=palette, order=order,
        size=4, jitter=True, alpha=0.65,
    )
    ax.set_xlabel("")
    style_ax(ax)


def main():
    """Build the combined stratification boxplot figure from pre-written data files.

    Reads the JSON files produced by the heteroscedasticity and asymmetry
    scoring scripts and assembles a single figure with one row per benchmark
    and one column per stratification type.

    Layout:
        - Column headers (above each column): "Heteroscedastic Strata" and
          "Asymmetric Strata", drawn once at the top of the figure.
        - Row labels (left of column 0): the benchmark name, drawn only when
          more than one benchmark is present.
        - Y-axis label on each panel carries the score-specific label.
        - Group identity is shown by x-axis tick labels ("Selected" / "Rest")
          and point colour; no separate legend is needed.
    """
    available_benchmarks = []
    data_by_benchmark_and_col = {}

    for benchmark in BENCHMARKS:
        col_data = []
        for template in DATA_FILENAME_TEMPLATES:
            col_data.append(load_boxplot_data(benchmark, template))
        if any(d is not None for d in col_data):
            available_benchmarks.append(benchmark)
            data_by_benchmark_and_col[benchmark] = col_data

    if not available_benchmarks:
        print("No boxplot data files found in", SUMMARY_DIR)
        return

    num_rows = len(available_benchmarks)
    num_cols = 2
    multiple_benchmarks = num_rows > 1

    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(CELL_WIDTH * num_cols, CELL_HEIGHT * num_rows),
        constrained_layout=True,
        squeeze=False,
    )

    for col_idx, col_label in enumerate(COLUMN_LABELS):
        axes[0, col_idx].set_title(col_label, fontsize=13, fontweight="bold", pad=8)

    for row_idx, benchmark in enumerate(available_benchmarks):
        col_data = data_by_benchmark_and_col[benchmark]

        if multiple_benchmarks:
            axes[row_idx, 0].set_ylabel(
                f"{benchmark}\n{SCORE_LABELS[0]}", fontsize=11, labelpad=10
            )
        else:
            axes[row_idx, 0].set_ylabel(SCORE_LABELS[0], fontsize=11, labelpad=10)

        for col_idx in range(num_cols):
            ax = axes[row_idx, col_idx]
            data = col_data[col_idx]

            if data is None:
                ax.set_visible(False)
                continue

            selected_scores = data.get("selected_scores", [])
            rest_scores = data.get("rest_scores", [])

            draw_boxplot_panel(ax, selected_scores, rest_scores)

            if col_idx > 0:
                ax.set_ylabel(SCORE_LABELS[col_idx], fontsize=11, labelpad=10)

    save_figure(fig, OUT_PATH)
    print(f"Saved: {OUT_PATH}.png / .eps")


if __name__ == "__main__":
    main()
