import pandas as pd
from typing import List, Optional

CALIBRATION_SCORE_DECIMALS = 3
CALIBRATION_METHOD_ORDER = [
    "Unconformalized",
    "Split Conformalized",
    "Cross Conformalized",
]


def _escape_latex_text(text: str) -> str:
    """Escape underscores in text for LaTeX rendering.

    Args:
        text: Input text that may contain underscores.

    Returns:
        Text with underscores escaped for LaTeX.
    """
    return text.replace("_", "\\_")


def _format_score_with_interval(
    mean_val: float, lower_val: float, upper_val: float, is_best: bool, decimals: int
) -> str:
    """Format a score with confidence interval for LaTeX table display.

    Args:
        mean_val: Mean score value.
        lower_val: Lower bound of confidence interval.
        upper_val: Upper bound of confidence interval.
        is_best: Whether this is the best score (for bold formatting).
        decimals: Number of digits after the decimal point.

    Returns:
        LaTeX-formatted string with score and interval.
    """
    mean_str = f"{mean_val:.{decimals}f}"
    interval_str = f"\\small{{[{lower_val:.{decimals}f}, {upper_val:.{decimals}f}]}}"

    if is_best:
        return f"\\normalsize{{\\textbf{{{mean_str}}}}} \\\\ {interval_str}"
    return f"\\normalsize{{{mean_str}}} \\\\ {interval_str}"


def _get_calibration_metrics_caption(rank_metrics: bool) -> str:
    """Get the caption text for calibration metrics tables.

    Args:
        rank_metrics: Whether the table reports ranks for every metric.

    Returns:
        LaTeX caption text describing calibration metrics analysis.
    """
    if rank_metrics:
        return (
            "Calibration performance rank by calibration metric. "
            "Chunked and global coverage deviation, McFadden's pseudo-$R^2$, and interval width "
            "are computed for intervals at 25\\%, 50\\% and 75\\% confidence on all LCbench datasets, "
            "then ranked across frameworks within each interval confidence and dataset. "
            "Individual ranks are then averaged by framework to demonstrate cross-confidence and cross-dataset performance."
        )
    return (
        "Calibration performance by metric. "
        "Chunked coverage deviation, global coverage deviation, and McFadden's pseudo-$R^2$ are averaged in native units "
        "across interval confidences and datasets. "
        "Interval width is ranked within each interval confidence and dataset, then averaged, "
        "because raw width does not share a scale across tasks."
    )


def _parse_and_group_entities(df_block: pd.DataFrame) -> dict:
    """Parse entity names and group by method and adapter.

    Args:
        df_block: DataFrame block containing tuner information.

    Returns:
        Dictionary grouping entities by method and adapter combinations.
    """
    grouped = {}

    for _, row in df_block.iterrows():
        tuner_name = row["tuner"]

        # Parse the tuner name to extract method and adapter
        if "unconformalized" in tuner_name.lower():
            method = "Unconformalized"
            adapter = "default"
        elif (
            "split conformalized" in tuner_name.lower()
            or "split_conformalized" in tuner_name.lower()
        ):
            method = "Split Conformalized"
            if "aci" in tuner_name.lower():
                if "dtaci" in tuner_name.lower():
                    adapter = "DtACI"
                else:
                    adapter = "ACI"
            else:
                adapter = "default"
        elif (
            "cross_conformalized" in tuner_name.lower()
            or "cross conformalized" in tuner_name.lower()
        ):
            method = "Cross Conformalized"
            if "aci" in tuner_name.lower():
                if "dtaci" in tuner_name.lower():
                    adapter = "DtACI"
                else:
                    adapter = "ACI"
            else:
                adapter = "default"
        else:
            # Default case - treat as is
            method = tuner_name
            adapter = "default"

        if method not in grouped:
            grouped[method] = {}

        grouped[method][adapter] = row

    return grouped


def _formatted_metric_minipage(
    row_data: pd.Series, metric: str, best_values: dict
) -> str:
    mean_col = metric
    lower_col = f"{metric}_lower"
    upper_col = f"{metric}_upper"
    if not all(col in row_data.index for col in [mean_col, lower_col, upper_col]):
        return "--"
    mean_val = row_data[mean_col]
    is_best = metric in best_values and mean_val == best_values[metric]
    formatted_metric = _format_score_with_interval(
        mean_val=mean_val,
        lower_val=row_data[lower_col],
        upper_val=row_data[upper_col],
        is_best=is_best,
        decimals=CALIBRATION_SCORE_DECIMALS,
    )
    return f"\\begin{{minipage}}{{3cm}}\\centering {formatted_metric} \\end{{minipage}}"


def _build_calibration_metrics_table_block(df_block: pd.DataFrame, caption: str) -> str:
    """Build a LaTeX table block for calibration metrics.

    Args:
        df_block: DataFrame containing calibration metrics data.
        caption: Caption text for the table.

    Returns:
        LaTeX table string for the calibration metrics.
    """
    target_metrics = [
        "global_target_coverage_deviation",
        "chunked_target_coverage_deviation",
        "mcfadden_r_squared",
        "width",
    ]
    metric_titles = {
        "global_target_coverage_deviation": "Global Target Coverage Deviation",
        "chunked_target_coverage_deviation": "Chunked Target Coverage Deviation",
        "mcfadden_r_squared": "McFadden $R^2$",
        "width": "Width",
    }

    available_metrics = []
    for metric in target_metrics:
        # Require mean aggregation to be under the original metric name only.
        mean_col = metric
        lower_col = f"{metric}_lower"
        upper_col = f"{metric}_upper"

        if all(col in df_block.columns for col in [mean_col, lower_col, upper_col]):
            available_metrics.append(metric)

    if not available_metrics:
        return ""

    # Find best (minimum) values for each metric to bold them
    best_values = {}
    for metric in available_metrics:
        best_values[metric] = df_block[metric].min()

    grouped_entities = _parse_and_group_entities(df_block)
    lines: List[str] = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\vspace{1em}",
        f"\\begin{{tabular}}{{@{{}}l*{{{len(available_metrics)}}}{{>{{\\centering\\arraybackslash}}p{{3cm}}}}@{{}}}}",
        "\\toprule",
    ]

    # Build header row
    header_parts = ["\\textbf{Entity}"]
    for metric in available_metrics:
        metric_title = metric_titles.get(metric, metric.replace("_", " ").title())
        header_parts.append(f"\\textbf{{{metric_title}}}")

    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")

    for method in CALIBRATION_METHOD_ORDER:
        if method not in grouped_entities:
            continue

        method_data = grouped_entities[method]

        # Add main method row (no adapter)
        if "default" in method_data:
            row_parts = [f"\\normalsize{{\\textbf{{{method}}}}}"]
            row_data = method_data["default"]

            for metric in available_metrics:
                row_parts.append(
                    _formatted_metric_minipage(row_data, metric, best_values)
                )

            lines.append(" & ".join(row_parts) + " \\\\")

        # Add adapter variants
        for adapter in sorted(method_data.keys()):
            if adapter == "default":
                continue

            row_parts = [f"\\normalsize{{\\quad + {adapter}}}"]
            row_data = method_data[adapter]

            for metric in available_metrics:
                row_parts.append(
                    _formatted_metric_minipage(row_data, metric, best_values)
                )

            lines.append(" & ".join(row_parts) + " \\\\")

        # Add spacing after each method group except the last
        if method != CALIBRATION_METHOD_ORDER[-1] and any(
            m in grouped_entities
            for m in CALIBRATION_METHOD_ORDER[
                CALIBRATION_METHOD_ORDER.index(method) + 1 :
            ]
        ):
            lines.append("")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\label{tab:calibration_metrics_by_entity}",
            "\\end{table}",
        ]
    )

    return "\n".join(lines)


def format_calibration_metrics_to_latex(
    results_df: pd.DataFrame,
    layout_breakout_col: Optional[str] = None,
    rank_metrics: bool = True,
) -> str:
    """Format calibration metrics results into LaTeX table format.

    Generates LaTeX tables showing calibration performance metrics with confidence
    intervals, optionally broken out by a specified column. Tables highlight the
    best performing methods and include proper LaTeX escaping.

    Args:
        results_df: DataFrame containing calibration metrics with columns for
            mean values, confidence intervals, and ranking information.
        layout_breakout_col: Optional column name to break the results into
            separate tables for each unique value in that column.
        rank_metrics: Whether the table reports ranks for every metric. Controls
            the caption; width is ranked in both table variants.

    Returns:
        LaTeX formatted string containing one or more tables with calibration metrics.
    """
    blocks: List[str] = []
    caption = _get_calibration_metrics_caption(rank_metrics)

    if layout_breakout_col and layout_breakout_col in results_df.columns:
        for l_val in sorted(results_df[layout_breakout_col].unique()):
            df_l = results_df[results_df[layout_breakout_col] == l_val]
            table_caption = f"{caption} - {_escape_latex_text(str(l_val))}"
            blocks.append(_build_calibration_metrics_table_block(df_l, table_caption))
    else:
        blocks.append(_build_calibration_metrics_table_block(results_df, caption))

    return "\n\n".join(blocks)
