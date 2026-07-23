import pandas as pd
from typing import List, Optional


def _escape_latex_text(text: str) -> str:
    """Escape underscores in text for LaTeX rendering.

    Args:
        text: Input text that may contain underscores.

    Returns:
        Text with underscores escaped for LaTeX.
    """
    return text.replace("_", "\\_")


def _format_score_with_interval(
    mean_val: float, lower_val: float, upper_val: float, is_best: bool
) -> str:
    """Format a score with confidence interval for LaTeX table display.

    Args:
        mean_val: Mean score value.
        lower_val: Lower bound of confidence interval.
        upper_val: Upper bound of confidence interval.
        is_best: Whether this is the best score (for bold formatting).

    Returns:
        LaTeX-formatted string with score and interval.
    """
    mean_str = f"{mean_val:.3f}"
    interval_str = f"\\small{{[{lower_val:.3f}, {upper_val:.3f}]}}"

    if is_best:
        return f"\\normalsize{{\\textbf{{{mean_str}}}}} \\\\ {interval_str}"
    return f"\\normalsize{{{mean_str}}} \\\\ {interval_str}"


def _get_calibration_metrics_caption() -> str:
    """Get the standard caption text for calibration metrics tables.

    Returns:
        LaTeX caption text describing calibration metrics analysis.
    """
    return (
        "Calibration performance rank by calibration metric. "
        "Metrics are computed for intervals at 25\\%, 50\\% and 75\\% confidence on all LCbench datasets, "
        "then ranked across frameworks within each interval confidence and dataset. "
        "Individual ranks are then averaged by framework to demonstrate cross-confidence and cross-dataset performance."
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


def _build_calibration_metrics_table_block(df_block: pd.DataFrame, caption: str) -> str:
    """Build a LaTeX table block for calibration metrics.

    Args:
        df_block: DataFrame containing calibration metrics data.
        caption: Caption text for the table.

    Returns:
        LaTeX table string for the calibration metrics.
    """
    target_metrics = ["chunked_target_coverage_deviation", "llr_statistic", "width"]

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
        # Use the original metric name for the mean column.
        mean_col = metric
        best_values[metric] = df_block[mean_col].min()

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
        metric_title = metric.replace("_", " ").title()
        header_parts.append(f"\\textbf{{{metric_title}}}")

    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")

    # Parse and group entities
    grouped_entities = _parse_and_group_entities(df_block)

    # Define method order
    method_order = ["Unconformalized", "Split Conformalized", "Cross Conformalized"]

    for method in method_order:
        if method not in grouped_entities:
            continue

        method_data = grouped_entities[method]

        # Add main method row (no adapter)
        if "default" in method_data:
            row_parts = [f"\\normalsize{{\\textbf{{{method}}}}}"]
            row_data = method_data["default"]

            for metric in available_metrics:
                mean_col = metric
                lower_col = f"{metric}_lower"
                upper_col = f"{metric}_upper"

                if all(
                    col in row_data.index for col in [mean_col, lower_col, upper_col]
                ):
                    mean_val = row_data[mean_col]
                    lower_val = row_data[lower_col]
                    upper_val = row_data[upper_col]

                    # Check if this is the best value for this metric
                    is_best = (
                        metric in best_values
                        and abs(mean_val - best_values[metric]) < 1e-10
                    )

                    formatted_metric = _format_score_with_interval(
                        mean_val, lower_val, upper_val, is_best
                    )
                    row_parts.append(
                        f"\\begin{{minipage}}{{3cm}}\\centering {formatted_metric} \\end{{minipage}}"
                    )
                else:
                    row_parts.append("--")

            lines.append(" & ".join(row_parts) + " \\\\")

        # Add adapter variants
        for adapter in sorted(method_data.keys()):
            if adapter == "default":
                continue

            row_parts = [f"\\normalsize{{\\quad + {adapter}}}"]
            row_data = method_data[adapter]

            for metric in available_metrics:
                mean_col = metric
                lower_col = f"{metric}_lower"
                upper_col = f"{metric}_upper"

                if all(
                    col in row_data.index for col in [mean_col, lower_col, upper_col]
                ):
                    mean_val = row_data[mean_col]
                    lower_val = row_data[lower_col]
                    upper_val = row_data[upper_col]

                    # Check if this is the best value for this metric
                    is_best = (
                        metric in best_values
                        and abs(mean_val - best_values[metric]) < 1e-10
                    )

                    formatted_metric = _format_score_with_interval(
                        mean_val, lower_val, upper_val, is_best
                    )
                    row_parts.append(
                        f"\\begin{{minipage}}{{3cm}}\\centering {formatted_metric} \\end{{minipage}}"
                    )
                else:
                    row_parts.append("--")

            lines.append(" & ".join(row_parts) + " \\\\")

        # Add spacing after each method group except the last
        if method != method_order[-1] and any(
            m in grouped_entities
            for m in method_order[method_order.index(method) + 1 :]
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

    Returns:
        LaTeX formatted string containing one or more tables with calibration metrics.
    """
    blocks: List[str] = []
    caption = _get_calibration_metrics_caption()

    if layout_breakout_col and layout_breakout_col in results_df.columns:
        for l_val in sorted(results_df[layout_breakout_col].unique()):
            df_l = results_df[results_df[layout_breakout_col] == l_val]
            table_caption = f"{caption} - {_escape_latex_text(str(l_val))}"
            blocks.append(_build_calibration_metrics_table_block(df_l, table_caption))
    else:
        blocks.append(_build_calibration_metrics_table_block(results_df, caption))

    return "\n\n".join(blocks)
