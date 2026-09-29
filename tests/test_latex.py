import pandas as pd
from hpobench.report.latex import format_calibration_metrics_to_latex


def test_latex_bolds_minimum_value_in_each_column():
    """Each column bolds the minimum among all rows in the results frame."""
    results = pd.DataFrame(
        [
            {
                "tuner": "Unconformalized GP",
                "global_target_coverage_deviation": 0.20,
                "global_target_coverage_deviation_lower": 0.10,
                "global_target_coverage_deviation_upper": 0.30,
                "chunked_target_coverage_deviation": 0.20,
                "chunked_target_coverage_deviation_lower": 0.10,
                "chunked_target_coverage_deviation_upper": 0.30,
                "mcfadden_r_squared": 0.010,
                "mcfadden_r_squared_lower": 0.001,
                "mcfadden_r_squared_upper": 0.020,
                "width": 2.5,
                "width_lower": 2.0,
                "width_upper": 3.0,
            },
            {
                "tuner": "Cross conformalized LBS",
                "global_target_coverage_deviation": 0.10,
                "global_target_coverage_deviation_lower": 0.05,
                "global_target_coverage_deviation_upper": 0.15,
                "chunked_target_coverage_deviation": 0.10,
                "chunked_target_coverage_deviation_lower": 0.05,
                "chunked_target_coverage_deviation_upper": 0.15,
                "mcfadden_r_squared": 0.040,
                "mcfadden_r_squared_lower": 0.020,
                "mcfadden_r_squared_upper": 0.060,
                "width": 1.2,
                "width_lower": 1.0,
                "width_upper": 1.4,
            },
        ]
    )
    latex = format_calibration_metrics_to_latex(results, rank_metrics=False)
    unconformalized_block = latex.split("Unconformalized")[1].split(
        "Cross Conformalized"
    )[0]
    cross_block = latex.split("Cross Conformalized")[1]

    assert "\\textbf{0.100}" in cross_block
    assert "\\textbf{1.200}" in cross_block
    assert "\\textbf{0.010}" in unconformalized_block
    assert "\\textbf{0.200}" not in unconformalized_block
    assert "\\textbf{2.500}" not in unconformalized_block
    assert "\\textbf{0.040}" not in cross_block


def test_latex_bolds_lowest_width_rank():
    """Ranked width bolds the smallest rank among the methods in the frame."""
    results = pd.DataFrame(
        [
            {
                "tuner": "Unconformalized GP",
                "width": 2.4167,
                "width_lower": 2.1,
                "width_upper": 2.8,
                "global_target_coverage_deviation": 1.0,
                "global_target_coverage_deviation_lower": 1.0,
                "global_target_coverage_deviation_upper": 1.0,
                "chunked_target_coverage_deviation": 1.0,
                "chunked_target_coverage_deviation_lower": 1.0,
                "chunked_target_coverage_deviation_upper": 1.0,
                "mcfadden_r_squared": 2.0,
                "mcfadden_r_squared_lower": 2.0,
                "mcfadden_r_squared_upper": 2.0,
            },
            {
                "tuner": "Split conformalized LBS",
                "width": 1.8333,
                "width_lower": 1.5,
                "width_upper": 2.2,
                "global_target_coverage_deviation": 2.0,
                "global_target_coverage_deviation_lower": 2.0,
                "global_target_coverage_deviation_upper": 2.0,
                "chunked_target_coverage_deviation": 2.0,
                "chunked_target_coverage_deviation_lower": 2.0,
                "chunked_target_coverage_deviation_upper": 2.0,
                "mcfadden_r_squared": 1.0,
                "mcfadden_r_squared_lower": 1.0,
                "mcfadden_r_squared_upper": 1.0,
            },
        ]
    )
    latex = format_calibration_metrics_to_latex(results, rank_metrics=True)
    split_block = latex.split("Split Conformalized")[1]
    unconformalized_block = latex.split("Unconformalized")[1].split(
        "Split Conformalized"
    )[0]
    assert "\\textbf{1.833}" in split_block
    assert "\\textbf{2.417}" not in unconformalized_block
