from hpobench.report.metrics import (
    friedman_test_runner,
    nemenyi_pairwise_test,
    wilcoxon_pairwise_test,
    permutation_pairwise_test,
)
from hpobench.utils import generate_hyperparameter_combinations
import pandas as pd
import logging
import os
from typing import List, Optional, Literal

from hpobench.utils import save_analysis_results, AnalysisPathManager
from hpobench.process import block_bootstrap
from hpobench.report.latex import (
    format_calibration_metrics_to_latex,
)
from hpobench.report.metrics import calculate_calibration_statistics_per_repetition


def _save_text_content(
    content: str,
    cache_path: str,
    run_start_str: str,
    filename: str,
    analysis_type: str = None,
    subfolder: str = None,
):
    """Save text content (like LaTeX) with proper path organization.

    Args:
        content: Text content to save
        cache_path: Base cache path
        run_start_str: Run identifier
        filename: Name of the file
        analysis_type: Analysis type (e.g., "01_coverage_analysis")
        subfolder: Optional subfolder (e.g., "latex_outputs")
    """
    if content:
        if analysis_type:
            path_manager = AnalysisPathManager(cache_path, run_start_str)
            analysis_data_path = path_manager.get_analysis_path(
                analysis_type, "data", subfolder
            )
        else:
            # Fallback to old behavior for backward compatibility
            analysis_data_path = os.path.join(cache_path, "data", run_start_str)
            os.makedirs(analysis_data_path, exist_ok=True)

        full_filename = os.path.join(analysis_data_path, filename)
        try:
            with open(full_filename, "w", encoding="utf-8") as f:
                f.write(content)
            logging.getLogger(__name__).info(f"Saved text content to {full_filename}")
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Failed to save text content to {full_filename}: {e}", exc_info=True
            )
    else:
        logging.getLogger(__name__).warning(
            f"Skipping save for {filename}: Content is empty or None."
        )


def generate_configs_per_repetition(
    search_space,
    n_configs,
    n_repetitions,
    base_seed,
    objective_function,
    seed_offset=0,
):
    """Generate hyperparameter configurations for multiple experimental repetitions.

    Args:
        search_space: Dictionary defining the hyperparameter search space.
        n_configs: Number of configurations to generate per repetition.
        n_repetitions: Number of experimental repetitions.
        base_seed: Base random seed for reproducible generation.
        objective_function: Objective function for evaluating configurations.
        seed_offset: Offset to add to base seed for variation.

    Returns:
        List of configuration lists, one per repetition.
    """
    configs_per_repetition = []
    for repetition in range(n_repetitions):
        configs = []
        consistent_configs = generate_hyperparameter_combinations(
            params=search_space,
            n_combinations=n_configs,
            random_state=base_seed + seed_offset + repetition,
        )

        performances = objective_function.predict_batch(consistent_configs)
        for combination, performance in zip(consistent_configs, performances):
            configs.append((combination, performance))

        configs_per_repetition.append(configs)
    return configs_per_repetition


def run_and_save_friedman(
    data: pd.DataFrame,
    breakout_col: List[str],
    across_col: str,
    entity_col: str,
    rank_col: str,
    alpha: float,
    cache_path: str,
    run_start_str: str,
    filename: str,
    analysis_type: str,
    subfolder: str = "statistical_tests",
):
    """Run Friedman statistical test and save results.

    Args:
        data: DataFrame containing experimental data.
        breakout_col: List of columns for grouping the analysis.
        across_col: Column representing blocks/groups for the test.
        entity_col: Column representing entities/treatments being compared.
        rank_col: Column containing rank values for comparison.
        alpha: Significance level for the test.
        cache_path: Base directory for saving results.
        run_start_str: Unique identifier for this analysis run.
        filename: Name of the output file.
        analysis_type: Type of analysis for path organization.
        subfolder: Subfolder name for organizing outputs.
    """
    logging.getLogger(__name__)
    results_df = friedman_test_runner(
        data=data,
        breakout_col=breakout_col,
        across_col=across_col,
        entity_col=entity_col,
        rank_col=rank_col,
        alpha=alpha,
    )
    save_analysis_results(
        results_df,
        cache_path,
        run_start_str,
        filename,
        analysis_type,
        subfolder,
    )


def run_and_save_nemenyi(
    data: pd.DataFrame,
    breakout_col: List[str],
    across_col: str,
    entity_col: str,
    rank_col: str,
    alpha: float,
    cache_path: str,
    run_start_str: str,
    filename: str,
    analysis_type: str,
    subfolder: str = "statistical_tests",
) -> pd.DataFrame:
    logging.getLogger(__name__)
    results_df = nemenyi_pairwise_test(
        data=data,
        breakout_col=breakout_col,
        across_col=across_col,
        entity_col=entity_col,
        rank_col=rank_col,
        alpha=alpha,
    )
    save_analysis_results(
        results_df,
        cache_path,
        run_start_str,
        filename,
        analysis_type,
        subfolder,
    )
    return results_df


def run_and_save_wilcoxon(
    data: pd.DataFrame,
    breakout_col: List[str],
    across_col: str,
    entity_col: str,
    rank_col: str,
    alpha: float,
    cache_path: str,
    run_start_str: str,
    filename: str,
    analysis_type: str,
    subfolder: str = "statistical_tests",
    correction_method: Literal[
        "bonferroni-holm", "benjamini-hochberg"
    ] = "benjamini-hochberg",
) -> pd.DataFrame:
    """Run Wilcoxon signed-rank pairwise tests with multiple testing correction and save results.

    Args:
        data: DataFrame with rank data
        breakout_col: List of columns for grouping data
        across_col: Column for datasets (e.g., 'dataset')
        entity_col: Column for algorithms/entities (e.g., 'tuner')
        rank_col: Column with ranks (lower is better)
        alpha: Significance level
        cache_path: Base cache path
        run_start_str: Run identifier
        filename: CSV filename to save
        analysis_type: Analysis type for path organization
        subfolder: Subfolder for saving results
        correction_method: Multiple testing correction method to use

    Returns:
        DataFrame with pairwise comparison results
    """
    results_df = wilcoxon_pairwise_test(
        data=data,
        breakout_col=breakout_col,
        across_col=across_col,
        entity_col=entity_col,
        rank_col=rank_col,
        alpha=alpha,
        correction_method=correction_method,
    )
    save_analysis_results(
        results_df,
        cache_path,
        run_start_str,
        filename,
        analysis_type,
        subfolder,
    )
    return results_df


def run_and_save_permutation_test(
    data: pd.DataFrame,
    breakout_col: List[str],
    across_col: str,
    entity_col: str,
    rank_col: str,
    alpha: float,
    cache_path: str,
    run_start_str: str,
    filename: str,
    analysis_type: str,
    subfolder: str = "statistical_tests",
    n_permutations: int = 10000,
    random_state: Optional[int] = None,
    correction_method: Literal[
        "bonferroni-holm", "benjamini-hochberg"
    ] = "benjamini-hochberg",
) -> pd.DataFrame:
    """Run permutation tests with multiple testing correction and save results.

    Args:
        data: DataFrame with rank data
        breakout_col: List of columns for grouping data
        across_col: Column for datasets (e.g., 'dataset')
        entity_col: Column for algorithms/entities (e.g., 'tuner')
        rank_col: Column with ranks (lower is better)
        alpha: Significance level
        cache_path: Base cache path
        run_start_str: Run identifier
        filename: CSV filename to save
        analysis_type: Analysis type for path organization
        subfolder: Subfolder for saving results
        n_permutations: Number of permutations for the test
        random_state: Random seed for reproducible results
        correction_method: Multiple testing correction method to use

    Returns:
        DataFrame with pairwise comparison results
    """
    results_df = permutation_pairwise_test(
        data=data,
        breakout_col=breakout_col,
        across_col=across_col,
        entity_col=entity_col,
        rank_col=rank_col,
        alpha=alpha,
        n_permutations=n_permutations,
        random_state=random_state,
        correction_method=correction_method,
    )
    save_analysis_results(
        results_df,
        cache_path,
        run_start_str,
        filename,
        analysis_type,
        subfolder,
    )
    return results_df


def run_statistical_tests_for_budget(
    data: pd.DataFrame,
    budget: int,
    norm_runtime_unit: str,
    analysis_components: List[str],
    cd_significance_method: str,
    bench_col: str,
    data_col: str,
    tuner_col: str,
    alpha: float,
    cache_path: str,
    run_start_str: str,
    analysis_type: str,
    random_state: Optional[int] = None,
    subfolder: str = "statistical_tests",
    filename_prefix: str = "",
    correction_method: Literal[
        "bonferroni-holm", "benjamini-hochberg"
    ] = "benjamini-hochberg",
) -> Optional[pd.DataFrame]:
    """Run the configured statistical tests for a single budget slice.

    Only tests benchmarks that have at least 3 datasets. Benchmarks with
    insufficient datasets are skipped gracefully and logged.

    Args:
        data: DataFrame containing the benchmark data
        budget: Budget value to analyze
        norm_runtime_unit: Column name for normalized runtime
        analysis_components: List of statistical tests to run
        cd_significance_method: Method for critical difference analysis
        bench_col: Column name for benchmark identifier
        data_col: Column name for dataset identifier
        tuner_col: Column name for tuner identifier
        alpha: Significance level for tests
        cache_path: Path for saving results
        run_start_str: Run identifier string
        analysis_type: Type of analysis
        random_state: Random seed for reproducibility
        subfolder: Subfolder for saving results
        filename_prefix: Prefix for output filenames
        correction_method: Multiple testing correction method to use

    Returns:
        The pairwise results DataFrame to be used for critical-difference
        plotting if the chosen `cd_significance_method` produced results, otherwise
        returns None.
    """
    logger = logging.getLogger(__name__)
    cd_results: Optional[pd.DataFrame] = None

    # Check which benchmarks have sufficient datasets for significance testing
    datasets_per_benchmark = data.groupby(bench_col)[data_col].nunique()
    valid_benchmarks = datasets_per_benchmark[
        datasets_per_benchmark >= 3
    ].index.tolist()
    invalid_benchmarks = datasets_per_benchmark[
        datasets_per_benchmark < 3
    ].index.tolist()

    if invalid_benchmarks:
        logger.info(
            f"Skipping statistical tests for benchmarks with insufficient datasets "
            f"(budget={budget}): {invalid_benchmarks}. Dataset counts: "
            f"{datasets_per_benchmark[datasets_per_benchmark < 3].to_dict()}"
        )

    if not valid_benchmarks:
        logger.warning(
            f"No benchmarks have sufficient datasets for statistical testing at budget={budget}"
        )
        return None

    logger.info(
        f"Running statistical tests for benchmarks with sufficient datasets "
        f"(budget={budget}): {valid_benchmarks}"
    )

    filtered_data = data[data[bench_col].isin(valid_benchmarks)]
    tuner_counts_per_benchmark = filtered_data.groupby(bench_col)[tuner_col].nunique()

    if (tuner_counts_per_benchmark > 3).all():
        if "friedman" in analysis_components:
            run_and_save_friedman(
                data=filtered_data,
                breakout_col=[bench_col],
                across_col=data_col,
                entity_col=tuner_col,
                rank_col="rank",
                alpha=alpha,
                cache_path=cache_path,
                run_start_str=run_start_str,
                filename=f"{filename_prefix}friedman_test_budget_{budget}.csv",
                analysis_type=analysis_type,
                subfolder=subfolder,
            )
    else:
        failing_benchmarks = tuner_counts_per_benchmark[tuner_counts_per_benchmark <= 3]
        logger.info(
            f"Skipping Friedman test for budget={budget} because some benchmarks "
            f"have 3 or fewer unique tuners. Failing benchmarks and their tuner counts: "
            f"{failing_benchmarks.to_dict()}"
        )

    if "nemenyi" in analysis_components:
        results_df = run_and_save_nemenyi(
            data=filtered_data,
            breakout_col=[bench_col],
            across_col=data_col,
            entity_col=tuner_col,
            rank_col="rank",
            alpha=alpha,
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename=f"{filename_prefix}nemenyi_pairwise_budget_{budget}.csv",
            analysis_type=analysis_type,
            subfolder=subfolder,
        )
        results_df[norm_runtime_unit] = budget
        if cd_significance_method == "nemenyi":
            cd_results = results_df

    if "wilcoxon" in analysis_components:
        results_df = run_and_save_wilcoxon(
            data=filtered_data,
            breakout_col=[bench_col],
            across_col=data_col,
            entity_col=tuner_col,
            rank_col="rank",
            alpha=alpha,
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename=f"{filename_prefix}wilcoxon_pairwise_budget_{budget}.csv",
            analysis_type=analysis_type,
            subfolder=subfolder,
            correction_method=correction_method,
        )
        results_df[norm_runtime_unit] = budget
        if cd_significance_method == "wilcoxon":
            cd_results = results_df

    if "permutation_test" in analysis_components:
        results_df = run_and_save_permutation_test(
            data=filtered_data,
            breakout_col=[bench_col],
            across_col=data_col,
            entity_col=tuner_col,
            rank_col="rank",
            alpha=alpha,
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename=f"{filename_prefix}permutation_pairwise_budget_{budget}.csv",
            analysis_type=analysis_type,
            subfolder=subfolder,
            random_state=random_state,
            correction_method=correction_method,
        )
        results_df[norm_runtime_unit] = budget
        if cd_significance_method == "permutation_test":
            cd_results = results_df

    logger.info(
        "Completed statistical tests for budget=%s; cd_method=%s; valid_benchmarks=%s",
        budget,
        cd_significance_method,
        valid_benchmarks,
    )
    return cd_results


def aggregate_and_save(
    data: pd.DataFrame,
    grouping_cols: List[str],
    breakout_cols: str,
    block_cols: List[str],
    metrics: List[str],
    cache_path: str,
    run_start_str: str,
    filename: str,
    analysis_type: str,
) -> pd.DataFrame:
    aggregated_results = block_bootstrap(
        data=data,
        breakout_cols=breakout_cols,
        block_cols=block_cols,
        aggregators=grouping_cols,
        metric_cols=metrics,
        n_bootstraps=1000,
    )

    save_analysis_results(
        aggregated_results,
        cache_path,
        run_start_str,
        filename,
        analysis_type,
        "aggregated_results",
    )
    return aggregated_results


def run_and_save_calibration_statistics(
    raw_benchmark_data: pd.DataFrame,
    aggregators: List[str],
    benchmark_col: str,
    tuner_column: str,
    breach_column: str,
    dataset_column: str,
    entity_column: str,
    budget_unit: str,
    cache_path: str,
    run_start_str: str,
    filename: str,
    analysis_type: str,
    n_bootstraps: int = 1000,
    latex_layout_breakout_col: Optional[str] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calculate calibration statistics and optionally generate LaTeX output.

    Args:
        raw_benchmark_data: Raw benchmark data
        aggregators: List of columns to aggregate by
        repetition_column: Column name for repetitions
        breach_column: Column name for binary breach indicators
        cache_path: Base cache path
        run_start_str: Run identifier
        filename: CSV filename to save
        analysis_type: Analysis type for path organization
        subfolder: Subfolder for saving results
        latex_layout_breakout_col: Optional column for LaTeX layout breakout
        random_state: Random state for reproducibility

    Returns:
        DataFrame with calibration statistics
    """
    metric_columns = [
        "winkler_score",
        "width",
        "miscoverage_penalty",
        "chunked_target_coverage_deviation",
        "llr_statistic",
    ]

    logger = logging.getLogger(__name__)
    # NOTE: These calculations don't align or filter the raw becnhmark data like in the
    # process.py module, this is fine since we always benchmark on the same number of iterations
    # and this function only gets passed the iteration benchmark data, but align in future (TODO).

    for rank_metrics in [True, False]:
        calibration_stats = calculate_calibration_statistics_per_repetition(
            raw_benchmark_data=raw_benchmark_data,
            aggregators=aggregators,
            breach_column=breach_column,
            entity_column=entity_column,
            metric_columns=metric_columns,
            budget_unit=budget_unit,
            random_state=random_state,
            rank_metrics=rank_metrics,
        )
        collapsed_calibration_stats = block_bootstrap(
            data=calibration_stats,
            breakout_cols=[benchmark_col],
            block_cols=[dataset_column],
            aggregators=[benchmark_col, tuner_column],
            metric_cols=metric_columns,
            n_bootstraps=n_bootstraps,
        )
        # Generate LaTeX table for calibration metrics
        latex_metrics_str = format_calibration_metrics_to_latex(
            collapsed_calibration_stats,
            layout_breakout_col=latex_layout_breakout_col,
        )

        if latex_metrics_str:
            latex_metrics_filename = f"{filename.replace('.csv', '')}_metrics_latex__ranked_{rank_metrics}.tex"
            _save_text_content(
                latex_metrics_str,
                cache_path,
                run_start_str,
                latex_metrics_filename,
                analysis_type,
                "latex_outputs",
            )
            logger.info("Generated LaTeX table for calibration metrics by entity")
