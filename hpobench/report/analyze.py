import pandas as pd
import logging
from typing import List, Literal, Optional
from hpobench.utils import AnalysisPathManager
from hpobench.config.schema import BenchmarkDataSchema
from hpobench.utils import save_analysis_results
from hpobench.plot import (
    plot_and_save,
    plot_paired_rank_and_cd,
    plot_joint_architecture_and_static,
)
from hpobench.process import (
    BenchmarkDataProcessor,
    rank_and_collapse_data,
)

from hpobench.report.utils import (
    run_statistical_tests_for_budget,
    aggregate_and_save,
    run_and_save_calibration_statistics,
)

logger = logging.getLogger(__name__)


def create_default_plotting_identifier(
    df: pd.DataFrame,
    tuner_col: str,
    estimator_architecture_col: str,
    sampler_col: str,
) -> pd.DataFrame:
    """Create plotting_identifier column with conditional logic for tuner fallback.

    Args:
        df: DataFrame to add plotting_identifier column to
        tuner_col: Column name for tuner identifier
        estimator_architecture_col: Column name for estimator architecture
        sampler_col: Column name for sampler

    Returns:
        DataFrame with plotting_identifier column added
    """
    df_copy = df.copy()
    df_copy["plotting_identifier"] = df_copy.apply(
        lambda row: row[tuner_col]
        if (
            (
                row[estimator_architecture_col] == ""
                or row[estimator_architecture_col] is None
            )
            and (row[sampler_col] == "" or row[sampler_col] is None)
        )
        else f"{row[estimator_architecture_col]}-{row[sampler_col]}",
        axis=1,
    )
    return df_copy


def compute_significance_results(
    benchmark_data: pd.DataFrame,
    norm_unit: str,
    tuner_col: str,
    estimator_architecture_col: str,
    sampler_col: str,
    analysis_components: List[str],
    cd_significance_method: Literal["nemenyi", "wilcoxon", "permutation_test"],
    bench_col: str,
    data_col: str,
    alpha: float,
    cache_path: str,
    run_start_str: str,
    analysis_type: str,
    correction_method: str,
    filename_prefix: str,
    is_global: bool = False,
) -> dict:
    """Helper function to compute significance results for a single dataset type.

    Args:
        dataset_results: Dataset-level relative results
        norm_unit: Normalized budget unit column name
        tuner_col: Column name for tuner identifier
        estimator_architecture_col: Column name for estimator architecture
        sampler_col: Column name for sampler
        analysis_components: List of analysis components
        cd_significance_method: Method for CD significance testing
        bench_col: Column name for benchmark identifier
        data_col: Column name for dataset identifier
        alpha: Significance level
        cache_path: Path for saving results
        run_start_str: Run identifier string
        analysis_type: Type of analysis
        correction_method: Multiple testing correction method
        filename_prefix: Prefix for output filenames
        is_global: Whether this is global dataset analysis

    Returns:
        Dictionary mapping budget to significance results
    """
    benchmark_data = create_default_plotting_identifier(
        df=benchmark_data,
        tuner_col=tuner_col,
        estimator_architecture_col=estimator_architecture_col,
        sampler_col=sampler_col,
    )

    significance_results = {}
    for budget in [50, 100]:
        budget_data = benchmark_data[benchmark_data[norm_unit] == budget]

        # Check dataset sufficiency based on analysis type
        if is_global:
            if budget_data[data_col].nunique() < 3:
                logger.info(
                    f"Skipping statistical tests for budget={budget}: no benchmark has at least 3 datasets."
                )
                continue
        else:
            datasets_per_benchmark = budget_data.groupby(bench_col)[data_col].nunique()
            benchmarks_with_sufficient_datasets = (datasets_per_benchmark >= 3).sum()
            if benchmarks_with_sufficient_datasets == 0:
                logger.info(
                    f"Skipping statistical tests for budget={budget}: no benchmark has at least 3 datasets. "
                    f"Dataset counts per benchmark: {datasets_per_benchmark.to_dict()}"
                )
                continue

        # Configure parameters based on analysis type
        if is_global:
            components = list(set(analysis_components + ["wilcoxon"]))
            prefix = f"global_{filename_prefix}"
        else:
            components = analysis_components
            prefix = filename_prefix

        cd_df = run_statistical_tests_for_budget(
            data=budget_data,
            budget=budget,
            norm_runtime_unit=norm_unit,
            analysis_components=components,
            cd_significance_method=cd_significance_method,
            bench_col=bench_col,
            data_col=data_col,
            tuner_col="plotting_identifier",
            alpha=alpha,
            cache_path=cache_path,
            run_start_str=run_start_str,
            analysis_type=analysis_type,
            random_state=42,
            filename_prefix=prefix,
            correction_method=correction_method,
        )
        if cd_df is not None:
            significance_results[budget] = cd_df

    return significance_results


def plot_cd_diagram(
    benchmark_data: pd.DataFrame,
    significance_results: dict,
    norm_unit: str,
    tuner_col: str,
    estimator_architecture_col: str,
    sampler_col: str,
    bench_col: str,
    alpha: float,
    cache_path: str,
    run_start_str: str,
    analysis_type: str,
    filename_suffix: str,
    is_global: bool = False,
    cd_budget: int = 100,
):
    """Helper function to plot CD diagram for a single result set.

    Args:
        bench_results: Benchmark-level relative results
        significance_results: Significance results dictionary
        norm_unit: Normalized budget unit column name
        tuner_col: Column name for tuner identifier
        estimator_architecture_col: Column name for estimator architecture
        sampler_col: Column name for sampler
        bench_col: Column name for benchmark identifier
        alpha: Significance level
        cache_path: Path for saving results
        run_start_str: Run identifier string
        analysis_type: Type of analysis
        filename_suffix: Suffix for output filenames
        is_global: Whether this is global analysis
        cd_budget: Budget value to use for CD analysis
    """
    if cd_budget not in significance_results:
        scope = "global " if is_global else ""
        logger.info(
            f"Skipping {scope}CD plot for budget={cd_budget}: no significance results available"
        )
        return

    bench_results_with_id = create_default_plotting_identifier(
        df=benchmark_data,
        tuner_col=tuner_col,
        estimator_architecture_col=estimator_architecture_col,
        sampler_col=sampler_col,
    )

    global_prefix = "global_" if is_global else ""
    filename_prefix = f"rank_vs_norm_{filename_suffix}_with_{global_prefix}cd"

    plot_paired_rank_and_cd(
        data=bench_results_with_id,
        significance_data=significance_results[cd_budget],
        x_col=norm_unit,
        entity_col="plotting_identifier",
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename_prefix=filename_prefix,
        analysis_type=analysis_type,
        subfolder="rank_analysis",
        row_measure=bench_col,
        cd_budget=cd_budget,
        alpha=alpha,
        x_label="% Budget Used",
        y_col_lower=None,
        y_col_upper=None,
        significance_plot_type="matrix",
    )


def analyze_main_benchmark(
    raw_benchmark_data: pd.DataFrame,
    cache_path: str,
    run_start_str: str,
    analysis_type: str,
    analysis_components: List[
        Literal[
            "friedman",
            "nemenyi",
            "wilcoxon",
            "permutation_test",
            "coverage",
            "dataset_performances",
            "rank_analysis",
            "sampler_comparison",
            "architecture_comparison",
            "conformalization_effect",
            "quantile_count_comparison",
            "search_tuning_effect_comparison",
            "num_candidates_comparison",
        ]
    ],
    schema: BenchmarkDataSchema,
    alpha: float = 0.05,
    starting_coverage_trial: Optional[int] = None,
    cd_significance_method: Literal[
        "nemenyi", "wilcoxon", "permutation_test"
    ] = "permutation_test",
    n_bootstraps: int = 1000,
    correction_method: Literal[
        "bonferroni-holm", "benjamini-hochberg"
    ] = "benjamini-hochberg",
):
    """Analyze HPO benchmark results with comprehensive statistical and visual analysis.

    Performs multi-faceted analysis of hyperparameter optimization benchmark data including
    statistical significance testing, ranking analysis, coverage assessment, and performance
    comparisons across different tuning configurations, samplers, and estimator architectures.
    Generates both statistical results and visualization plots for each analysis component.

    The function processes raw benchmark data through multiple analytical lenses:
    - Statistical tests (Friedman, Nemenyi) to assess tuner performance differences
    - Coverage analysis for conformal prediction breach rates
    - Rank-based performance analysis across runtime and iteration budgets
    - Architecture and sampler comparison breakdowns
    - Conformalization effect assessment for conformal vs non-conformal methods

    Args:
        raw_benchmark_data: DataFrame containing benchmark results.
        cache_path: Root directory path for saving analysis outputs and plots.
        run_start_str: Timestamp string identifying this experimental run for file organization.
        analysis_type: Category label for analysis (e.g., "coverage_analysis", "sampler_variation").
        alpha: Significance level for statistical tests. Defaults to 0.05.
        starting_coverage_trial: Optional starting trial for coverage analysis.
        cd_significance_method: Method for critical difference diagrams ("nemenyi", "wilcoxon", or "permutation_test").
        correction_method: Multiple testing correction method ("bonferroni-holm" or "benjamini-hochberg").
        analysis_components: List of analysis types to execute. Valid options:
            - "friedman": Friedman test for overall statistical significance
            - "nemenyi": Nemenyi post-hoc test for pairwise comparisons
            - "wilcoxon": Wilcoxon signed-rank test with multiple testing correction for pairwise comparisons
            - "permutation_test": Permutation test with multiple testing correction for pairwise comparisons
            - "coverage": Coverage breach rate analysis for conformal prediction
            - "dataset_performances": Per-dataset performance trajectory plots
            - "rank_analysis": Ranking evolution across runtime and iteration budgets
            - "sampler_comparison": Performance comparison partitioned by sampler
            - "architecture_comparison": Performance comparison by estimator architecture
            - "conformalization_effect": Conformal vs non-conformal method comparison
            - "quantile_count_comparison": Performance comparison across different quantile counts by architecture
            - "search_tuning_effect_comparison": Performance comparison of searcher tuning framework effects by architecture
            - "num_candidates_comparison": Performance comparison across different numbers of candidates by architecture

    Side Effects:
        - Generates and saves statistical test results as CSV files
        - Creates performance visualization plots in organized subdirectories
        - Logs analysis progress and completion status
        - Saves aggregated results for different budget cross-sections

    Note:
        Coverage analysis is only performed for single-dataset benchmarks to ensure
        meaningful coverage rate calculations. Multi-dataset benchmarks will skip
        coverage components with a warning message.
    """
    rep_col = schema.rep_col
    tuner_col = schema.tuner_col
    bench_col = schema.bench_col
    data_col = schema.data_col
    sampler_col = schema.sampler_col
    confidence_level_col = schema.confidence_level_col
    estimator_architecture_col = schema.estimator_architecture_col
    runtime_unit = schema.runtime_unit
    iter_unit = schema.iter_unit
    norm_runtime_unit = schema.norm_runtime_unit
    norm_iter_unit = schema.norm_iter_unit
    breach_col = schema.breach_col
    n_pre_conformal_trials_col = schema.n_pre_conformal_trials_col
    n_quantiles_col = schema.sampler_n_quantiles_col
    n_candidates_col = schema.n_candidates_col
    searcher_tuning_framework_col = schema.tuner_searcher_tuning_framework_col

    processor = BenchmarkDataProcessor(schema=schema)

    raw_benchmark_data[data_col] = raw_benchmark_data[data_col].astype(str)

    # 1. Create broad-use processed data:
    # 1.1 Dataset-level relativized runtime results:
    dataset_relative_runtime_results = processor.process_performance_records(
        raw_benchmark_data=raw_benchmark_data,
        budget_unit=runtime_unit,
        relativize_budget=True,
        collapse_repetitions=True,
        collapse_datasets=False,
        extra_ranking_cols=None,
        n_bootstraps=n_bootstraps,
    )
    save_analysis_results(
        df=dataset_relative_runtime_results,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="dataset_relative_runtime_results.csv",
        analysis_type=analysis_type,
    )

    # 1.2 Dataset-level iterative results:
    dataset_absolute_iterative_results = processor.process_performance_records(
        raw_benchmark_data=raw_benchmark_data,
        budget_unit=iter_unit,
        relativize_budget=False,
        collapse_repetitions=True,
        collapse_datasets=False,
        extra_ranking_cols=None,
        n_bootstraps=n_bootstraps,
    )
    save_analysis_results(
        df=dataset_absolute_iterative_results,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="dataset_absolute_iterative_results.csv",
        analysis_type=analysis_type,
    )

    # 1.2.1 Dataset-level relativized iterative results:
    dataset_relative_iterative_results = processor.process_performance_records(
        raw_benchmark_data=raw_benchmark_data,
        budget_unit=iter_unit,
        relativize_budget=True,
        collapse_repetitions=True,
        collapse_datasets=False,
        extra_ranking_cols=None,
        n_bootstraps=n_bootstraps,
    )
    save_analysis_results(
        df=dataset_relative_iterative_results,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="dataset_relative_iterative_results.csv",
        analysis_type=analysis_type,
    )

    # 1.3 Benchmark-level relativized runtime results:
    bench_relative_runtime_results = processor.process_performance_records(
        raw_benchmark_data=raw_benchmark_data,
        budget_unit=runtime_unit,
        relativize_budget=True,
        collapse_repetitions=True,
        collapse_datasets=True,
        extra_ranking_cols=None,
        n_bootstraps=n_bootstraps,
    )
    save_analysis_results(
        df=bench_relative_runtime_results,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="bench_relative_runtime_results.csv",
        analysis_type=analysis_type,
    )

    # 1.4 Benchmark-level relative iterative results:
    bench_relative_iterative_results = processor.process_performance_records(
        raw_benchmark_data=raw_benchmark_data,
        budget_unit=iter_unit,
        relativize_budget=True,
        collapse_repetitions=True,
        collapse_datasets=True,
        extra_ranking_cols=None,
        n_bootstraps=n_bootstraps,
    )
    save_analysis_results(
        df=bench_relative_iterative_results,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="bench_relative_iterative_results.csv",
        analysis_type=analysis_type,
    )

    # 1.5 Benchmark-level iterative results:
    bench_absolute_iterative_results = processor.process_performance_records(
        raw_benchmark_data=raw_benchmark_data,
        budget_unit=iter_unit,
        relativize_budget=False,
        collapse_repetitions=True,
        collapse_datasets=True,
        extra_ranking_cols=None,
        n_bootstraps=n_bootstraps,
    )
    save_analysis_results(
        df=bench_absolute_iterative_results,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="bench_absolute_iterative_results.csv",
        analysis_type=analysis_type,
    )

    # 1.6 Cross benchmark relative runtime results:
    global_raw_benchmark_data = raw_benchmark_data.copy()
    global_raw_benchmark_data[data_col] = (
        global_raw_benchmark_data[bench_col] + "_" + global_raw_benchmark_data[data_col]
    )
    unique_benchmarks = sorted(global_raw_benchmark_data[bench_col].unique())
    global_raw_benchmark_data[bench_col] = " + ".join(unique_benchmarks)
    global_bench_relative_runtime_results = processor.process_performance_records(
        raw_benchmark_data=global_raw_benchmark_data,
        budget_unit=runtime_unit,
        relativize_budget=True,
        collapse_repetitions=True,
        collapse_datasets=True,
        extra_ranking_cols=None,
        n_bootstraps=n_bootstraps,
    )

    # 1.7 Cross benchmark relative iterative results:
    global_bench_relative_iterative_results = processor.process_performance_records(
        raw_benchmark_data=global_raw_benchmark_data,
        budget_unit=iter_unit,
        relativize_budget=True,
        collapse_repetitions=True,
        collapse_datasets=True,
        extra_ranking_cols=None,
        n_bootstraps=n_bootstraps,
    )

    # 1.8 Cross benchmark dataset relative runtime results:
    global_dataset_relative_runtime_results = processor.process_performance_records(
        raw_benchmark_data=global_raw_benchmark_data,
        budget_unit=runtime_unit,
        relativize_budget=True,
        collapse_repetitions=True,
        collapse_datasets=False,
        extra_ranking_cols=None,
        n_bootstraps=n_bootstraps,
    )

    # 1.9 Cross benchmark dataset relative iterative results:
    global_dataset_relative_iterative_results = processor.process_performance_records(
        raw_benchmark_data=global_raw_benchmark_data,
        budget_unit=iter_unit,
        relativize_budget=True,
        collapse_repetitions=True,
        collapse_datasets=False,
        extra_ranking_cols=None,
        n_bootstraps=n_bootstraps,
    )

    # 2. Carry out component analysis:
    # 2.2 Coverage analysis plots:
    if "coverage" in analysis_components:
        if starting_coverage_trial is not None:
            raw_benchmark_data_adj = raw_benchmark_data[
                raw_benchmark_data[iter_unit] >= starting_coverage_trial
            ]
        else:
            raw_benchmark_data_adj = raw_benchmark_data.copy()

        if raw_benchmark_data_adj[bench_col].nunique() == 1:
            dataset_absolute_iterative_results_adj = (
                processor.process_performance_records(
                    raw_benchmark_data=raw_benchmark_data_adj,
                    budget_unit=iter_unit,
                    relativize_budget=False,
                    collapse_repetitions=True,
                    collapse_datasets=False,
                    extra_ranking_cols=[confidence_level_col],
                    n_bootstraps=n_bootstraps,
                )
            )
            plot_and_save(
                data=dataset_absolute_iterative_results_adj,
                x_col=iter_unit,
                y_cols=["cumulative_coverage_error", "rolling_coverage_error"],
                entity_col=tuner_col,
                col_measure=confidence_level_col,
                row_measure=data_col,
                cache_path=cache_path,
                run_start_str=run_start_str,
                filename_prefix="coverage_per_dataset",
                analysis_type=analysis_type,
                subfolder="coverage_breach_rates",
                y_cols_lower=None,
                y_cols_upper=None,
                share_y_axis=False,
                col_measure_label="Confidence Level",
                hide_col_and_row_labels=False,
            )

            # Snapshot plot (pick one dataset, and only show CV+ vs. Unconformalized):
            # NOTE: All hardcoded
            if "7593" in dataset_absolute_iterative_results_adj[data_col].unique():
                snapshot_data = dataset_absolute_iterative_results_adj[
                    dataset_absolute_iterative_results_adj[data_col] == "7593"
                ]
            else:
                random_data_col_value = (
                    dataset_absolute_iterative_results_adj[data_col]
                    .sample(n=1, random_state=42)
                    .iloc[0]
                )
                snapshot_data = dataset_absolute_iterative_results_adj[
                    dataset_absolute_iterative_results_adj[data_col]
                    == random_data_col_value
                ]
            snapshot_data = snapshot_data[
                ~snapshot_data[tuner_col].str.contains("Split")
            ]
            plot_and_save(
                data=snapshot_data,
                x_col=iter_unit,
                y_cols=["cumulative_coverage_error", "rolling_coverage_error"],
                entity_col=tuner_col,
                col_measure=confidence_level_col,
                row_measure=data_col,
                cache_path=cache_path,
                run_start_str=run_start_str,
                filename_prefix="coverage_per_dataset_snapshot",
                analysis_type=analysis_type,
                subfolder="coverage_breach_rates",
                y_cols_lower=None,
                y_cols_upper=None,
                share_y_axis=False,
                col_measure_label="Confidence Level",
                hide_col_and_row_labels=False,
            )

        else:
            # TODO: Change so this loops through each benchmark slice like in later sections:
            logger.warning(
                "Skipping coverage plots: can only plot configurations with a single benchmark."
            )

        # TODO: This function is a bit archaic and not in line with the processor in process.py
        # you must ensure that the entity_column uniquely identifies each variant to rank across:
        raw_benchmark_data_filtered = raw_benchmark_data_adj[
            ~(raw_benchmark_data_adj[breach_col].isna())
        ]
        run_and_save_calibration_statistics(
            raw_benchmark_data=raw_benchmark_data_filtered,
            aggregators=[
                bench_col,
                data_col,
                tuner_col,
                rep_col,
                sampler_col,
                confidence_level_col,
                estimator_architecture_col,
            ],
            benchmark_col=bench_col,
            tuner_column=tuner_col,
            breach_column="breach_status",
            dataset_column=data_col,
            entity_column=tuner_col,
            budget_unit=iter_unit,
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename="calibration_statistics.csv",
            analysis_type=analysis_type,
            latex_layout_breakout_col=None,  # Can be modified to include estimator_architecture if needed
            n_bootstraps=n_bootstraps,
            random_state=1234,
        )

    # Dataset level analysis:
    if "dataset_performances" in analysis_components:
        for benchmark in dataset_absolute_iterative_results[bench_col].unique():
            benchmark_data = dataset_absolute_iterative_results[
                dataset_absolute_iterative_results[bench_col] == benchmark
            ]
            plot_and_save(
                data=benchmark_data,
                x_col=iter_unit,
                y_cols=["best_performance", "rank"],
                entity_col=tuner_col,
                col_measure=data_col,
                row_measure=bench_col,
                cache_path=cache_path,
                run_start_str=run_start_str,
                filename_prefix=f"perf_vs_iter__{benchmark}",
                analysis_type=analysis_type,
                subfolder="dataset_performances",
                y_cols_lower=None,
                y_cols_upper=None,
                share_y_axis=False,
                hide_col_and_row_labels=False,
            )

    # Rank analysis:
    if "rank_analysis" in analysis_components:
        bench_relative_runtime_results_filled_bounds = (
            bench_relative_runtime_results.copy()
        )
        bench_relative_runtime_results_filled_bounds[
            "rank_lower"
        ] = bench_relative_runtime_results_filled_bounds["rank_lower"].fillna(
            bench_relative_runtime_results_filled_bounds["rank"]
        )
        bench_relative_runtime_results_filled_bounds = (
            create_default_plotting_identifier(
                df=bench_relative_runtime_results_filled_bounds,
                tuner_col=tuner_col,
                estimator_architecture_col=estimator_architecture_col,
                sampler_col=sampler_col,
            )
        )
        plot_and_save(
            data=bench_relative_runtime_results_filled_bounds,
            x_col=norm_runtime_unit,
            y_cols=["rank"],
            entity_col="plotting_identifier",
            col_measure=bench_col,
            row_measure=None,
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename_prefix="rank_vs_norm_runtime",
            analysis_type=analysis_type,
            subfolder="rank_analysis",
            y_cols_lower=["rank_lower"],
            y_cols_upper=["rank_upper"],
            share_y_axis=False,
            x_label="% Budget Used",
        )
        bench_relative_iterative_results_filled_bounds = (
            bench_relative_iterative_results.copy()
        )
        bench_relative_iterative_results_filled_bounds[
            "rank_lower"
        ] = bench_relative_iterative_results_filled_bounds["rank_lower"].fillna(
            bench_relative_iterative_results_filled_bounds["rank"]
        )
        bench_relative_iterative_results_filled_bounds = (
            create_default_plotting_identifier(
                df=bench_relative_iterative_results_filled_bounds,
                tuner_col=tuner_col,
                estimator_architecture_col=estimator_architecture_col,
                sampler_col=sampler_col,
            )
        )
        plot_and_save(
            data=bench_relative_iterative_results_filled_bounds,
            x_col=norm_iter_unit,
            y_cols=["rank"],
            entity_col="plotting_identifier",
            col_measure=bench_col,
            row_measure=None,
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename_prefix="rank_vs_iteration",
            analysis_type=analysis_type,
            subfolder="rank_analysis",
            y_cols_lower=["rank_lower"],
            y_cols_upper=["rank_upper"],
            share_y_axis=False,
            x_label="% Budget Used",
        )

        # Plot global figures across benchmarks:
        global_bench_relative_iterative_results = create_default_plotting_identifier(
            df=global_bench_relative_iterative_results,
            tuner_col=tuner_col,
            estimator_architecture_col=estimator_architecture_col,
            sampler_col=sampler_col,
        )
        plot_and_save(
            data=global_bench_relative_iterative_results,
            x_col=norm_iter_unit,
            y_cols=["rank"],
            entity_col="plotting_identifier",
            col_measure=bench_col,
            row_measure=None,
            cache_path=cache_path,
            run_start_str=run_start_str,
            filename_prefix="global_rank_vs_norm_iteration",
            analysis_type=analysis_type,
            subfolder="rank_analysis",
            y_cols_lower=None,
            y_cols_upper=None,
            share_y_axis=False,
            x_label="% Budget Used",
        )

        # 2.1 Significance Analysis for Runtime Data:
        significance_results_for_cd = compute_significance_results(
            benchmark_data=dataset_relative_runtime_results,
            norm_unit=norm_runtime_unit,
            tuner_col=tuner_col,
            estimator_architecture_col=estimator_architecture_col,
            sampler_col=sampler_col,
            analysis_components=analysis_components,
            cd_significance_method=cd_significance_method,
            bench_col=bench_col,
            data_col=data_col,
            alpha=alpha,
            cache_path=cache_path,
            run_start_str=run_start_str,
            analysis_type=analysis_type,
            correction_method=correction_method,
            filename_prefix="runtime_",
            is_global=False,
        )

        global_significance_results_for_cd = compute_significance_results(
            benchmark_data=global_dataset_relative_runtime_results,
            norm_unit=norm_runtime_unit,
            tuner_col=tuner_col,
            estimator_architecture_col=estimator_architecture_col,
            sampler_col=sampler_col,
            analysis_components=analysis_components,
            cd_significance_method="wilcoxon",
            bench_col=bench_col,
            data_col=data_col,
            alpha=alpha,
            cache_path=cache_path,
            run_start_str=run_start_str,
            analysis_type=analysis_type,
            correction_method=correction_method,
            filename_prefix="runtime_",
            is_global=True,
        )

        # Plot CD diagrams for runtime data
        cd_budget = 100

        # Plot local CD diagrams if method is in analysis components
        if cd_significance_method in analysis_components:
            plot_cd_diagram(
                benchmark_data=bench_relative_runtime_results,
                significance_results=significance_results_for_cd,
                norm_unit=norm_runtime_unit,
                tuner_col=tuner_col,
                estimator_architecture_col=estimator_architecture_col,
                sampler_col=sampler_col,
                bench_col=bench_col,
                alpha=alpha,
                cache_path=cache_path,
                run_start_str=run_start_str,
                analysis_type=analysis_type,
                filename_suffix="runtime",
                is_global=False,
                cd_budget=cd_budget,
            )

        # Plot global CD diagrams
        plot_cd_diagram(
            benchmark_data=global_bench_relative_runtime_results,
            significance_results=global_significance_results_for_cd,
            norm_unit=norm_runtime_unit,
            tuner_col=tuner_col,
            estimator_architecture_col=estimator_architecture_col,
            sampler_col=sampler_col,
            bench_col=bench_col,
            alpha=alpha,
            cache_path=cache_path,
            run_start_str=run_start_str,
            analysis_type=analysis_type,
            filename_suffix="runtime",
            is_global=True,
            cd_budget=cd_budget,
        )

        # 2.2 Significance Analysis for Iterative Data:
        significance_results_for_cd_iterative = compute_significance_results(
            benchmark_data=dataset_relative_iterative_results,
            norm_unit=norm_iter_unit,
            tuner_col=tuner_col,
            estimator_architecture_col=estimator_architecture_col,
            sampler_col=sampler_col,
            analysis_components=analysis_components,
            cd_significance_method=cd_significance_method,
            bench_col=bench_col,
            data_col=data_col,
            alpha=alpha,
            cache_path=cache_path,
            run_start_str=run_start_str,
            analysis_type=analysis_type,
            correction_method=correction_method,
            filename_prefix="iterative_",
            is_global=False,
        )

        global_significance_results_for_cd_iterative = compute_significance_results(
            benchmark_data=global_dataset_relative_iterative_results,
            norm_unit=norm_iter_unit,
            tuner_col=tuner_col,
            estimator_architecture_col=estimator_architecture_col,
            sampler_col=sampler_col,
            analysis_components=analysis_components,
            cd_significance_method="wilcoxon",
            bench_col=bench_col,
            data_col=data_col,
            alpha=alpha,
            cache_path=cache_path,
            run_start_str=run_start_str,
            analysis_type=analysis_type,
            correction_method=correction_method,
            filename_prefix="iterative_",
            is_global=True,
        )

        # Plot CD diagrams for iterative data
        # Plot local CD diagrams if method is in analysis components
        if cd_significance_method in analysis_components:
            plot_cd_diagram(
                benchmark_data=bench_relative_iterative_results,
                significance_results=significance_results_for_cd_iterative,
                norm_unit=norm_iter_unit,
                tuner_col=tuner_col,
                estimator_architecture_col=estimator_architecture_col,
                sampler_col=sampler_col,
                bench_col=bench_col,
                alpha=alpha,
                cache_path=cache_path,
                run_start_str=run_start_str,
                analysis_type=analysis_type,
                filename_suffix="iterative",
                is_global=False,
                cd_budget=cd_budget,
            )

        # Plot global CD diagrams
        plot_cd_diagram(
            benchmark_data=global_bench_relative_iterative_results,
            significance_results=global_significance_results_for_cd_iterative,
            norm_unit=norm_iter_unit,
            tuner_col=tuner_col,
            estimator_architecture_col=estimator_architecture_col,
            sampler_col=sampler_col,
            bench_col=bench_col,
            alpha=alpha,
            cache_path=cache_path,
            run_start_str=run_start_str,
            analysis_type=analysis_type,
            filename_suffix="iterative",
            is_global=True,
            cd_budget=cd_budget,
        )

    # NOTE: For next two breakout plots, values are first ranked by benchmark
    # and then split by sampler or architecture on column axis of plots, but
    # the rank is not only between the lines on a given plot, it's global, and
    # then split in post.
    # Sampler comparison plots:
    if "sampler_comparison" in analysis_components:
        for dataset, x_col in [
            (bench_relative_runtime_results, norm_runtime_unit),
            (bench_relative_iterative_results, norm_iter_unit),
        ]:
            dataset["plotting_identifier"] = dataset[estimator_architecture_col]
            plot_and_save(
                data=dataset,
                x_col=x_col,
                y_cols=["rank"],
                entity_col="plotting_identifier",
                col_measure=sampler_col,
                row_measure=bench_col,
                cache_path=cache_path,
                run_start_str=run_start_str,
                filename_prefix=f"sampler_partitioned_perf_vs_{x_col}",
                analysis_type=analysis_type,
                subfolder="sampler_comparison",
                y_cols_lower=["rank_lower"],
                y_cols_upper=["rank_upper"],
                share_y_axis=True,
                x_label="% Budget Used",
            )

    # Architecture comparison plots:
    if "architecture_comparison" in analysis_components:
        for dataset, x_col in [
            (bench_relative_runtime_results, norm_runtime_unit),
            (bench_relative_iterative_results, norm_iter_unit),
        ]:
            dataset["plotting_identifier"] = dataset[sampler_col]
            plot_and_save(
                data=dataset,
                x_col=x_col,
                y_cols=["rank"],
                entity_col="plotting_identifier",
                col_measure=estimator_architecture_col,
                row_measure=bench_col,
                cache_path=cache_path,
                run_start_str=run_start_str,
                filename_prefix=f"architecture_partitioned_perf_vs_{x_col}",
                analysis_type=analysis_type,
                subfolder="architecture_comparison",
                y_cols_lower=["rank_lower"],
                y_cols_upper=["rank_upper"],
                share_y_axis=True,
                x_label="% Budget Used",
            )

    # Conformalization effect analysis:
    if "conformalization_effect" in analysis_components:
        for benchmark in raw_benchmark_data[bench_col].unique():
            benchmark_data = raw_benchmark_data[
                raw_benchmark_data[bench_col] == benchmark
            ]
            # NOTE: Raw data for this component is expected to contain ccqr_optimization
            # variants where we want to rank a same architecture, repeated for
            # many samplers, ranked across varying levels of n_preconformal_trials, hence
            # 'extra_ranking_cols=[estimator_architecture_col, sampler_col]':
            for budget_unit in [runtime_unit, iter_unit]:
                conformalized_vs_nonconformalized_results = (
                    processor.process_performance_records(
                        raw_benchmark_data=benchmark_data,
                        budget_unit=budget_unit,
                        relativize_budget=True,
                        collapse_repetitions=True,
                        collapse_datasets=True,
                        extra_ranking_cols=[estimator_architecture_col, sampler_col],
                        n_bootstraps=n_bootstraps,
                    )
                )
                conformalized_vs_nonconformalized_results[
                    "plotting_identifier"
                ] = conformalized_vs_nonconformalized_results[
                    n_pre_conformal_trials_col
                ].apply(
                    lambda x: "Unconformalized" if x > 32 else "Conformalized + DtACI"
                )
                plot_and_save(
                    data=conformalized_vs_nonconformalized_results,
                    x_col=f"normalized_{budget_unit}",
                    y_cols=["rank"],
                    entity_col="plotting_identifier",
                    col_measure=sampler_col,
                    row_measure=estimator_architecture_col,
                    cache_path=cache_path,
                    run_start_str=run_start_str,
                    filename_prefix=f"perf_vs_{budget_unit}_n_pre_conformal_trials__{benchmark}",
                    analysis_type=analysis_type,
                    subfolder="conformalization_effect",
                    y_cols_lower=["rank_lower"],
                    y_cols_upper=["rank_upper"],
                    share_y_axis=False,
                    x_label="% Budget Used",
                )

    # Quantile count comparison analysis:
    if "quantile_count_comparison" in analysis_components:
        if len(raw_benchmark_data[sampler_col].unique()) > 1:
            raise ValueError(
                "Quantile count comparison analysis requires only one sampler."
            )
        for benchmark in raw_benchmark_data[bench_col].unique():
            benchmark_data = raw_benchmark_data[
                raw_benchmark_data[bench_col] == benchmark
            ]
            # NOTE: Raw data for this component is expected to contain ccqr_optimization
            # variants where we want to rank a same sampler, repeated for
            # many architectures, ranked across varying levels of n_quantiles, hence
            # 'extra_ranking_cols=[estimator_architecture_col, sampler_col]':
            for budget_unit in [runtime_unit, iter_unit]:
                quantile_count_comparison_results = (
                    processor.process_performance_records(
                        raw_benchmark_data=benchmark_data,
                        budget_unit=budget_unit,
                        relativize_budget=True,
                        collapse_repetitions=True,
                        collapse_datasets=True,
                        extra_ranking_cols=[estimator_architecture_col, sampler_col],
                        n_bootstraps=n_bootstraps,
                    )
                )
                quantile_count_comparison_results[
                    "plotting_identifier"
                ] = quantile_count_comparison_results[n_quantiles_col].apply(
                    lambda x: f"{int(x)} Quantiles"
                )
                plot_and_save(
                    data=quantile_count_comparison_results,
                    x_col=f"normalized_{budget_unit}",
                    y_cols=["rank"],
                    entity_col="plotting_identifier",
                    col_measure=estimator_architecture_col,
                    row_measure=bench_col,
                    cache_path=cache_path,
                    run_start_str=run_start_str,
                    filename_prefix=f"perf_vs_{budget_unit}_quantile_count_variation__{benchmark}",
                    analysis_type=analysis_type,
                    subfolder="quantile_count_comparison",
                    y_cols_lower=["rank_lower"],
                    y_cols_upper=["rank_upper"],
                    share_y_axis=False,
                    x_label="% Budget Used",
                )

    if "search_tuning_effect_comparison" in analysis_components:
        if len(raw_benchmark_data[sampler_col].unique()) > 1:
            raise ValueError(
                "Search tuning effect comparison analysis requires only one sampler."
            )
        for benchmark in raw_benchmark_data[bench_col].unique():
            benchmark_data = raw_benchmark_data[
                raw_benchmark_data[bench_col] == benchmark
            ]
            # NOTE: Raw data for this component is expected to contain ccqr_optimization
            # variants where we want to rank a same architecture across varying
            # levels of tuning, hence 'extra_ranking_cols=[estimator_architecture_col]':
            for budget_unit in [runtime_unit, iter_unit]:
                search_tuning_effect_comparison_results = (
                    processor.process_performance_records(
                        raw_benchmark_data=benchmark_data,
                        budget_unit=budget_unit,
                        relativize_budget=True,
                        collapse_repetitions=True,
                        collapse_datasets=True,
                        extra_ranking_cols=[estimator_architecture_col],
                        n_bootstraps=n_bootstraps,
                    )
                )
                plot_and_save(
                    data=search_tuning_effect_comparison_results,
                    x_col=f"normalized_{budget_unit}",
                    y_cols=["rank"],
                    entity_col=searcher_tuning_framework_col,
                    col_measure=estimator_architecture_col,
                    row_measure=bench_col,
                    cache_path=cache_path,
                    run_start_str=run_start_str,
                    filename_prefix=f"perf_vs_{budget_unit}_search_tuning_effect__{benchmark}",
                    analysis_type=analysis_type,
                    subfolder="search_tuning_effect_comparison",
                    y_cols_lower=["rank_lower"],
                    y_cols_upper=["rank_upper"],
                    share_y_axis=False,
                    x_label="% Budget Used",
                )

    # Number of candidates comparison analysis:
    if "num_candidates_comparison" in analysis_components:
        if len(raw_benchmark_data[sampler_col].unique()) > 1:
            raise ValueError(
                "Number of candidates comparison analysis requires only one sampler."
            )
        for benchmark in raw_benchmark_data[bench_col].unique():
            benchmark_data = raw_benchmark_data[
                raw_benchmark_data[bench_col] == benchmark
            ]
            # NOTE: Raw data for this component is expected to contain ccqr_optimization
            # variants where we want to rank a same sampler, repeated for
            # many architectures, ranked across varying levels of n_candidates, hence
            # 'extra_ranking_cols=[estimator_architecture_col, sampler_col]':
            for budget_unit in [runtime_unit, iter_unit]:
                num_candidates_comparison_results = (
                    processor.process_performance_records(
                        raw_benchmark_data=benchmark_data,
                        budget_unit=budget_unit,
                        relativize_budget=True,
                        collapse_repetitions=True,
                        collapse_datasets=True,
                        extra_ranking_cols=[estimator_architecture_col, sampler_col],
                        n_bootstraps=n_bootstraps,
                    )
                )
                num_candidates_comparison_results[
                    "plotting_identifier"
                ] = num_candidates_comparison_results[n_candidates_col].apply(
                    lambda x: f"{int(x)} Candidates"
                )
                plot_and_save(
                    data=num_candidates_comparison_results,
                    x_col=f"normalized_{budget_unit}",
                    y_cols=["rank"],
                    entity_col="plotting_identifier",
                    col_measure=estimator_architecture_col,
                    row_measure=sampler_col,
                    cache_path=cache_path,
                    run_start_str=run_start_str,
                    filename_prefix=f"perf_vs_{budget_unit}_num_candidates_variation__{benchmark}",
                    analysis_type=analysis_type,
                    subfolder="num_candidates_comparison",
                    y_cols_lower=["rank_lower"],
                    y_cols_upper=["rank_upper"],
                    share_y_axis=False,
                    x_label="% Budget Used",
                )


def analyze_searcher_tuning_effect(
    static_raw_benchmark_data: pd.DataFrame,
    cache_path: str,
    run_start_str: str,
    analysis_type: str,
    schema: BenchmarkDataSchema,
):
    """Analyze the effect of searcher tuning iterations on search estimator performance.

    Args:
        static_raw_benchmark_data: DataFrame containing static benchmark results.
        cache_path: Root directory for saving analysis outputs.
        run_start_str: Timestamp identifier for this experimental run.
        analysis_type: Analysis category label for file organization.
        schema: Data schema defining column names and structure.
    """
    estimator_architecture_col = schema.estimator_architecture_col
    repetition_column = schema.rep_col
    tuning_iterations_column = schema.tuning_iterations_col
    estimator_error_column = schema.estimator_error_col
    bench_col = schema.bench_col
    data_col = schema.data_col
    data_size_col = schema.data_size_col

    filtered_df = rank_and_collapse_data(
        static_raw_benchmark_data=static_raw_benchmark_data,
        aggregators=[
            schema.bench_col,
            schema.data_col,
            schema.data_size_col,
            schema.rep_col,
            schema.estimator_architecture_col,
            schema.tuning_iterations_col,
        ],
        comparison_col=tuning_iterations_column,
        metric_col=estimator_error_column,
        repetition_col=repetition_column,
    )

    save_analysis_results(
        filtered_df,
        cache_path,
        run_start_str,
        "filtered_ranks.csv",
        analysis_type,
    )

    path_manager = AnalysisPathManager(cache_path, run_start_str)
    tuning_plots_path = path_manager.get_analysis_path(
        analysis_type, "plots", "tuning_effect"
    )

    # TODO: Remove dependancy on this legacy function:
    aggregated_df = aggregate_and_save(
        data=filtered_df,
        grouping_cols=[
            schema.bench_col,
            schema.data_size_col,
            schema.estimator_architecture_col,
            schema.tuning_iterations_col,
        ],
        breakout_cols=[bench_col],
        block_cols=[data_col],
        metrics=["rank"],
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="tuning_effect_aggregated_results.csv",
        analysis_type=analysis_type,
    )
    plot_and_save(
        data=aggregated_df,
        x_col=tuning_iterations_column,
        y_cols=["rank"],
        entity_col=estimator_architecture_col,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename_prefix="tuning_effect_vs_data_size",
        analysis_type=analysis_type,
        subfolder="tuning_effect",
        col_measure=data_size_col,
        row_measure=bench_col,
        y_cols_lower=None,
        y_cols_upper=None,
        share_y_axis=False,
        add_markers=True,
        hide_col_and_row_labels=False,
    )

    logger.info(f"Tuning rank comparison plots saved in {tuning_plots_path}")


def analyze_searcher_estimator_comparison(
    static_raw_benchmark_data: pd.DataFrame,
    cache_path: str,
    run_start_str: str,
    analysis_type: str,
    schema: BenchmarkDataSchema,
):
    """Compare baseline performance across different searcher estimator architectures.

    Analyzes the inherent performance differences between estimator architectures (e.g.,
    Random Forest, XGBoost, Neural Networks) when used without hyperparameter optimization.
    This provides baseline comparisons to understand which architectures perform better
    out-of-the-box before any tuning effort is applied.

    The analysis focuses on:
    1. Filtering results to only include non-tuned configurations (tuning_iterations == 0)
    2. Ranking estimator architectures by performance within experimental conditions
    3. Statistical testing to identify significant architecture differences
    4. Visualization of performance patterns across data sizes and benchmarks

    Args:
        static_raw_benchmark_data: DataFrame containing static benchmark results
        cache_path: Root directory for saving analysis outputs.
        run_start_str: Timestamp identifier for this experimental run.
        analysis_type: Analysis category label for file organization.
        schema: Data schema defining column names and structure.

    Note:
        Only analyzes configurations with tuning_iterations == 0 to isolate the effect
        of estimator architecture choice from hyperparameter optimization effects.
    """
    estimator_architecture_col = schema.estimator_architecture_col
    repetition_column = schema.rep_col
    estimator_error_column = schema.estimator_error_col
    bench_col = schema.bench_col
    data_col = schema.data_col
    data_size_col = schema.data_size_col

    # Filter results to only include non-tuned configurations:
    non_tuned_results_df = static_raw_benchmark_data[
        static_raw_benchmark_data[schema.tuning_iterations_col] == 0
    ]
    # Rank and collapse the data:
    filtered_df = rank_and_collapse_data(
        static_raw_benchmark_data=non_tuned_results_df,
        aggregators=[
            schema.bench_col,
            schema.data_col,
            schema.data_size_col,
            schema.rep_col,
            schema.estimator_architecture_col,
            schema.tuning_iterations_col,
        ],
        comparison_col=estimator_architecture_col,
        metric_col=estimator_error_column,
        repetition_col=repetition_column,
    )

    save_analysis_results(
        filtered_df,
        cache_path,
        run_start_str,
        "non_tuned_filtered_ranks.csv",
        analysis_type,
    )

    aggregated_df = aggregate_and_save(
        data=filtered_df,
        grouping_cols=[
            schema.bench_col,
            schema.data_size_col,
            schema.estimator_architecture_col,
            schema.tuning_iterations_col,
        ],
        breakout_cols=[bench_col],
        block_cols=[data_col],
        metrics=["rank"],
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="estimator_comparison_aggregated_results.csv",
        analysis_type=analysis_type,
    )
    plot_and_save(
        data=aggregated_df,
        x_col=data_size_col,
        y_cols=["rank"],
        entity_col=estimator_architecture_col,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename_prefix="estimator_comparison_vs_data_size",
        analysis_type=analysis_type,
        subfolder="estimator_comparison",
        col_measure=bench_col,
        row_measure=None,
        y_cols_lower=None,
        y_cols_upper=None,
        share_y_axis=False,
        add_markers=True,
        col_measure_label="Benchmark",
        row_measure_label="Surrogate Architecture",
        x_label="Training Data Size",
        x_axis_start=0,
    )


def analyze_joint_architecture_and_static(
    main_raw_data: pd.DataFrame,
    static_raw_data: pd.DataFrame,
    cache_path: str,
    run_start_str: str,
    analysis_type: str,
    schema: BenchmarkDataSchema,
) -> None:
    """Process both main architecture variation and static benchmarks, then plot jointly."""
    processor = BenchmarkDataProcessor(schema=schema)
    
    bench_relative_iterative_results = processor.process_performance_records(
        raw_benchmark_data=main_raw_data,
        budget_unit=schema.iter_unit,
        relativize_budget=True,
        collapse_repetitions=True,
        collapse_datasets=True,
        extra_ranking_cols=None,
        n_bootstraps=1000,
    )
    
    bench_relative_iterative_results["rank_lower"] = bench_relative_iterative_results["rank_lower"].fillna(bench_relative_iterative_results["rank"])

    non_tuned_results_df = static_raw_data[
        static_raw_data[schema.tuning_iterations_col] == 0
    ]

    filtered_static_df = rank_and_collapse_data(
        static_raw_benchmark_data=non_tuned_results_df,
        aggregators=[
            schema.bench_col,
            schema.data_col,
            schema.data_size_col,
            schema.rep_col,
            schema.estimator_architecture_col,
            schema.tuning_iterations_col,
        ],
        comparison_col=schema.estimator_architecture_col,
        metric_col=schema.estimator_error_col,
        repetition_col=schema.rep_col,
    )

    aggregated_static_df = aggregate_and_save(
        data=filtered_static_df,
        grouping_cols=[
            schema.bench_col,
            schema.data_size_col,
            schema.estimator_architecture_col,
            schema.tuning_iterations_col,
        ],
        breakout_cols=[schema.bench_col],
        block_cols=[schema.data_col],
        metrics=["rank"],
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="joint_static_aggregated_results.csv",
        analysis_type=analysis_type,
    )

    plot_joint_architecture_and_static(
        main_processed_df=bench_relative_iterative_results,
        static_processed_df=aggregated_static_df,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename_prefix="joint_architecture_and_static",
        analysis_type=analysis_type,
        subfolder="joint_analysis",
        schema=schema,
    )
