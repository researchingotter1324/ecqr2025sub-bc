import pandas as pd
import numpy as np
import logging
from typing import List, Optional
from hpobench.config.schema import BenchmarkDataSchema


logger = logging.getLogger(__name__)


class BenchmarkDataProcessor:
    """
    Class-based HPO benchmark data processor that eliminates column aggregator confusion.

    Centralizes column management through BenchmarkDataSchema and provides clear
    methods for each processing step without passing column lists around.
    """

    def __init__(self, schema: BenchmarkDataSchema = None):
        """Initialize the benchmark data processor with column schema configuration.

        Args:
            schema: BenchmarkDataSchema defining column names and structure.
                Uses default schema if None provided.
        """
        if schema is None:
            schema = BenchmarkDataSchema()
        self.schema = schema

        # Extract all column names from schema
        self.rep_col = schema.rep_col
        self.perf_col = schema.perf_col
        self.tuner_col = schema.tuner_col
        self.bench_col = schema.bench_col
        self.data_col = schema.data_col
        self.sampler_col = schema.sampler_col
        self.confidence_level_col = schema.confidence_level_col
        self.estimator_architecture_col = schema.estimator_architecture_col
        self.sampler_n_quantiles = schema.sampler_n_quantiles_col
        self.n_candidates = schema.n_candidates_col
        self.sampler_adapter = schema.sampler_adapter_col
        self.tuner_searcher_tuning_framework = (
            schema.tuner_searcher_tuning_framework_col
        )
        self.n_pre_conformal_trials = schema.n_pre_conformal_trials_col
        self.runtime_unit = schema.runtime_unit
        self.iter_unit = schema.iter_unit
        self.breach_column = schema.breach_col
        self.extreme_quantile_used_col = schema.extreme_quantile_used_col
        self.cumulative_extreme_quantile_rate_col = "cumulative_extreme_quantile_rate"
        self.cumulative_coverage_error_col = schema.cumulative_coverage_error_col
        self.rolling_coverage_error_col = schema.rolling_coverage_error_col

        self.tuner_cols = [
            self.tuner_col,
            self.sampler_col,
            self.confidence_level_col,
            self.estimator_architecture_col,
            self.sampler_n_quantiles,
            self.n_candidates,
            self.sampler_adapter,
            self.tuner_searcher_tuning_framework,
            self.n_pre_conformal_trials,
        ]
        self.benchmark_and_data_cols = [self.bench_col, self.data_col]
        self.tuner_level = self.benchmark_and_data_cols + self.tuner_cols
        self.repetition_level = self.tuner_level + [self.rep_col]

        self.dataset_level = [self.bench_col, self.data_col]

        self.all_relevant_cols = [
            self.rep_col,
            self.perf_col,
            self.tuner_col,
            self.bench_col,
            self.data_col,
            self.sampler_col,
            self.confidence_level_col,
            self.estimator_architecture_col,
            self.sampler_n_quantiles,
            self.n_candidates,
            self.sampler_adapter,
            self.tuner_searcher_tuning_framework,
            self.n_pre_conformal_trials,
            self.runtime_unit,
            self.iter_unit,
            self.breach_column,
            self.extreme_quantile_used_col,
        ]

    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean benchmark data by handling missing values.

        Args:
            data: Raw benchmark data DataFrame.

        Returns:
            Cleaned DataFrame with NaN values filled in tuner columns.

        Raises:
            ValueError: If required columns are missing from input data.
        """
        data_copy = data.copy()

        for col in self.all_relevant_cols:
            if col not in data_copy.columns:
                raise ValueError(f"Missing column in input data: {col}")

        # Only fill NaN values in tuner columns with empty strings
        data_copy[self.tuner_cols] = data_copy[self.tuner_cols].fillna("")

        return data_copy

    def accumulate_best_performances(
        self, data: pd.DataFrame, budget_unit: str
    ) -> pd.DataFrame:
        """
        Tracks the best performance achieved so far during progression.
        """
        data_cleaned = data.copy()
        sorted_data = data_cleaned.sort_values(
            by=self.repetition_level + [budget_unit],
            ascending=True,
        ).reset_index(drop=True)

        sorted_data["best_performance"] = sorted_data.groupby(self.repetition_level)[
            self.perf_col
        ].transform("cummin")

        return sorted_data

    def align_tuners_to_shared_budget(
        self, data: pd.DataFrame, budget_unit: str
    ) -> pd.DataFrame:
        """
        Aligns tuner experiments to common budget intervals for fair comparison.

        Restricts analysis to budget ranges where all tuners have data.
        """
        data_cleaned = data.copy()

        # Calculate budget ranges per experiment run
        data_cleaned["max_budget_per_run"] = data_cleaned.groupby(
            self.repetition_level
        )[budget_unit].transform("max")

        data_cleaned["min_budget_per_run"] = data_cleaned.groupby(
            self.repetition_level
        )[budget_unit].transform("min")

        # Find shared budget range per dataset
        data_cleaned["max_shared_budget_per_dataset"] = data_cleaned.groupby(
            self.dataset_level
        )["max_budget_per_run"].transform("min")

        data_cleaned["min_shared_budget_per_dataset"] = data_cleaned.groupby(
            self.dataset_level
        )["min_budget_per_run"].transform("max")

        # Filter to shared budget range
        aligned_data = data_cleaned[
            (data_cleaned[budget_unit] >= data_cleaned["min_shared_budget_per_dataset"])
            & (
                data_cleaned[budget_unit]
                <= data_cleaned["max_shared_budget_per_dataset"]
            )
        ].copy()

        # Clean up temporary columns
        aligned_data = aligned_data.drop(
            columns=[
                "max_budget_per_run",
                "min_budget_per_run",
                "max_shared_budget_per_dataset",
                "min_shared_budget_per_dataset",
            ]
        )

        return aligned_data

    def calculate_ranks(
        self,
        data: pd.DataFrame,
        budget_unit: str,
        extra_ranking_cols: List[str] = None,
        rank_ascending: bool = True,
        metric_column: str = "best_performance",
    ) -> pd.DataFrame:
        """
        Computes performance ranks for tuner comparison.

        Ranks are calculated within each dataset/budget combination,
        excluding comparison factors to enable fair comparison.
        """
        data_cleaned = data.copy()

        ranking_cols = self.repetition_level + [budget_unit]
        ranking_cols = [col for col in ranking_cols if col not in self.tuner_cols]
        if extra_ranking_cols:
            ranking_cols = list(set(ranking_cols + extra_ranking_cols))

        data_cleaned["rank"] = data_cleaned.groupby(ranking_cols)[metric_column].rank(
            method="average", ascending=rank_ascending
        )
        return data_cleaned

    def accumulate_breaches(
        self, data: pd.DataFrame, budget_unit: str, rolling_window: int = 20
    ) -> pd.DataFrame:
        """
        Tracks cumulative and rolling breach rates for constraint violations.
        """
        data_cleaned = data.copy()
        data_cleaned[self.breach_column] = data_cleaned[self.breach_column].replace(
            {"": np.nan, None: np.nan}
        )
        # Convert to numeric, coercing errors to NaN
        data_cleaned[self.breach_column] = pd.to_numeric(
            data_cleaned[self.breach_column], errors="coerce"
        )

        sorted_data = data_cleaned.sort_values(
            by=self.repetition_level + [budget_unit],
            ascending=True,
        ).reset_index(drop=True)

        sorted_data["cumulative_breach_rate"] = (
            sorted_data.groupby(self.repetition_level)[self.breach_column]
            .expanding()
            .mean()
            .reset_index(level=self.repetition_level, drop=True)
        )

        sorted_data["rolling_breach_rate"] = (
            sorted_data.groupby(self.repetition_level)[self.breach_column]
            .rolling(window=rolling_window, min_periods=rolling_window)
            .mean()
            .reset_index(level=self.repetition_level, drop=True)
        )

        # Always create coverage error columns, fill with NaN if not available
        confidence_vals = sorted_data[self.confidence_level_col]
        has_valid_confidence = (~confidence_vals.isin([None, ""])).all()

        if has_valid_confidence:
            logger.debug("Creating coverage error columns with calculated values")
            sorted_data[self.cumulative_coverage_error_col] = abs(
                (1 - sorted_data["cumulative_breach_rate"])
                - sorted_data[self.confidence_level_col].astype(float)
            )
            sorted_data[self.rolling_coverage_error_col] = abs(
                (1 - sorted_data["rolling_breach_rate"])
                - sorted_data[self.confidence_level_col].astype(float)
            )
        else:
            logger.debug("Creating coverage error columns with NaN values")
            sorted_data[self.cumulative_coverage_error_col] = np.nan
            sorted_data[self.rolling_coverage_error_col] = np.nan

        return sorted_data

    def accumulate_extreme_quantile_rate(
        self, data: pd.DataFrame, budget_unit: str
    ) -> pd.DataFrame:
        """Tracks the cumulative rate at which configurations were acquired via the
        lowest (extreme) quantile bound.

        Mirrors the structure of ``accumulate_breaches``: the binary
        ``extreme_quantile_used`` indicator is expanded into a per-repetition
        cumulative mean, producing ``cumulative_extreme_quantile_rate``.
        Rows where the indicator is absent (e.g. non-ccqr tuners, warm-start
        trials) receive NaN so that they remain distinguishable from genuine
        zeros.
        """
        data_cleaned = data.copy()
        data_cleaned[self.extreme_quantile_used_col] = data_cleaned[
            self.extreme_quantile_used_col
        ].replace({"": np.nan, None: np.nan})
        data_cleaned[self.extreme_quantile_used_col] = pd.to_numeric(
            data_cleaned[self.extreme_quantile_used_col], errors="coerce"
        )

        sorted_data = data_cleaned.sort_values(
            by=self.repetition_level + [budget_unit],
            ascending=True,
        ).reset_index(drop=True)

        sorted_data[self.cumulative_extreme_quantile_rate_col] = (
            sorted_data.groupby(self.repetition_level)[self.extreme_quantile_used_col]
            .expanding()
            .mean()
            .reset_index(level=self.repetition_level, drop=True)
        )

        return sorted_data

    def time_discretize_data(
        self, data: pd.DataFrame, budget_unit: str = "runtime"
    ) -> pd.DataFrame:
        """
        Discretizes continuous runtime data into uniform time intervals.
        """
        data_cleaned = data.copy()

        discretized_slices = []

        # Group by dataset level to discretize within each dataset
        for _, dataset_group in data_cleaned.groupby(self.dataset_level):
            max_runtime = dataset_group[budget_unit].max()
            rounding_increment = -(len(str(int(max_runtime))) - 3)
            runtime_values = np.arange(
                0, max_runtime, max(1, 10 ** (-rounding_increment))
            )
            expanded_df = pd.DataFrame({budget_unit: runtime_values}).astype(int)

            # Process each individual experiment run
            for _, run_group in dataset_group.groupby(self.repetition_level):
                run_group = run_group.copy()
                run_group[budget_unit] = (
                    run_group[budget_unit].round(rounding_increment).astype(int)
                )

                # Aggregate duplicates and accumulate performance
                run_group = run_group.groupby(
                    self.repetition_level + [budget_unit], as_index=False
                ).agg({self.perf_col: "min"})
                run_max_time = max(run_group[budget_unit])
                run_min_time = min(run_group[budget_unit])
                run_group = self.accumulate_best_performances(run_group, budget_unit)

                # Merge with full time grid and forward fill
                run_group = pd.merge(expanded_df, run_group, how="left", on=budget_unit)
                run_group = run_group.sort_values(by=budget_unit).reset_index(drop=True)
                run_group = run_group.ffill()
                run_group[self.repetition_level] = run_group[
                    self.repetition_level
                ].bfill()

                run_group.loc[
                    run_group[budget_unit] < run_min_time, "best_performance"
                ] = np.nan
                run_group.loc[
                    run_group[budget_unit] > run_max_time, "best_performance"
                ] = np.nan

                discretized_slices.append(run_group)

        discretized_data = pd.concat(discretized_slices, ignore_index=True)

        # Calculate observation fill rate and filter
        discretized_data["observation_fill_rate"] = discretized_data.groupby(
            self.tuner_level + [budget_unit]
        )["best_performance"].transform(lambda x: (x.notna().sum()) / len(x))

        discretized_data = discretized_data[
            discretized_data["observation_fill_rate"] == 1
        ].drop(columns=["observation_fill_rate"])

        return discretized_data

    def standardize_budget_to_percentage(
        self, data: pd.DataFrame, budget_unit: str, metrics_to_keep: List[str]
    ) -> pd.DataFrame:
        """
        Normalizes budget units to 0-100 scale for cross-experiment comparison.
        """
        data_cleaned = data.copy()

        # Normalize budget within each experiment run
        normalized_col = f"normalized_{budget_unit}"
        data_cleaned[normalized_col] = data_cleaned.groupby(self.repetition_level)[
            budget_unit
        ].transform(
            lambda x: 100 * (x - x.min()) / (x.max() - x.min())
            if x.max() > x.min()
            else 0
        )
        data_cleaned[normalized_col] = data_cleaned[normalized_col].round().astype(int)

        # Create standardized results
        results = []
        for _, group in data_cleaned.groupby(self.repetition_level):
            # Create 0-100 percentage grid
            percentage_grid = pd.DataFrame({normalized_col: np.arange(0, 101)})

            columns_to_keep = (
                self.repetition_level + [normalized_col, budget_unit] + metrics_to_keep
            )
            group_subset = group[columns_to_keep].drop_duplicates(
                subset=[normalized_col]
            )

            # Merge and forward fill
            merged_group = pd.merge(
                percentage_grid, group_subset, how="left", on=normalized_col
            )
            merged_group = merged_group.sort_values(by=normalized_col).reset_index(
                drop=True
            )

            columns_to_fill = self.repetition_level + metrics_to_keep
            merged_group[columns_to_fill] = merged_group[columns_to_fill].ffill()

            results.append(merged_group)

        if not results:
            return pd.DataFrame()

        return pd.concat(results, ignore_index=True)

    def collapse_across_repetitions(
        self, data: pd.DataFrame, metrics: List[str], budget_unit: str
    ) -> pd.DataFrame:
        """
        Collapses HPO benchmark data by computing means across repetitions.

        Groups data without repetition column and calculates mean for each metric.
        """
        data_cleaned = data.copy()

        aggregations = {metric: "mean" for metric in metrics}
        processed_data = data_cleaned.groupby(
            self.tuner_level + [budget_unit], as_index=False
        ).agg(aggregations)

        # Remove '_mean' suffix from aggregated metric columns if present
        rename_dict = {
            f"{metric}_mean": metric
            for metric in metrics
            if f"{metric}_mean" in processed_data.columns
        }
        if rename_dict:
            processed_data = processed_data.rename(columns=rename_dict)

        return processed_data

    def process_iteration_budget_data(
        self,
        data: pd.DataFrame,
        additional_tuner_breakout_cols: List[str] = None,
        relativize_budget: bool = False,
        collapse_repetitions: bool = False,
        collapse_datasets: bool = False,
        n_bootstraps: int = 1000,
    ) -> pd.DataFrame:
        """
        Process data with iteration-based budget.
        """
        # Step 1: Accumulate best performances
        accumulated_data = self.accumulate_best_performances(data, self.iter_unit)

        # Step 2: Align tuners to shared budget ranges
        aligned_data = self.align_tuners_to_shared_budget(
            accumulated_data, self.iter_unit
        )

        # Step 3: Calculate ranks
        ranked_data = self.calculate_ranks(
            aligned_data,
            self.iter_unit,
            extra_ranking_cols=additional_tuner_breakout_cols,
            rank_ascending=True,
            metric_column="best_performance",
        )

        breach_data = self.accumulate_breaches(ranked_data, self.iter_unit)
        final_data = self.accumulate_extreme_quantile_rate(breach_data, self.iter_unit)

        # Step 5: Relativize budget if requested
        budget_unit = self.iter_unit
        if relativize_budget:
            metrics = ["rank", "best_performance"]
        else:
            metrics = [
                "rank",
                "best_performance",
                self.cumulative_coverage_error_col,
                self.rolling_coverage_error_col,
                self.cumulative_extreme_quantile_rate_col,
            ]
        if relativize_budget:
            final_data = self.standardize_budget_to_percentage(
                final_data, budget_unit, metrics
            )
            budget_unit = f"normalized_{self.iter_unit}"

        if collapse_repetitions:
            final_data = self.collapse_across_repetitions(
                final_data, metrics, budget_unit=budget_unit
            )

        if collapse_datasets:
            final_data = block_bootstrap(
                final_data,
                breakout_cols=[self.bench_col],
                block_cols=[self.data_col],
                aggregators=[
                    col
                    for col in self.tuner_level + [budget_unit]
                    if col != self.data_col
                ],
                metric_cols=metrics,
                n_bootstraps=n_bootstraps,
                random_state=1234,
            )

        return final_data

    def process_runtime_budget_data(
        self,
        data: pd.DataFrame,
        additional_tuner_breakout_cols: List[str] = None,
        relativize_budget: bool = False,
        collapse_repetitions: bool = False,
        collapse_datasets: bool = False,
        n_bootstraps: int = 1000,
    ) -> pd.DataFrame:
        """
        Process data with runtime-based budget.
        """
        # Step 1: Discretize time data
        discretized_data = self.time_discretize_data(data, self.runtime_unit)

        # Step 2: Align tuners to shared budget ranges
        aligned_data = self.align_tuners_to_shared_budget(
            discretized_data, self.runtime_unit
        )

        # Step 3: Calculate ranks
        final_data = self.calculate_ranks(
            aligned_data,
            self.runtime_unit,
            extra_ranking_cols=additional_tuner_breakout_cols,
            rank_ascending=True,
            metric_column="best_performance",
        )

        budget_unit = self.runtime_unit
        # Step 4: Relativize budget if requested
        if relativize_budget:
            final_data = self.standardize_budget_to_percentage(
                final_data, self.runtime_unit, ["rank", "best_performance"]
            )
            budget_unit = f"normalized_{self.runtime_unit}"

        if collapse_repetitions:
            final_data = self.collapse_across_repetitions(
                final_data, ["rank", "best_performance"], budget_unit=budget_unit
            )

        if collapse_datasets:
            final_data = block_bootstrap(
                final_data,
                breakout_cols=[self.bench_col],
                block_cols=[self.data_col],
                aggregators=[
                    col
                    for col in self.tuner_level + [budget_unit]
                    if col != self.data_col
                ],
                metric_cols=["rank", "best_performance"],
                n_bootstraps=n_bootstraps,
                random_state=1234,
            )

        return final_data

    def process_performance_records(
        self,
        raw_benchmark_data: pd.DataFrame,
        budget_unit: str,
        extra_ranking_cols: List[str] = None,
        relativize_budget: bool = False,
        collapse_repetitions: bool = False,
        collapse_datasets: bool = False,
        n_bootstraps: int = 1000,
    ) -> pd.DataFrame:
        """Process HPO benchmark data through complete analysis pipeline.

        Main entry point that orchestrates data processing based on budget type,
        handling both iteration-based and runtime-based budgets.

        Args:
            raw_benchmark_data: Raw experimental results DataFrame.
            budget_unit: Type of budget unit (iteration or runtime).
            extra_ranking_cols: Additional columns to include in ranking.
            relativize_budget: Whether to normalize budget to percentage scale.
            collapse_repetitions: Whether to aggregate across repetitions.
            collapse_datasets: Whether to aggregate across datasets.
            n_bootstraps: Number of bootstrap samples for confidence intervals.

        Returns:
            Processed DataFrame with performance metrics and rankings.

        Raises:
            ValueError: If budget_unit is not supported.
        """
        data_cleaned = self._validate_and_clean_data(raw_benchmark_data)

        if budget_unit == self.iter_unit:
            return self.process_iteration_budget_data(
                data_cleaned,
                extra_ranking_cols,
                relativize_budget,
                collapse_repetitions,
                collapse_datasets,
                n_bootstraps=n_bootstraps,
            )
        elif budget_unit == self.runtime_unit:
            return self.process_runtime_budget_data(
                data_cleaned,
                additional_tuner_breakout_cols=extra_ranking_cols,
                relativize_budget=relativize_budget,
                collapse_repetitions=collapse_repetitions,
                collapse_datasets=collapse_datasets,
                n_bootstraps=n_bootstraps,
            )
        else:
            raise ValueError(f"Unsupported budget unit: {budget_unit}")


def block_bootstrap(
    data: pd.DataFrame,
    breakout_cols: List[str],
    block_cols: List[str],
    aggregators: List[str],
    metric_cols: List[str],
    n_bootstraps: int,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Compute block-bootstrap percentile intervals for grouped metrics.

    Performs a block bootstrap within each combination of `breakout_cols`.
    Blocks are defined by `block_cols` and are resampled with replacement to
    create bootstrap samples. For each bootstrap iteration the mean of the
    `metric_cols` is computed per `aggregators` group. The function returns
    the original group means plus 5th and 95th percentile bounds computed
    over the bootstrap iterations (named "<metric>_lower" and "<metric>_upper").

    Args:
        data: Input DataFrame containing metrics and grouping columns.
        breakout_cols: Columns defining independent breakout groups; bootstrapping
            is performed separately within each breakout group.
        block_cols: Columns whose combined values define a block/key that is
            resampled (preserves within-block dependencies).
        aggregators: Columns to group by when computing means (e.g. experimental
            factors to aggregate over).
        metric_cols: Numeric metric columns to aggregate and compute percentiles for.
        n_bootstraps: Number of bootstrap iterations to perform per breakout group.
        random_state: Optional seed for numpy's RNG to make results reproducible.

    Returns:
        DataFrame: A table containing the original means for each `aggregators`
        group and, for each metric in `metric_cols`, the lower (5th pct) and
        upper (95th pct) bootstrap bounds named "<metric>_lower" and
        "<metric>_upper". If no valid bootstrap could be performed for a
        group, the bounds are NaN.
    """

    if random_state is not None:
        np.random.seed(random_state)

    data_with_key = data.copy()
    data_with_key["_bootstrap_key"] = (
        data_with_key[block_cols].astype(str).agg("_".join, axis=1)
    )

    # Calculate original means for each aggregator group.
    # Keep the original metric column names for the mean aggregation (no "_mean" suffix).
    original_means = data_with_key.groupby(aggregators, as_index=False)[
        metric_cols
    ].mean()

    all_bootstrap_results = []

    # Group by all breakout columns
    for breakout_values, breakout_group in data_with_key.groupby(breakout_cols):
        unique_keys = breakout_group["_bootstrap_key"].unique()
        if len(unique_keys) < 2:
            continue

        bootstrap_iterations = []
        for _ in range(n_bootstraps):
            sampled_keys = np.random.choice(
                unique_keys, size=len(unique_keys), replace=True
            )
            bootstrap_sample = pd.concat(
                [
                    breakout_group[breakout_group["_bootstrap_key"] == key]
                    for key in sampled_keys
                ],
                ignore_index=True,
            )
            bootstrap_means = bootstrap_sample.groupby(aggregators, as_index=False)[
                metric_cols
            ].mean()
            bootstrap_iterations.append(bootstrap_means)

        if bootstrap_iterations:
            combined_bootstrap = pd.concat(bootstrap_iterations, ignore_index=True)
            # Add breakout columns to keep track of group
            if isinstance(breakout_values, tuple):
                for col, val in zip(breakout_cols, breakout_values):
                    combined_bootstrap[col] = val
            else:
                combined_bootstrap[breakout_cols[0]] = breakout_values
            all_bootstrap_results.append(combined_bootstrap)

    if all_bootstrap_results:
        all_bootstrap_data = pd.concat(all_bootstrap_results, ignore_index=True)

        def p5(x):
            return np.percentile(x, 5)

        def p95(x):
            return np.percentile(x, 95)

        agg_dict = {col: [p5, p95] for col in metric_cols}
        group_cols = list(set(breakout_cols + aggregators))
        percentile_results = all_bootstrap_data.groupby(group_cols, as_index=False).agg(
            agg_dict
        )

        new_cols = []
        for col in percentile_results.columns:
            if isinstance(col, tuple):
                if col[1] == "p5":
                    new_cols.append(f"{col[0]}_lower")
                elif col[1] == "p95":
                    new_cols.append(f"{col[0]}_upper")
                else:
                    new_cols.append("_".join([str(c) for c in col if c]))
            else:
                new_cols.append(col)
        percentile_results.columns = new_cols
    else:
        percentile_results = original_means[aggregators].copy()
        for metric in metric_cols:
            percentile_results[f"{metric}_lower"] = np.nan
            percentile_results[f"{metric}_upper"] = np.nan

    final_results = original_means.merge(
        percentile_results, on=aggregators, how="outer"
    )

    return final_results


def rank_and_collapse_data(
    static_raw_benchmark_data: pd.DataFrame,
    aggregators: list[str],
    comparison_col: str,
    metric_col: str,
    repetition_col: str,
) -> pd.DataFrame:
    """
    Ranks data within groups and collapses the results by averaging ranks across repetitions.

    Args:
        static_raw_benchmark_data: Input DataFrame containing the data to be ranked and collapsed.
        aggregators: Total identifiers for grouping, may be inclusive of other cols in below inputs.
        comparison_col: Column containing the entities to rank over (eg. tuners).
        metric_col: Column containing the values to be ranked.
        repetition_col: Column containing experiment repetition values.

    Returns:
        pd.DataFrame: DataFrame with mean ranks collapsed across repetitions for each group and comparison.
    """
    # Get the columns to group by during ranking (all grouping columns, except the thing to rank over):
    ranking_aggregators = [col for col in aggregators if col != comparison_col]
    ranked_df = static_raw_benchmark_data.copy()
    ranked_df["rank"] = ranked_df.groupby(ranking_aggregators)[metric_col].rank(
        method="average", ascending=True
    )

    # Collapse ranks by averaging across repetitions:
    collapsing_aggregators = [col for col in aggregators if col != repetition_col]
    collapsed_df = (
        ranked_df.groupby(collapsing_aggregators, observed=True)["rank"]
        .mean()
        .reset_index()
    )

    return collapsed_df
