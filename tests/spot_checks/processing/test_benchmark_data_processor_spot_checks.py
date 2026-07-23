"""
Spot check tests for BenchmarkDataProcessor methods.

These tests are designed for qualitative inspection of outputs rather than strict assertions.
Each test method saves its output as CSV files for manual inspection to verify behavior.
"""

import pandas as pd
import os
import itertools


class TestBenchmarkDataProcessorSpotChecks:
    """Spot check tests for BenchmarkDataProcessor methods."""

    def save_output(
        self, data: pd.DataFrame, method_name: str, params: dict, output_dir: str
    ):
        """Save test output to CSV file."""
        # Create subdirectory for the method
        method_dir = os.path.join(output_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)

        # Create filename with parameters
        param_str = "_".join([f"{k}_{v}" for k, v in params.items()])
        filename = f"{method_name}_{param_str}.csv"
        filepath = os.path.join(method_dir, filename)

        # Save the data
        data.to_csv(filepath, index=False)
        print(f"Saved {method_name} output to {filepath} (shape: {data.shape})")

    def test_save_input_data(self, dummy_processing_raw_data, spot_check_output_dir):
        """Save input data to CSV file."""
        self.save_output(
            dummy_processing_raw_data, "input_data", {}, spot_check_output_dir
        )

    def test_process_performance_records_spot_check(
        self,
        benchmark_data_processor,
        dummy_processing_raw_data,
        spot_check_output_dir,
        benchmark_data_schema,
    ):
        """
        Spot check test for process_performance_records method.

        Tests all combinations of the main parameters:
        - budget_unit: iteration, runtime
        - extra_ranking_cols: None, [estimator_architecture_col]
        - relativize_budget: False, True
        - collapse_repetitions: False, True
        - collapse_datasets: False, True
        """
        # Parameter combinations
        budget_units = [
            benchmark_data_schema.iter_unit,
            benchmark_data_schema.runtime_unit,
        ]
        extra_ranking_options = [
            None,
            [benchmark_data_schema.estimator_architecture_col],
        ]
        relativize_options = [False, True]
        collapse_repetitions_options = [False, True]
        collapse_datasets_options = [False, True]

        # Test all combinations
        for (
            budget_unit,
            extra_ranking,
            relativize,
            collapse_reps,
            collapse_datasets,
        ) in itertools.product(
            budget_units,
            extra_ranking_options,
            relativize_options,
            collapse_repetitions_options,
            collapse_datasets_options,
        ):
            params = {
                "budget_unit": budget_unit,
                "extra_ranking": "estimator_arch" if extra_ranking else "None",
                "relativize": relativize,
                "collapse_reps": collapse_reps,
                "collapse_datasets": collapse_datasets,
            }

            result = benchmark_data_processor.process_performance_records(
                raw_benchmark_data=dummy_processing_raw_data,
                budget_unit=budget_unit,
                extra_ranking_cols=extra_ranking,
                relativize_budget=relativize,
                collapse_repetitions=collapse_reps,
                collapse_datasets=collapse_datasets,
            )

            self.save_output(
                result, "process_performance_records", params, spot_check_output_dir
            )

    def test_accumulate_best_performances_spot_check(
        self,
        benchmark_data_processor,
        dummy_processing_raw_data,
        spot_check_output_dir,
        benchmark_data_schema,
    ):
        """
        Spot check test for accumulate_best_performances method.
        """
        result = benchmark_data_processor.accumulate_best_performances(
            dummy_processing_raw_data, benchmark_data_schema.iter_unit
        )

        params = {"budget_unit": benchmark_data_schema.iter_unit}
        self.save_output(
            result, "accumulate_best_performances", params, spot_check_output_dir
        )

    def test_align_tuners_to_shared_budget_spot_check(
        self,
        benchmark_data_processor,
        dummy_processing_raw_data,
        spot_check_output_dir,
        benchmark_data_schema,
    ):
        """
        Spot check test for align_tuners_to_shared_budget method.
        """
        for budget_unit in [
            benchmark_data_schema.iter_unit,
            benchmark_data_schema.runtime_unit,
        ]:
            if budget_unit == benchmark_data_schema.iter_unit:
                # For iteration budget: accumulate best performances first
                preprocessed_data = (
                    benchmark_data_processor.accumulate_best_performances(
                        dummy_processing_raw_data, budget_unit
                    )
                )
            else:
                # For runtime budget: time discretize data first
                preprocessed_data = benchmark_data_processor.time_discretize_data(
                    dummy_processing_raw_data, budget_unit
                )

            result = benchmark_data_processor.align_tuners_to_shared_budget(
                preprocessed_data, budget_unit
            )

            params = {"budget_unit": budget_unit}
            self.save_output(
                result, "align_tuners_to_shared_budget", params, spot_check_output_dir
            )

    def test_calculate_ranks_spot_check(
        self,
        benchmark_data_processor,
        dummy_processing_raw_data,
        spot_check_output_dir,
        benchmark_data_schema,
    ):
        """
        Spot check test for calculate_ranks method.
        """
        # First preprocess and align data according to budget type
        for budget_unit in [
            benchmark_data_schema.iter_unit,
            benchmark_data_schema.runtime_unit,
        ]:
            for extra_ranking_cols in [
                None,
                [benchmark_data_schema.estimator_architecture_col],
            ]:
                params = {
                    "budget_unit": budget_unit,
                    "extra_ranking": "estimator_arch" if extra_ranking_cols else "None",
                }

                if budget_unit == benchmark_data_schema.iter_unit:
                    # For iteration budget: accumulate best performances first
                    preprocessed_data = (
                        benchmark_data_processor.accumulate_best_performances(
                            dummy_processing_raw_data, budget_unit
                        )
                    )
                else:
                    # For runtime budget: time discretize data first
                    preprocessed_data = benchmark_data_processor.time_discretize_data(
                        dummy_processing_raw_data, budget_unit
                    )

                aligned = benchmark_data_processor.align_tuners_to_shared_budget(
                    preprocessed_data, budget_unit
                )
                result = benchmark_data_processor.calculate_ranks(
                    aligned,
                    budget_unit,
                    extra_ranking_cols=extra_ranking_cols,
                    rank_ascending=True,
                    metric_column="best_performance",
                )

                self.save_output(
                    result, "calculate_ranks", params, spot_check_output_dir
                )

    def test_accumulate_breaches_spot_check(
        self,
        benchmark_data_processor,
        dummy_processing_raw_data,
        spot_check_output_dir,
        benchmark_data_schema,
    ):
        """
        Spot check test for accumulate_breaches method.
        """
        budget_unit = benchmark_data_schema.iter_unit
        accumulated = benchmark_data_processor.accumulate_best_performances(
            dummy_processing_raw_data, budget_unit
        )
        aligned = benchmark_data_processor.align_tuners_to_shared_budget(
            accumulated, budget_unit
        )
        ranked = benchmark_data_processor.calculate_ranks(aligned, budget_unit)
        result = benchmark_data_processor.accumulate_breaches(ranked, budget_unit)

        params = {"budget_unit": budget_unit}
        self.save_output(result, "accumulate_breaches", params, spot_check_output_dir)

    def test_time_discretize_data_spot_check(
        self,
        benchmark_data_processor,
        dummy_processing_raw_data,
        spot_check_output_dir,
        benchmark_data_schema,
    ):
        """
        Spot check test for time_discretize_data method.
        """
        result = benchmark_data_processor.time_discretize_data(
            dummy_processing_raw_data, benchmark_data_schema.runtime_unit
        )

        params = {"budget_unit": benchmark_data_schema.runtime_unit}
        self.save_output(result, "time_discretize_data", params, spot_check_output_dir)

    def test_standardize_budget_to_percentage_spot_check(
        self,
        benchmark_data_processor,
        dummy_processing_raw_data,
        spot_check_output_dir,
        benchmark_data_schema,
    ):
        """
        Spot check test for standardize_budget_to_percentage method.
        """
        for budget_unit in [
            benchmark_data_schema.iter_unit,
            benchmark_data_schema.runtime_unit,
        ]:
            if budget_unit == benchmark_data_schema.iter_unit:
                # For iteration budget: full flow through accumulate_breaches
                preprocessed_data = (
                    benchmark_data_processor.accumulate_best_performances(
                        dummy_processing_raw_data, budget_unit
                    )
                )
                aligned = benchmark_data_processor.align_tuners_to_shared_budget(
                    preprocessed_data, budget_unit
                )
                ranked = benchmark_data_processor.calculate_ranks(aligned, budget_unit)
                processed_data = benchmark_data_processor.accumulate_breaches(
                    ranked, budget_unit
                )
                metrics_to_keep = ["rank", "best_performance"]
            else:
                # For runtime budget: through calculate_ranks only
                preprocessed_data = benchmark_data_processor.time_discretize_data(
                    dummy_processing_raw_data, budget_unit
                )
                aligned = benchmark_data_processor.align_tuners_to_shared_budget(
                    preprocessed_data, budget_unit
                )
                processed_data = benchmark_data_processor.calculate_ranks(
                    aligned, budget_unit
                )
                metrics_to_keep = ["rank", "best_performance"]

            result = benchmark_data_processor.standardize_budget_to_percentage(
                processed_data, budget_unit, metrics_to_keep
            )

            params = {"budget_unit": budget_unit}
            self.save_output(
                result,
                "standardize_budget_to_percentage",
                params,
                spot_check_output_dir,
            )

    def test_collapse_across_repetitions_spot_check(
        self,
        benchmark_data_processor,
        dummy_processing_raw_data,
        spot_check_output_dir,
        benchmark_data_schema,
    ):
        """
        Spot check test for collapse_across_repetitions method.
        """
        for budget_unit in [
            benchmark_data_schema.iter_unit,
            benchmark_data_schema.runtime_unit,
        ]:
            if budget_unit == benchmark_data_schema.iter_unit:
                # For iteration budget: full flow through accumulate_breaches
                preprocessed_data = (
                    benchmark_data_processor.accumulate_best_performances(
                        dummy_processing_raw_data, budget_unit
                    )
                )
                aligned = benchmark_data_processor.align_tuners_to_shared_budget(
                    preprocessed_data, budget_unit
                )
                ranked = benchmark_data_processor.calculate_ranks(aligned, budget_unit)
                processed_data = benchmark_data_processor.accumulate_breaches(
                    ranked, budget_unit
                )
                metrics = ["rank", "best_performance"]
            else:
                # For runtime budget: through calculate_ranks only
                preprocessed_data = benchmark_data_processor.time_discretize_data(
                    dummy_processing_raw_data, budget_unit
                )
                aligned = benchmark_data_processor.align_tuners_to_shared_budget(
                    preprocessed_data, budget_unit
                )
                processed_data = benchmark_data_processor.calculate_ranks(
                    aligned, budget_unit
                )
                metrics = ["rank", "best_performance"]

            result = benchmark_data_processor.collapse_across_repetitions(
                processed_data, metrics, budget_unit
            )

            params = {"budget_unit": budget_unit}
            self.save_output(
                result, "collapse_across_repetitions", params, spot_check_output_dir
            )

    def test_validate_and_clean_data_spot_check(
        self, benchmark_data_processor, dummy_processing_raw_data, spot_check_output_dir
    ):
        """
        Spot check test for validate_and_clean_data method.
        """
        result = benchmark_data_processor._validate_and_clean_data(
            dummy_processing_raw_data
        )

        params = {}
        self.save_output(
            result, "validate_and_clean_data", params, spot_check_output_dir
        )
