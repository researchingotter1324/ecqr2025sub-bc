from hpobench.config.tuner_configurations import (
    PRECONFORMAL_COMPARISON_CONFIGURATIONS,
    EXTERNAL_TUNING_CONFIGURATIONS,
    LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS,
    ARCHITECTURE_VARIATION_CONFIGURATIONS,
    SAMPLER_VARIATION_CONFIGURATIONS,
    COVERAGE_ANALYSIS_CONFIGURATIONS,
    STATIC_ANALYSIS_ESTIMATOR_ARCHITECTURES,
    QUANTILE_COUNT_VARIATION_CONFIGURATIONS,
    SEARCH_TUNING_EFFECT_CONFIGURATIONS,
)
from hpobench.config.constants import ExperimentParameters
from hpobench.report.analyze import (
    analyze_searcher_tuning_effect,
    analyze_searcher_estimator_comparison,
)
from hpobench.config.schema import BenchmarkDataSchema
from hpobench.report.orchestrate import (
    run_and_analyze_main_benchmark,
    run_static_benchmark,
)
from hpobench.utils import setup_environment
import numpy as np
import random

BASE_RANDOM_STATE = 42
np.random.seed(BASE_RANDOM_STATE)
random.seed(BASE_RANDOM_STATE)

experiment_params = ExperimentParameters()

# Granular run section control
run_sections = {
    "run_coverage_analysis": False,
    "run_sampler_variation_analysis": False,
    "run_architecture_variation_analysis": False,
    "run_external_tuning_analysis": True,
    "run_heteroscedastic_external_tuning_analysis": False,
    "run_skew_external_tuning_analysis": False,
    "run_preconformal_comparison_analysis": False,
    "run_static_analysis": False,
    "run_quantile_count_comparison": False,
    "run_search_tuning_effect_comparison": False,
}


CACHE_PATH = "cache/"
run_start_str, logger = setup_environment(cache_path=CACHE_PATH)

schema = BenchmarkDataSchema()


def main():
    """Main entry point for running benchmarks (task bodies inlined)."""
    if not any(run_sections.values()):
        logger.warning(
            "No tasks to run. Please enable at least one section in run_sections."
        )
        return

    logger.info("Starting sequential execution of enabled tasks")

    # Coverage analysis
    if run_sections.get("run_coverage_analysis", False):
        name = "coverage_analysis"
        logger.info("Starting coverage analysis")
        try:
            run_and_analyze_main_benchmark(
                parallelize=True,
                benchmarks=["LCBench-L"],
                tuning_configurations=COVERAGE_ANALYSIS_CONFIGURATIONS,
                n_warm_starts=experiment_params.n_coverage_warm_starts,
                n_trials=experiment_params.n_trials,
                timeout=experiment_params.timeout,
                base_random_state=BASE_RANDOM_STATE,
                schema=schema,
                cache_path=CACHE_PATH,
                run_start_str=run_start_str,
                analysis_type="01_coverage_analysis",
                max_n_instances_per_benchmark=experiment_params.default_max_n_instances,
                n_repetitions=experiment_params.large_n_repetitions_per_tuner_config,
                starting_coverage_trial=32,
                analysis_components=["coverage"],
            )
            logger.info(f"Completed task: {name}")
        except Exception as e:
            logger.error(f"Error in task {name}: {e}", exc_info=True)

    # Sampler variation
    if run_sections.get("run_sampler_variation_analysis", False):
        name = "sampler_variation"
        logger.info("Starting sampler variation analysis")
        try:
            run_and_analyze_main_benchmark(
                parallelize=True,
                benchmarks=["LCBench-L"],
                tuning_configurations=SAMPLER_VARIATION_CONFIGURATIONS,
                n_warm_starts=experiment_params.n_warm_starts,
                n_trials=experiment_params.n_trials,
                timeout=experiment_params.timeout,
                base_random_state=BASE_RANDOM_STATE,
                cache_path=CACHE_PATH,
                run_start_str=run_start_str,
                analysis_type="02_sampler_variation",
                max_n_instances_per_benchmark=experiment_params.default_max_n_instances,
                n_repetitions=experiment_params.large_n_repetitions_per_tuner_config,
                analysis_components=["rank_analysis"],
                schema=schema,
            )
            logger.info(f"Completed task: {name}")
        except Exception as e:
            logger.error(f"Error in task {name}: {e}", exc_info=True)

    # Architecture variation
    if run_sections.get("run_architecture_variation_analysis", False):
        name = "architecture_variation"
        logger.info("Starting architecture variation analysis")
        try:
            run_and_analyze_main_benchmark(
                parallelize=True,
                benchmarks=["LCBench-L"],
                tuning_configurations=ARCHITECTURE_VARIATION_CONFIGURATIONS,
                n_warm_starts=experiment_params.n_warm_starts,
                n_trials=experiment_params.n_trials,
                timeout=experiment_params.timeout,
                base_random_state=BASE_RANDOM_STATE,
                cache_path=CACHE_PATH,
                run_start_str=run_start_str,
                analysis_type="03_architecture_variation",
                max_n_instances_per_benchmark=experiment_params.default_max_n_instances,
                n_repetitions=experiment_params.large_n_repetitions_per_tuner_config,
                analysis_components=[
                    "architecture_comparison",
                    "rank_analysis",
                    "sampler_comparison",
                ],
                schema=schema,
            )
            logger.info(f"Completed task: {name}")
        except Exception as e:
            logger.error(f"Error in task {name}: {e}", exc_info=True)

    # External tuning
    if run_sections.get("run_external_tuning_analysis", False):
        name = "external_tuning"
        logger.info("Starting external tuning analysis")
        try:
            run_and_analyze_main_benchmark(
                parallelize=True,
                benchmarks=[
                    # "jahs201",
                    "LCBench-L",
                    # "rbv2_aknn-L",
                ],
                tuning_configurations=LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS
                + EXTERNAL_TUNING_CONFIGURATIONS,
                n_warm_starts=experiment_params.n_warm_starts,
                n_trials=experiment_params.n_trials,
                timeout=experiment_params.timeout,
                base_random_state=BASE_RANDOM_STATE,
                cache_path=CACHE_PATH,
                run_start_str=run_start_str,
                analysis_type="04_external_tuning",
                max_n_instances_per_benchmark=experiment_params.default_max_n_instances,
                n_repetitions=experiment_params.medium_n_repetitions_per_tuner_config,
                analysis_components=[
                    "permutation_test",
                    "rank_analysis",
                    "dataset_performances",
                ],
                schema=schema,
            )
            logger.info(f"Completed task: {name}")
        except Exception as e:
            logger.error(f"Error in task {name}: {e}", exc_info=True)

    # Heteroscedastic external tuning
    if run_sections.get("run_heteroscedastic_external_tuning_analysis", False):
        name = "heteroscedastic_external_tuning"
        logger.info("Starting heteroscedastic external tuning analysis")
        try:
            run_and_analyze_main_benchmark(
                parallelize=True,
                benchmarks=["LCBench-H",
                #  "rbv2_aknn-H"
                 ],
                tuning_configurations=LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS
                + EXTERNAL_TUNING_CONFIGURATIONS,
                n_warm_starts=experiment_params.n_warm_starts,
                n_trials=experiment_params.n_trials,
                timeout=experiment_params.timeout,
                base_random_state=BASE_RANDOM_STATE,
                cache_path=CACHE_PATH,
                run_start_str=run_start_str,
                analysis_type="04_heteroskedastic_external_tuning",
                max_n_instances_per_benchmark=experiment_params.default_max_n_instances,
                n_repetitions=experiment_params.medium_n_repetitions_per_tuner_config,
                analysis_components=[
                    "permutation_test",
                    "rank_analysis",
                    "dataset_performances",
                ],
                schema=schema,
            )
            logger.info(f"Completed task: {name}")
        except Exception as e:
            logger.error(f"Error in task {name}: {e}", exc_info=True)

    # Skew external tuning
    if run_sections.get("run_skew_external_tuning_analysis", False):
        name = "skew_external_tuning"
        logger.info("Starting skew external tuning analysis")
        try:
            run_and_analyze_main_benchmark(
                parallelize=True,
                benchmarks=["LCBench-A", 
                # "rbv2_aknn-A"
                ],
                tuning_configurations=LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS
                + EXTERNAL_TUNING_CONFIGURATIONS,
                n_warm_starts=experiment_params.n_warm_starts,
                n_trials=experiment_params.n_trials,
                timeout=experiment_params.timeout,
                base_random_state=BASE_RANDOM_STATE,
                cache_path=CACHE_PATH,
                run_start_str=run_start_str,
                analysis_type="04_skew_external_tuning",
                max_n_instances_per_benchmark=experiment_params.default_max_n_instances,
                n_repetitions=experiment_params.medium_n_repetitions_per_tuner_config,
                analysis_components=[
                    "permutation_test",
                    "rank_analysis",
                    "dataset_performances",
                ],
                schema=schema,
            )
            logger.info(f"Completed task: {name}")
        except Exception as e:
            logger.error(f"Error in task {name}: {e}", exc_info=True)

    # Preconformal comparison
    if run_sections.get("run_preconformal_comparison_analysis", False):
        name = "preconformal_comparison"
        logger.info("Starting pre-conformal comparison analysis")
        try:
            run_and_analyze_main_benchmark(
                parallelize=True,
                benchmarks=["LCBench-L"],
                tuning_configurations=PRECONFORMAL_COMPARISON_CONFIGURATIONS,
                n_warm_starts=experiment_params.n_warm_starts,
                n_trials=experiment_params.n_trials,
                timeout=experiment_params.timeout,
                base_random_state=BASE_RANDOM_STATE,
                cache_path=CACHE_PATH,
                run_start_str=run_start_str,
                analysis_type="05_preconformal_comparison",
                max_n_instances_per_benchmark=experiment_params.default_max_n_instances,
                n_repetitions=experiment_params.medium_n_repetitions_per_tuner_config,
                analysis_components=["friedman", "nemenyi", "conformalization_effect"],
                schema=schema,
            )
            logger.info(f"Completed task: {name}")
        except Exception as e:
            logger.error(f"Error in task {name}: {e}", exc_info=True)

    # Quantile count comparison
    if run_sections.get("run_quantile_count_comparison", False):
        name = "quantile_count_comparison"
        logger.info("Starting quantile count comparison")
        try:
            run_and_analyze_main_benchmark(
                parallelize=True,
                benchmarks=["LCBench-L"],
                tuning_configurations=QUANTILE_COUNT_VARIATION_CONFIGURATIONS,
                n_warm_starts=experiment_params.n_warm_starts,
                n_trials=experiment_params.n_trials,
                timeout=experiment_params.timeout,
                base_random_state=BASE_RANDOM_STATE,
                cache_path=CACHE_PATH,
                run_start_str=run_start_str,
                analysis_type="06_quantile_count_comparison",
                max_n_instances_per_benchmark=experiment_params.default_max_n_instances,
                n_repetitions=experiment_params.large_n_repetitions_per_tuner_config,
                analysis_components=["quantile_count_comparison"],
                schema=schema,
            )
            logger.info(f"Completed task: {name}")
        except Exception as e:
            logger.error(f"Error in task {name}: {e}", exc_info=True)

    # Search tuning effect
    if run_sections.get("run_search_tuning_effect_comparison", False):
        name = "search_tuning_effect"
        logger.info("Starting search tuning effect comparison")
        try:
            run_and_analyze_main_benchmark(
                parallelize=True,
                benchmarks=["LCBench-L"],
                tuning_configurations=SEARCH_TUNING_EFFECT_CONFIGURATIONS,
                n_warm_starts=experiment_params.n_warm_starts,
                n_trials=experiment_params.n_trials,
                timeout=experiment_params.timeout,
                base_random_state=BASE_RANDOM_STATE,
                cache_path=CACHE_PATH,
                run_start_str=run_start_str,
                analysis_type="07_search_tuning_effect",
                max_n_instances_per_benchmark=experiment_params.default_max_n_instances,
                n_repetitions=experiment_params.large_n_repetitions_per_tuner_config,
                analysis_components=["search_tuning_effect_comparison"],
                schema=schema,
            )
            logger.info(f"Completed task: {name}")
        except Exception as e:
            logger.error(f"Error in task {name}: {e}", exc_info=True)

    # Static analysis (estimator error analysis)
    if run_sections.get("run_static_analysis", False):
        name = "static_analysis"
        logger.info("Starting Estimator Error Analysis (STATIC configs)...")
        try:
            static_results = run_static_benchmark(
                benchmarks=["LCBench-L"],
                data_size_range=experiment_params.static_data_sizes,
                estimator_architectures=STATIC_ANALYSIS_ESTIMATOR_ARCHITECTURES,
                n_repetitions_per_estimator=experiment_params.large_n_repetitions_per_tuner_config,
                tuning_iterations_range=experiment_params.static_tuning_iterations,
                alpha=0.4,
                n_pre_conformal_trials=min(experiment_params.static_tuning_iterations)
                - 1,
                max_n_instances=experiment_params.default_max_n_instances,
                base_random_state=BASE_RANDOM_STATE,
            )

            logger.info("Starting Tuning Effect Analysis...")
            analyze_searcher_tuning_effect(
                static_raw_benchmark_data=static_results,
                cache_path=CACHE_PATH,
                run_start_str=run_start_str,
                analysis_type="05_static_analysis",
                schema=schema,
            )
            logger.info("Tuning Effect Analysis finished.")

            logger.info("Starting Estimator Comparison Analysis...")
            analyze_searcher_estimator_comparison(
                static_raw_benchmark_data=static_results,
                cache_path=CACHE_PATH,
                run_start_str=run_start_str,
                analysis_type="05_static_analysis",
                schema=schema,
            )
            logger.info("Estimator Comparison Analysis finished.")
        except Exception as e:
            logger.error(f"Error in task {name}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
