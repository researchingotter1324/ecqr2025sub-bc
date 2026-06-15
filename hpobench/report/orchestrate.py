import pandas as pd
import concurrent.futures
import multiprocessing
from datetime import datetime
import os
import logging
from typing import Literal, Optional
import gc
import numpy as np
from sklearn.metrics import mean_pinball_loss

try:
    from ccqr_optimization.selection.conformalization import QuantileConformalEstimator
    from ccqr_optimization.utils.configurations.encoding import ConfigurationEncoder
    from ccqr_optimization.selection.sampling.utils import initialize_quantile_alphas
except ImportError:
    raise ImportError(
        "ccqr_optimization is a core dependency of this repository, but it is not automatically installed via pyproject.toml, please refer to the README.md for instructions on how to install this separately"
    )
from hpobench.config.config_types import (
    ExperimentConfig,
    TunerConfig,
)
from hpobench.tune import setup_ccqr_optimization_params
from hpobench.report.utils import generate_configs_per_repetition
from hpobench.utils import (
    generate_hyperparameter_combinations,
    add_runtime,
)
from hpobench.prepare import (
    setup_yahpo_instance_configs,
    setup_jahs201_configs,
    setup_nas301_configs,
)
from hpobench.config.schema import BenchmarkDataSchema
from hpobench.config.constants import Aliases, ExperimentParameters
from hpobench.config.utils import fmt_float

from hpobench.tune import tune
from hpobench.report.analyze import (
    analyze_main_benchmark,
    analyze_joint_architecture_and_static,
    analyze_joint_candidates_and_extreme_quantile,
    analyze_ei_architecture,
)

logger = logging.getLogger(__name__)

aliases = Aliases()


def load_experiment_configs(
    benchmarks: list[
        Literal[
            "jahs201",
            "lcbench",
            "rbv2_aknn",
            "LCBench-L",
            "LCBench-H",
            "LCBench-A",
            "rbv2_aknn-L",
            "rbv2_aknn-H",
            "rbv2_aknn-A",
            "nas301",
        ]
    ],
    tuning_configurations: list[TunerConfig],
    n_warm_starts: int,
    n_trials: int,
    timeout: Optional[float],
    max_n_instances_per_benchmark: int = 10,
    datasets_per_benchmark: Optional[list[list[str]]] = None,
) -> list[ExperimentConfig]:
    """Load and configure benchmark instances for hyperparameter optimization experiments.

    This function sets up experiment configurations for different HPO benchmarks, handling
    the specific initialization requirements for YAHPO (RBVS2 XGBoost, LCBench) and JAHS-Bench-201
    datasets. For JAHS-Bench-201, it automatically selects datasets up to the specified limit,
    prioritizing CIFAR-10, Fashion-MNIST, and colorectal histology datasets.

    Args:
        benchmarks: List of benchmark names to initialize.         Supported benchmarks are:
            - "jahs201": JAHS-Bench-201 neural architecture search benchmark
            - "lcbench": Learning Curves Benchmark for machine learning algorithms
            - "rbv2_aknn": RBVS2 XGBoost benchmark from YAHPO suite
            - "LCBench-L": LCBench subset with largest datasets
            - "LCBench-H": LCBench subset with most heteroscedastic datasets
            - "LCBench-A": LCBench subset with most skewed datasets
            - "rbv2_aknn-L": RBV2 XGBoost subset with largest datasets
            - "rbv2_aknn-H": RBV2 XGBoost subset with most heteroscedastic datasets
            - "rbv2_aknn-A": RBV2 XGBoost subset with most skewed datasets
        tuning_configurations: List of tuner configurations defining the HPO algorithms
            and their parameters to be evaluated on each benchmark instance.
        n_warm_starts: Number of initial random hyperparameter configurations to generate
            for each tuner to ensure fair comparison across different optimization methods.
        n_trials: Total number of hyperparameter evaluation trials per tuner configuration,
            including warm start trials.
        timeout: Maximum time in seconds allowed for each individual hyperparameter
            evaluation. None for no timeout limit.
        max_n_instances_per_benchmark: Maximum number of dataset instances to use per
            benchmark.
        datasets_per_benchmark: Optional list of lists, each containing specific dataset
            identifiers to use for the corresponding benchmark. If provided, overrides
            the default dataset selection logic for benchmarks.

    Returns:
        List of ExperimentConfig objects, each containing a benchmark instance paired
        with its search space, objective function, and tuning parameters. The number
        of configs returned depends on the benchmarks selected and max_n_instances_per_benchmark.
    """
    logger.info("Setting up benchmark instances...")

    experiment_configs = []
    for i, benchmark in enumerate(benchmarks):
        if benchmark in [
            "rbv2_aknn",
            "lcbench",
            "LCBench-L",
            "LCBench-H",
            "LCBench-A",
            "rbv2_aknn-L",
            "rbv2_aknn-H",
            "rbv2_aknn-A",
        ]:
            configs = setup_yahpo_instance_configs(
                benchmark=benchmark,
                tuning_configurations=tuning_configurations,
                n_warm_starts=n_warm_starts,
                n_trials=n_trials,
                timeout=timeout,
                max_n_instances=max_n_instances_per_benchmark,
            )
            experiment_configs.extend(configs)

    if "jahs201" in benchmarks:
        idx = benchmarks.index("jahs201")
        all_datasets = ["cifar10", "fashion_mnist", "colorectal_histology"]
        if (
            datasets_per_benchmark is not None
            and datasets_per_benchmark[idx] is not None
        ):
            selected_datasets = datasets_per_benchmark[idx]
        elif max_n_instances_per_benchmark < len(all_datasets):
            selected_datasets = all_datasets[:max_n_instances_per_benchmark]
        else:
            selected_datasets = all_datasets

        configs = setup_jahs201_configs(
            datasets=selected_datasets,
            tuning_configurations=tuning_configurations,
            n_warm_starts=n_warm_starts,
            n_trials=n_trials,
            timeout=timeout,
        )
        experiment_configs.extend(configs)

    if "nas301" in benchmarks:
        idx = benchmarks.index("nas301")
        all_datasets = ["CIFAR10"]  # NAS-301 only has CIFAR10 dataset
        if (
            datasets_per_benchmark is not None
            and datasets_per_benchmark[idx] is not None
        ):
            selected_datasets = datasets_per_benchmark[idx]
        elif max_n_instances_per_benchmark < len(all_datasets):
            selected_datasets = all_datasets[:max_n_instances_per_benchmark]
        else:
            selected_datasets = all_datasets

        configs = setup_nas301_configs(
            datasets=selected_datasets,
            tuning_configurations=tuning_configurations,
            n_warm_starts=n_warm_starts,
            n_trials=n_trials,
            timeout=timeout,
        )
        experiment_configs.extend(configs)

    return experiment_configs


def process_single_experiment_config(
    experiment_config: ExperimentConfig,
    n_repetitions: int,
    base_random_state: int,
) -> pd.DataFrame:
    dataset_name = experiment_config.dataset_identifier
    logger.info(f"Worker | Dataset: {dataset_name}")

    logger.info(f"Worker | Initializing generator for dataset: {dataset_name}...")
    experiment_config.objective_function.initialize()

    logger.info(
        f"Worker | Generating {experiment_config.n_warm_starts} warm start configurations for dataset: {dataset_name}"
    )
    # NOTE: Warm starts are identical per repetition, so all models
    # will have the same starting hyperparameter configurations, but
    # a new set of warm starts needs to be generated per dataset and
    # per repetition.
    warm_start_configs_per_repetition = []
    for repetition in range(n_repetitions):
        consistent_warm_starts = generate_hyperparameter_combinations(
            params=experiment_config.search_space,
            n_combinations=experiment_config.n_warm_starts,
            random_state=base_random_state + repetition,
        )
        warm_start_configs = []
        for combination in consistent_warm_starts:
            performance = experiment_config.objective_function.predict(configuration=combination)
            warm_start_configs.append((combination, performance))
        warm_start_configs_per_repetition.append(warm_start_configs)
    logger.info(
        f"Worker | Generated {len(warm_start_configs_per_repetition[0])} warm start configurations."
    )

    dataset_benchmark_data = pd.DataFrame()

    for tuner in experiment_config.tuner_configurations:
        logger.info(f"Worker | Tuner: {tuner}")
        for repetition in range(n_repetitions):
            logger.info(f"Worker | Repetition: {repetition}")
            tune_start = datetime.now()

            # Grab the warm start configurations for this repetition (shared by all tuners):
            historical_performance = tune(
                performance_generator=experiment_config.objective_function,
                tuner_config=tuner,
                n_trials=experiment_config.n_trials,
                timeout=experiment_config.timeout,
                params=experiment_config.search_space,
                warm_start_configs=warm_start_configs_per_repetition[repetition],
                random_state=base_random_state + repetition,
            )

            historical_performance = add_runtime(
                experiment_log=historical_performance,
                tune_start=tune_start,
                performance_generator=experiment_config.objective_function,
                n_warm_starts=len(warm_start_configs_per_repetition[repetition]),
            )

            aliased_benchmark_identifier = (
                aliases.benchmark_aliases[experiment_config.benchmark_identifier]
                if experiment_config.benchmark_identifier
                in aliases.benchmark_aliases
                else experiment_config.benchmark_identifier
            )
            historical_performance[
                "benchmark_identifier"
            ] = aliased_benchmark_identifier
            historical_performance["dataset"] = dataset_name
            historical_performance["tuner"] = tuner.tuner_identifier
            historical_performance["repetition"] = repetition + 1
            historical_performance[
                "searcher_tuning_framework"
            ] = tuner.searcher_tuning_framework

            if tuner.tuner.backend == "ccqr_optimization":
                sampler = tuner.tuner.searcher.sampler
                sampler_name = sampler.__class__.__name__
                has_local_search_attr = hasattr(sampler, "local_search")
                use_local_search_ccqr = getattr(sampler, "local_search", None) is not None

                if hasattr(sampler, "interval_width") and sampler.interval_width is not None:
                    confidence_level = fmt_float(sampler.interval_width)
                else:
                    confidence_level = ""

                estimator_architecture = (
                    tuner.tuner.searcher.quantile_estimator_architecture
                )

                if hasattr(tuner.tuner.searcher, "n_pre_conformal_trials"):
                    n_pre_conformal_trials = (
                        tuner.tuner.searcher.n_pre_conformal_trials
                    )
                else:
                    n_pre_conformal_trials = ""

                if hasattr(sampler, "max_quantiles"):
                    sampler_n_quantiles = sampler.max_quantiles
                elif hasattr(sampler, "n_quantiles"):
                    sampler_n_quantiles = sampler.n_quantiles
                else:
                    sampler_n_quantiles = ""

                if hasattr(sampler, "adapter"):
                    if sampler.adapter is None:
                        sampler_adapter = "None"
                    else:
                        sampler_adapter = type(sampler.adapter).__name__
                else:
                    sampler_adapter = ""

                if tuner.searcher_tuning_framework is None:
                    tuner_searcher_tuning_framework = "None"
                else:
                    tuner_searcher_tuning_framework = str(
                        tuner.searcher_tuning_framework
                    )
            else:
                # NOTE: Use "" instead of None or NaN to avoid bad groupby behavior
                sampler_name = ""
                use_local_search_ccqr = False  # non-ccqr backends never have local search
                confidence_level = ""
                estimator_architecture = ""
                n_pre_conformal_trials = ""
                sampler_n_quantiles = ""
                sampler_adapter = ""
                tuner_searcher_tuning_framework = ""

            aliased_estimator_architecture = (
                aliases.architecture_aliases[estimator_architecture]
                if estimator_architecture in aliases.architecture_aliases
                else estimator_architecture
            )
            aliased_sampler_name = (
                aliases.sampler_aliases[sampler_name]
                if sampler_name in aliases.sampler_aliases
                else sampler_name
            )
            if tuner.tuner.backend == "ccqr_optimization":
                if sampler_name == "ThompsonSampler":
                    if sampler.enable_optimistic_sampling:
                        aliased_sampler_name = "OBS"
                elif sampler_name in ("LowerBoundSampler", "PessimisticLowerBoundSampler"):
                    # Build a stable, fully-qualified suffix so each distinct
                    # (iw, c, beta_decay) combination gets its own plotting group.
                    parts = []
                    if hasattr(sampler, "interval_width") and sampler.interval_width is not None:
                        parts.append(f"iw{fmt_float(sampler.interval_width)}")
                    if hasattr(sampler, "c") and sampler.c is not None:
                        parts.append(f"c{fmt_float(sampler.c)}")
                    if hasattr(sampler, "beta_decay") and sampler.beta_decay:
                        decay_parts = sampler.beta_decay.split("_")
                        parts.append("".join(p[0] for p in decay_parts))
                    if parts:
                        aliased_sampler_name = f"{aliased_sampler_name}-{'_'.join(parts)}"
            if tuner.tuner.backend == "ccqr_optimization":
                if has_local_search_attr and not use_local_search_ccqr:
                    aliased_estimator_architecture = f"NL-{aliased_estimator_architecture}"
            historical_performance[
                "estimator_architecture"
            ] = aliased_estimator_architecture
            historical_performance["confidence_level"] = confidence_level
            historical_performance["sampler"] = aliased_sampler_name
            historical_performance[
                "n_pre_conformal_trials"
            ] = n_pre_conformal_trials
            historical_performance["sampler_n_quantiles"] = sampler_n_quantiles
            historical_performance["n_candidates"] = tuner.n_candidates
            historical_performance["sampler_adapter"] = sampler_adapter
            historical_performance[
                "tuner_searcher_tuning_framework"
            ] = tuner_searcher_tuning_framework

            dataset_benchmark_data = pd.concat(
                [dataset_benchmark_data, historical_performance], axis=0
            )

    # Free up memory after processing each experiment config:
    experiment_config.objective_function = None
    gc.collect()

    return dataset_benchmark_data


def run_main_benchmark(
    experiment_configs: list[ExperimentConfig],
    n_repetitions: int,
    cache_path: str,
    run_start_str: str,
    base_random_state: Optional[int] = None,
    parallelize: bool = False,
) -> pd.DataFrame:
    """Execute the core hyperparameter optimization benchmark experiments."""
    logger.info("Running HPO benchmark...")

    incremental_data_path = os.path.join(cache_path, f"data/{run_start_str}")
    os.makedirs(incremental_data_path, exist_ok=True)
    incremental_csv_path = os.path.join(incremental_data_path, "incremental_raw_benchmark_data.csv")

    raw_benchmark_data = pd.DataFrame()

    if parallelize:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
        logger.info(f"Running in parallel across {len(experiment_configs)} datasets using {max_workers} workers.")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_single_experiment_config,
                    experiment_config=config,
                    n_repetitions=n_repetitions,
                    base_random_state=base_random_state
                ): config
                for config in experiment_configs
            }
            
            for future in concurrent.futures.as_completed(futures):
                config = futures[future]
                dataset_df = future.result()
                raw_benchmark_data = pd.concat([raw_benchmark_data, dataset_df], axis=0)
                
                write_header = not os.path.exists(incremental_csv_path)
                dataset_df.to_csv(incremental_csv_path, mode='a', header=write_header, index=False)
                logger.info(f"Successfully processed and saved dataset: {config.dataset_identifier}")
    else:
        logger.info("Running sequentially.")
        for experiment_config in experiment_configs:
            dataset_df = process_single_experiment_config(
                experiment_config=experiment_config,
                n_repetitions=n_repetitions,
                base_random_state=base_random_state
            )
            raw_benchmark_data = pd.concat([raw_benchmark_data, dataset_df], axis=0)
            
            write_header = not os.path.exists(incremental_csv_path)
            dataset_df.to_csv(incremental_csv_path, mode='a', header=write_header, index=False)
            logger.info(f"Successfully processed and saved dataset: {experiment_config.dataset_identifier}")

    final_data_path = os.path.join(cache_path, f"data/{run_start_str}")
    os.makedirs(final_data_path, exist_ok=True)
    final_filename = os.path.join(final_data_path, "raw_benchmark_data.csv")
    raw_benchmark_data.to_csv(final_filename, index=False)
    logger.info(
        f"Final raw benchmark data saved to {final_filename} ({len(raw_benchmark_data)} rows)."
    )
    return raw_benchmark_data

def run_and_analyze_main_benchmark(
    benchmarks: list[
        Literal[
            "jahs201",
            "lcbench",
            "rbv2_aknn",
            "LCBench-L",
            "LCBench-H",
            "LCBench-A",
            "rbv2_aknn-L",
            "rbv2_aknn-H",
            "rbv2_aknn-A",
        ]
    ],
    tuning_configurations: list[TunerConfig],
    n_warm_starts: int,
    n_trials: int,
    timeout: Optional[float],
    base_random_state: int,
    schema: BenchmarkDataSchema,
    cache_path: str,
    run_start_str: str,
    analysis_type: str,
    analysis_components: list[
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
    max_n_instances_per_benchmark: int = 10,
    starting_coverage_trial: Optional[int] = None,
    n_repetitions: int = 10,
    datasets_per_benchmark: Optional[list[list[str]]] = None,
    parallelize: bool = False,
) -> pd.DataFrame:
    """
    Complete end-to-end hyperparameter optimization benchmark pipeline with analysis.

    The function supports various analysis types corresponding to different research
    questions in the HPO literature:
    - Coverage analysis: Evaluates conformal prediction interval validity
    - Sampler variation: Compares different acquisition functions and sampling strategies
    - Architecture variation: Studies impact of surrogate model architectures
    - External tuning: Benchmarks against established HPO frameworks
    - Preconformal comparison: Analyzes effect of pre-conformal training phases

    Args:
        benchmarks: List of benchmark datasets to evaluate. Each benchmark provides
            different characteristics (search space dimensionality, evaluation cost, etc.):
            - "jahs201": Neural architecture search with expensive evaluations
            - "lcbench": Classical ML algorithms with learning curve data
            - "rbv2_aknn": Gradient boosting hyperparameter optimization
            - "LCBench-L": LCBench subset with largest datasets
            - "LCBench-H": LCBench subset with most heteroscedastic datasets
            - "LCBench-A": LCBench subset with most skewed datasets
            - "rbv2_aknn-L": RBV2 XGBoost subset with largest datasets
            - "rbv2_aknn-H": RBV2 XGBoost subset with most heteroscedastic datasets
            - "rbv2_aknn-A": RBV2 XGBoost subset with most skewed datasets
        tuning_configurations: HPO algorithms and their parameter settings to compare.
            Should include both ccqr_optimization-based methods and baseline algorithms for
            comprehensive evaluation.
        n_warm_starts: Number of random initial configurations per tuner to ensure
            fair comparison. Typically 10-20 for small search spaces, more for complex ones.
        n_trials: Total hyperparameter evaluations per tuner run. Should be sufficient
            to reach convergence - typically 100-500 depending on search space complexity.
        timeout: Per-evaluation time limit in seconds. Critical for expensive benchmarks
            like JAHS-Bench-201 where individual evaluations can take minutes.
        base_random_state: Seed for reproducible experiments. All randomness in the
            experimental pipeline derives from this seed to ensure exact reproducibility.
        cache_path: Directory for storing experimental data, plots, and analysis results.
            Should have sufficient space (several GB for large experiments).
        run_start_str: Unique identifier for this experimental run, typically a timestamp.
            Used to organize results and prevent conflicts between concurrent experiments.
        analysis_type: Identifier for the type of analysis being performed, used in
            result organization and plot titles. Examples: "01_coverage_analysis",
            "02_sampler_variation", "03_architecture_variation".
        analysis_components: List of specific analyses to perform on the experimental data:
            - "friedman": Friedman test for overall statistical differences
            - "nemenyi": Post-hoc Nemenyi test for pairwise comparisons
            - "coverage": Conformal prediction interval coverage validation
            - "dataset_performances": Per-dataset performance breakdowns
            - "rank_analysis": Algorithm ranking analysis across datasets
            - "sampler_comparison": Detailed comparison of sampling strategies
            - "architecture_comparison": Analysis of surrogate model architectures
            - "conformalization_effect": Impact analysis of conformalization
        max_n_instances_per_benchmark: Limit on dataset instances per benchmark to
            control experimental scope and runtime. Use smaller values for initial
            experiments or when computational resources are limited.
        n_repetitions: Number of independent experimental repetitions for statistical
            validity. Minimum 10 recommended for meaningful confidence intervals,
            30+ for publication-quality results.
        datasets_per_benchmark: Optional list of lists, each containing specific dataset
            identifiers to use for the corresponding benchmark. If provided, overrides
            the default dataset selection logic for benchmarks.

    Returns:
        Complete experimental dataset as DataFrame with all trial results, performance
        metrics, metadata, and derived features needed for analysis. This data serves
        as input to the analysis functions and can be used for custom analysis.
    """
    experiment_configs = load_experiment_configs(
        benchmarks=benchmarks,
        tuning_configurations=tuning_configurations,
        n_warm_starts=n_warm_starts,
        n_trials=n_trials,
        timeout=timeout,
        max_n_instances_per_benchmark=max_n_instances_per_benchmark,
        datasets_per_benchmark=datasets_per_benchmark,
    )

    raw_benchmark_data = run_main_benchmark(
        experiment_configs=experiment_configs,
        n_repetitions=n_repetitions,
        base_random_state=base_random_state,
        cache_path=cache_path,
        run_start_str=run_start_str,
        parallelize=parallelize,
    )

    analyze_main_benchmark(
        raw_benchmark_data=raw_benchmark_data,
        cache_path=cache_path,
        run_start_str=run_start_str,
        analysis_type=analysis_type,
        analysis_components=analysis_components,
        schema=schema,
        starting_coverage_trial=starting_coverage_trial,
    )

    return raw_benchmark_data


def run_and_analyze_joint_benchmark(
    benchmarks: list[
        Literal[
            "jahs201",
            "lcbench",
            "rbv2_aknn",
            "LCBench-L",
            "LCBench-H",
            "LCBench-A",
            "rbv2_aknn-L",
            "rbv2_aknn-H",
            "rbv2_aknn-A",
        ]
    ],
    tuning_configurations: list[TunerConfig],
    n_warm_starts: int,
    n_trials: int,
    timeout: Optional[float],
    base_random_state: int,
    cache_path: str,
    run_start_str: str,
    analysis_type: str,
    schema: BenchmarkDataSchema,
    experiment_params: ExperimentParameters,
    static_architectures: list[str],
    max_n_instances_per_benchmark: int = 10,
    n_repetitions: int = 10,
    datasets_per_benchmark: Optional[list[list[str]]] = None,
    parallelize: bool = False,
) -> None:
    """Execute main architecture variation and static benchmark, then plot jointly."""
    experiment_configs = load_experiment_configs(
        benchmarks=benchmarks,
        tuning_configurations=tuning_configurations,
        n_warm_starts=n_warm_starts,
        n_trials=n_trials,
        timeout=timeout,
        max_n_instances_per_benchmark=max_n_instances_per_benchmark,
        datasets_per_benchmark=datasets_per_benchmark,
    )

    raw_main_data = run_main_benchmark(
        experiment_configs=experiment_configs,
        n_repetitions=n_repetitions,
        base_random_state=base_random_state,
        cache_path=cache_path,
        run_start_str=run_start_str,
        parallelize=parallelize,
    )

    static_results = run_static_benchmark(
        benchmarks=benchmarks,
        data_size_range=experiment_params.static_data_sizes,
        estimator_architectures=static_architectures,
        n_repetitions_per_estimator=experiment_params.large_n_repetitions_per_tuner_config,
        tuning_iterations_range=experiment_params.static_tuning_iterations,
        n_quantiles=experiment_params.static_n_quantiles,
        n_pre_conformal_trials=min(experiment_params.static_tuning_iterations) - 1 if min(experiment_params.static_tuning_iterations) > 0 else 0,
        max_n_instances=experiment_params.default_max_n_instances,
        base_random_state=base_random_state,
    )

    analyze_joint_architecture_and_static(
        main_raw_data=raw_main_data,
        static_raw_data=static_results,
        cache_path=cache_path,
        run_start_str=run_start_str,
        analysis_type=analysis_type,
        schema=schema,
    )


def run_and_analyze_joint_candidates_extreme_quantile_benchmark(
    benchmarks: list[
        Literal[
            "jahs201",
            "lcbench",
            "rbv2_aknn",
            "LCBench-L",
            "LCBench-H",
            "LCBench-A",
            "rbv2_aknn-L",
            "rbv2_aknn-H",
            "rbv2_aknn-A",
        ]
    ],
    tuning_configurations: list[TunerConfig],
    n_warm_starts: int,
    n_trials: int,
    timeout: Optional[float],
    base_random_state: int,
    cache_path: str,
    run_start_str: str,
    analysis_type: str,
    schema: BenchmarkDataSchema,
    max_n_instances_per_benchmark: int = 10,
    n_repetitions: int = 10,
    datasets_per_benchmark: Optional[list[list[str]]] = None,
    parallelize: bool = False,
) -> None:
    """Execute the candidate-count benchmark, then jointly plot search ranks and extreme-quantile usage.

    The tuning configurations must all share a single estimator architecture and a
    single sampler, varying only in the number of candidates; this is enforced
    downstream in the analysis step.
    """
    experiment_configs = load_experiment_configs(
        benchmarks=benchmarks,
        tuning_configurations=tuning_configurations,
        n_warm_starts=n_warm_starts,
        n_trials=n_trials,
        timeout=timeout,
        max_n_instances_per_benchmark=max_n_instances_per_benchmark,
        datasets_per_benchmark=datasets_per_benchmark,
    )

    raw_benchmark_data = run_main_benchmark(
        experiment_configs=experiment_configs,
        n_repetitions=n_repetitions,
        base_random_state=base_random_state,
        cache_path=cache_path,
        run_start_str=run_start_str,
        parallelize=parallelize,
    )

    analyze_joint_candidates_and_extreme_quantile(
        raw_benchmark_data=raw_benchmark_data,
        cache_path=cache_path,
        run_start_str=run_start_str,
        analysis_type=analysis_type,
        schema=schema,
    )


def run_and_analyze_ei_architecture_benchmark(
    benchmarks: list[
        Literal[
            "jahs201",
            "lcbench",
            "rbv2_aknn",
            "LCBench-L",
            "LCBench-H",
            "LCBench-A",
            "rbv2_aknn-L",
            "rbv2_aknn-H",
            "rbv2_aknn-A",
        ]
    ],
    tuning_configurations: list[TunerConfig],
    n_warm_starts: int,
    n_trials: int,
    timeout: Optional[float],
    base_random_state: int,
    cache_path: str,
    run_start_str: str,
    analysis_type: str,
    schema: BenchmarkDataSchema,
    max_n_instances_per_benchmark: int = 10,
    n_repetitions: int = 10,
    datasets_per_benchmark: Optional[list[list[str]]] = None,
    parallelize: bool = False,
) -> None:
    """Run benchmark with EI-based configurations and produce the EI architecture tri-plot.

    The tuning configurations must share exactly one EI sampler across multiple
    estimator architectures; this is enforced downstream in the analysis step.
    """
    experiment_configs = load_experiment_configs(
        benchmarks=benchmarks,
        tuning_configurations=tuning_configurations,
        n_warm_starts=n_warm_starts,
        n_trials=n_trials,
        timeout=timeout,
        max_n_instances_per_benchmark=max_n_instances_per_benchmark,
        datasets_per_benchmark=datasets_per_benchmark,
    )

    raw_benchmark_data = run_main_benchmark(
        experiment_configs=experiment_configs,
        n_repetitions=n_repetitions,
        base_random_state=base_random_state,
        cache_path=cache_path,
        run_start_str=run_start_str,
        parallelize=parallelize,
    )

    analyze_ei_architecture(
        raw_benchmark_data=raw_benchmark_data,
        cache_path=cache_path,
        run_start_str=run_start_str,
        analysis_type=analysis_type,
        schema=schema,
    )


def run_static_benchmark(
    benchmarks: list[
        Literal[
            "jahs201",
            "LCBench-L",
            "LCBench-H",
            "LCBench-A",
            "rbv2_aknn-L",
            "rbv2_aknn-H",
            "rbv2_aknn-A",
        ]
    ],
    data_size_range: list[int],
    estimator_architectures: list[str],
    n_repetitions_per_estimator: int,
    tuning_iterations_range: list[int],
    n_quantiles: int,
    n_pre_conformal_trials: int,
    max_n_instances: int,
    base_random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Evaluate conformal prediction estimator architectures in controlled static setting.

    Supports lcbench, rbv2_aknn, and jahs201 benchmark variants.

    This function performs a controlled evaluation of different quantile estimator
    architectures for conformal prediction by training on fixed datasets of varying
    sizes and evaluating prediction interval quality on holdout sets. Unlike the main
    benchmark which focuses on optimization performance, this analysis isolates the
    impact of estimator choice and training data size on conformal prediction accuracy.

    The experimental design follows a train/calibration/test split paradigm:
    1. Generate training configurations from the search space
    2. Split training data into model training and conformal calibration sets
    3. Train quantile estimators with optional hyperparameter tuning
    4. Generate holdout test configurations for unbiased evaluation
    5. Evaluate holdout prediction interval quality using pinball loss metrics

    Alpha values are derived from ``n_quantiles`` using the same
    ``initialize_quantile_alphas`` logic used by the CCQR samplers, so the spread
    of quantile levels evaluated here mirrors the HPO benchmark. The mean pinball
    loss is then averaged across all quantile pairs, giving a holistic picture of
    distributional prediction quality rather than focusing on a single pair of tails.

    Args:
        benchmarks: List of benchmark names to evaluate. Supported benchmarks are:
            - "LCBench-L": LCBench subset with largest datasets
            - "LCBench-H": LCBench subset with most heteroscedastic datasets
            - "LCBench-A": LCBench subset with most skewed datasets
            - "rbv2_aknn-L": RBV2 XGBoost subset with largest datasets
            - "rbv2_aknn-H": RBV2 XGBoost subset with most heteroscedastic datasets
            - "rbv2_aknn-A": RBV2 XGBoost subset with most skewed datasets
            - "jahs201": JAHS-Bench-201 neural architecture search benchmark
        data_size_range: List of training dataset sizes to evaluate. Allows studying
            how estimator performance scales with available data, typically ranging
            from small (50-100) to moderate (500-1000) sample sizes.
        estimator_architectures: List of quantile estimator architecture (ccqr_optimization package)
            names to compare. Examples: "lgb", "rf", "linear", "nn". Each architecture
            implements different inductive biases for performance surface modeling.
        n_repetitions_per_estimator: Number of independent experimental repetitions
            per estimator-dataset-size combination to ensure statistical reliability
            of the comparison results.
        tuning_iterations_range: List of hyperparameter tuning iteration counts to
            evaluate the effect of estimator tuning on prediction quality. Include 0
            for no tuning and positive values for tuned estimators.
        n_quantiles: Number of quantiles (must be even) used for the static benchmark.
            Alpha values are derived via ``initialize_quantile_alphas(n_quantiles)``,
            mirroring the logic used by CCQR samplers. The mean pinball loss is then
            averaged across all resulting quantile pairs so that the evaluation covers
            the full predicted distribution rather than a single pair of tails.
        n_pre_conformal_trials: Minimum number of observations required before enabling
            conformal prediction. Below this threshold, the estimator returns uninformative
            wide intervals as a safety mechanism.
        max_n_instances: Maximum number of dataset instances to use per benchmark,
            controlling the scope of the evaluation. Use smaller values for initial
            analysis or resource-limited environments.

    Returns:
        DataFrame containing estimator evaluation results with columns:
        - 'estimator_architecture': Name of the quantile estimator architecture tested
        - 'dataset': Dataset instance identifier within the benchmark
        - 'benchmark_identifier': Benchmark name for organization
        - 'repetition': Experimental repetition number for statistical analysis
        - 'tuning_iterations': Number of hyperparameter tuning iterations applied
        - 'data_size': Training dataset size used for this evaluation
        - 'n_quantiles': Number of quantiles used to derive the alpha spread
        - 'mean_pinball_loss': Mean pinball loss averaged across all quantile pairs,
          measuring full-distribution prediction quality (lower is better)
    """
    alphas = initialize_quantile_alphas(n_quantiles)

    estimator_error_results = []

    # Process each benchmark
    for benchmark in benchmarks:
        logger.info(f"Processing benchmark: {benchmark}")
        experiment_configs = []
        if benchmark == "LCBench-L":
            # NOTE: Below we use setup function as shortcut, but we are only interested in
            # the yahpo generator and param space generation, the other inputs are
            # just placeholders:
            yahpo_configs = setup_yahpo_instance_configs(
                benchmark="LCBench-L",  # hard coded, leave as is
                tuning_configurations=[],  # placeholder
                n_warm_starts=1,  # placeholder
                n_trials=0,  # placeholder
                timeout=100000,  # placeholder
                max_n_instances=max_n_instances,
            )
            experiment_configs.extend(yahpo_configs)

        elif benchmark == "LCBench-H":
            yahpo_configs = setup_yahpo_instance_configs(
                benchmark="LCBench-H",  # hard coded, leave as is
                tuning_configurations=[],  # placeholder
                n_warm_starts=1,  # placeholder
                n_trials=0,  # placeholder
                timeout=100000,  # placeholder
                max_n_instances=max_n_instances,
            )
            experiment_configs.extend(yahpo_configs)

        elif benchmark == "LCBench-A":
            yahpo_configs = setup_yahpo_instance_configs(
                benchmark="LCBench-A",  # hard coded, leave as is
                tuning_configurations=[],  # placeholder
                n_warm_starts=1,  # placeholder
                n_trials=0,  # placeholder
                timeout=100000,  # placeholder
                max_n_instances=max_n_instances,
            )
            experiment_configs.extend(yahpo_configs)

        elif benchmark == "rbv2_aknn-L":
            yahpo_configs = setup_yahpo_instance_configs(
                benchmark="rbv2_aknn-L",  # hard coded, leave as is
                tuning_configurations=[],  # placeholder
                n_warm_starts=1,  # placeholder
                n_trials=0,  # placeholder
                timeout=100000,  # placeholder
                max_n_instances=max_n_instances,
            )
            experiment_configs.extend(yahpo_configs)

        elif benchmark == "rbv2_aknn-H":
            yahpo_configs = setup_yahpo_instance_configs(
                benchmark="rbv2_aknn-H",  # hard coded, leave as is
                tuning_configurations=[],  # placeholder
                n_warm_starts=1,  # placeholder
                n_trials=0,  # placeholder
                timeout=100000,  # placeholder
                max_n_instances=max_n_instances,
            )
            experiment_configs.extend(yahpo_configs)

        elif benchmark == "rbv2_aknn-A":
            yahpo_configs = setup_yahpo_instance_configs(
                benchmark="rbv2_aknn-A",  # hard coded, leave as is
                tuning_configurations=[],  # placeholder
                n_warm_starts=1,  # placeholder
                n_trials=0,  # placeholder
                timeout=100000,  # placeholder
                max_n_instances=max_n_instances,
            )
            experiment_configs.extend(yahpo_configs)

        elif benchmark == "jahs201":
            # NOTE: Use setup function as shortcut, but we are only interested in
            # the objective function and search space generation, the other inputs are
            # just placeholders:
            all_datasets = ["cifar10", "fashion_mnist", "colorectal_histology"]
            selected_datasets = (
                all_datasets[:max_n_instances]
                if max_n_instances < len(all_datasets)
                else all_datasets
            )
            jahs201_configs = setup_jahs201_configs(
                datasets=selected_datasets,
                tuning_configurations=[],  # placeholder
                n_warm_starts=1,  # placeholder
                n_trials=0,  # placeholder
                timeout=100000,  # placeholder
            )
            experiment_configs.extend(jahs201_configs)

        else:
            raise ValueError(f"Unsupported benchmark: {benchmark}")

        # NOTE: Create single population per experiment (single source of truth)
        populations = {}
        for experiment_config in experiment_configs:
            logger.info(
                f"Preparing population for experiment: {experiment_config.dataset_identifier}"
            )

            # Create single large population that will be sampled from for both experiment and holdout
            population = generate_configs_per_repetition(
                search_space=experiment_config.search_space,
                n_configs=50000,
                n_repetitions=1,
                base_seed=base_random_state,
                objective_function=experiment_config.objective_function,
                seed_offset=0,
            )[
                0
            ]  # Take first (and only) repetition

            populations[experiment_config.dataset_identifier] = population

        for data_size in data_size_range:
            logger.info(f"Loop Level | Dataset Size: {data_size}")
            for experiment_config in experiment_configs:
                logger.info(
                    f"Loop Level | Dataset: {experiment_config.dataset_identifier}"
                )

                population = populations[experiment_config.dataset_identifier]

                experiment_configs_per_repetition = []
                holdout_configs_per_repetition = []
                for repetition in range(n_repetitions_per_estimator):
                    random_seed = np.random.RandomState(base_random_state + repetition)

                    experiment_indices = random_seed.choice(
                        len(population), size=data_size, replace=False
                    )
                    holdout_indices = np.setdiff1d(
                        np.arange(len(population)), experiment_indices
                    )
                    sampled_experiment = [population[i] for i in experiment_indices]
                    sampled_holdout = [population[i] for i in holdout_indices]

                    experiment_configs_per_repetition.append(sampled_experiment)
                    holdout_configs_per_repetition.append(sampled_holdout)

                # Train the searcher on the warm start configurations and
                # evaluate on the holdout configurations:
                for estimator_architecture in estimator_architectures:
                    logger.info(f"Loop Level | Tuner: {estimator_architecture}")
                    for tuning_iterations in tuning_iterations_range:
                        for repetition in range(n_repetitions_per_estimator):
                            logger.info(f"Loop Level | Repetition: {repetition}")

                            experiment_data = experiment_configs_per_repetition[
                                repetition
                            ]
                            X_experiment = [cfg for cfg, _ in experiment_data]
                            y_experiment = [perf for _, perf in experiment_data]

                            holdout_data = holdout_configs_per_repetition[repetition]
                            X_holdout = [cfg for cfg, _ in holdout_data]
                            y_holdout = [perf for _, perf in holdout_data]

                            encoder = ConfigurationEncoder(
                                search_space=setup_ccqr_optimization_params(
                                    experiment_config.search_space
                                )
                            )
                            X_experiment_encoded = np.array(
                                encoder.transform(X_experiment)
                            )
                            X_holdout_encoded = np.array(encoder.transform(X_holdout))

                            searcher = QuantileConformalEstimator(
                                quantile_estimator_architecture=estimator_architecture,
                                alphas=alphas,
                                n_pre_conformal_trials=n_pre_conformal_trials,
                                calibration_split_strategy="train_test_split",
                                normalize_features=True,
                            )

                            searcher.fit(
                                X=X_experiment_encoded,
                                y=np.array(y_experiment),
                                tuning_iterations=tuning_iterations,
                                min_obs_for_tuning=n_pre_conformal_trials,
                                random_state=base_random_state + repetition,
                            )

                            # Evaluate on holdout configurations across all quantile pairs:
                            holdout_predicted_intervals = searcher.predict_intervals(
                                X=X_holdout_encoded,
                            )

                            pair_losses = []
                            for interval_idx, alpha_val in enumerate(alphas):
                                lower_quantile = alpha_val / 2
                                upper_quantile = 1 - lower_quantile
                                lo_y_pred = holdout_predicted_intervals[interval_idx].lower_bounds
                                hi_y_pred = holdout_predicted_intervals[interval_idx].upper_bounds
                                lo_score = mean_pinball_loss(
                                    y_holdout, lo_y_pred, alpha=lower_quantile
                                )
                                hi_score = mean_pinball_loss(
                                    y_holdout, hi_y_pred, alpha=upper_quantile
                                )
                                pair_losses.append((lo_score + hi_score) / 2)
                            mean_loss = float(np.mean(pair_losses))

                            aliased_estimator_architecture = (
                                aliases.architecture_aliases[estimator_architecture]
                                if estimator_architecture
                                in aliases.architecture_aliases
                                else estimator_architecture
                            )
                            aliased_benchmark_identifier = (
                                aliases.benchmark_aliases[
                                    experiment_config.benchmark_identifier
                                ]
                                if experiment_config.benchmark_identifier
                                in aliases.benchmark_aliases
                                else experiment_config.benchmark_identifier
                            )
                            # Create dictionary with results:
                            results = {
                                "estimator_architecture": aliased_estimator_architecture,
                                "dataset": experiment_config.dataset_identifier,
                                "benchmark_identifier": aliased_benchmark_identifier,
                                "repetition": repetition,
                                "tuning_iterations": tuning_iterations,
                                "data_size": data_size,
                                "n_quantiles": n_quantiles,
                                "mean_pinball_loss": mean_loss,
                            }
                            estimator_error_results.append(results)

        # Clean up populations to free memory
        for experiment_config in experiment_configs:
            experiment_config.objective_function = None
        del populations
        gc.collect()

    logger.info("Estimator Error Analysis finished.")
    return pd.DataFrame(estimator_error_results)
