from hpobench.report.analyze import analyze_main_benchmark
from hpobench.utils import save_analysis_results
import pandas as pd


def test_analyze_main_benchmark(
    multi_benchmark_raw_data, dummy_processing_raw_data, benchmark_data_schema
):
    run_start = pd.Timestamp.now()
    run_start_str = run_start.strftime("%Y-%m-%d_%H-%M-%S")
    cache_path = "tests/spot_checks/analysis/"
    analyze_main_benchmark(
        raw_benchmark_data=multi_benchmark_raw_data,
        cache_path=cache_path,
        run_start_str=run_start_str,
        analysis_type="test-general",
        analysis_components=[
            "friedman",
            "nemenyi",
            "wilcoxon",
            "permutation_test",
            "dataset_performances",
            "rank_analysis",
            "sampler_comparison",
            "architecture_comparison",
        ],
        schema=benchmark_data_schema,
        alpha=0.05,
        starting_coverage_trial=None,
        cd_significance_method="permutation_test",
        n_bootstraps=50,
    )
    save_analysis_results(
        df=multi_benchmark_raw_data,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="raw_benchmark_data.csv",
        analysis_type="test-general",
    )

    # Coverage with a single benchmark and varying confidence levels for plot:
    unique_benchmarks = dummy_processing_raw_data[
        benchmark_data_schema.bench_col
    ].unique()
    single_benchmark_processing_raw_data = dummy_processing_raw_data[
        dummy_processing_raw_data[benchmark_data_schema.bench_col]
        == unique_benchmarks[0]
    ]
    high_conf_data = single_benchmark_processing_raw_data.copy()
    high_conf_data[benchmark_data_schema.confidence_level_col] = 0.8
    low_conf_data = single_benchmark_processing_raw_data.copy()
    low_conf_data[benchmark_data_schema.confidence_level_col] = 0.2
    multi_conf_data = pd.concat([high_conf_data, low_conf_data], axis=0)
    value_map = {
        "QGBM tuner 1": "Conformalized + DtACI",
        "QGBM tuner 2": "Conformalized",
    }
    multi_conf_data[benchmark_data_schema.tuner_col] = multi_conf_data[
        benchmark_data_schema.tuner_col
    ].replace(value_map)
    analyze_main_benchmark(
        raw_benchmark_data=multi_conf_data,
        cache_path=cache_path,
        run_start_str=run_start_str,
        analysis_type="test-coverage",
        analysis_components=["coverage"],
        schema=benchmark_data_schema,
        alpha=0.05,
        starting_coverage_trial=None,
        cd_significance_method="permutation_test",
        n_bootstraps=50,
    )
    save_analysis_results(
        df=multi_conf_data,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="raw_benchmark_data.csv",
        analysis_type="test-coverage",
    )

    # For complex ranking components, we need ad hoc raw data:
    conformal_data = dummy_processing_raw_data.copy()
    conformal_data[benchmark_data_schema.n_pre_conformal_trials_col] = 32
    non_conformal_data = dummy_processing_raw_data.copy()
    non_conformal_data[benchmark_data_schema.n_pre_conformal_trials_col] = 10000
    non_conformal_data[benchmark_data_schema.tuner_col] = (
        non_conformal_data[benchmark_data_schema.tuner_col] + "_non_conformal"
    )
    multi_conformal_data = pd.concat([conformal_data, non_conformal_data], axis=0)
    multi_conformal_data[benchmark_data_schema.sampler_col] = "some_sampler"
    analyze_main_benchmark(
        raw_benchmark_data=multi_conformal_data,
        cache_path=cache_path,
        run_start_str=run_start_str,
        analysis_type="test-preconformal-trials-plot",
        analysis_components=["conformalization_effect"],
        schema=benchmark_data_schema,
        alpha=0.05,
        starting_coverage_trial=None,
        cd_significance_method="permutation_test",
        n_bootstraps=50,
    )
    save_analysis_results(
        df=multi_conformal_data,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="raw_benchmark_data.csv",
        analysis_type="test-preconformal-trials-plot",
    )

    # Real world raw benchmark data snippet:
    toy_raw_benchmark_data = pd.read_csv(
        "tests/spot_checks/analysis/toy_raw_benchmark_data.csv"
    )
    analyze_main_benchmark(
        raw_benchmark_data=toy_raw_benchmark_data,
        cache_path=cache_path,
        run_start_str=run_start_str,
        analysis_type="test-toy-data",
        analysis_components=[
            "friedman",
            "nemenyi",
            "wilcoxon",
            "permutation_test",
            "dataset_performances",
            "rank_analysis",
            "sampler_comparison",
            "architecture_comparison",
        ],
        schema=benchmark_data_schema,
        alpha=0.05,
        starting_coverage_trial=None,
        cd_significance_method="permutation_test",
        n_bootstraps=50,
    )

    # Create toy data for rank analysis CD diagram testing with 10 datasets
    base_data = dummy_processing_raw_data.copy()
    # Create two tuners: tuner 1 with better performance, tuner 2 with worse performance
    tuner1_data = base_data.copy()
    tuner1_data[benchmark_data_schema.tuner_col] = "Tuner 1"
    tuner1_data[benchmark_data_schema.estimator_architecture_col] = "Tuner 1"
    tuner1_data[benchmark_data_schema.sampler_col] = "Sampler 1"
    # Keep original performance for tuner 1 (better performance)

    tuner2_data = base_data.copy()
    tuner2_data[benchmark_data_schema.tuner_col] = "Tuner 2"
    tuner2_data[benchmark_data_schema.estimator_architecture_col] = "Tuner 2"
    tuner2_data[benchmark_data_schema.sampler_col] = "Sampler 2"
    # Make tuner 2 have worse performance by adding a penalty
    tuner2_data[benchmark_data_schema.perf_col] = (
        tuner2_data[benchmark_data_schema.perf_col] + 20
    )

    tuner3_data = base_data.copy()
    tuner3_data[benchmark_data_schema.tuner_col] = "Tuner 3"
    tuner3_data[benchmark_data_schema.estimator_architecture_col] = "Tuner 3"
    tuner3_data[benchmark_data_schema.sampler_col] = "Sampler 3"
    tuner3_data[benchmark_data_schema.perf_col] = (
        tuner3_data[benchmark_data_schema.perf_col] + 10
    )

    tuner4_data = base_data.copy()
    tuner4_data[benchmark_data_schema.tuner_col] = "Tuner 4"
    tuner4_data[benchmark_data_schema.estimator_architecture_col] = "Tuner 4"
    tuner4_data[benchmark_data_schema.sampler_col] = "Sampler 4"
    tuner4_data[benchmark_data_schema.perf_col] = (
        tuner4_data[benchmark_data_schema.perf_col] + 5
    )

    # Combine tuner data
    combined_data = pd.concat(
        [tuner1_data, tuner2_data, tuner3_data, tuner4_data], ignore_index=True
    )

    # Create 15 datasets by copying the data with different dataset IDs
    rank_analysis_data = []
    for i in range(15):
        dataset_data = combined_data.copy()
        dataset_data[benchmark_data_schema.data_col] = f"dataset_{i}"
        rank_analysis_data.append(dataset_data)

    rank_analysis_data = pd.concat(rank_analysis_data, ignore_index=True)

    analyze_main_benchmark(
        raw_benchmark_data=rank_analysis_data,
        cache_path=cache_path,
        run_start_str=run_start_str,
        analysis_type="test-rank-analysis-cd",
        analysis_components=["rank_analysis"],
        schema=benchmark_data_schema,
        alpha=0.05,
        starting_coverage_trial=None,
        cd_significance_method="permutation_test",
        n_bootstraps=50,
    )
    save_analysis_results(
        df=rank_analysis_data,
        cache_path=cache_path,
        run_start_str=run_start_str,
        filename="raw_benchmark_data.csv",
        analysis_type="test-rank-analysis-cd",
    )
