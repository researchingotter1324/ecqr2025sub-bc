import numpy as np
import pytest
import pandas as pd
from hpobench.generation.generate import BlackBoxGenerator
from hpobench.config.config_types import FloatRange
import os


@pytest.fixture
def dummy_processing_raw_data():
    data = {
        "performance": [
            # lcbench dataset 3945 (19 rows)
            -83.3992,
            -91.3902,
            -89.9583,
            -88.5413,
            -79.0829,
            -87.9277,
            -80.241,
            -87.4604,
            -84.0319,
            -71.6547,
            -89.9583,
            -88.5413,
            -79.0829,
            -83.757,
            -81.358,
            -87.4604,
            -84.0319,
            -71.6547,
            -84.2562,
            # lcbench dataset 7593 (19 rows with noise)
            -85.1234,
            -89.7641,
            -91.2145,
            -86.8921,
            -81.4567,
            -89.3455,
            -82.6789,
            -85.9123,
            -86.4567,
            -74.1234,
            -91.2145,
            -86.8921,
            -81.4567,
            -85.2341,
            -83.7892,
            -85.9123,
            -86.4567,
            -74.1234,
            -86.7891,
            # nahs201 dataset cifar10 (8 rows)
            -84.0319,
            -71.6547,
            -89.9583,
            -88.5413,
            -79.0829,
            -88.5413,
            -79.0829,
            -83.757,
            # nahs201 dataset imagenet (8 rows with noise)
            -86.4521,
            -73.8912,
            -87.6734,
            -90.1256,
            -77.4891,
            -90.1256,
            -77.4891,
            -81.9823,
        ],
        "iteration": [
            # lcbench dataset 3945
            1,
            2,
            3,
            4,
            5,
            6,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            5,
            1,
            2,
            3,
            4,
            # lcbench dataset 7593 (same pattern)
            1,
            2,
            3,
            4,
            5,
            6,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            5,
            1,
            2,
            3,
            4,
            # nahs201 dataset cifar10
            1,
            2,
            1,
            2,
            1,
            2,
            1,
            2,
            # nahs201 dataset imagenet (same pattern)
            1,
            2,
            1,
            2,
            1,
            2,
            1,
            2,
        ],
        "runtime": [
            # lcbench dataset 3945
            256.8546,
            257.8,
            259.8,
            259.9,
            260,
            260.1,
            254.2,
            257.5,
            258,
            259,
            257.1,
            258.2,
            258.3,
            258.5,
            259.1,
            256,
            257.2,
            258.5,
            258.9,
            # lcbench dataset 7593 (slightly different timing)
            248.3421,
            249.2,
            251.3,
            251.4,
            251.7,
            251.9,
            246.8,
            249.1,
            249.6,
            250.5,
            249.7,
            250.8,
            250.9,
            251.1,
            251.7,
            248.6,
            249.8,
            251.1,
            251.5,
            # nahs201 dataset cifar10
            50,
            50.5,
            50.1,
            52,
            50.3,
            52.2,
            50.3,
            50.8,
            # nahs201 dataset imagenet (slightly different timing)
            47.2,
            47.8,
            47.4,
            49.3,
            47.6,
            49.5,
            47.6,
            48.1,
        ],
        "benchmark_identifier": ["lcbench"] * 38 + ["nahs201"] * 16,
        "dataset": [3945] * 19 + [7593] * 19 + ["cifar10"] * 8 + ["imagenet"] * 8,
        "tuner": ["tuner 1"] * 10
        + ["tuner 2"] * 9
        + ["tuner 1"] * 10
        + ["tuner 2"] * 9
        + ["tuner 1"] * 4
        + ["tuner 2"] * 4
        + ["tuner 1"] * 4
        + ["tuner 2"] * 4,
        "repetition": [1] * 6
        + [2] * 4
        + [1] * 5
        + [2] * 4
        + [1] * 6
        + [2] * 4
        + [1] * 5
        + [2] * 4
        + [1] * 2
        + [2] * 2
        + [1] * 2
        + [2] * 2
        + [1] * 2
        + [2] * 2
        + [1] * 2
        + [2] * 2,
        "sampler": ["TS"] * 54,
        "confidence_level": [0.2] * 54,  # Empty strings for non-ccqr_optimization tuners
        "estimator_architecture": ["QGBM"] * 10
        + ["QRF"] * 9
        + ["QGBM"] * 10
        + ["QRF"] * 9
        + ["QGBM"] * 4
        + ["QRF"] * 4
        + ["QGBM"] * 4
        + ["QRF"] * 4,
        # Added columns to match run_main_benchmark output format
        "searcher_tuning_framework": [""] * 54,
        "n_pre_conformal_trials": [32] * 54,
        "sampler_n_quantiles": [4] * 54,
        "sampler_adapter": ["DtACI"] * 54,
        "tuner_searcher_tuning_framework": [""] * 54,
        "breach_status": [
            # lcbench dataset 3945
            np.nan,  # iteration 1 -> empty
            1,
            0,
            1,
            0,
            1,
            np.nan,  # iteration 1
            0,
            1,
            0,
            np.nan,  # iteration 1
            1,
            0,
            1,
            0,
            np.nan,  # iteration 1
            1,
            0,
            1,
            # lcbench dataset 7593 (similar pattern with slight variations)
            np.nan,  # iteration 1 -> empty
            0,
            1,
            0,
            1,
            0,
            np.nan,  # iteration 1
            1,
            0,
            1,
            np.nan,  # iteration 1
            0,
            1,
            0,
            1,
            np.nan,  # iteration 1
            0,
            1,
            0,
            # nahs201 dataset cifar10
            np.nan,  # iteration 1
            0,
            np.nan,  # iteration 1
            1,
            np.nan,  # iteration 1
            0,
            np.nan,  # iteration 1
            1,
            # nahs201 dataset imagenet (similar pattern)
            np.nan,  # iteration 1
            1,
            np.nan,  # iteration 1
            0,
            np.nan,  # iteration 1
            1,
            np.nan,  # iteration 1
            0,
        ],
        "miscoverage_penalty": [
            # lcbench dataset 3945
            np.nan,  # matches breach == ""
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            np.nan,
            0.01,
            0.02,
            0.03,
            np.nan,
            0.01,
            0.02,
            0.03,
            0.04,
            np.nan,
            0.01,
            0.02,
            0.03,
            # lcbench dataset 7593 (with slight noise)
            np.nan,
            0.015,
            0.018,
            0.032,
            0.038,
            0.048,
            np.nan,
            0.012,
            0.024,
            0.036,
            np.nan,
            0.013,
            0.019,
            0.031,
            0.042,
            np.nan,
            0.014,
            0.021,
            0.029,
            # nahs201 dataset cifar10
            np.nan,
            0.01,
            np.nan,
            0.02,
            np.nan,
            0.01,
            np.nan,
            0.02,
            # nahs201 dataset imagenet (with slight noise)
            np.nan,
            0.016,
            np.nan,
            0.023,
            np.nan,
            0.017,
            np.nan,
            0.025,
        ],
        "winkler_score": [
            # lcbench dataset 3945
            np.nan,
            0.10,
            0.12,
            0.15,
            0.18,
            0.2,
            np.nan,
            0.11,
            0.13,
            0.16,
            np.nan,
            0.10,
            0.12,
            0.15,
            0.18,
            np.nan,
            0.11,
            0.13,
            0.16,
            # lcbench dataset 7593 (with noise)
            np.nan,
            0.105,
            0.118,
            0.152,
            0.175,
            0.195,
            np.nan,
            0.108,
            0.125,
            0.162,
            np.nan,
            0.103,
            0.115,
            0.148,
            0.172,
            np.nan,
            0.107,
            0.122,
            0.158,
            # nahs201 dataset cifar10
            np.nan,
            0.10,
            np.nan,
            0.12,
            np.nan,
            0.11,
            np.nan,
            0.13,
            # nahs201 dataset imagenet (with noise)
            np.nan,
            0.106,
            np.nan,
            0.124,
            np.nan,
            0.114,
            np.nan,
            0.135,
        ],
        "width": [
            # lcbench dataset 3945
            np.nan,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            np.nan,
            0.22,
            0.27,
            0.32,
            np.nan,
            0.21,
            0.26,
            0.31,
            0.36,
            np.nan,
            0.22,
            0.27,
            0.33,
            # lcbench dataset 7593 (with noise)
            np.nan,
            0.205,
            0.248,
            0.295,
            0.342,
            0.385,
            np.nan,
            0.218,
            0.265,
            0.312,
            np.nan,
            0.208,
            0.253,
            0.298,
            0.345,
            np.nan,
            0.215,
            0.262,
            0.318,
            # nahs201 dataset cifar10
            np.nan,
            0.20,
            np.nan,
            0.25,
            np.nan,
            0.22,
            np.nan,
            0.27,
            # nahs201 dataset imagenet (with noise)
            np.nan,
            0.212,
            np.nan,
            0.258,
            np.nan,
            0.234,
            np.nan,
            0.283,
        ],
        "tabularized_configuration": [
            # lcbench dataset 3945 (original configurations)
            [343.0, 0.0143, 0.074, 290.0, 0.63, 2.0, 0.019],
            [439.0, 0.0184, 0.303, 142.0, 0.40, 4.0, 0.073],
            [59.0, 0.0813, 0.569, 146.0, 0.15, 2.0, 0.087],
            [337.0, 0.0436, 0.712, 433.0, 0.93, 3.0, 0.087],
            [347.0, 0.0016, 0.084, 276.0, 0.50, 1.0, 0.007],
            [343.0, 0.0143, 0.074, 290.0, 0.63, 2.0, 0.019],
            [439.0, 0.0184, 0.303, 142.0, 0.40, 4.0, 0.073],
            [59.0, 0.0813, 0.569, 146.0, 0.15, 2.0, 0.087],
            [337.0, 0.0436, 0.712, 433.0, 0.93, 3.0, 0.087],
            [347.0, 0.0016, 0.084, 276.0, 0.50, 1.0, 0.007],
            [343.0, 0.0143, 0.074, 290.0, 0.63, 2.0, 0.019],
            [439.0, 0.0184, 0.303, 142.0, 0.40, 4.0, 0.073],
            [59.0, 0.0813, 0.569, 146.0, 0.15, 2.0, 0.087],
            [337.0, 0.0436, 0.712, 433.0, 0.93, 3.0, 0.087],
            [347.0, 0.0016, 0.084, 276.0, 0.50, 1.0, 0.007],
            [343.0, 0.0143, 0.074, 290.0, 0.63, 2.0, 0.019],
            [439.0, 0.0184, 0.303, 142.0, 0.40, 4.0, 0.073],
            [59.0, 0.0813, 0.569, 146.0, 0.15, 2.0, 0.087],
            [337.0, 0.0436, 0.712, 433.0, 0.93, 3.0, 0.087],
            # lcbench dataset 7593 (configurations with slight variations)
            [351.2, 0.0156, 0.081, 295.3, 0.67, 2.0, 0.022],
            [446.1, 0.0197, 0.318, 148.7, 0.43, 4.0, 0.078],
            [62.8, 0.0854, 0.582, 152.4, 0.18, 2.0, 0.093],
            [342.5, 0.0463, 0.729, 441.2, 0.96, 3.0, 0.091],
            [354.9, 0.0019, 0.089, 283.1, 0.53, 1.0, 0.009],
            [351.2, 0.0156, 0.081, 295.3, 0.67, 2.0, 0.022],
            [446.1, 0.0197, 0.318, 148.7, 0.43, 4.0, 0.078],
            [62.8, 0.0854, 0.582, 152.4, 0.18, 2.0, 0.093],
            [342.5, 0.0463, 0.729, 441.2, 0.96, 3.0, 0.091],
            [354.9, 0.0019, 0.089, 283.1, 0.53, 1.0, 0.009],
            [351.2, 0.0156, 0.081, 295.3, 0.67, 2.0, 0.022],
            [446.1, 0.0197, 0.318, 148.7, 0.43, 4.0, 0.078],
            [62.8, 0.0854, 0.582, 152.4, 0.18, 2.0, 0.093],
            [342.5, 0.0463, 0.729, 441.2, 0.96, 3.0, 0.091],
            [354.9, 0.0019, 0.089, 283.1, 0.53, 1.0, 0.009],
            [351.2, 0.0156, 0.081, 295.3, 0.67, 2.0, 0.022],
            [446.1, 0.0197, 0.318, 148.7, 0.43, 4.0, 0.078],
            [62.8, 0.0854, 0.582, 152.4, 0.18, 2.0, 0.093],
            [342.5, 0.0463, 0.729, 441.2, 0.96, 3.0, 0.091],
            # nahs201 dataset cifar10 (original configurations)
            [347.0, 0.0016, 0.084, 276.0, 0.50, 1.0, 0.007],
            [343.0, 0.0143, 0.074, 290.0, 0.63, 2.0, 0.019],
            [439.0, 0.0184, 0.303, 142.0, 0.40, 4.0, 0.073],
            [59.0, 0.0813, 0.569, 146.0, 0.15, 2.0, 0.087],
            [337.0, 0.0436, 0.712, 433.0, 0.93, 3.0, 0.087],
            [347.0, 0.0016, 0.084, 276.0, 0.50, 1.0, 0.007],
            [343.0, 0.0143, 0.074, 290.0, 0.63, 2.0, 0.019],
            [439.0, 0.0184, 0.303, 142.0, 0.40, 4.0, 0.073],
            # nahs201 dataset imagenet (configurations with variations)
            [358.4, 0.0021, 0.091, 284.2, 0.54, 1.0, 0.011],
            [349.7, 0.0167, 0.082, 297.8, 0.69, 2.0, 0.024],
            [453.6, 0.0209, 0.325, 155.3, 0.47, 4.0, 0.081],
            [67.4, 0.0876, 0.594, 159.1, 0.21, 2.0, 0.095],
            [348.9, 0.0481, 0.741, 448.7, 0.99, 3.0, 0.094],
            [358.4, 0.0021, 0.091, 284.2, 0.54, 1.0, 0.011],
            [349.7, 0.0167, 0.082, 297.8, 0.69, 2.0, 0.024],
            [453.6, 0.0209, 0.325, 155.3, 0.47, 4.0, 0.081],
        ],
    }
    df = pd.DataFrame(data)
    # Enforce uniqueness per x per entity in both iteration and normalized_runtime subsets
    iter_mask = df["iteration"].notna()
    if iter_mask.any():
        df_iter = df.loc[iter_mask].drop_duplicates(
            subset=[
                "benchmark_identifier",
                "dataset",
                "tuner",
                "iteration",
            ]
        )
    else:
        df_iter = df.loc[[]]
    rt_mask = (
        df["normalized_runtime"].notna()
        if "normalized_runtime" in df.columns
        else pd.Series(False, index=df.index)
    )
    if rt_mask.any():
        df_rt = df.loc[rt_mask].drop_duplicates(
            subset=[
                "benchmark_identifier",
                "dataset",
                "tuner",
                "normalized_runtime",
            ]
        )
    else:
        df_rt = df.loc[[]]
    return pd.concat([df_iter, df_rt], ignore_index=True)


@pytest.fixture
def benchmark_data_schema():
    """Schema fixture for testing."""
    from hpobench.config.schema import BenchmarkDataSchema

    return BenchmarkDataSchema()


@pytest.fixture
def benchmark_data_processor(benchmark_data_schema):
    """Processor fixture for testing."""
    from hpobench.process import BenchmarkDataProcessor

    return BenchmarkDataProcessor(schema=benchmark_data_schema)


@pytest.fixture
def spot_check_output_dir():
    """Output directory for spot check test results."""
    base_dir = "tests/spot_checks/processing"
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


@pytest.fixture
def processed_benchmark_data_with_ranks():
    """
    Processed benchmark data with ranks suitable for plot_and_save testing.

    Contains:
    - 2 benchmarks (lcbench, nahs201)
    - 2-3 datasets per benchmark
    - 3 tuners with different samplers and architectures
    - Multiple budget points (iteration and normalized_runtime)
    - Rank columns with confidence intervals
    - Best performance metrics
    """
    np.random.seed(42)

    data = []
    benchmarks = ["lcbench", "nahs201"]
    datasets_map = {
        "lcbench": ["dataset_3945", "dataset_7593"],
        "nahs201": ["cifar10", "imagenet", "colorectal_histology"],
    }
    tuners = ["QGBM_tuner_1", "QGBM_tuner_2", "QGBM_tuner_3"]
    samplers = ["GBRT", "TPE", "random"]
    architectures = ["QGBM", "MLP", "ResNet"]
    confidence_levels = [0.5, 0.75, 0.9]

    # Budget points
    iterations = [1, 2, 5, 10, 20, 50]
    normalized_runtimes = [10, 25, 50, 75, 100]

    for bench in benchmarks:
        for dataset in datasets_map[bench]:
            for i, tuner in enumerate(tuners):
                sampler = samplers[i]
                architecture = architectures[i]
                confidence = confidence_levels[i % len(confidence_levels)]

                # Generate performance trajectory that improves over budget
                base_perf = np.random.uniform(0.65, 0.85)
                improvement_rate = np.random.uniform(0.001, 0.005)

                # Iteration budget data
                for iteration in iterations:
                    # Performance improves with more iterations
                    performance = min(
                        0.95,
                        base_perf
                        + improvement_rate * iteration
                        + np.random.normal(0, 0.01),
                    )

                    # Best performance (monotonically improving)
                    best_performance = max(
                        [
                            base_perf + improvement_rate * it
                            for it in range(1, iteration + 1)
                        ]
                    )

                    # Rank within this benchmark/dataset (1-3)
                    rank = i + 1 + np.random.uniform(-0.2, 0.2)
                    rank_lower = max(1.0, rank - 0.3)
                    rank_upper = min(3.0, rank + 0.3)

                    # Runtime increases with iterations
                    runtime = 50 + iteration * 10 + np.random.uniform(-5, 5)

                    data.append(
                        {
                            "benchmark_identifier": bench,
                            "dataset": dataset,
                            "tuner": tuner,
                            "sampler": sampler,
                            "estimator_architecture": architecture,
                            "confidence_level": confidence,
                            "iteration": iteration,
                            "runtime": runtime,
                            "performance": performance,
                            "best_performance": best_performance,
                            "rank": rank,
                            "rank_lower": rank_lower,
                            "rank_upper": rank_upper,
                            "cumulative_coverage_error": abs(confidence - 0.9)
                            * np.random.uniform(0.8, 1.2),
                            "rolling_coverage_error": abs(confidence - 0.9)
                            * np.random.uniform(0.5, 1.5),
                        }
                    )

                # Normalized runtime budget data
                for norm_runtime in normalized_runtimes:
                    # Map normalized runtime to actual iteration for consistency
                    equiv_iteration = int(norm_runtime / 10)

                    performance = min(
                        0.95,
                        base_perf
                        + improvement_rate * equiv_iteration
                        + np.random.normal(0, 0.01),
                    )
                    best_performance = max(
                        [
                            base_perf + improvement_rate * it
                            for it in range(1, equiv_iteration + 1)
                        ]
                    )

                    rank = i + 1 + np.random.uniform(-0.2, 0.2)
                    rank_lower = max(1.0, rank - 0.3)
                    rank_upper = min(3.0, rank + 0.3)

                    data.append(
                        {
                            "benchmark_identifier": bench,
                            "dataset": dataset,
                            "tuner": tuner,
                            "sampler": sampler,
                            "estimator_architecture": architecture,
                            "confidence_level": confidence,
                            "normalized_runtime": norm_runtime,
                            "performance": performance,
                            "best_performance": best_performance,
                            "rank": rank,
                            "rank_lower": rank_lower,
                            "rank_upper": rank_upper,
                            "cumulative_coverage_error": abs(confidence - 0.9)
                            * np.random.uniform(0.8, 1.2),
                            "rolling_coverage_error": abs(confidence - 0.9)
                            * np.random.uniform(0.5, 1.5),
                        }
                    )

    df = pd.DataFrame(data)
    # Ensure uniqueness per x per entity for both budgets to match plotting expectations
    parts = []
    if "iteration" in df.columns:
        df_iter = df.dropna(subset=["iteration"]).drop_duplicates(
            subset=["benchmark_identifier", "dataset", "tuner", "iteration"],
            keep="first",
        )
        parts.append(df_iter)
    if "normalized_runtime" in df.columns:
        df_rt = df.dropna(subset=["normalized_runtime"]).drop_duplicates(
            subset=["benchmark_identifier", "dataset", "tuner", "normalized_runtime"],
            keep="first",
        )
        parts.append(df_rt)
    if parts:
        return pd.concat(parts, ignore_index=True)
    return df


@pytest.fixture
def dataset_level_data():
    """
    Dataset-level processed data for dataset performance plotting.

    Contains iteration-based trajectories for individual datasets.
    """
    np.random.seed(123)

    data = []
    benchmarks = ["lcbench", "nahs201"]
    datasets_map = {
        "lcbench": ["dataset_3945", "dataset_7593"],
        "nahs201": ["cifar10", "imagenet"],
    }
    tuners = ["QGBM_tuner_1", "QGBM_tuner_2", "QGBM_tuner_3"]
    iterations = [1, 2, 5, 10, 20, 50, 100]

    for bench in benchmarks:
        for dataset in datasets_map[bench]:
            for i, tuner in enumerate(tuners):
                base_perf = np.random.uniform(0.70, 0.90)
                improvement_rate = np.random.uniform(0.001, 0.003)

                for iteration in iterations:
                    performance = min(
                        0.95,
                        base_perf
                        + improvement_rate * iteration
                        + np.random.normal(0, 0.005),
                    )
                    best_performance = max(
                        [
                            base_perf + improvement_rate * it
                            for it in range(1, iteration + 1)
                        ]
                    )

                    # Rank based on performance within dataset
                    rank = i + 1 + np.random.uniform(-0.1, 0.1)

                    data.append(
                        {
                            "benchmark_identifier": bench,
                            "dataset": dataset,
                            "tuner": tuner,
                            "iteration": iteration,
                            "performance": performance,
                            "best_performance": best_performance,
                            "rank": rank,
                        }
                    )

    return pd.DataFrame(data)


@pytest.fixture
def coverage_analysis_data():
    """
    Data suitable for coverage analysis plotting with breach rates.

    Contains breach status and coverage metrics over iterations.
    """
    np.random.seed(456)

    data = []
    benchmarks = ["conformal_bench"]
    datasets = ["dataset_A", "dataset_B"]
    tuners = ["Conformalized_QGBM", "Conformalized_TPE", "Non_Conformalized"]
    confidence_levels = [0.5, 0.75, 0.9]
    iterations = range(1, 51)  # 1 to 50 iterations

    for bench in benchmarks:
        for dataset in datasets:
            for tuner in tuners:
                for conf_level in confidence_levels:
                    cumulative_breaches = 0

                    for iteration in iterations:
                        # Simulate breach probability based on confidence level and tuner
                        if "Non_Conformalized" in tuner:
                            breach_prob = (
                                0.3  # Higher breach rate for non-conformalized
                            )
                        else:
                            breach_prob = max(
                                0.05, 1 - conf_level + np.random.uniform(-0.05, 0.05)
                            )

                        breach_occurred = np.random.binomial(1, breach_prob)
                        cumulative_breaches += breach_occurred

                        # Coverage errors
                        cumulative_coverage_error = cumulative_breaches / iteration

                        # Rolling window coverage error (last 10 iterations)
                        window_start = max(1, iteration - 9)
                        window_breaches = np.random.binomial(
                            iteration - window_start + 1, breach_prob
                        )
                        rolling_coverage_error = window_breaches / (
                            iteration - window_start + 1
                        )

                        data.append(
                            {
                                "benchmark_identifier": bench,
                                "dataset": dataset,
                                "tuner": tuner,
                                "confidence_level": conf_level,
                                "iteration": iteration,
                                "cumulative_coverage_error": cumulative_coverage_error,
                                "rolling_coverage_error": rolling_coverage_error,
                            }
                        )

    return pd.DataFrame(data)


@pytest.fixture
def significance_test_data():
    """
    Statistical significance test results for critical difference plotting.

    Mimics output from nemenyi_pairwise_test or wilcoxon_pairwise_test.
    """
    return pd.DataFrame(
        [
            {
                "benchmark_identifier": "lcbench",
                "entity1": "QGBM_tuner_1",
                "entity2": "QGBM_tuner_2",
                "mean_rank_1": 1.8,
                "mean_rank_2": 2.3,
                "p_value": 0.02,
                "significant": True,
                "better_entity": "QGBM_tuner_1",
            },
            {
                "benchmark_identifier": "lcbench",
                "entity1": "QGBM_tuner_1",
                "entity2": "QGBM_tuner_3",
                "mean_rank_1": 1.8,
                "mean_rank_2": 2.7,
                "p_value": 0.001,
                "significant": True,
                "better_entity": "QGBM_tuner_1",
            },
            {
                "benchmark_identifier": "lcbench",
                "entity1": "QGBM_tuner_2",
                "entity2": "QGBM_tuner_3",
                "mean_rank_1": 2.3,
                "mean_rank_2": 2.7,
                "p_value": 0.15,
                "significant": False,
                "better_entity": "QGBM_tuner_2",
            },
            {
                "benchmark_identifier": "nahs201",
                "entity1": "QGBM_tuner_1",
                "entity2": "QGBM_tuner_2",
                "mean_rank_1": 1.5,
                "mean_rank_2": 2.1,
                "p_value": 0.03,
                "significant": True,
                "better_entity": "QGBM_tuner_1",
            },
            {
                "benchmark_identifier": "nahs201",
                "entity1": "QGBM_tuner_1",
                "entity2": "QGBM_tuner_3",
                "mean_rank_1": 1.5,
                "mean_rank_2": 2.9,
                "p_value": 0.005,
                "significant": True,
                "better_entity": "QGBM_tuner_1",
            },
            {
                "benchmark_identifier": "nahs201",
                "entity1": "QGBM_tuner_2",
                "entity2": "QGBM_tuner_3",
                "mean_rank_1": 2.1,
                "mean_rank_2": 2.9,
                "p_value": 0.08,
                "significant": False,
                "better_entity": "QGBM_tuner_2",
            },
        ]
    )


@pytest.fixture
def sampler_breakout_data():
    """
    Data partitioned by sampler for sampler comparison plots.

    Same structure as processed_benchmark_data_with_ranks but organized
    to highlight sampler differences.
    """
    np.random.seed(789)

    data = []
    benchmarks = ["lcbench", "nahs201"]
    datasets_map = {
        "lcbench": ["dataset_3945", "dataset_7593"],
        "nahs201": ["cifar10", "imagenet"],
    }

    # Different tuner configurations per sampler
    sampler_configs = {
        "GBRT": ["GBRT_standard", "GBRT_adaptive", "GBRT_optimized"],
        "TPE": ["TPE_standard", "TPE_multivariate", "TPE_hyperband"],
        "random": ["Random_uniform", "Random_sobol", "Random_lhs"],
    }

    normalized_runtimes = [10, 25, 50, 75, 100]

    for bench in benchmarks:
        for dataset in datasets_map[bench]:
            for sampler, tuners in sampler_configs.items():
                for i, tuner in enumerate(tuners):
                    # Different performance characteristics per sampler
                    if sampler == "GBRT":
                        base_rank = 1.5 + i * 0.3
                    elif sampler == "TPE":
                        base_rank = 2.0 + i * 0.3
                    else:  # random
                        base_rank = 2.5 + i * 0.3

                    for norm_runtime in normalized_runtimes:
                        # Ranks improve with more budget, but at different rates per sampler
                        improvement = (norm_runtime / 100) * 0.5
                        rank = max(
                            1.0, base_rank - improvement + np.random.uniform(-0.1, 0.1)
                        )
                        rank_lower = max(1.0, rank - 0.2)
                        rank_upper = min(4.0, rank + 0.2)

                        data.append(
                            {
                                "benchmark_identifier": bench,
                                "dataset": dataset,
                                "tuner": tuner,
                                "sampler": sampler,
                                "estimator_architecture": "QGBM",
                                "normalized_runtime": norm_runtime,
                                "rank": rank,
                                "rank_lower": rank_lower,
                                "rank_upper": rank_upper,
                            }
                        )

    return pd.DataFrame(data)


@pytest.fixture
def architecture_breakout_data():
    """
    Data partitioned by estimator architecture for architecture comparison plots.
    """
    np.random.seed(101)

    data = []
    benchmarks = ["lcbench", "nahs201"]
    datasets_map = {
        "lcbench": ["dataset_3945", "dataset_7593"],
        "nahs201": ["cifar10", "imagenet"],
    }

    # Different tuner configurations per architecture
    arch_configs = {
        "QGBM": ["QGBM_base", "QGBM_tuned", "QGBM_ensemble"],
        "MLP": ["MLP_shallow", "MLP_deep", "MLP_wide"],
        "ResNet": ["ResNet18", "ResNet34", "ResNet50"],
    }

    normalized_runtimes = [10, 25, 50, 75, 100]

    for bench in benchmarks:
        for dataset in datasets_map[bench]:
            for arch, tuners in arch_configs.items():
                for i, tuner in enumerate(tuners):
                    # Different performance characteristics per architecture
                    if arch == "QGBM":
                        base_rank = 1.8 + i * 0.2
                    elif arch == "MLP":
                        base_rank = 2.2 + i * 0.2
                    else:  # ResNet
                        base_rank = 2.0 + i * 0.2

                    for norm_runtime in normalized_runtimes:
                        improvement = (norm_runtime / 100) * 0.4
                        rank = max(
                            1.0,
                            base_rank - improvement + np.random.uniform(-0.15, 0.15),
                        )
                        rank_lower = max(1.0, rank - 0.25)
                        rank_upper = min(4.0, rank + 0.25)

                        data.append(
                            {
                                "benchmark_identifier": bench,
                                "dataset": dataset,
                                "tuner": tuner,
                                "sampler": "GBRT",
                                "estimator_architecture": arch,
                                "normalized_runtime": norm_runtime,
                                "rank": rank,
                                "rank_lower": rank_lower,
                                "rank_upper": rank_upper,
                            }
                        )

    return pd.DataFrame(data)


@pytest.fixture
def multi_benchmark_raw_data(dummy_processing_raw_data):
    """
    Create multi-benchmark dataset by duplicating dummy_processing_raw_data with different benchmark_identifier.
    This serves as the base fixture for dynamic data transformations in plotting tests.
    """
    # Create a copy of the original data
    df_original = dummy_processing_raw_data.copy()

    # Create a second benchmark with modified benchmark_identifier
    df_duplicate = dummy_processing_raw_data.copy()
    df_duplicate["benchmark_identifier"] = df_duplicate["benchmark_identifier"].replace(
        "lcbench", "hpobench"
    )
    df_duplicate["benchmark_identifier"] = df_duplicate["benchmark_identifier"].replace(
        "nahs201", "nas_bench"
    )

    # Concatenate the datasets
    combined_df = pd.concat([df_original, df_duplicate], ignore_index=True).reset_index(
        drop=True
    )

    return combined_df


@pytest.fixture
def dynamic_plotting_data_transformer(benchmark_data_processor, benchmark_data_schema):
    """
    Factory fixture that provides data transformation functions for plotting tests.
    This replaces hardcoded fixtures with dynamic transformations based on processing pipeline.
    """

    def transform_for_rank_analysis(raw_data):
        """Transform data for basic rank analysis plotting (like analyze.py rank_analysis)."""
        # Process for normalized runtime budget with ranks and confidence intervals
        runtime_data = benchmark_data_processor.process_performance_records(
            raw_benchmark_data=raw_data,
            budget_unit=benchmark_data_schema.runtime_unit,
            relativize_budget=True,
            collapse_repetitions=True,
            collapse_datasets=False,
            extra_ranking_cols=None,
        )
        return runtime_data

    def transform_for_iteration_performance(raw_data):
        """Transform data for iteration-based performance plotting (like analyze.py dataset_performances)."""
        # Process for iteration budget with best performance tracking
        iter_data = benchmark_data_processor.process_performance_records(
            raw_benchmark_data=raw_data,
            budget_unit=benchmark_data_schema.iter_unit,
            relativize_budget=False,
            collapse_repetitions=True,
            collapse_datasets=False,
            extra_ranking_cols=None,
        )
        return iter_data

    def transform_for_sampler_comparison(raw_data):
        """Transform data for sampler comparison plotting (like analyze.py sampler_comparison)."""
        # Process with sampler as partitioning dimension
        sampler_data = benchmark_data_processor.process_performance_records(
            raw_benchmark_data=raw_data,
            budget_unit=benchmark_data_schema.runtime_unit,
            relativize_budget=True,
            collapse_repetitions=True,
            collapse_datasets=False,
            extra_ranking_cols=None,
        )
        return sampler_data

    def transform_for_architecture_comparison(raw_data):
        """Transform data for architecture comparison plotting (like analyze.py architecture_comparison)."""
        # Process with architecture as partitioning dimension
        arch_data = benchmark_data_processor.process_performance_records(
            raw_benchmark_data=raw_data,
            budget_unit=benchmark_data_schema.runtime_unit,
            relativize_budget=True,
            collapse_repetitions=True,
            collapse_datasets=False,
            extra_ranking_cols=None,
        )
        return arch_data

    def transform_for_conformalization_effect(raw_data):
        """Transform data for conformalization effect plotting (like analyze.py conformalization_effect)."""
        # Process with both sampler and architecture as ranking dimensions
        conformal_data = benchmark_data_processor.process_performance_records(
            raw_benchmark_data=raw_data,
            budget_unit=benchmark_data_schema.runtime_unit,
            relativize_budget=True,
            collapse_repetitions=True,
            collapse_datasets=True,
            extra_ranking_cols=[
                benchmark_data_schema.estimator_architecture_col,
                benchmark_data_schema.sampler_col,
            ],
        )
        return conformal_data

    def transform_for_coverage_analysis(raw_data):
        """Transform data for coverage analysis plotting."""
        # Filter to conformal data and process for coverage metrics
        conformal_raw = raw_data[raw_data["breach_status"].notna()].copy()
        if conformal_raw.empty:
            # If no conformal data, create mock coverage metrics from existing data
            coverage_data = raw_data.copy()
            coverage_data["cumulative_coverage_error"] = np.random.uniform(
                0, 0.1, len(coverage_data)
            )
            coverage_data["rolling_coverage_error"] = np.random.uniform(
                0, 0.05, len(coverage_data)
            )
            coverage_data["confidence_level"] = "0.75"
            return coverage_data

        # Process conformal data for coverage analysis
        processed_data = benchmark_data_processor.process_performance_records(
            raw_benchmark_data=conformal_raw,
            budget_unit=benchmark_data_schema.iter_unit,
            relativize_budget=False,
            collapse_repetitions=False,
            collapse_datasets=False,
            extra_ranking_cols=None,
        )

        # Add mock coverage error columns if not present
        if "cumulative_coverage_error" not in processed_data.columns:
            processed_data["cumulative_coverage_error"] = np.random.uniform(
                0, 0.1, len(processed_data)
            )
        if "rolling_coverage_error" not in processed_data.columns:
            processed_data["rolling_coverage_error"] = np.random.uniform(
                0, 0.05, len(processed_data)
            )
        if "confidence_level" not in processed_data.columns:
            processed_data["confidence_level"] = "0.75"

        return processed_data

    def transform_for_significance_testing(raw_data):
        """Transform data for significance testing and CD diagrams."""
        # Process data similar to rank analysis but ensure proper format for significance tests
        sig_data = benchmark_data_processor.process_performance_records(
            raw_benchmark_data=raw_data,
            budget_unit=benchmark_data_schema.runtime_unit,
            relativize_budget=True,
            collapse_repetitions=True,
            collapse_datasets=False,
            extra_ranking_cols=None,
        )

        # Ensure benchmark_identifier column exists (might be renamed during processing)
        if (
            "benchmark_identifier" not in sig_data.columns
            and benchmark_data_schema.bench_col in sig_data.columns
        ):
            sig_data["benchmark_identifier"] = sig_data[benchmark_data_schema.bench_col]
        elif "benchmark_identifier" not in sig_data.columns:
            # If neither exists, copy from raw data
            if "benchmark_identifier" in raw_data.columns:
                # Create a mapping from processed data back to raw data benchmark identifiers
                sig_data["benchmark_identifier"] = raw_data[
                    "benchmark_identifier"
                ].iloc[
                    0
                ]  # Use first value as fallback

        # Create mock significance test results per benchmark with expected column names
        mock_significance_data = []
        for bench in sig_data["benchmark_identifier"].unique():
            bench_df = sig_data[sig_data["benchmark_identifier"] == bench]
            tuners = bench_df["tuner"].unique()
            for i, tuner1 in enumerate(tuners):
                for j, tuner2 in enumerate(tuners):
                    if i < j:  # Only create upper triangle
                        mock_significance_data.append(
                            {
                                "benchmark_identifier": bench,
                                "entity1": tuner1,
                                "entity2": tuner2,
                                "p_value": np.random.uniform(0.01, 0.3),
                                "statistic": np.random.uniform(-2, 2),
                                "significant": np.random.random() > 0.5,
                            }
                        )

        significance_df = pd.DataFrame(mock_significance_data)
        return sig_data, significance_df

    # Return dictionary of transformation functions
    return {
        "rank_analysis": transform_for_rank_analysis,
        "iteration_performance": transform_for_iteration_performance,
        "sampler_comparison": transform_for_sampler_comparison,
        "architecture_comparison": transform_for_architecture_comparison,
        "conformalization_effect": transform_for_conformalization_effect,
        "coverage_analysis": transform_for_coverage_analysis,
        "significance_testing": transform_for_significance_testing,
    }


@pytest.fixture
def conformalization_effect_data():
    """
    Data for conformalization effect analysis comparing conformalized vs non-conformalized methods.
    """
    np.random.seed(202)

    data = []
    benchmarks = ["conformal_benchmark"]
    datasets = ["dataset_A", "dataset_B", "dataset_C"]

    # Tuner configurations showing conformalization effect
    tuner_configs = [
        ("Non_Conformalized_GBRT", "GBRT", "QGBM"),
        ("Conformalized_GBRT", "GBRT", "QGBM"),
        ("Non_Conformalized_TPE", "TPE", "MLP"),
        ("Conformalized_TPE", "TPE", "MLP"),
    ]

    normalized_runtimes = [10, 25, 50, 75, 100]

    for bench in benchmarks:
        for dataset in datasets:
            for tuner, sampler, arch in tuner_configs:
                for norm_runtime in normalized_runtimes:
                    # Conformalized methods generally perform slightly worse initially
                    # but converge to similar performance with more budget
                    if "Conformalized" in tuner:
                        base_rank = 2.2
                        convergence_rate = 0.006
                    else:
                        base_rank = 1.8
                        convergence_rate = 0.004

                    improvement = norm_runtime * convergence_rate
                    rank = max(
                        1.0, base_rank - improvement + np.random.uniform(-0.1, 0.1)
                    )
                    rank_lower = max(1.0, rank - 0.2)
                    rank_upper = min(4.0, rank + 0.2)

                    data.append(
                        {
                            "benchmark_identifier": bench,
                            "dataset": dataset,
                            "tuner": tuner,
                            "sampler": sampler,
                            "estimator_architecture": arch,
                            "normalized_runtime": norm_runtime,
                            "rank": rank,
                            "rank_lower": rank_lower,
                            "rank_upper": rank_upper,
                        }
                    )

    return pd.DataFrame(data)


@pytest.fixture
def toy_relativized_runtime_data():
    data = {
        "benchmark_identifier": [
            "bench_A",
            "bench_A",
            "bench_A",
            "bench_A",
            "bench_A",
            "bench_A",
            "bench_A",
            "bench_A",
            "bench_A",
            "bench_B",
            "bench_B",
            "bench_B",
            "bench_B",
            "bench_B",
            "bench_B",
            "bench_B",
            "bench_B",
            "bench_B",
        ],
        "dataset": [
            "dataset_1",
            "dataset_1",
            "dataset_1",
            "dataset_2",
            "dataset_2",
            "dataset_2",
            "dataset_3",
            "dataset_3",
            "dataset_3",
            "dataset_4",
            "dataset_4",
            "dataset_4",
            "dataset_5",
            "dataset_5",
            "dataset_5",
            "dataset_6",
            "dataset_6",
            "dataset_6",
        ],
        "tuner": [
            "tuner_X",
            "tuner_Y",
            "tuner_Z",
            "tuner_X",
            "tuner_Y",
            "tuner_Z",
            "tuner_X",
            "tuner_Y",
            "tuner_Z",
            "tuner_X",
            "tuner_Y",
            "tuner_Z",
            "tuner_X",
            "tuner_Y",
            "tuner_Z",
            "tuner_X",
            "tuner_Y",
            "tuner_Z",
        ],
        "sampler": [
            "TPE",
            "TPE",
            "Random",
            "Random",
            "TPE",
            "Random",
            "TPE",
            "Random",
            "TPE",
            "TPE",
            "Random",
            "TPE",
            "Random",
            "TPE",
            "Random",
            "TPE",
            "Random",
            "TPE",
        ],
        "confidence_level": [
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
        ],
        "rank": [1, 2, 3, 1.5, 1.5, 3, 2, 3, 1, 1, 2, 3, 3, 1.5, 1.5, 3, 1, 2],
    }

    return pd.DataFrame(data)


@pytest.fixture
def grouping_columns():
    return [
        "benchmark_identifier",
        "dataset",
        "tuner",
        "repetition",
        "sampler",
        "confidence_level",
        "estimator_architecture",
    ]


@pytest.fixture
def repetition_column():
    return "repetition"


@pytest.fixture
def performance_column():
    return "performance"


@pytest.fixture
def tuner_column():
    return "tuner"


@pytest.fixture
def sampler_column():
    return "sampler"


@pytest.fixture
def confidence_level_column():
    return "confidence_level"


@pytest.fixture
def estimator_architecture_column():
    return "estimator_architecture"


@pytest.fixture
def benchmark_column():
    return "benchmark_identifier"


@pytest.fixture
def budget_units():
    return ["iteration", "runtime"]


@pytest.fixture
def dataset_aggregators():
    return ["benchmark_identifier", "dataset"]


@pytest.fixture
def entity_columns():
    return ["benchmark_identifier", "dataset", "tuner"]


@pytest.fixture
def dummy_blackbox_generator():
    return {
        "generator": "rastrigin",
    }


@pytest.fixture
def dummy_jahs201_generator():
    return {
        "dataset": "cifar10",
    }


@pytest.fixture
def dummy_yahpo_generator():
    return {
        "dataset": "167168",
    }


@pytest.fixture
def dummy_nas301_generator():
    return {
        "instance_value": "CIFAR10",
    }


@pytest.fixture
def small_param_space():
    """Create a small parameter search space for testing."""
    return {
        "x": FloatRange(lower=0, upper=100.0),
        "y": FloatRange(lower=0, upper=100.0),
    }


@pytest.fixture
def performance_generator():
    """Create a BlackBoxGenerator with rastrigin for testing."""
    return BlackBoxGenerator(generator="rastrigin")


@pytest.fixture
def warm_start_configs(performance_generator):
    """Create a set of warm start configurations for testing with actual performance values."""
    configs = [
        {"x": 0.0, "y": 0.0},
        {"x": 1.0, "y": 1.0},
        {"x": 10.0, "y": 20.0},
        {"x": 30.0, "y": 40.0},
        {"x": 50.0, "y": 60.0},
        {"x": 70.0, "y": 80.0},
        {"x": 90.0, "y": 100.0},
        {"x": 75.0, "y": 25.0},
        {"x": 25.0, "y": 75.0},
        {"x": 45.0, "y": 55.0},
    ]

    # Get actual performances from the generator instead of hardcoding
    return [(config, performance_generator.predict(config)) for config in configs]


# Fixtures for metrics testing


@pytest.fixture
def extreme_significant_data():
    """Data where entities have extremely significant differences (zero variance)."""
    data = []
    # Use more datasets to get extremely small p-values in Wilcoxon test
    datasets = [f"dataset{i}" for i in range(1, 21)]  # 20 datasets

    # Entity A always ranks 1, B always ranks 2, C always ranks 3
    for dataset in datasets:
        data.extend(
            [
                {"dataset": dataset, "entity": "entity_A", "rank": 1.0},
                {"dataset": dataset, "entity": "entity_B", "rank": 2.0},
                {"dataset": dataset, "entity": "entity_C", "rank": 3.0},
            ]
        )

    return pd.DataFrame(data)


@pytest.fixture
def identical_ranks_data():
    """Data where all entities have identical ranks (no differences)."""
    data = []
    datasets = [f"dataset{i}" for i in range(1, 21)]  # 20 datasets
    entities = ["entity_A", "entity_B", "entity_C"]

    # All entities always get the same rank
    for dataset in datasets:
        for entity in entities:
            data.append({"dataset": dataset, "entity": entity, "rank": 2.0})

    return pd.DataFrame(data)


@pytest.fixture
def realistic_significant_data():
    """Realistic data with significant but variable differences."""
    np.random.seed(42)
    data = []
    datasets = [f"dataset{i}" for i in range(1, 21)]  # 20 datasets

    for dataset in datasets:
        # Entity A is consistently better (lower ranks) but with some variance
        rank_A = np.random.normal(1.2, 0.2)
        # Entity B is consistently middle
        rank_B = np.random.normal(2.5, 0.3)
        # Entity C is consistently worse
        rank_C = np.random.normal(3.8, 0.2)

        data.extend(
            [
                {"dataset": dataset, "entity": "entity_A", "rank": max(1.0, rank_A)},
                {"dataset": dataset, "entity": "entity_B", "rank": rank_B},
                {"dataset": dataset, "entity": "entity_C", "rank": max(1.0, rank_C)},
            ]
        )

    return pd.DataFrame(data)


@pytest.fixture
def realistic_insignificant_data():
    """Realistic data with small, insignificant differences."""
    np.random.seed(123)
    data = []
    datasets = [f"dataset{i}" for i in range(1, 21)]  # 20 datasets

    for dataset in datasets:
        # All entities have similar performance with high variance
        base_rank = np.random.normal(2.0, 0.1)
        data.extend(
            [
                {
                    "dataset": dataset,
                    "entity": "entity_A",
                    "rank": max(1.0, base_rank + np.random.normal(0, 0.4)),
                },
                {
                    "dataset": dataset,
                    "entity": "entity_B",
                    "rank": max(1.0, base_rank + np.random.normal(0, 0.4)),
                },
                {
                    "dataset": dataset,
                    "entity": "entity_C",
                    "rank": max(1.0, base_rank + np.random.normal(0, 0.4)),
                },
            ]
        )

    return pd.DataFrame(data)


@pytest.fixture
def grouped_test_data():
    """Data with breakout groups for testing grouped statistical tests."""
    data = []
    datasets = [f"dataset{i}" for i in range(1, 21)]  # 20 datasets
    entities = ["entity_A", "entity_B", "entity_C"]
    groups = ["group1", "group2"]

    for group in groups:
        for dataset in datasets:
            if group == "group1":
                # Group 1: significant differences
                ranks = {"entity_A": 1.0, "entity_B": 2.0, "entity_C": 3.0}
            else:
                # Group 2: no differences
                ranks = {"entity_A": 2.0, "entity_B": 2.0, "entity_C": 2.0}

            for entity in entities:
                data.append(
                    {
                        "dataset": dataset,
                        "entity": entity,
                        "rank": ranks[entity],
                        "group": group,
                    }
                )

    return pd.DataFrame(data)


@pytest.fixture
def insufficient_data():
    """Data with insufficient samples for statistical tests."""
    return pd.DataFrame(
        [
            {"dataset": "dataset1", "entity": "entity_A", "rank": 1.0},
            {"dataset": "dataset1", "entity": "entity_B", "rank": 2.0},
        ]
    )


@pytest.fixture
def independent_X_y_data():
    """X features completely independent of y outcome for likelihood ratio testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 3

    # Independent random features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=["feature1", "feature2", "feature3"],
    )
    # Random binary outcome
    y = pd.Series(np.random.binomial(1, 0.5, n_samples))

    return X, y


@pytest.fixture
def dependent_X_y_data():
    """X features with functional relationship to y outcome for likelihood ratio testing."""
    np.random.seed(42)
    n_samples = 100

    # Create features with relationship to outcome
    feature1 = np.random.randn(n_samples)
    feature2 = np.random.randn(n_samples)
    feature3 = np.random.randn(n_samples)

    # Create y with strong relationship to features
    linear_combination = 2 * feature1 + 1.5 * feature2 - 0.8 * feature3
    probabilities = 1 / (1 + np.exp(-linear_combination))  # sigmoid
    y = pd.Series(np.random.binomial(1, probabilities))

    X = pd.DataFrame({"feature1": feature1, "feature2": feature2, "feature3": feature3})

    return X, y


@pytest.fixture
def single_class_y_data():
    """Data with only one class in y for testing edge cases."""
    np.random.seed(42)
    n_samples = 50
    n_features = 3

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=["feature1", "feature2", "feature3"],
    )
    # All outcomes are the same class
    y = pd.Series(np.ones(n_samples, dtype=int))

    return X, y
