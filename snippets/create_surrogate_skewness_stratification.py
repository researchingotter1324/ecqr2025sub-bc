import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from stratification_utils import (
    get_benchmark_task_ids,
    sample_benchmark_data,
    validate_dataset,
    select_top_datasets,
    save_stratification,
)

BENCHMARKS = ["rbv2_aknn", "lcbench"]
TOP_COUNT = 5
TOP_PERCENT = None
MAX_PERFECT_ACC_RATIO = 0.05
MIN_RUNTIME = 8


def calculate_conditional_asymmetry(X: np.ndarray, y: np.ndarray) -> float:
    """Calculate conditional asymmetry using quantile skew ratio."""
    if len(X) == 0 or len(y) == 0:
        return 0.0

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_neighbors = 100
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(
        X_scaled
    )

    skew_ratios = []

    for i in range(len(X_scaled)):
        distances, indices = nbrs.kneighbors([X_scaled[i]])
        neighbor_indices = indices[0]
        local_y = y[neighbor_indices]

        q95 = np.quantile(local_y, 0.95)
        q50 = np.quantile(local_y, 0.5)
        q05 = np.quantile(local_y, 0.05)

        denominator = q50 - q05
        numerator = q95 - q50
        if numerator > 0 and denominator > 0:
            skew_ratio = numerator / denominator
            skew_ratios.append(np.log(skew_ratio))

    return np.median([abs(ratio) for ratio in skew_ratios])


def calculate_overall_asymmetry(y: np.ndarray) -> float:
    """Calculate the overall skewness of the entire Y sample."""
    skewness = stats.skew(y)
    return skewness if np.isfinite(skewness) else 0.0


def calculate_summary_stats(y: np.ndarray) -> dict:
    """Calculate comprehensive summary statistics for the y values."""
    stats_dict = {
        "count": len(y),
        "mean": np.mean(y),
        "std": np.std(y),
        "min": np.min(y),
        "max": np.max(y),
        "q05": np.quantile(y, 0.05),
        "q25": np.quantile(y, 0.25),
        "q50": np.quantile(y, 0.50),
        "q75": np.quantile(y, 0.75),
        "q95": np.quantile(y, 0.95),
        "skewness": stats.skew(y),
        "kurtosis": stats.kurtosis(y),
        "range": np.max(y) - np.min(y),
        "iqr": np.quantile(y, 0.75) - np.quantile(y, 0.25),
    }

    for key in stats_dict:
        if not np.isfinite(stats_dict[key]):
            stats_dict[key] = 0.0

    return stats_dict


def create_skewness_stratification(
    benchmark_name: str,
    task_ids: list,
    top_count: int = None,
    top_percent: float = None,
    max_perfect_acc_ratio: float = 0.01,
) -> list:
    """Create stratification based on highest conditional asymmetry datasets."""
    scores = {}
    for task_id in task_ids:
        tabularized_configurations, accuracies, runtimes = sample_benchmark_data(
            benchmark_name=benchmark_name, task_id=task_id
        )

        if validate_dataset(
            accuracies=accuracies,
            runtimes=runtimes,
            max_perfect_acc_ratio=max_perfect_acc_ratio,
            min_avg_runtime=MIN_RUNTIME,
        ):
            score = calculate_conditional_asymmetry(
                X=tabularized_configurations, y=accuracies
            )
            if score > 0:
                scores[task_id] = score

    return select_top_datasets(
        scores=scores, top_count=top_count, top_percent=top_percent
    )


def main():
    for benchmark in BENCHMARKS:
        task_ids = get_benchmark_task_ids(benchmark_name=benchmark)
        top_skewed = create_skewness_stratification(
            benchmark_name=benchmark,
            task_ids=task_ids,
            top_count=TOP_COUNT,
            top_percent=TOP_PERCENT,
            max_perfect_acc_ratio=MAX_PERFECT_ACC_RATIO,
        )

        if top_skewed:
            output_file = f"top_asymmetric_datasets_{benchmark}.json"
            save_stratification(task_ids=top_skewed, output_file=output_file)


if __name__ == "__main__":
    main()
