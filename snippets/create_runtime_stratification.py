import numpy as np
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


def create_runtime_stratification(
    benchmark_name: str,
    task_ids: list,
    top_count: int = None,
    top_percent: float = None,
    max_perfect_acc_ratio: float = 0.01,
) -> list:
    """Create stratification based on highest average runtime datasets."""
    scores = {}
    for task_id in task_ids:
        _X_num, _X_cat_str, _num_names, _cat_names, accuracies, runtimes = sample_benchmark_data(
            benchmark_name=benchmark_name, task_id=task_id
        )

        if (
            validate_dataset(
                accuracies=accuracies,
                runtimes=runtimes,
                max_perfect_acc_ratio=max_perfect_acc_ratio,
                min_avg_runtime=0,  # No minimum runtime requirement for runtime stratification
            )
            and len(runtimes) > 0
        ):
            scores[task_id] = np.mean(runtimes)

    return select_top_datasets(
        scores=scores, top_count=top_count, top_percent=top_percent
    )


def main():
    for benchmark in BENCHMARKS:
        task_ids = get_benchmark_task_ids(benchmark_name=benchmark)
        top_runtime_datasets = create_runtime_stratification(
            benchmark_name=benchmark,
            task_ids=task_ids,
            top_count=TOP_COUNT,
            top_percent=TOP_PERCENT,
            max_perfect_acc_ratio=MAX_PERFECT_ACC_RATIO,
        )

        if top_runtime_datasets:
            output_file = f"top_runtime_datasets_{benchmark}.json"
            save_stratification(task_ids=top_runtime_datasets, output_file=output_file)


if __name__ == "__main__":
    main()
