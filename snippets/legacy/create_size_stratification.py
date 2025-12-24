import json
from typing import List, Union
import warnings
import time
import threading

from yahpo_gym import BenchmarkSet, local_config
import openml
import logging

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Configure yahpo_gym data path
local_config.init_config()
local_config.set_data_path("yahpo_bench_data")


class TimeoutException(Exception):
    pass


def run_with_timeout(func, args, timeout_seconds):
    """Run a function with a timeout using threading."""
    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = func(*args)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        # Thread is still running, timeout occurred
        raise TimeoutException(f"Operation timed out after {timeout_seconds} seconds")

    if exception[0]:
        raise exception[0]

    return result[0]


def get_benchmark_task_ids(benchmark_name: str) -> List[str]:
    benchmark_set = BenchmarkSet(benchmark_name)
    return benchmark_set.instances


def _fetch_dataset_size_internal(task_id: str) -> int:
    """Internal function to fetch dataset size."""
    # Configure OpenML to use working endpoint
    openml.config.server = "https://www.openml.org/api/v1/xml"

    task = openml.tasks.get_task(int(task_id))
    dataset_id = task.dataset_id

    # Get size from metadata instead of downloading the full dataset
    dataset = openml.datasets.get_dataset(dataset_id)
    size = int(dataset.qualities["NumberOfInstances"])
    return size


def fetch_dataset_size(
    task_id: str, max_retries: int = 3, delay: float = 1.0, timeout_seconds: int = 60
) -> int:
    """Fetch dataset size with retry logic and timeout for API failures."""
    for attempt in range(max_retries):
        try:
            result = run_with_timeout(
                _fetch_dataset_size_internal, (task_id,), timeout_seconds
            )
            return result

        except Exception as e:
            if attempt < max_retries - 1:
                print(
                    f"  Attempt {attempt + 1} failed for Task {task_id}: {str(e)[:100]}..."
                )
                print(f"  Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(
                    f"  Failed to fetch Task {task_id} after {max_retries} attempts: {str(e)[:100]}..."
                )
                return None


def create_size_stratification(
    task_ids: List[str],
    top_count: Union[int, None] = None,
    top_percent: Union[float, None] = None,
) -> List[str]:
    """
    Create stratification based on either top N datasets or top X percent.

    Args:
        task_ids: List of task IDs to process
        top_count: Number of top datasets to select (mutually exclusive with top_percent)
        top_percent: Percentage of top datasets to select (mutually exclusive with top_count)
    """
    if top_count is not None and top_percent is not None:
        raise ValueError("Cannot specify both top_count and top_percent")
    if top_count is None and top_percent is None:
        raise ValueError("Must specify either top_count or top_percent")

    dataset_sizes = {}

    for i, task_id in enumerate(task_ids, 1):
        print(f"Processing {i}/{len(task_ids)}: Task {task_id}")

        size = fetch_dataset_size(task_id)
        if size is not None:
            dataset_sizes[task_id] = size
            print(f"  Task {task_id}: {size:,} samples")
        else:
            print(f"  Task {task_id}: Skipped due to errors")

    if not dataset_sizes:
        print("No datasets were successfully processed!")
        return []

    # Sort by size (descending)
    sorted_tasks = sorted(dataset_sizes.items(), key=lambda x: x[1], reverse=True)

    # Determine how many to select
    if top_count is not None:
        n_top = min(top_count, len(sorted_tasks))
        selection_desc = f"top {n_top} largest datasets"
    else:
        n_top = max(1, int(len(sorted_tasks) * top_percent / 100))
        selection_desc = f"top {top_percent}% largest datasets"

    top_tasks = [task_id for task_id, _ in sorted_tasks[:n_top]]

    print(
        f"\n{selection_desc.title()} (from {len(dataset_sizes)} successfully processed):"
    )
    for i, (task_id, size) in enumerate(sorted_tasks[:n_top], 1):
        print(f"  {i}. Task {task_id}: {size:,} samples")

    return top_tasks


def save_stratification(task_ids: List[str], output_file: str = None):
    if output_file is None:
        output_file = "top_largest_datasets.json"

    with open(output_file, "w") as f:
        json.dump(task_ids, f, indent=2)

    print(f"\nSaved {len(task_ids)} task IDs to {output_file}")


def main():
    # Configuration - define parameters directly
    top_count = 5
    top_percent = None
    benchmarks = ["lcbench", "rbv2_aknn"]

    for benchmark in benchmarks:
        print(f"\nCreating {benchmark} size stratification...")

        try:
            task_ids = get_benchmark_task_ids(benchmark)
            top_largest = create_size_stratification(
                task_ids, top_count=top_count, top_percent=top_percent
            )

            if top_largest:
                output_file = f"top_largest_datasets_{benchmark}.json"
                save_stratification(top_largest, output_file)
                print(f"Completed {benchmark}: {len(top_largest)} datasets selected")
            else:
                print(
                    f"No datasets were successfully processed for {benchmark}. Check your network connection and try again."
                )
        except Exception as e:
            print(f"Error processing {benchmark}: {str(e)}")


if __name__ == "__main__":
    main()
