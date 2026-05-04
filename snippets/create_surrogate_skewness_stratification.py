import json
import os
import warnings
import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors

from stratification_utils import (
    get_benchmark_task_ids,
    sample_benchmark_data,
    validate_dataset,
    select_top_datasets,
    save_stratification,
    gower_distance_matrix,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

SEED = 42

BENCHMARKS = ["lcbench"]
TOP_COUNT = 5
MAX_PERFECT_ACC_RATIO = 0.05
MIN_RUNTIME = 8

MIN_SAMPLES = 40
MIN_K = 20

SUMMARY_DIR = os.path.join("cache", "snippets_outputs", "summary")


def adaptive_k(num_samples: int) -> int:
    """Compute an adaptive neighbourhood size for local asymmetry estimation.

    The neighbourhood is set to roughly 5 % of the dataset, capped at 100 to
    prevent patches from spanning a large fraction of the hyperparameter space
    and losing local sensitivity.  The lower bound of MIN_SAMPLES // 2 ensures
    that Q(0.25) in each patch is based on at least 5 order statistics.

    Args:
        num_samples: Total number of observations in the dataset.

    Returns:
        Neighbourhood size k in [MIN_K, 100].
    """
    return max(MIN_K, min(100, num_samples // 20))


def bowley_asymmetry(patches: np.ndarray) -> np.ndarray:
    """Compute the Bowley (1920) quartile skewness for each local patch.

    This is the Groeneveld-Meeden (1984) measure evaluated at u = 0.25:

        γ = [Q(0.25) + Q(0.75) − 2·Q(0.5)] / [Q(0.75) − Q(0.25)]

    The numerator measures how far the midpoint of the inter-quartile interval
    lies from the median; the denominator is the IQR, which normalises γ to
    (−1, 1) for any distribution with positive spread.  u = 0.25 is the
    canonical, best-studied instantiation of the Groeneveld-Meeden family and
    is directly citable from both Bowley (1920) and Groeneveld & Meeden (1984).

    Patches with IQR < 1e-12 (effectively constant distributions) produce a
    degenerate score and are masked out; the caller receives NaN for those rows
    and is responsible for filtering them.

    Args:
        patches: Performance values for every local patch, shape (n, k).

    Returns:
        Per-patch Bowley asymmetry, shape (n,).  Degenerate patches yield NaN.
    """
    q25 = np.quantile(patches, 0.25, axis=1)
    q50 = np.quantile(patches, 0.50, axis=1)
    q75 = np.quantile(patches, 0.75, axis=1)
    iqr = q75 - q25
    degenerate = iqr < 1e-12
    gamma = np.where(degenerate, np.nan, (q25 + q75 - 2.0 * q50) / np.where(degenerate, 1.0, iqr))
    return gamma


def mean_absolute_local_asymmetry(
    X_num: np.ndarray,
    X_cat_str: np.ndarray,
    performance: np.ndarray,
) -> float:
    """Compute the mean absolute local asymmetry of the performance surface.

    For each observed configuration, its k nearest neighbours in hyperparameter
    space are identified using Gower (1971) distance. Gower distance handles
    mixed-type hyperparameter spaces without requiring a user-defined scale and
    guarantees each original hyperparameter contributes equally regardless of
    category count or measurement scale.

    Within each local patch the Bowley (1920) quartile asymmetry is computed
    (see bowley_asymmetry). The anchor point is excluded from its own patch by
    requesting k+1 neighbours and dropping the self at distance 0.

    Args:
        X_num: Numeric feature matrix, shape (n, n_numeric).
        X_cat_str: Categorical feature matrix, shape (n, n_cat), dtype object.
            Raw string values — no OHE required.
        performance: Observed performance values, shape (n,).

    Returns:
        Mean absolute Bowley asymmetry over all non-degenerate patches, in [0, 1).

    Raises:
        ValueError: If fewer than MIN_SAMPLES observations are present, or if
            all patches are degenerate (constant performance surface).
    """
    n = len(performance)
    if n < MIN_SAMPLES:
        raise ValueError(
            f"Insufficient samples for asymmetry scoring: {n} < {MIN_SAMPLES}."
        )

    distance_matrix = gower_distance_matrix(X_num, X_cat_str)
    k = adaptive_k(n)

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="precomputed", algorithm="brute")
    nbrs.fit(distance_matrix)
    _, neighbour_indices = nbrs.kneighbors(distance_matrix)
    neighbour_indices = neighbour_indices[:, 1:]

    patches = np.stack([performance[idx] for idx in neighbour_indices])
    gamma = bowley_asymmetry(patches)

    valid = ~np.isnan(gamma)
    if not valid.any():
        raise ValueError(
            "All local patches have zero IQR; performance surface is constant."
        )

    return float(np.abs(gamma[valid]).mean())


def score_and_select_tasks(
    benchmark: str,
    task_ids: list,
    top_count: int,
) -> tuple[list, dict]:
    """Score every valid task in the benchmark and select the top-scoring ones.

    Tasks that fail dataset validation are skipped silently (validation failure
    is expected and not an error).  Tasks where mean_absolute_local_asymmetry
    raises a ValueError are logged as warnings and excluded from the scored pool.
    Both cases result in the task being absent from the analysis entirely; no
    sentinel score is injected.

    Args:
        benchmark: YAHPO benchmark scenario name (e.g. "lcbench").
        task_ids: List of task/instance identifiers to evaluate.
        top_count: Number of highest-scoring tasks to select.

    Returns:
        Tuple of (selected_task_ids, score_by_task) where selected_task_ids is
        the list of selected task IDs and score_by_task maps every scored task
        ID to its asymmetry score.
    """
    score_by_task = {}

    for task_id in task_ids:
        X_num, X_cat_str, _num_names, _cat_names, performance, runtimes = sample_benchmark_data(
            benchmark_name=benchmark, task_id=task_id
        )
        if not validate_dataset(
            accuracies=performance,
            runtimes=runtimes,
            max_perfect_acc_ratio=MAX_PERFECT_ACC_RATIO,
            min_avg_runtime=MIN_RUNTIME,
        ):
            continue

        try:
            score = mean_absolute_local_asymmetry(X_num, X_cat_str, performance)
        except ValueError as exc:
            logger.warning("Skipping task %s (%s): %s", task_id, benchmark, exc)
            continue

        score_by_task[task_id] = score

    selected_task_ids = select_top_datasets(scores=score_by_task, top_count=top_count)
    return selected_task_ids, score_by_task


def save_boxplot_data(selected_scores: list, rest_scores: list, benchmark: str):
    """Write the selected/rest score arrays for one benchmark to a JSON file.

    The file is read by create_stratification_boxplots.py to produce the
    combined figure without re-running the scoring pipeline.

    Args:
        selected_scores: Asymmetry scores for the selected tasks.
        rest_scores: Asymmetry scores for the non-selected tasks.
        benchmark: Benchmark name, used to construct the output filename.
    """
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    path = os.path.join(SUMMARY_DIR, f"asymmetry_boxplot_{benchmark}.json")
    with open(path, "w") as fh:
        json.dump({"selected_scores": selected_scores, "rest_scores": rest_scores}, fh, indent=2)


def main():
    """Run the local asymmetry stratification for all configured benchmarks.

    For each benchmark, scores every valid task, saves the stratification JSON
    and a JSON file with the selected/rest score arrays for downstream boxplot
    generation.
    """
    np.random.seed(SEED)

    for benchmark in BENCHMARKS:
        task_ids = get_benchmark_task_ids(benchmark_name=benchmark)
        selected, score_by_task = score_and_select_tasks(
            benchmark=benchmark,
            task_ids=task_ids,
            top_count=TOP_COUNT,
        )
        if not selected:
            continue

        save_stratification(
            task_ids=selected,
            output_file=f"top_asymmetric_datasets_{benchmark}.json",
        )
        save_boxplot_data(
            selected_scores=[score_by_task[t] for t in selected if t in score_by_task],
            rest_scores=[score_by_task[t] for t in score_by_task if t not in selected],
            benchmark=benchmark,
        )

        print(f"[{benchmark}] selected: {selected}")


if __name__ == "__main__":
    main()
