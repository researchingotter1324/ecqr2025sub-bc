"""
Skewness Stratification for HPO Surrogate Benchmarks
=====================================================
Selects benchmark tasks whose response-surface exhibits extreme *conditional*
skewness — i.e. the distribution of the target metric inside local
hyperparameter-space neighbourhoods is consistently asymmetric.

Score: mean absolute Moors (1988) skewness across all k-NN patches
-------------------------------------------------------------------
For every point xᵢ we find its k nearest neighbours (Gower distance,
mixed-type HP space), compute the Moors skewness of the local target values:

    Moors(y_local) = [(Q.875 - Q.625) - (Q.375 - Q.125)] / (Q.875 - Q.125)

then take the *absolute value* (direction of skew is irrelevant; we care only
about magnitude) and average across all points.  This gives one scalar per
task that measures how non-symmetric the response surface is on average.

k is adaptive: k = max(20, min(100, n // 20)) — roughly 5 % of n, capped.

Statistical validation
----------------------
Mann-Whitney U test (one-sided: selected > rest) on the per-task scores.
Effect size: Cliff's delta.  No p-value correction applied.
"""

import os
import json
import warnings
import logging
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, bootstrap
from sklearn.neighbors import NearestNeighbors

from stratification_utils import (
    get_benchmark_task_ids,
    sample_benchmark_data,
    validate_dataset,
    select_top_datasets,
    save_stratification,
    gower_distance_matrix,
)
from plot_utils import group_score_boxplot, target_kde_per_task

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

SEED = 42
np.random.seed(SEED)

BENCHMARKS = ["rbv2_aknn", "lcbench"]
TOP_COUNT = 5
MAX_PERFECT_ACC_RATIO = 0.05
MIN_RUNTIME = 8

OUT_ROOT = os.path.join("cache", "snippets_outputs", "skewness")
SUMMARY_DIR = os.path.join("cache", "snippets_outputs", "summary")


# ── Score computation ─────────────────────────────────────────────────────────

def _adaptive_k(num_samples: int) -> int:
    return max(20, min(100, num_samples // 20))


def _moors_skewness(values: np.ndarray) -> float:
    """
    Moors (1988) quantile skewness, bounded in [-1, 1].
    Returns 0 when the denominator is effectively zero (degenerate patch).
    """
    q = np.quantile(values, [0.125, 0.375, 0.625, 0.875])
    spread = q[3] - q[0]
    if spread < 1e-12:
        return 0.0
    return ((q[3] - q[2]) - (q[1] - q[0])) / spread


def mean_absolute_local_skewness(
    configs: np.ndarray,
    feature_groups: list,
    performance: np.ndarray,
) -> float:
    """
    Compute the mean |Moors skewness| over all k-NN local patches.

    Uses Gower distance so that the mixed-type HP space is handled correctly —
    each hyperparameter (continuous, integer, or categorical) contributes
    equally regardless of its cardinality or encoding width.

    Returns 0.0 if there are too few samples to form meaningful patches.
    """
    num_samples = len(performance)
    if num_samples < 40:
        return 0.0

    distance_matrix = gower_distance_matrix(configs, feature_groups)
    k = _adaptive_k(num_samples)

    nbrs = NearestNeighbors(n_neighbors=k, metric="precomputed", algorithm="brute")
    nbrs.fit(distance_matrix)
    _, neighbour_indices = nbrs.kneighbors(distance_matrix)

    local_skewness_values = np.array(
        [abs(_moors_skewness(performance[idx])) for idx in neighbour_indices]
    )
    return float(np.mean(local_skewness_values))


# ── Statistical validation ────────────────────────────────────────────────────

def _cliffs_delta(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cliff's delta ∈ [-1, 1]: P(a > b) - P(b > a)."""
    dominance = sum(
        1 if a > b else (-1 if a < b else 0)
        for a in group_a for b in group_b
    )
    return dominance / (len(group_a) * len(group_b))


def compare_selected_vs_rest(
    selected_task_ids: list,
    score_by_task: dict,
) -> dict:
    """
    Mann-Whitney U test (one-sided: selected scores > rest scores) and
    Cliff's delta effect size.
    """
    selected_scores = np.array([score_by_task[t] for t in selected_task_ids if t in score_by_task])
    rest_scores = np.array([score_by_task[t] for t in score_by_task if t not in selected_task_ids])

    if len(selected_scores) < 2 or len(rest_scores) < 2:
        return {"note": "insufficient tasks for statistical comparison"}

    u_stat, p_value = mannwhitneyu(selected_scores, rest_scores, alternative="greater")
    delta = _cliffs_delta(selected_scores, rest_scores)

    return {
        "n_selected":             len(selected_scores),
        "n_rest":                 len(rest_scores),
        "selected_score_median":  float(np.median(selected_scores)),
        "rest_score_median":      float(np.median(rest_scores)),
        "mwu_statistic":          float(u_stat),
        "mwu_p_value":            float(p_value),
        "cliffs_delta":           float(delta),
    }


# ── Per-benchmark stratification ──────────────────────────────────────────────

def score_and_select_tasks(
    benchmark: str,
    task_ids: list,
    top_count: int,
) -> tuple[list, dict, dict]:
    """
    Score every valid task in the benchmark and return the top-scoring ones.

    Returns
    -------
    selected_task_ids : list[str]
    score_by_task     : dict[task_id -> float]   (all scored tasks)
    performance_by_task : dict[task_id -> np.ndarray]  (for plotting)
    """
    score_by_task = {}
    performance_by_task = {}

    for task_id in task_ids:
        configs, feature_groups, performance, runtimes = sample_benchmark_data(
            benchmark_name=benchmark, task_id=task_id
        )
        if not validate_dataset(
            accuracies=performance,
            runtimes=runtimes,
            max_perfect_acc_ratio=MAX_PERFECT_ACC_RATIO,
            min_avg_runtime=MIN_RUNTIME,
        ):
            continue

        score = mean_absolute_local_skewness(configs, feature_groups, performance)
        if score > 0:
            score_by_task[task_id] = score
            performance_by_task[task_id] = performance

    selected_task_ids = select_top_datasets(scores=score_by_task, top_count=top_count)
    return selected_task_ids, score_by_task, performance_by_task


# ── Summary table ─────────────────────────────────────────────────────────────

def save_summary_csv(
    selected_task_ids: list,
    score_by_task: dict,
    performance_by_task: dict,
    test_result: dict,
    benchmark: str,
):
    rows = []
    for task_id, performance in performance_by_task.items():
        q25, q50, q75 = np.quantile(performance, [0.25, 0.50, 0.75])
        mean_ci = bootstrap(
            (performance,), np.mean,
            confidence_level=0.95, n_resamples=999,
            random_state=SEED, method="percentile",
        ).confidence_interval
        rows.append({
            "task_id":          task_id,
            "selected":         task_id in selected_task_ids,
            "mean_abs_moors":   round(score_by_task[task_id], 6),
            "n":                len(performance),
            "mean_perf":        round(float(np.mean(performance)), 6),
            "mean_perf_ci_lo":  round(float(mean_ci.low), 6),
            "mean_perf_ci_hi":  round(float(mean_ci.high), 6),
            "median_perf":      round(float(q50), 6),
            "iqr_perf":         round(float(q75 - q25), 6),
        })

    df = pd.DataFrame(rows).sort_values("mean_abs_moors", ascending=False)
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    df.to_csv(os.path.join(SUMMARY_DIR, f"skewness_summary_{benchmark}.csv"), index=False)

    test_path = os.path.join(SUMMARY_DIR, f"skewness_group_test_{benchmark}.json")
    with open(test_path, "w") as fh:
        json.dump(test_result, fh, indent=2, default=str)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(SEED)

    # Accumulate results across benchmarks so both plots are produced in one call
    boxplot_data = {}
    kde_data = {}

    for benchmark in BENCHMARKS:
        task_ids = get_benchmark_task_ids(benchmark_name=benchmark)
        selected, score_by_task, performance_by_task = score_and_select_tasks(
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

        test_result = compare_selected_vs_rest(selected, score_by_task)
        save_summary_csv(selected, score_by_task, performance_by_task, test_result, benchmark)

        boxplot_data[benchmark] = {
            "selected_scores": [score_by_task[t] for t in selected if t in score_by_task],
            "rest_scores":     [score_by_task[t] for t in score_by_task if t not in selected],
            "mwu_p_value":     test_result.get("mwu_p_value", float("nan")),
            "cliffs_delta":    test_result.get("cliffs_delta", float("nan")),
        }
        kde_data[benchmark] = {
            "selected_task_ids": selected,
            "task_performance":  performance_by_task,
        }

        print(f"[{benchmark}] selected: {selected}")
        print(f"  MWU p = {test_result.get('mwu_p_value', 'N/A'):.4e}  "
              f"Cliff's delta = {test_result.get('cliffs_delta', 'N/A'):.3f}")

    # One figure per plot type, covering all benchmarks as columns
    if boxplot_data:
        os.makedirs(OUT_ROOT, exist_ok=True)
        group_score_boxplot(
            results_by_benchmark=boxplot_data,
            score_label="|Moors skewness score| (mean over patches)",
            out_path=os.path.join(OUT_ROOT, "selected_vs_rest_scores"),
        )
    if kde_data:
        target_kde_per_task(
            results_by_benchmark=kde_data,
            out_path=os.path.join(OUT_ROOT, "task_performance_distributions"),
        )


if __name__ == "__main__":
    main()
