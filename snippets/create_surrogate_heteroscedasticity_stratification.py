import json
import os
import warnings
import logging
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from stratification_utils import (
    get_benchmark_task_ids,
    sample_benchmark_data,
    validate_dataset,
    select_top_datasets,
    save_stratification,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

SEED = 42

BENCHMARKS = ["lcbench"]
TOP_COUNT = 5
MAX_PERFECT_ACC_RATIO = 0.05
MIN_RUNTIME = 8

TEST_SIZE = 0.2

SUMMARY_DIR = os.path.join("cache", "snippets_outputs", "summary")


def build_ols_matrix(
    X_num: np.ndarray,
    X_cat_str: np.ndarray,
) -> np.ndarray:
    """Assemble the OLS regressor matrix from numeric and categorical arrays.

    Numeric columns are used as-is (no scaling — linear regression does not
    require it). Categorical features are one-hot encoded with the full K
    dummies per feature; statsmodels OLS with add_constant handles any
    resulting rank deficiency via its internal pivoting.

    The encoder is fit on the data passed in, so this must be called on the
    full dataset before splitting to ensure train and test share the same
    column layout.

    Args:
        X_num: Numeric feature matrix, shape (n, n_numeric). No constant columns.
        X_cat_str: Categorical feature matrix, shape (n, n_cat), dtype object.
            No constant columns.

    Returns:
        OLS regressor matrix, shape (n, n_numeric + sum_K).
    """
    if X_cat_str.shape[1] == 0:
        return X_num.copy()

    enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
    X_ohe = enc.fit_transform(X_cat_str)
    return np.hstack([X_num, X_ohe])


def fit_ols_and_get_residuals(
    X_num: np.ndarray,
    X_cat_str: np.ndarray,
    performance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit OLS on a train split and return the test regressor matrix and residuals.

    The full OLS matrix (numerics + OHE categoricals) is assembled before
    splitting so the OHE column layout is identical for train and test. Data
    is split 80/20 with a fixed seed.

    Args:
        X_num: Numeric feature matrix, shape (n, n_numeric).
        X_cat_str: Categorical feature matrix, shape (n, n_cat), dtype object.
        performance: Target values, shape (n,).

    Returns:
        Tuple of (X_test_ols, residuals) where X_test_ols is the OLS regressor
        matrix for the test split and residuals are the held-out OLS residuals.
    """
    X_ols = build_ols_matrix(X_num, X_cat_str)
    X_train, X_test, y_train, y_test = train_test_split(
        X_ols, performance, test_size=TEST_SIZE, random_state=SEED
    )

    ols = sm.OLS(y_train, sm.add_constant(X_train, has_constant="add")).fit()
    predicted = ols.predict(sm.add_constant(X_test, has_constant="add"))
    residuals = y_test - predicted

    return X_test, residuals


def breusch_pagan_r_squared(
    X_bp: np.ndarray,
    residuals: np.ndarray,
) -> float:
    """Breusch-Pagan (1979) auxiliary regression R^2 for heteroscedasticity scoring.

    Auxiliary OLS: e^2 ~ X_bp (with intercept). R^2 measures how much of the
    variance in squared residuals is explained by the hyperparameter features,
    serving as a scale-free heteroscedasticity score comparable across datasets.

    Args:
        X_bp: Regressor matrix, shape (n, p).
        residuals: Held-out OLS residuals, shape (n,).

    Returns:
        R^2 of the auxiliary regression, bounded in [0, 1].
    """
    squared_residuals = residuals ** 2
    aux_ols = sm.OLS(squared_residuals, sm.add_constant(X_bp, has_constant="add")).fit()
    return float(aux_ols.rsquared)


def heteroscedasticity_score(
    X_num: np.ndarray,
    X_cat_str: np.ndarray,
    performance: np.ndarray,
) -> float:
    """Fit OLS, compute held-out residuals, return Breusch-Pagan R^2.

    Args:
        X_num: Numeric feature matrix, shape (n, n_numeric).
        X_cat_str: Categorical feature matrix, shape (n, n_cat), dtype object.
        performance: Observed performance values, shape (n,).

    Returns:
        Breusch-Pagan auxiliary R^2 in [0, 1].

    Raises:
        ValueError: If fewer than 30 samples are present.
    """
    if len(performance) < 30:
        raise ValueError(
            f"Insufficient samples for heteroscedasticity scoring ({len(performance)} < 30)"
        )
    X_bp, residuals = fit_ols_and_get_residuals(X_num, X_cat_str, performance)
    return breusch_pagan_r_squared(X_bp, residuals)


def score_and_select_tasks(
    benchmark: str,
    task_ids: list,
    top_count: int,
) -> tuple[list, dict]:
    """Score every valid task in the benchmark and select the top-scoring ones.

    Tasks that fail dataset validation are skipped silently. Tasks where
    heteroscedasticity_score raises are logged as warnings and excluded from
    the scored pool; no sentinel score is injected.

    Args:
        benchmark: YAHPO benchmark scenario name (e.g. "lcbench").
        task_ids: List of task/instance identifiers to evaluate.
        top_count: Number of highest-scoring tasks to select.

    Returns:
        Tuple of (selected_task_ids, score_by_task) where selected_task_ids is
        the list of selected task IDs and score_by_task maps every scored task
        ID to its Breusch-Pagan R^2.
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
            r_sq = heteroscedasticity_score(X_num, X_cat_str, performance)
        except Exception as exc:
            logger.warning("[%s/%s] heteroscedasticity scoring failed, skipping: %s",
                           benchmark, task_id, exc)
            continue

        score_by_task[task_id] = r_sq

    selected_task_ids = select_top_datasets(scores=score_by_task, top_count=top_count)
    return selected_task_ids, score_by_task


def save_boxplot_data(boxplot_data: dict, benchmark: str):
    """Write the selected/rest score arrays for one benchmark to a JSON file.

    The file is read by create_stratification_boxplots.py to produce the
    combined figure without re-running the scoring pipeline.

    Args:
        boxplot_data: Dict with keys ``selected_scores`` and ``rest_scores``,
            each a list of floats.
        benchmark: Benchmark name, used to construct the output filename.
    """
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    path = os.path.join(SUMMARY_DIR, f"heteroscedasticity_boxplot_{benchmark}.json")
    with open(path, "w") as fh:
        json.dump(boxplot_data, fh, indent=2)


def main():
    """Run heteroscedasticity scoring for all configured benchmarks.

    For each benchmark, scores every valid task, saves the stratification JSON
    and a JSON file containing the selected and rest score arrays for downstream
    boxplot generation.
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
            output_file=f"top_heteroscedastic_datasets_{benchmark}.json",
        )

        save_boxplot_data(
            boxplot_data={
                "selected_scores": [score_by_task[t] for t in selected if t in score_by_task],
                "rest_scores":     [score_by_task[t] for t in score_by_task if t not in selected],
            },
            benchmark=benchmark,
        )

        print(f"[{benchmark}] selected: {selected}")


if __name__ == "__main__":
    main()
