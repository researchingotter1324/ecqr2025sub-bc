"""
Heteroscedasticity Stratification for HPO Surrogate Benchmarks
==============================================================
Selects benchmark tasks whose response-surface exhibits extreme
heteroscedasticity as it would be experienced by a Gaussian process surrogate
— i.e. the variance of the GP residuals varies systematically with the
predicted mean.

Why a GP surrogate, not a simpler estimator
--------------------------------------------
Heteroscedasticity is only meaningful relative to a specific model.  Because
the wider HPO-bench framework uses GP-based Bayesian optimisation (Optuna's
GPSampler), we use the same GP model to estimate the mean function.  Residuals
from *this* model are exactly the error signal that would be misrepresented
if the GP assumed homoscedastic noise.

GP model
---------
We use the exact same GP as Optuna's GPSampler:
  - Matern-5/2 ARD kernel
  - Kernel hyperparameters fitted by maximising the penalised marginal
    log-likelihood (L-BFGS-B, Gamma priors on kernel scale and noise variance)
  - Input normalisation: numerics min-max scaled to [0,1], categoricals as
    integer indices (Hamming distance in the kernel)

The GP is fitted on a subsample of up to MAX_GP_SAMPLES points to keep
the O(n³) Cholesky cost tractable.  Predictions are then made on the same
subsample.

Heteroscedasticity test: Breusch-Pagan (1979)
----------------------------------------------
After obtaining GP residuals eᵢ = yᵢ − μ̂(xᵢ), we run the Breusch-Pagan
auxiliary OLS regression:

    eᵢ² = α₀ + α₁xᵢ₁ + … + αₚxᵢₚ + εᵢ

and compute the test statistic:

    LM = n · R²  ~  χ²(p)  under H₀ (homoscedasticity)

where n is the number of observations and p is the number of predictors.
The task-level score is LM (larger = stronger heteroscedasticity signal).
The p-value from χ²(p) is also stored.

This is the textbook Breusch-Pagan test as in the original paper:
    Breusch, T. S. & Pagan, A. R. (1979).  A simple test for heteroscedasticity
    and random coefficient variation.  Econometrica 47(5), 1287–1294.

Why LM = nR² and not raw R²?
  - Raw R² grows mechanically with the number of regressors p; LM does not.
  - LM has a known null distribution (χ²(p)), giving a proper p-value.
  - LM is consistent: it → ∞ under any fixed alternative as n → ∞.

Statistical validation of the stratification
--------------------------------------------
Mann-Whitney U test (one-sided: selected > rest) on LM scores.
Effect size: Cliff's delta.  No p-value correction applied.
"""

import os
import json
import warnings
import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, bootstrap, chi2

from stratification_utils import (
    get_benchmark_task_ids,
    sample_benchmark_data,
    validate_dataset,
    select_top_datasets,
    save_stratification,
    normalise_for_gp,
)
from optuna_gp import fit_gp
from plot_utils import group_score_boxplot, target_kde_per_task

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

SEED = 42
np.random.seed(SEED)

BENCHMARKS = ["rbv2_aknn", "lcbench"]
TOP_COUNT = 5
MAX_PERFECT_ACC_RATIO = 0.05
MIN_RUNTIME = 8
MAX_GP_SAMPLES = 500   # subsample cap for O(n³) GP fit

OUT_ROOT = os.path.join("cache", "snippets_outputs", "heteroscedasticity")
SUMMARY_DIR = os.path.join("cache", "snippets_outputs", "summary")


# ── GP fit and residuals ──────────────────────────────────────────────────────

def _standardise(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = values.mean()
    std = max(values.std(), 1e-10)
    return (values - mean) / std, mean, std


def _subsample(X: np.ndarray, y: np.ndarray, max_n: int) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= max_n:
        return X, y
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y), max_n, replace=False)
    return X[idx], y[idx]


def fit_gp_and_get_residuals(
    configs: np.ndarray,
    feature_groups: list,
    performance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit the Optuna GP to (configs, performance) and return residuals.

    The GP is the same model used by Optuna's GPSampler:
      - Matern-5/2 ARD kernel
      - MLL-fitted hyperparameters with Gamma priors
      - Inputs normalised: numerics to [0,1], categoricals as integer indices

    Returns
    -------
    X_gp    : shape (n_sub, p) — GP input matrix (normalised, one col per HP)
    predicted  : shape (n_sub,)  — GP posterior mean predictions
    residuals  : shape (n_sub,)  — raw residuals e = y - predicted (original scale)
    """
    X_gp, is_categorical = normalise_for_gp(configs, feature_groups)
    X_sub, y_sub = _subsample(X_gp, performance, MAX_GP_SAMPLES)

    # Standardise targets (GP fits better on zero-mean, unit-variance targets)
    y_std, y_mean, y_scale = _standardise(y_sub)

    gpr = fit_gp(
        X=X_sub.astype(np.float64),
        Y=y_std.astype(np.float64),
        is_categorical=is_categorical,
    )

    import torch
    X_torch = torch.from_numpy(X_sub.astype(np.float64))
    predicted_std, _ = gpr.posterior(X_torch)
    predicted = predicted_std.detach().cpu().numpy() * y_scale + y_mean
    residuals = y_sub - predicted

    return X_sub, predicted, residuals


# ── Breusch-Pagan test ────────────────────────────────────────────────────────

def breusch_pagan_lm(
    X_normalised: np.ndarray,
    residuals: np.ndarray,
) -> tuple[float, float]:
    """
    Breusch-Pagan (1979) Lagrange Multiplier test for heteroscedasticity.

    Auxiliary OLS: e² ~ X_normalised (with intercept)
    Test statistic: LM = n · R²  ~  χ²(p) under H₀

    Parameters
    ----------
    X_normalised : shape (n, p) — GP input features, already normalised to [0,1]
    residuals    : shape (n,)   — GP residuals e = y - GP_mean

    Returns
    -------
    lm_statistic : float — nR²; larger means stronger heteroscedasticity
    p_value      : float — P(χ²(p) ≥ LM) under H₀ of homoscedasticity
    """
    squared_residuals = residuals ** 2
    X_with_intercept = sm.add_constant(X_normalised, has_constant="add")

    aux_ols = sm.OLS(squared_residuals, X_with_intercept).fit()
    n = len(residuals)
    p = X_normalised.shape[1]

    lm = n * aux_ols.rsquared
    p_value = float(chi2.sf(lm, df=p))
    return float(lm), p_value


# ── Per-task score ────────────────────────────────────────────────────────────

def heteroscedasticity_score(
    configs: np.ndarray,
    feature_groups: list,
    performance: np.ndarray,
) -> tuple[float, float]:
    """
    Fit Optuna GP, compute residuals, run Breusch-Pagan test.

    Returns (lm_statistic, p_value).  Returns (0.0, 1.0) if there are too few
    samples for a reliable fit.
    """
    if len(performance) < 30:
        return 0.0, 1.0
    try:
        X_gp, predicted, residuals = fit_gp_and_get_residuals(
            configs, feature_groups, performance
        )
        return breusch_pagan_lm(X_gp, residuals)
    except Exception as exc:
        logging.getLogger(__name__).warning(f"GP fit failed: {exc}")
        return 0.0, 1.0


# ── Statistical validation ────────────────────────────────────────────────────

def _cliffs_delta(group_a: np.ndarray, group_b: np.ndarray) -> float:
    dominance = sum(
        1 if a > b else (-1 if a < b else 0)
        for a in group_a for b in group_b
    )
    return dominance / (len(group_a) * len(group_b))


def compare_selected_vs_rest(
    selected_task_ids: list,
    score_by_task: dict,
) -> dict:
    """Mann-Whitney U (one-sided: selected > rest) + Cliff's delta."""
    selected_scores = np.array([score_by_task[t] for t in selected_task_ids if t in score_by_task])
    rest_scores = np.array([score_by_task[t] for t in score_by_task if t not in selected_task_ids])

    if len(selected_scores) < 2 or len(rest_scores) < 2:
        return {"note": "insufficient tasks for statistical comparison"}

    u_stat, p_value = mannwhitneyu(selected_scores, rest_scores, alternative="greater")
    delta = _cliffs_delta(selected_scores, rest_scores)

    return {
        "n_selected":            len(selected_scores),
        "n_rest":                len(rest_scores),
        "selected_lm_median":    float(np.median(selected_scores)),
        "rest_lm_median":        float(np.median(rest_scores)),
        "mwu_statistic":         float(u_stat),
        "mwu_p_value":           float(p_value),
        "cliffs_delta":          float(delta),
    }


# ── Per-benchmark stratification ──────────────────────────────────────────────

def score_and_select_tasks(
    benchmark: str,
    task_ids: list,
    top_count: int,
) -> tuple[list, dict, dict, dict]:
    score_by_task = {}
    bp_pvalue_by_task = {}
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

        lm, bp_pval = heteroscedasticity_score(configs, feature_groups, performance)
        if lm > 0:
            score_by_task[task_id] = lm
            bp_pvalue_by_task[task_id] = bp_pval
            performance_by_task[task_id] = performance

    selected_task_ids = select_top_datasets(scores=score_by_task, top_count=top_count)
    return selected_task_ids, score_by_task, bp_pvalue_by_task, performance_by_task


# ── Summary table ─────────────────────────────────────────────────────────────

def save_summary_csv(
    selected_task_ids: list,
    score_by_task: dict,
    bp_pvalue_by_task: dict,
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
            "task_id":         task_id,
            "selected":        task_id in selected_task_ids,
            "bp_lm_statistic": round(score_by_task[task_id], 4),
            "bp_p_value":      float(bp_pvalue_by_task.get(task_id, float("nan"))),
            "n":               len(performance),
            "mean_perf":       round(float(np.mean(performance)), 6),
            "mean_perf_ci_lo": round(float(mean_ci.low), 6),
            "mean_perf_ci_hi": round(float(mean_ci.high), 6),
            "median_perf":     round(float(q50), 6),
            "iqr_perf":        round(float(q75 - q25), 6),
        })

    df = pd.DataFrame(rows).sort_values("bp_lm_statistic", ascending=False)
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    df.to_csv(
        os.path.join(SUMMARY_DIR, f"heteroscedasticity_summary_{benchmark}.csv"),
        index=False,
    )
    with open(
        os.path.join(SUMMARY_DIR, f"heteroscedasticity_group_test_{benchmark}.json"), "w"
    ) as fh:
        json.dump(test_result, fh, indent=2, default=str)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(SEED)

    boxplot_data = {}
    kde_data = {}

    for benchmark in BENCHMARKS:
        task_ids = get_benchmark_task_ids(benchmark_name=benchmark)
        selected, score_by_task, bp_pvalue_by_task, performance_by_task = score_and_select_tasks(
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

        test_result = compare_selected_vs_rest(selected, score_by_task)
        save_summary_csv(
            selected, score_by_task, bp_pvalue_by_task,
            performance_by_task, test_result, benchmark,
        )

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

    if boxplot_data:
        os.makedirs(OUT_ROOT, exist_ok=True)
        group_score_boxplot(
            results_by_benchmark=boxplot_data,
            score_label="Breusch-Pagan LM statistic (n·R²)",
            out_path=os.path.join(OUT_ROOT, "selected_vs_rest_scores"),
        )
    if kde_data:
        target_kde_per_task(
            results_by_benchmark=kde_data,
            out_path=os.path.join(OUT_ROOT, "task_performance_distributions"),
        )


if __name__ == "__main__":
    main()
