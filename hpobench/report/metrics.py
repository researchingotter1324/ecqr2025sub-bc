import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Literal
from scikit_posthocs import posthoc_nemenyi_friedman
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from hpobench.utils import get_group_dict
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests


logger = logging.getLogger(__name__)


def friedman_test_runner(
    data: pd.DataFrame,
    across_col: str,
    entity_col: str,
    rank_col: str,
    breakout_col: Optional[list[str]] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Performs Friedman tests, optionally grouped by `breakout_col`.

    For each group (or the entire DataFrame if no `breakout_col`), data is
    pivoted: `index=across_col`, `columns=entity_col`, `values=rank_col`.
    The `rank_col` in the input `data` for each group should be unique
    for `across_col` and `entity_col` combinations (e.g., pre-aggregated ranks).

    The pivoted matrix for the Friedman test will have unique `across_col`
    values as rows and unique `entity_col` values as columns. It requires
    at least 2 rows and 3 columns.

    The baseline alpha significance level is used without any multiple
    comparison corrections.

    Generally, this test would check whether there is a significant difference in
    the ranks of the entities, across the across_col values.

    Args:
        data: DataFrame with `across_col`, `entity_col`, `rank_col`,
            and any `breakout_col` columns.
        across_col: Column for blocks/groups (e.g., 'dataset').
        entity_col: Column for entities/treatments (e.g., 'tuner').
        rank_col: Column with ranks (ranks should be calculated to measure
            differences between entities).
        breakout_col: Optional list of columns for grouping data.
            A test is run per group.
        alpha: Significance level.

    Returns:
        DataFrame of test results per group, including
              'statistic', 'p_value', 'significant'.
    """
    results = []
    group_iter = (
        data.groupby(breakout_col) if breakout_col is not None else [(None, data)]
    )
    for within_group, group_df in group_iter:
        pivot_df = group_df.pivot(index=across_col, columns=entity_col, values=rank_col)
        if pivot_df.shape[0] < 2 or pivot_df.shape[1] < 3:
            raise ValueError(
                f"Friedman test requires at least 2 blocks and 3 entities; group {within_group} has {pivot_df.shape[0]} blocks and {pivot_df.shape[1]} entities"
            )
        stat, p = friedmanchisquare(
            *[pivot_df[col].dropna() for col in pivot_df.columns]
        )
        group_dict = get_group_dict(breakout_col, within_group)
        results.append({**group_dict, "statistic": stat, "p_value": p})

    results_df = pd.DataFrame(results)
    if not results_df.empty and "p_value" in results_df.columns:
        results_df["significant"] = results_df["p_value"] < alpha
    else:
        results_df["significant"] = []

    return results_df


def nemenyi_pairwise_test(
    data: pd.DataFrame,
    across_col: str,
    entity_col: str,
    rank_col: str,
    breakout_col: Optional[list[str]] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Performs Nemenyi pairwise post-hoc tests, optionally grouped by `breakout_col`.

    This test is typically used after a significant Friedman test to determine
    which specific pairs of entities differ significantly. It compares
    all possible pairs of entities within each group. The `across_col`
    serves as the blocking factor, consistent with the Friedman test.

    The input data's `rank_col` should contain values where lower indicates better
    performance or rank. These can be raw performance metrics or pre-calculated
    ranks (ranks assigned within each block defined by `across_col`, where a
    lower rank is better). The `posthoc_nemenyi_friedman` function internally
    re-ranks the `rank_col` values within each block (defined by `across_col`).
    If pre-calculated ranks (lower is better) are provided, this re-ranking
    will preserve their relative order.

    The test requires at least 2 entities and 2 blocks per group for meaningful
    comparisons.

    Args:
        data: DataFrame with `across_col`, `entity_col`, `rank_col`,
            and any `breakout_col` columns.
        across_col: Column for blocks/groups (e.g., 'dataset') - same as Friedman test.
        entity_col: Column for entities/treatments (e.g., 'tuner') being compared.
        rank_col: Column with performance values or pre-calculated ranks.
        breakout_col: Optional list of columns for grouping data.
            A separate test is run for each group.
        alpha: Significance level for determining statistical significance.

    Returns:
        DataFrame with pairwise comparison results including:
        - Group identifiers (if breakout_col provided)
        - entity1, entity2: The two entities being compared
        - mean_rank_1, mean_rank_2: Mean ranks for each entity
        - p_value: Statistical significance of the difference
        - significant: Boolean indicator if p_value < alpha
        - better_entity: Entity with lower (better) mean rank
    """
    results = []
    group_iter = (
        data.groupby(breakout_col) if breakout_col is not None else [(None, data)]
    )

    for within_group, group_df_view in group_iter:
        if group_df_view[entity_col].nunique() < 2:
            raise ValueError(
                f"Nemenyi test requires at least 2 entities; group {within_group} has {group_df_view[entity_col].nunique()}"
            )

        if group_df_view[across_col].nunique() < 2:
            raise ValueError(
                f"Nemenyi test requires at least 2 blocks/datasets; group {within_group} has {group_df_view[across_col].nunique()}"
            )

        group_df = group_df_view.copy()
        unique_blocks = group_df[across_col].unique()
        block_to_id = {block: i for i, block in enumerate(unique_blocks)}
        group_df["_block_id"] = group_df[across_col].map(block_to_id)

        p_value_matrix = posthoc_nemenyi_friedman(
            a=group_df,
            y_col=rank_col,
            block_col=across_col,
            group_col=entity_col,
            block_id_col="_block_id",
            melted=True,
            sort=True,
        )
        entities = p_value_matrix.index.tolist()
        group_dict = get_group_dict(breakout_col, within_group)

        mean_ranks = group_df.groupby(entity_col)[rank_col].mean()
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i < j:
                    p_value = p_value_matrix.loc[e1, e2]
                    rank1 = mean_ranks.get(e1, np.nan)
                    rank2 = mean_ranks.get(e2, np.nan)
                    results.append(
                        {
                            **group_dict,
                            "entity1": e1,
                            "entity2": e2,
                            "mean_rank_1": rank1,
                            "mean_rank_2": rank2,
                            "p_value": p_value,
                            "significant": p_value < alpha,
                            "better_entity": e1 if rank1 < rank2 else e2,
                        }
                    )

    return pd.DataFrame(results)


def wilcoxon_pairwise_test(
    data: pd.DataFrame,
    across_col: str,
    entity_col: str,
    rank_col: str,
    breakout_col: Optional[List[str]] = None,
    alpha: float = 0.05,
    correction_method: Literal[
        "bonferroni-holm", "benjamini-hochberg"
    ] = "benjamini-hochberg",
) -> pd.DataFrame:
    """Performs Wilcoxon signed-rank pairwise tests with multiple testing correction.

    For each algorithm pair, computes mean rank per dataset, then applies
    Wilcoxon signed-rank test across datasets. Uses either Holm-Bonferroni (FWER)
    or Benjamini-Hochberg (FDR) correction to control for multiple comparisons.

    Args:
        data: DataFrame with `across_col`, `entity_col`, `rank_col`,
            and any `breakout_col` columns.
        across_col: Column for datasets (e.g., 'dataset').
        entity_col: Column for algorithms/entities (e.g., 'tuner').
        rank_col: Column with ranks (lower is better).
        breakout_col: Optional list of columns for grouping data.
        alpha: Significance level for determining statistical significance.
        correction_method: Multiple testing correction method to use.

    Returns:
        DataFrame with pairwise comparison results including:
        - Group identifiers (if breakout_col provided)
        - entity1, entity2: The two entities being compared
        - mean_rank_1, mean_rank_2: Mean ranks for each entity
        - mean_rank_difference: Difference in mean ranks (entity1 - entity2)
        - p_value: Raw p-value from Wilcoxon test
        - p_value_corrected: Multiple testing corrected p-value
        - significant: Boolean indicator if corrected p_value < alpha
        - better_entity: Entity with lower (better) mean rank
    """
    results = []
    group_iter = (
        data.groupby(breakout_col) if breakout_col is not None else [(None, data)]
    )

    for within_group, group_df in group_iter:
        if group_df[entity_col].nunique() < 2:
            raise ValueError(
                f"Wilcoxon test requires at least 2 entities; group {within_group} has {group_df[entity_col].nunique()}"
            )

        if group_df[across_col].nunique() < 2:
            raise ValueError(
                f"Wilcoxon test requires at least 2 datasets; group {within_group} has {group_df[across_col].nunique()}"
            )

        # Collapse repetitions into mean rank per algorithm per dataset
        mean_ranks_df = (
            group_df.groupby([across_col, entity_col])[rank_col].mean().reset_index()
        )

        # Pivot to get algorithms as columns
        pivot_df = mean_ranks_df.pivot(
            index=across_col, columns=entity_col, values=rank_col
        )

        # Get all entities
        entities = pivot_df.columns.tolist()
        group_dict = get_group_dict(breakout_col, within_group)

        # Store pairwise results for this group
        pairwise_results = []

        # Compute all pairwise comparisons
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i < j:
                    # Get paired observations (one per dataset)
                    e1_ranks = pivot_df[e1].dropna()
                    e2_ranks = pivot_df[e2].dropna()

                    # Find common datasets
                    common_datasets = e1_ranks.index.intersection(e2_ranks.index)
                    if len(common_datasets) < 2:
                        raise ValueError(
                            f"Wilcoxon test requires at least 2 common datasets between {e1} and {e2} in {within_group}; found {len(common_datasets)}"
                        )

                    e1_common = e1_ranks.loc[common_datasets]
                    e2_common = e2_ranks.loc[common_datasets]

                    # Compute Wilcoxon signed-rank test
                    try:
                        # Test if distributions are different
                        stat, p_value = wilcoxon(
                            e1_common, e2_common, alternative="two-sided"
                        )
                    except ValueError as e:
                        # Handle case where differences are all zero
                        logger.warning(f"Wilcoxon test failed for {e1} vs {e2}: {e}")
                        p_value = 1.0

                    mean_rank_1 = e1_common.mean()
                    mean_rank_2 = e2_common.mean()
                    mean_rank_diff = mean_rank_1 - mean_rank_2

                    pairwise_results.append(
                        {
                            **group_dict,
                            "entity1": e1,
                            "entity2": e2,
                            "mean_rank_1": mean_rank_1,
                            "mean_rank_2": mean_rank_2,
                            "mean_rank_difference": mean_rank_diff,
                            "p_value": p_value,
                            "better_entity": e1 if mean_rank_1 < mean_rank_2 else e2,
                        }
                    )

        # Apply multiple testing correction within this group
        if pairwise_results:
            p_values = [result["p_value"] for result in pairwise_results]
            method_map = {"bonferroni-holm": "holm", "benjamini-hochberg": "fdr_bh"}
            reject, p_corrected, _, _ = multipletests(
                p_values, alpha=alpha, method=method_map[correction_method]
            )

            for i, result in enumerate(pairwise_results):
                result["p_value_corrected"] = p_corrected[i]
                result["significant"] = reject[i]

            results.extend(pairwise_results)

    return pd.DataFrame(results)


def permutation_pairwise_test(
    data: pd.DataFrame,
    across_col: str,
    entity_col: str,
    rank_col: str,
    breakout_col: Optional[List[str]] = None,
    alpha: float = 0.05,
    n_permutations: int = 10000,
    random_state: Optional[int] = None,
    correction_method: Literal[
        "bonferroni-holm", "benjamini-hochberg"
    ] = "benjamini-hochberg",
) -> pd.DataFrame:
    """Performs permutation tests for pairwise comparisons with multiple testing correction.

    For each dataset, computes mean rank difference between algorithms.
    Uses permutation test by randomly flipping signs of differences to build
    null distribution. Applies either Holm-Bonferroni (FWER) or Benjamini-Hochberg
    (FDR) correction across all pairs.

    Args:
        data: DataFrame with `across_col`, `entity_col`, `rank_col`,
            and any `breakout_col` columns.
        across_col: Column for datasets (e.g., 'dataset').
        entity_col: Column for algorithms/entities (e.g., 'tuner').
        rank_col: Column with ranks (lower is better).
        breakout_col: Optional list of columns for grouping data.
        alpha: Significance level for determining statistical significance.
        n_permutations: Number of permutations for the test.
        random_state: Random seed for reproducible results.
        correction_method: Multiple testing correction method to use.

    Returns:
        DataFrame with pairwise comparison results including:
        - Group identifiers (if breakout_col provided)
        - entity1, entity2: The two entities being compared
        - mean_rank_1, mean_rank_2: Mean ranks for each entity
        - mean_rank_difference: Difference in mean ranks (entity1 - entity2)
        - p_value: Two-sided p-value from permutation test
        - p_value_corrected: Multiple testing corrected p-value
        - significant: Boolean indicator if corrected p_value < alpha
        - better_entity: Entity with lower (better) mean rank
        - ci_lower, ci_upper: Confidence interval from permutation distribution
    """
    if random_state is not None:
        np.random.seed(random_state)

    results = []
    group_iter = (
        data.groupby(breakout_col) if breakout_col is not None else [(None, data)]
    )

    for within_group, group_df in group_iter:
        if group_df[entity_col].nunique() < 2:
            raise ValueError(
                f"Permutation test requires at least 2 entities; group {within_group} has {group_df[entity_col].nunique()}"
            )

        if group_df[across_col].nunique() < 2:
            raise ValueError(
                f"Permutation test requires at least 2 datasets; group {within_group} has {group_df[across_col].nunique()}"
            )

        # Collapse repetitions into mean rank per algorithm per dataset
        mean_ranks_df = (
            group_df.groupby([across_col, entity_col])[rank_col].mean().reset_index()
        )

        # Pivot to get algorithms as columns
        pivot_df = mean_ranks_df.pivot(
            index=across_col, columns=entity_col, values=rank_col
        )

        # Get all entities
        entities = pivot_df.columns.tolist()
        group_dict = get_group_dict(breakout_col, within_group)

        # Store pairwise results for this group
        pairwise_results = []

        # Compute all pairwise comparisons
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i < j:
                    # Get paired observations (one per dataset)
                    e1_ranks = pivot_df[e1].dropna()
                    e2_ranks = pivot_df[e2].dropna()

                    # Find common datasets
                    common_datasets = e1_ranks.index.intersection(e2_ranks.index)
                    if len(common_datasets) < 2:
                        raise ValueError(
                            f"Permutation test requires at least 2 common datasets between {e1} and {e2} in {within_group}; found {len(common_datasets)}"
                        )

                    e1_common = e1_ranks.loc[common_datasets]
                    e2_common = e2_ranks.loc[common_datasets]

                    # Compute observed differences per dataset
                    differences = e1_common.values - e2_common.values
                    observed_mean_diff = np.mean(differences)

                    # Permutation test: randomly flip signs
                    null_distribution = []
                    for _ in range(n_permutations):
                        # Randomly flip signs of differences
                        signs = np.random.choice([-1, 1], size=len(differences))
                        permuted_diff = np.mean(differences * signs)
                        null_distribution.append(permuted_diff)

                    null_distribution = np.array(null_distribution)

                    # Compute two-sided p-value
                    p_value = np.mean(
                        np.abs(null_distribution) >= np.abs(observed_mean_diff)
                    )

                    # Compute confidence interval (95% by default)
                    ci_alpha = 1 - 0.95  # 95% CI
                    ci_lower = np.percentile(null_distribution, 100 * ci_alpha / 2)
                    ci_upper = np.percentile(
                        null_distribution, 100 * (1 - ci_alpha / 2)
                    )

                    mean_rank_1 = e1_common.mean()
                    mean_rank_2 = e2_common.mean()

                    pairwise_results.append(
                        {
                            **group_dict,
                            "entity1": e1,
                            "entity2": e2,
                            "mean_rank_1": mean_rank_1,
                            "mean_rank_2": mean_rank_2,
                            "mean_rank_difference": observed_mean_diff,
                            "p_value": p_value,
                            "ci_lower": ci_lower,
                            "ci_upper": ci_upper,
                            "better_entity": e1 if mean_rank_1 < mean_rank_2 else e2,
                        }
                    )

        # Apply multiple testing correction within this group
        if pairwise_results:
            p_values = [result["p_value"] for result in pairwise_results]
            method_map = {"bonferroni-holm": "holm", "benjamini-hochberg": "fdr_bh"}
            reject, p_corrected, _, _ = multipletests(
                p_values, alpha=alpha, method=method_map[correction_method]
            )

            for i, result in enumerate(pairwise_results):
                result["p_value_corrected"] = p_corrected[i]
                result["significant"] = reject[i]

            results.extend(pairwise_results)

    return pd.DataFrame(results)


def _log_likelihood(model, X_input, y):
    """Compute log-likelihood for logistic regression model predictions.

    Args:
        model: Fitted logistic regression model.
        X_input: Feature matrix for predictions.
        y: True binary labels.

    Returns:
        Log-likelihood value for the model predictions.
    """
    eps = 1e-15
    probs = model.predict_proba(X_input)
    return np.sum(y * np.log(probs[:, 1] + eps) + (1 - y) * np.log(probs[:, 0] + eps))


def _compute_likelihood_ratio_statistic(
    X: pd.DataFrame, y: pd.Series, random_state: Optional[int] = None
) -> float:
    """Compute likelihood ratio test statistic for logistic regression models.

    Fits two logistic regression models:
    1. Null model (intercept only)
    2. Full model (with all features)

    Computes the likelihood ratio test statistic: 2 * (log_likelihood_full - log_likelihood_null)

    Args:
        X: Feature matrix (configuration features).
        y: Binary outcome vector (breach indicators).
        random_state: Random seed for reproducible results.

    Returns:
        Likelihood ratio test statistic, or nan if data contains only one class.
    """
    # Check if y contains only one class
    if len(y.unique()) < 2:
        return np.nan

    # Validate input dimensions
    if len(X) != len(y):
        logger.warning(
            f"Feature matrix and target length mismatch: {len(X)} vs {len(y)}"
        )
        return np.nan

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit null model (intercept only)
    null_model = LogisticRegression(
        fit_intercept=True, random_state=random_state, max_iter=1000
    )
    intercept_only = np.ones((len(X_scaled), 1))
    null_model.fit(intercept_only, y)
    ll_null = _log_likelihood(null_model, intercept_only, y)

    # Fit full model
    full_model = LogisticRegression(
        fit_intercept=True, random_state=random_state, max_iter=1000
    )
    full_model.fit(X_scaled, y)
    ll_full = _log_likelihood(full_model, X_scaled, y)

    llr_stat = 2 * (ll_full - ll_null)

    # Validate result
    if not np.isfinite(llr_stat) or llr_stat < 0:
        logger.warning(f"Invalid LLR statistic computed: {llr_stat}")
        return np.nan

    return llr_stat


def _calculate_chunked_target_coverage_deviation(
    group: pd.DataFrame, breach_column: str
) -> pd.Series:
    """Compute absolute deviation between observed breach rate and target confidence level per chunk.

    Splits the group into fixed-size chunks, calculates breach rate and confidence level
    for each chunk, and stores the deviation at the start index of each chunk. Used for
    analyzing calibration of constraint coverage over budget progression.

    Args:
        group: DataFrame containing experiment records for a single group.
        breach_column: Column name indicating constraint breaches.

    Returns:
        Series with chunked target coverage deviation values (NaN for non-chunk start indices).
    """
    n_obs = len(group)
    chunk_size = 10
    n_chunks = n_obs // chunk_size

    # Fix: Use positional index instead of group.index to avoid misalignment when reset_index is applied
    chunked_deviations = pd.Series([np.nan] * n_obs, index=range(n_obs))

    if n_chunks > 3:
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = start_idx + chunk_size
            if start_idx >= n_obs:
                break
            chunk_data = group.iloc[start_idx:end_idx]
            chunk_breach_rate = chunk_data[breach_column].mean()
            miscoverage_level = 1 - float(chunk_data["confidence_level"].iloc[0])
            if (
                pd.isna(miscoverage_level)
                or miscoverage_level is None
                or miscoverage_level == "None"
                or miscoverage_level == ""
            ):
                break
            if pd.notna(chunk_breach_rate):
                target_coverage_deviation = abs(chunk_breach_rate - miscoverage_level)
            else:
                target_coverage_deviation = np.nan
            chunked_deviations.iloc[start_idx] = target_coverage_deviation

    return chunked_deviations


def calculate_calibration_statistics_per_repetition(
    raw_benchmark_data: pd.DataFrame,
    aggregators: List[str],
    breach_column: str,
    entity_column: str,
    metric_columns: List[str],
    budget_unit: str,
    random_state: Optional[int] = None,
    rank_metrics: bool = True,
) -> pd.DataFrame:
    """Calculate calibration statistics for conformal prediction methods.

    Computes various calibration metrics including chunked target coverage deviation
    and likelihood ratio statistics for evaluating conformal prediction performance.

    Args:
        raw_benchmark_data: Raw benchmark results with breach indicators and configurations.
        aggregators: List of aggregation column names for grouping.
        breach_column: Column name containing binary breach indicators.
        entity_column: Column name for entities being compared (e.g., tuners).
        metric_columns: List of metric column names to compute.
        budget_unit: Column name for budget/iteration unit.
        random_state: Random seed for reproducible likelihood ratio computations.
        rank_metrics: Whether to rank metrics within groups.

    Returns:
        DataFrame with averaged calibration statistics per repetition group.
    """
    sorted_experiment_log = raw_benchmark_data.sort_values(
        by=aggregators + [budget_unit],
        ascending=True,
    ).reset_index(drop=True)
    # Fix: Use manual concatenation instead of groupby.apply to avoid pandas version compatibility issues
    # The original groupby.apply creates a wide DataFrame instead of properly stacking Series
    chunked_deviation_results = []
    for name, group in sorted_experiment_log.groupby(aggregators):
        series_result = _calculate_chunked_target_coverage_deviation(
            group, breach_column
        )
        # Reset index to align with the original DataFrame
        series_result.index = group.index
        chunked_deviation_results.append(series_result)

    chunked_deviations = pd.concat(chunked_deviation_results).sort_index()
    sorted_experiment_log["chunked_target_coverage_deviation"] = chunked_deviations

    score_columns = [col for col in metric_columns if col != "llr_statistic"]

    avg_scores_per_repetition = (
        sorted_experiment_log.groupby(aggregators)
        .agg({col: "mean" for col in score_columns})
        .reset_index()
    )

    if "llr_statistic" in metric_columns:
        # Fix: Don't pre-compute feature matrix, extract features per group to avoid index mismatch
        def compute_group_llr(grp):
            # Extract features directly from group to avoid index mismatch issues
            group_features = np.vstack(grp["tabularized_configuration"].values)
            return _compute_likelihood_ratio_statistic(
                pd.DataFrame(group_features), grp[breach_column], random_state
            )

        llr_series = sorted_experiment_log.groupby(aggregators).apply(compute_group_llr)
        llr_df = llr_series.reset_index(name="llr_statistic")
        avg_scores_per_repetition = avg_scores_per_repetition.merge(
            llr_df, on=aggregators, how="left"
        )

        score_columns.append("llr_statistic")

    # NOTE: Rank is computed after raw scores are averaged across iterations (differs from search results)
    if rank_metrics:
        for metric_column in score_columns:
            rank_ascending = True
            rank_groupers = [col for col in aggregators if col != entity_column]
            avg_scores_per_repetition[
                metric_column
            ] = avg_scores_per_repetition.groupby(rank_groupers)[metric_column].rank(
                method="average",
                ascending=rank_ascending,
            )

    return avg_scores_per_repetition
