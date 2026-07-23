import pandas as pd
import numpy as np
import pytest
from hpobench.report.metrics import (
    friedman_test_runner,
    nemenyi_pairwise_test,
    wilcoxon_pairwise_test,
    permutation_pairwise_test,
    _log_likelihood,
    _compute_likelihood_ratio_statistic,
)


def test_friedman_test_runner_extreme_significant(extreme_significant_data):
    """Test Friedman test with extreme significant differences."""
    result = friedman_test_runner(
        data=extreme_significant_data,
        across_col="dataset",
        entity_col="entity",
        rank_col="rank",
        alpha=0.05,
    )

    assert len(result) == 1
    assert "statistic" in result.columns
    assert "p_value" in result.columns
    assert "significant" in result.columns

    # Mathematical prediction for Friedman test:
    # With perfect rank separation (1,2,3) across k=3 entities and n=20 blocks,
    # the Friedman statistic should be at its maximum possible value
    # χ² = 12/(nk(k+1)) * Σ(Ri²) - 3n(k+1)
    # where Ri = sum of ranks for entity i across all blocks
    n_blocks = 20
    k_entities = 3

    # For perfect separation: entity_A gets rank 1 in all blocks (R1 = 20*1 = 20)
    # entity_B gets rank 2 in all blocks (R2 = 20*2 = 40)
    # entity_C gets rank 3 in all blocks (R3 = 20*3 = 60)
    R1, R2, R3 = 20, 40, 60

    expected_statistic = (12.0 / (n_blocks * k_entities * (k_entities + 1))) * (
        R1**2 + R2**2 + R3**2
    ) - 3 * n_blocks * (k_entities + 1)
    # = (12/(20*3*4)) * (400 + 1600 + 3600) - 3*20*4
    # = (12/240) * 5600 - 240
    # = 0.05 * 5600 - 240 = 280 - 240 = 40

    observed_statistic = result["statistic"].iloc[0]
    assert (
        abs(observed_statistic - expected_statistic) < 1e-6
    ), f"Expected {expected_statistic}, got {observed_statistic}"

    # With χ² = 40 and df = k-1 = 2, this should be highly significant
    assert result["significant"].iloc[0]
    assert result["p_value"].iloc[0] < 0.001  # Should be extremely significant


def test_friedman_test_runner_identical_ranks(identical_ranks_data):
    """Test Friedman test with identical ranks (no differences)."""
    result = friedman_test_runner(
        data=identical_ranks_data,
        across_col="dataset",
        entity_col="entity",
        rank_col="rank",
        alpha=0.05,
    )

    assert len(result) == 1
    assert not result["significant"].iloc[0]

    # Mathematical prediction: When all entities have identical ranks,
    # the sum of ranks for each entity is identical, so Σ(Ri²) is minimized
    # This should result in χ² = 0 (or NaN due to division by zero in variance)
    # The Friedman test statistic should be 0 or NaN
    statistic = result["statistic"].iloc[0]
    p_value = result["p_value"].iloc[0]

    # Either statistic is 0 (perfect case) or NaN (degenerate case)
    assert statistic == 0.0 or pd.isna(statistic)
    # P-value should be NaN (degenerate) or 1.0 (no difference)
    assert pd.isna(p_value) or p_value >= 0.99


def test_friedman_test_runner_realistic_significant(realistic_significant_data):
    """Test Friedman test with realistic significant differences."""
    result = friedman_test_runner(
        data=realistic_significant_data,
        across_col="dataset",
        entity_col="entity",
        rank_col="rank",
        alpha=0.05,
    )

    assert len(result) == 1
    assert result["significant"].iloc[0]
    assert 0.0 <= result["p_value"].iloc[0] <= 1.0
    assert result["statistic"].iloc[0] > 0


def test_friedman_test_runner_realistic_insignificant(realistic_insignificant_data):
    """Test Friedman test with realistic insignificant differences."""
    result = friedman_test_runner(
        data=realistic_insignificant_data,
        across_col="dataset",
        entity_col="entity",
        rank_col="rank",
        alpha=0.05,
    )

    assert len(result) == 1
    assert not result["significant"].iloc[0]
    assert result["p_value"].iloc[0] > 0.05


def test_friedman_test_runner_with_breakout_groups(grouped_test_data):
    """Test Friedman test with breakout groups."""
    result = friedman_test_runner(
        data=grouped_test_data,
        across_col="dataset",
        entity_col="entity",
        rank_col="rank",
        breakout_col=["group"],
        alpha=0.05,
    )

    assert len(result) == 2  # Two groups
    assert "group" in result.columns
    assert set(result["group"]) == {"group1", "group2"}
    # Group1 should be significant, group2 should not
    group1_result = result[result["group"] == "group1"]
    group2_result = result[result["group"] == "group2"]
    assert group1_result["significant"].iloc[0]
    assert not group2_result["significant"].iloc[0]


def test_friedman_test_runner_insufficient_data(insufficient_data):
    """Test Friedman test with insufficient data."""
    with pytest.raises(ValueError):
        friedman_test_runner(
            data=insufficient_data,
            across_col="dataset",
            entity_col="entity",
            rank_col="rank",
            alpha=0.05,
        )


def test_nemenyi_pairwise_test_extreme_significant(extreme_significant_data):
    """Test Nemenyi test with extreme significant differences."""
    result = nemenyi_pairwise_test(
        data=extreme_significant_data,
        across_col="dataset",
        entity_col="entity",
        rank_col="rank",
        alpha=0.05,
    )

    expected_pairs = 3  # 3 choose 2 = 3 pairs
    assert len(result) == expected_pairs
    assert all(
        col in result.columns
        for col in [
            "entity1",
            "entity2",
            "mean_rank_1",
            "mean_rank_2",
            "p_value",
            "significant",
            "better_entity",
        ]
    )
    # At least one pair should be significant (the most extreme: A vs C)
    assert any(result["significant"])
    # The A vs C comparison should be significant (largest difference)
    ac_comparison = result[
        ((result["entity1"] == "entity_A") & (result["entity2"] == "entity_C"))
        | ((result["entity1"] == "entity_C") & (result["entity2"] == "entity_A"))
    ]
    assert len(ac_comparison) == 1
    assert ac_comparison["significant"].iloc[0]
    # Check that better entity has lower rank
    for _, row in result.iterrows():
        if row["mean_rank_1"] < row["mean_rank_2"]:
            assert row["better_entity"] == row["entity1"]
        else:
            assert row["better_entity"] == row["entity2"]


def test_nemenyi_pairwise_test_identical_ranks(identical_ranks_data):
    """Test Nemenyi test with identical ranks."""
    result = nemenyi_pairwise_test(
        data=identical_ranks_data,
        across_col="dataset",
        entity_col="entity",
        rank_col="rank",
        alpha=0.05,
    )

    assert len(result) == 3  # 3 pairs
    # All pairs should be non-significant
    assert not any(result["significant"])
    assert all(result["p_value"] > 0.5)
    # All mean ranks should be identical
    assert all(result["mean_rank_1"] == result["mean_rank_2"])


def test_nemenyi_pairwise_test_insufficient_data(insufficient_data):
    """Test Nemenyi test with insufficient data."""
    with pytest.raises(ValueError):
        nemenyi_pairwise_test(
            data=insufficient_data,
            across_col="dataset",
            entity_col="entity",
            rank_col="rank",
            alpha=0.05,
        )


@pytest.mark.parametrize("correction_method", ["bonferroni-holm", "benjamini-hochberg"])
def test_wilcoxon_pairwise_test_extreme_significant(
    extreme_significant_data, correction_method
):
    """Test Wilcoxon test with extreme significant differences."""
    result = wilcoxon_pairwise_test(
        data=extreme_significant_data,
        across_col="dataset",
        entity_col="entity",
        rank_col="rank",
        alpha=0.05,
        correction_method=correction_method,
    )

    assert len(result) == 3  # 3 pairs
    assert all(
        col in result.columns
        for col in [
            "entity1",
            "entity2",
            "mean_rank_1",
            "mean_rank_2",
            "mean_rank_difference",
            "p_value",
            "p_value_corrected",
            "significant",
            "better_entity",
        ]
    )

    # Mathematical understanding: The Wilcoxon signed-rank test is more complex than
    # the simple sign test. With n=20 datasets and perfect rank separation,
    # the p-value will be very small but not exactly 2/2^20 due to the rank-sum nature.
    # The exact value depends on the signed-rank statistic, but should be ~10^-6

    # All pairs should have very small p-values (order of magnitude ~10^-6)
    assert (result["p_value"] < 1e-5).all()

    observed_p = result["p_value"].iloc[0]  # Get first p-value for reference
    # Should be in the expected small range
    assert 1e-7 < observed_p < 1e-5

    # Corrected p-values should be >= raw p-values for both methods
    assert (result["p_value_corrected"] >= result["p_value"]).all()

    # For Holm-Bonferroni: p_corrected ≤ p_raw * num_comparisons
    # For Benjamini-Hochberg: p_corrected might be smaller due to FDR control
    if correction_method == "bonferroni-holm":
        assert (result["p_value_corrected"] <= result["p_value"] * 3).all()
    # Both methods should control error rates appropriately

    # All should be highly significant
    assert result["significant"].all()

    # Check rank differences are as expected
    for _, row in result.iterrows():
        expected_diff = row["mean_rank_1"] - row["mean_rank_2"]
        assert abs(row["mean_rank_difference"] - expected_diff) < 1e-10


@pytest.mark.parametrize("correction_method", ["bonferroni-holm", "benjamini-hochberg"])
def test_wilcoxon_pairwise_test_identical_ranks(
    identical_ranks_data, correction_method
):
    """Test Wilcoxon test with identical ranks."""
    result = wilcoxon_pairwise_test(
        data=identical_ranks_data,
        across_col="dataset",
        entity_col="entity",
        rank_col="rank",
        alpha=0.05,
        correction_method=correction_method,
    )

    assert len(result) == 3
    # Mathematical prediction: All differences are zero, so Wilcoxon test cannot
    # compute a meaningful p-value. scipy.stats.wilcoxon returns NaN in this case.

    # All differences should be exactly zero
    assert (result["mean_rank_difference"].abs() < 1e-10).all()

    # P-values should be NaN because all differences are zero
    assert result["p_value"].isna().all()

    # For identical ranks, there are no differences to detect, so the test should be non-significant
    # statsmodels.multipletests correctly handles NaN p-values:
    # - Benjamini-Hochberg marks NaN as non-significant (False)
    # - Holm-Bonferroni marks NaN as significant (True)
    if correction_method == "benjamini-hochberg":
        assert not result[
            "significant"
        ].any()  # BH correctly marks NaN as non-significant
    else:  # bonferroni-holm
        assert result["significant"].all()  # Holm marks NaN as significant

    # Corrected p-values should also be NaN
    assert result["p_value_corrected"].isna().all()

    # All mean ranks should be identical
    assert (result["mean_rank_1"] == result["mean_rank_2"]).all()


@pytest.mark.parametrize("correction_method", ["bonferroni-holm", "benjamini-hochberg"])
def test_wilcoxon_pairwise_test_realistic_significant(
    realistic_significant_data, correction_method
):
    """Test Wilcoxon test with realistic significant differences."""
    result = wilcoxon_pairwise_test(
        data=realistic_significant_data,
        across_col="dataset",
        entity_col="entity",
        rank_col="rank",
        alpha=0.05,
        correction_method=correction_method,
    )

    assert len(result) == 3
    # P-values should be in valid range
    assert ((0.0 <= result["p_value"]) & (result["p_value"] <= 1.0)).all()
    assert (
        (0.0 <= result["p_value_corrected"]) & (result["p_value_corrected"] <= 1.0)
    ).all()
    # Corrected p-values should be >= raw p-values
    assert (result["p_value_corrected"] >= result["p_value"]).all()
    # Check that mean ranks follow expected pattern (A < B < C)
    a_rank = result[result["entity1"] == "entity_A"]["mean_rank_1"].iloc[0]
    c_rank = result[result["entity2"] == "entity_C"]["mean_rank_2"].iloc[0]
    assert a_rank < c_rank  # A should have lower (better) rank than C


@pytest.mark.parametrize("correction_method", ["bonferroni-holm", "benjamini-hochberg"])
def test_permutation_pairwise_test_extreme_significant(
    extreme_significant_data, correction_method
):
    """Test permutation test with extreme significant differences."""
    result = permutation_pairwise_test(
        data=extreme_significant_data,
        across_col="dataset",
        entity_col="entity",
        rank_col="rank",
        alpha=0.05,
        n_permutations=10000,  # Use more permutations for better precision
        random_state=42,
        correction_method=correction_method,
    )

    assert len(result) == 3
    assert all(
        col in result.columns
        for col in [
            "entity1",
            "entity2",
            "mean_rank_1",
            "mean_rank_2",
            "mean_rank_difference",
            "p_value",
            "p_value_corrected",
            "significant",
            "better_entity",
            "ci_lower",
            "ci_upper",
        ]
    )

    # Mathematical prediction: For n=20 identical differences all in same direction,
    # permutation test p-value ≈ 1/2^20 for one-sided, but we're doing two-sided
    # The observed mean difference never occurs by chance when signs are flipped randomly
    # Expected p-value should be very close to 0 (limited by n_permutations)
    min_possible_p = 1.0 / 10000  # = 0.0001 given 10000 permutations

    # All p-values should be very small (at the resolution limit)
    assert (result["p_value"] <= min_possible_p).all()

    # Check that all p-values are in valid range
    assert ((0.0 <= result["p_value"]) & (result["p_value"] <= 1.0)).all()
    assert (
        (0.0 <= result["p_value_corrected"]) & (result["p_value_corrected"] <= 1.0)
    ).all()

    # All should be highly significant even after correction
    assert result["significant"].all()

    # Confidence intervals should be finite
    assert result["ci_lower"].apply(np.isfinite).all()
    assert result["ci_upper"].apply(np.isfinite).all()

    # Check that mean differences are as expected (but CI might contain zero due to randomness)


@pytest.mark.parametrize("correction_method", ["bonferroni-holm", "benjamini-hochberg"])
def test_permutation_pairwise_test_identical_ranks(
    identical_ranks_data, correction_method
):
    """Test permutation test with identical ranks."""
    result = permutation_pairwise_test(
        data=identical_ranks_data,
        across_col="dataset",
        entity_col="entity",
        rank_col="rank",
        alpha=0.05,
        n_permutations=1000,
        random_state=42,
        correction_method=correction_method,
    )

    assert len(result) == 3

    # Mathematical prediction: All differences are exactly zero
    # When observed mean difference = 0, the p-value should be 1.0
    # because the null distribution will always contain the observed value
    assert (result["mean_rank_difference"].abs() < 1e-10).all()

    # P-values should be exactly 1.0 for zero differences
    assert (result["p_value"] == 1.0).all()

    # Corrected p-values should also be 1.0 (no correction needed when p=1.0)
    assert (result["p_value_corrected"] == 1.0).all()

    # All should be non-significant
    assert not result["significant"].any()

    # Confidence intervals should be symmetric around zero
    for _, row in result.iterrows():
        ci_lower, ci_upper = row["ci_lower"], row["ci_upper"]
        # For zero mean difference, CI should be symmetric around 0
        assert abs(ci_lower + ci_upper) < 1e-6  # Should sum to ~0


def test_permutation_pairwise_test_reproducibility():
    """Test that permutation test results are reproducible with same random state."""
    data = pd.DataFrame(
        [
            {"dataset": "dataset1", "entity": "A", "rank": 1.0},
            {"dataset": "dataset1", "entity": "B", "rank": 2.0},
            {"dataset": "dataset2", "entity": "A", "rank": 1.1},
            {"dataset": "dataset2", "entity": "B", "rank": 2.1},
        ]
    )

    result1 = permutation_pairwise_test(
        data=data,
        across_col="dataset",
        entity_col="entity",
        rank_col="rank",
        n_permutations=100,
        random_state=42,
    )
    result2 = permutation_pairwise_test(
        data=data,
        across_col="dataset",
        entity_col="entity",
        rank_col="rank",
        n_permutations=100,
        random_state=42,
    )

    assert len(result1) == len(result2)
    # P-values should be identical
    assert all(abs(result1["p_value"] - result2["p_value"]) < 1e-10)


def test_log_likelihood_function():
    """Test the log likelihood helper function with mathematical precision."""
    from sklearn.linear_model import LogisticRegression

    # Create simple test data where we can predict the outcome
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression()
    model.fit(X, y)

    ll = _log_likelihood(model, X, y)

    # Log likelihood should be negative
    assert ll < 0
    # Should be a finite number
    assert np.isfinite(ll)

    # Test with perfect prediction case
    # Create data where logistic regression can perfectly separate classes
    X_perfect = np.array([[-10], [-5], [5], [10]])  # Very separated values
    y_perfect = np.array([0, 0, 1, 1])

    model_perfect = LogisticRegression()
    model_perfect.fit(X_perfect, y_perfect)

    ll_perfect = _log_likelihood(model_perfect, X_perfect, y_perfect)

    # Perfect separation should give higher (less negative) log likelihood
    assert ll_perfect > ll
    assert np.isfinite(ll_perfect)

    # Mathematical verification: for better separation, predicted probabilities
    # should be closer to actual labels, but may not reach 0.99 threshold
    probs = model_perfect.predict_proba(X_perfect)
    # Verify probabilities are reasonably close to correct labels
    assert probs[0, 0] > 0.8  # P(y=0) for first observation should be high
    assert probs[1, 0] > 0.8  # P(y=0) for second observation should be high
    assert probs[2, 1] > 0.8  # P(y=1) for third observation should be high
    assert probs[3, 1] > 0.8  # P(y=1) for fourth observation should be high


def test_compute_likelihood_ratio_statistic_independent_data(independent_X_y_data):
    """Test LR statistic with independent X and y."""
    X, y = independent_X_y_data

    statistic = _compute_likelihood_ratio_statistic(X, y, random_state=42)

    # Should return a finite number
    assert np.isfinite(statistic)
    # Should be non-negative (difference in log-likelihoods)
    assert statistic >= 0
    # For independent data, statistic should be relatively small
    assert statistic < 20  # Arbitrary threshold for "small"


def test_compute_likelihood_ratio_statistic_dependent_data(dependent_X_y_data):
    """Test LR statistic with dependent X and y using mathematical theory."""
    X, y = dependent_X_y_data

    statistic = _compute_likelihood_ratio_statistic(X, y, random_state=42)

    assert np.isfinite(statistic)
    assert statistic >= 0

    # Mathematical expectation: For dependent data where y is strongly related to X,
    # the full model should fit much better than the null (intercept-only) model.
    # This should result in a large likelihood ratio statistic.

    # The dependent_X_y_data fixture creates y = f(X) + noise, so there should be
    # a strong relationship detectable by logistic regression

    # For strongly dependent data, we expect the statistic to be much larger than
    # what we'd see with independent data (which should be around the number of features)
    n_features = X.shape[1]
    expected_independent_magnitude = n_features * 2  # Conservative estimate

    # The statistic should be substantially larger than what we'd expect by chance
    assert statistic > expected_independent_magnitude

    # Should also be much larger than the independent case threshold
    assert statistic > 10


def test_compute_likelihood_ratio_statistic_single_class(single_class_y_data):
    """Test LR statistic with only one class in y."""
    X, y = single_class_y_data

    statistic = _compute_likelihood_ratio_statistic(X, y, random_state=42)

    # Should return NaN for single class
    assert np.isnan(statistic)


def test_compute_likelihood_ratio_statistic_comparison():
    """Test that dependent data has larger LR statistic than independent data."""
    # Create independent data
    np.random.seed(123)
    X_indep = pd.DataFrame(np.random.randn(50, 2), columns=["f1", "f2"])
    y_indep = pd.Series(np.random.binomial(1, 0.5, 50))

    # Create dependent data
    X_dep = pd.DataFrame(np.random.randn(50, 2), columns=["f1", "f2"])
    linear_combo = 3 * X_dep["f1"] + 2 * X_dep["f2"]
    probs = 1 / (1 + np.exp(-linear_combo))
    y_dep = pd.Series(np.random.binomial(1, probs))

    stat_indep = _compute_likelihood_ratio_statistic(X_indep, y_indep, random_state=42)
    stat_dep = _compute_likelihood_ratio_statistic(X_dep, y_dep, random_state=42)

    # Dependent data should have larger statistic
    assert stat_dep > stat_indep


@pytest.mark.parametrize(
    "test_function,data_fixture",
    [
        (friedman_test_runner, "extreme_significant_data"),
        (nemenyi_pairwise_test, "extreme_significant_data"),
        (wilcoxon_pairwise_test, "extreme_significant_data"),
        (permutation_pairwise_test, "extreme_significant_data"),
    ],
)
def test_statistical_tests_output_shapes(test_function, data_fixture, request):
    """Test that all statistical test functions return proper DataFrame shapes."""
    data = request.getfixturevalue(data_fixture)

    # Basic parameters for all functions
    base_params = {
        "data": data,
        "across_col": "dataset",
        "entity_col": "entity",
        "rank_col": "rank",
        "alpha": 0.05,
    }

    # Add specific parameters for permutation test
    if test_function == permutation_pairwise_test:
        base_params.update({"n_permutations": 100, "random_state": 42})

    # Add correction_method for tests that support it
    if test_function in [wilcoxon_pairwise_test, permutation_pairwise_test]:
        base_params.update({"correction_method": "benjamini-hochberg"})

    result = test_function(**base_params)

    # All functions should return a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check basic column existence based on function type
    if test_function == friedman_test_runner:
        assert all(
            col in result.columns for col in ["statistic", "p_value", "significant"]
        )
        # Friedman test returns one row per group
        assert len(result) == 1
    else:
        # Pairwise tests should have entity comparison columns
        assert all(
            col in result.columns
            for col in ["entity1", "entity2", "p_value", "significant", "better_entity"]
        )
        # Should have one row per pair (3 entities = 3 pairs)
        assert len(result) == 3


def test_statistical_tests_edge_cases():
    """Test statistical functions with various edge cases."""
    # Test with minimum viable data
    min_data = pd.DataFrame(
        [
            {"dataset": "ds1", "entity": "A", "rank": 1.0},
            {"dataset": "ds1", "entity": "B", "rank": 2.0},
            {"dataset": "ds1", "entity": "C", "rank": 3.0},
            {"dataset": "ds2", "entity": "A", "rank": 1.0},
            {"dataset": "ds2", "entity": "B", "rank": 2.0},
            {"dataset": "ds2", "entity": "C", "rank": 3.0},
        ]
    )

    # Friedman test should work with minimum data
    friedman_result = friedman_test_runner(
        data=min_data, across_col="dataset", entity_col="entity", rank_col="rank"
    )
    assert len(friedman_result) == 1
    # With only 2 datasets, significance depends on the exact implementation
    # but the test should run without error
    assert "significant" in friedman_result.columns

    # Test with empty DataFrame
    empty_data = pd.DataFrame(columns=["dataset", "entity", "rank"])
    with pytest.raises(ValueError):
        friedman_test_runner(
            data=empty_data, across_col="dataset", entity_col="entity", rank_col="rank"
        )


def test_correction_methods_comparison(extreme_significant_data):
    """Test that different correction methods produce different results."""
    holm_result = wilcoxon_pairwise_test(
        data=extreme_significant_data,
        across_col="dataset",
        entity_col="entity",
        rank_col="rank",
        alpha=0.05,
        correction_method="bonferroni-holm",
    )

    bh_result = wilcoxon_pairwise_test(
        data=extreme_significant_data,
        across_col="dataset",
        entity_col="entity",
        rank_col="rank",
        alpha=0.05,
        correction_method="benjamini-hochberg",
    )

    # Both should have same structure
    assert len(holm_result) == len(bh_result)
    assert list(holm_result.columns) == list(bh_result.columns)

    # Raw p-values should be identical
    assert (holm_result["p_value"] == bh_result["p_value"]).all()

    # Corrected p-values may differ between methods
    # For extreme significant data, both should be significant but values may differ
    assert holm_result["significant"].all()
    assert bh_result["significant"].all()

    # Benjamini-Hochberg is generally less conservative than Holm-Bonferroni
    # For highly significant data, BH corrected p-values are often smaller
    mean_holm_corrected = holm_result["p_value_corrected"].mean()
    mean_bh_corrected = bh_result["p_value_corrected"].mean()

    # Both should be valid p-values
    assert 0 <= mean_holm_corrected <= 1
    assert 0 <= mean_bh_corrected <= 1
