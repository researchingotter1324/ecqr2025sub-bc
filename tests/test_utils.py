import pytest
from hpobench.utils import generate_hyperparameter_combinations
from hpobench.config.config_types import IntRange, FloatRange, CategoricalRange


@pytest.mark.parametrize("n_combinations", [1, 2, 3, 4, 5])
def test_generate_hyperparameter_combinations(n_combinations):
    # Test params with all types
    params = {
        "int_param": IntRange(lower=1, upper=10),
        "float_param": FloatRange(lower=0.0, upper=1.0),
        "cat_param": CategoricalRange(choices=["a", "b", "c"]),
    }

    # Test with fixed random seed
    combinations = generate_hyperparameter_combinations(
        params, n_combinations=n_combinations, random_state=42
    )

    assert len(combinations) == n_combinations

    for combo in combinations:
        # Check keys
        assert set(combo.keys()) == {"int_param", "float_param", "cat_param"}

        # Check value types and ranges
        assert isinstance(combo["int_param"], int)
        assert (
            params["int_param"].lower <= combo["int_param"] <= params["int_param"].upper
        )

        assert isinstance(combo["float_param"], float)
        assert (
            params["float_param"].lower
            <= combo["float_param"]
            <= params["float_param"].upper
        )

        assert combo["cat_param"] in ["a", "b", "c"]


def test_generate_hyperparameter_combinations_same_seed():
    params = {
        "int_param": IntRange(lower=1, upper=10),
        "float_param": FloatRange(lower=0.0, upper=1.0),
        "cat_param": CategoricalRange(choices=["a", "b", "c"]),
    }
    n_combinations = 5
    random_state = 42

    combinations1 = generate_hyperparameter_combinations(
        params, n_combinations=n_combinations, random_state=random_state
    )
    combinations2 = generate_hyperparameter_combinations(
        params, n_combinations=n_combinations, random_state=random_state
    )

    assert combinations1 == combinations2
