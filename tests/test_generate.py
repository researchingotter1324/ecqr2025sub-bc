# import numpy as np
from hpobench.generation.generate import (
    BlackBoxGenerator,
    Jahs201Generator,
    YahpoGenerator,
    NAS301Generator,
)
from hpobench.config.benchmark_data import JAHS201_SEARCH_SPACE, NAS301_SEARCH_SPACE
from hpobench.utils import generate_hyperparameter_combinations
from yahpo_gym import BenchmarkSet


def test_blackbox_generator_predict_reproducibility():
    generator = BlackBoxGenerator(generator="rastrigin")
    configuration = {"param1": 1.0, "param2": 2.0, "param3": 3.0}

    result1 = generator.predict(configuration)
    result2 = generator.predict(configuration)

    assert result1 == result2

    result1 = generator.predict_runtime(configuration)
    result2 = generator.predict_runtime(configuration)

    assert result1 == result2


def test_jahs201_generator_predict__reproducibility():
    generator = Jahs201Generator(dataset="cifar10", lazy=True)
    configuration = generate_hyperparameter_combinations(
        params=JAHS201_SEARCH_SPACE, n_combinations=1, random_state=1234
    )[0]

    result1 = generator.predict(configuration)
    result2 = generator.predict(configuration)

    assert result1 == result2

    result1 = generator.predict_runtime(configuration)
    result2 = generator.predict_runtime(configuration)

    assert result1 == result2


def test_nas301_generator_predict_reproducibility():
    """Test NAS-301 generator prediction reproducibility."""
    benchmark_set = BenchmarkSet("nb301", active_session=False, check=False)
    full_config_space = benchmark_set.get_opt_space(drop_fidelity_params=True)
    
    generator = NAS301Generator(
        instance_value="CIFAR10",
        instance_name=None,
        fidelity_space={},
        config_space=full_config_space,
    )
    
    configuration = generate_hyperparameter_combinations(
        params=NAS301_SEARCH_SPACE, n_combinations=1, random_state=1234
    )[0]
    
    result1 = generator.predict(configuration)
    result2 = generator.predict(configuration)
    assert result1 == result2
    
    result1 = generator.predict_runtime(configuration)
    result2 = generator.predict_runtime(configuration)
    assert result1 == result2

