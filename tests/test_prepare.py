from hpobench.prepare import (
    setup_yahpo_instance_configs,
    setup_jahs201_configs,
    setup_blackbox_configs,
    setup_nas301_configs,
)
from hpobench.config.config_types import (
    ExperimentConfig,
)
from hpobench.config.benchmark_data import (
    JAHS201_SEARCH_SPACE,
    BLACK_BOX_SEARCH_SPACE,
    NAS301_SEARCH_SPACE,
)
from hpobench.config.config_types import CCQRModel
from hpobench.config.tuner_configurations import cv_conformal_searcher, TunerConfig
from hpobench.config.utils import create_searcher_config_id

DEV_TUNING_CONFIGURATIONS = [
    TunerConfig(
        tuner=CCQRModel(backend="ccqr_optimization", searcher=cv_conformal_searcher),
        tuner_identifier=create_searcher_config_id(cv_conformal_searcher),
        searcher_tuning_framework=None,
    )
]


def test_setup_yahpo_instance_configs():

    n_instances = 5

    configs = setup_yahpo_instance_configs(
        benchmark="lcbench",
        tuning_configurations=DEV_TUNING_CONFIGURATIONS,
        n_warm_starts=5,
        n_trials=10,
        timeout=3600,
        max_n_instances=n_instances,
    )

    assert len(configs) == n_instances
    assert all(isinstance(c, ExperimentConfig) for c in configs)
    assert all(c.benchmark_identifier == "lcbench" for c in configs)


def test_setup_jahs201_configs():
    datasets = ["cifar10"]

    configs = setup_jahs201_configs(
        datasets=datasets,
        tuning_configurations=DEV_TUNING_CONFIGURATIONS,
        n_warm_starts=5,
        n_trials=10,
        timeout=3600,
    )

    assert len(configs) == len(datasets)
    assert all(isinstance(c, ExperimentConfig) for c in configs)
    assert all(c.search_space == JAHS201_SEARCH_SPACE for c in configs)
    assert all(c.benchmark_identifier == "JAHS-201" for c in configs)
    assert [c.dataset_identifier for c in configs] == datasets


def test_setup_blackbox_configs():
    functions = ["hartmann"]

    configs = setup_blackbox_configs(
        functions=functions,
        tuning_configurations=DEV_TUNING_CONFIGURATIONS,
        n_warm_starts=5,
        n_trials=10,
        timeout=3600,
    )

    assert len(configs) == len(functions)
    assert all(isinstance(c, ExperimentConfig) for c in configs)
    assert all(c.search_space == BLACK_BOX_SEARCH_SPACE for c in configs)
    assert all(c.benchmark_identifier == "blackbox" for c in configs)
    assert [c.dataset_identifier for c in configs] == functions


def test_setup_nas301_configs():
    """Test NAS-301 configuration setup."""
    datasets = ["CIFAR10"]

    configs = setup_nas301_configs(
        datasets=datasets,
        tuning_configurations=DEV_TUNING_CONFIGURATIONS,
        n_warm_starts=5,
        n_trials=10,
        timeout=3600,
    )

    assert len(configs) == len(datasets)
    assert all(isinstance(c, ExperimentConfig) for c in configs)
    assert all(c.search_space == NAS301_SEARCH_SPACE for c in configs)
    assert all(c.benchmark_identifier == "nas301" for c in configs)
    assert [c.dataset_identifier for c in configs] == datasets

    # Test that the generator is properly configured with maximum fidelity
    generator = configs[0].objective_function
    assert generator.dataset == "nb301"
    assert hasattr(generator, "default_fidelities")
    assert generator.default_fidelities["epoch"] == 97  # Maximum fidelity for NAS-301


def test_yahpo_instance_configs_use_maximum_fidelity():
    """Test that YAHPO instance configs use maximum fidelity values."""
    test_cases = [
        ("lcbench", {"epoch": 50}),
        ("rbv2_aknn", {"repl": 10, "trainsize": 1.0}),
    ]

    for benchmark, expected_max_fidelity in test_cases:
        configs = setup_yahpo_instance_configs(
            benchmark=benchmark,
            tuning_configurations=DEV_TUNING_CONFIGURATIONS,
            n_warm_starts=5,
            n_trials=10,
            timeout=3600,
            max_n_instances=1,  # Just test one instance
        )

        assert len(configs) >= 1
        generator = configs[0].objective_function

        # Verify all fidelity parameters use maximum values
        for param_name, expected_max in expected_max_fidelity.items():
            assert (
                param_name in generator.fidelity_space
            ), f"Missing fidelity parameter {param_name}"
            actual_value = generator.fidelity_space[param_name]
            assert (
                actual_value == expected_max
            ), f"{benchmark} {param_name}: expected {expected_max}, got {actual_value}"
