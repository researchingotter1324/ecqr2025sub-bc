from typing import Union, Optional, List, Any
from hpobench.config.config_types import CCQRModel

try:
    from ccqr_optimization.selection.acquisition import (
        QuantileConformalSearcher,
    )
    from copy import deepcopy
    from ccqr_optimization.selection.sampling.bound_samplers import (
        LowerBoundSampler,
    )
    from ccqr_optimization.selection.sampling.expected_improvement_samplers import (
        ExpectedImprovementSampler,
    )
    from ccqr_optimization.selection.sampling.thompson_samplers import ThompsonSampler
except ImportError:
    raise ImportError(
        "ccqr_optimization is a core dependency of this repository, but it is not automatically installed via pyproject.toml, please refer to the README.md for instructions on how to install this separately"
    )
from hpobench.config.config_types import TunerConfig
from hpobench.config.config_types import (
    OptunaModel,
    SMACModel,
)


def create_searcher_config_id(
    searcher: Union[QuantileConformalSearcher, str],
    custom_prefix: Optional[str] = None,
    searcher_tuning_framework: Optional[str] = None,
) -> str:
    """Create a configuration ID string based on searcher properties.

    Args:
        searcher: A QuantileConformalSearcher instance or a string identifier for the searcher.
        custom_prefix: Optional prefix to add to the config ID.
        searcher_tuning_framework: Optional indicator for tuning framework (e.g., 'fixed', 'reward_cost').

    Returns:
        A formatted string for the configuration ID.
    """
    config_id = f"{custom_prefix}-" if custom_prefix else ""

    if isinstance(searcher, str):
        return searcher.upper()

    if not hasattr(searcher, "quantile_estimator_architecture"):
        raise ValueError(
            "Input must be a QuantileConformalSearcher instance or a string"
        )

    sampler = searcher.sampler
    quantile_arch = searcher.quantile_estimator_architecture
    n_pre_conformal_trials = (
        searcher.n_pre_conformal_trials
        if hasattr(searcher, "n_pre_conformal_trials")
        else None
    )

    sampler_class_name = sampler.__class__.__name__
    sampler_acronym = "".join(c for c in sampler_class_name if c.isupper())

    adapter_name = None
    if hasattr(sampler, "adapters") and sampler.adapters:
        adapter_name = sampler.adapters[0].__class__.__name__
    elif hasattr(sampler, "adapter"):
        if isinstance(sampler.adapter, str):
            adapter_name = sampler.adapter
        elif sampler.adapter is not None:
            adapter_name = sampler.adapter.__class__.__name__

    if quantile_arch:
        quantile_arch_upper = quantile_arch.upper()
        if adapter_name:
            config_id += f"{quantile_arch_upper}-{adapter_name} {sampler_acronym}"
        else:
            config_id += f"{quantile_arch_upper} {sampler_acronym}"
    else:
        config_id += f"{sampler_acronym}"

    if hasattr(sampler, "c"):
        config_id += f" c={sampler.c}"

    if hasattr(sampler, "beta_decay") and sampler.beta_decay:
        decay_parts = sampler.beta_decay.split("_")
        decay_acronym = "".join(part[0] for part in decay_parts)
        config_id += f" {decay_acronym}"

    if (
        hasattr(sampler, "enable_optimistic_sampling")
        and sampler.enable_optimistic_sampling
    ):
        config_id += " OPT"

    if hasattr(sampler, "n_quantiles"):
        config_id += f" nq={sampler.n_quantiles}"

    if hasattr(sampler, "num_ei_samples"):
        config_id += f" ns={sampler.num_ei_samples}"

    if n_pre_conformal_trials and n_pre_conformal_trials != 20:
        config_id += f" pre={n_pre_conformal_trials}"

    if searcher_tuning_framework == "fixed":
        config_id += " TUNED-F"
    elif searcher_tuning_framework == "reward_cost":
        config_id += " TUNED-A"
    elif searcher_tuning_framework:
        config_id += " TUNED"

    return config_id


def build_static_tuning_configurations(
    quantile_architectures: List[Any],
    searcher_tuning_frameworks: List[Optional[str]],
    n_pre_conformal_trials: int = 20,
) -> List[TunerConfig]:
    """Build static tuning configurations for given quantile architectures and tuning frameworks.

    Args:
        quantile_architectures: List of quantile estimator architectures.
        searcher_tuning_frameworks: List of tuning framework identifiers.
        n_pre_conformal_trials: Number of pre-conformal trials.

    Returns:
        List of static tuning configuration objects.
    """
    # NOTE: Sampler is irrelevant for static tuning analysis,
    # since we only consider the first trial:

    placeholder_sampler = LowerBoundSampler(
        interval_width=0.9,
        adapter="DtACI",
        c=1,
        beta_decay="logarithmic_decay",
    )
    sampler_copy = deepcopy(placeholder_sampler)
    return [
        TunerConfig(
            tuner=CCQRModel(
                backend="ccqr_optimization",
                searcher=QuantileConformalSearcher(
                    quantile_estimator_architecture=arch,
                    sampler=sampler_copy,
                    n_pre_conformal_trials=n_pre_conformal_trials,
                ),
            ),
            tuner_identifier=create_searcher_config_id(
                QuantileConformalSearcher(
                    quantile_estimator_architecture=arch,
                    sampler=sampler_copy,
                    n_pre_conformal_trials=n_pre_conformal_trials,
                ),
                searcher_tuning_framework=framework,
            ),
            searcher_tuning_framework=framework,
        )
        for arch in quantile_architectures
        for framework in searcher_tuning_frameworks
    ]


def build_sampler_variation_configurations(
    samplers: List[
        Union[
            ThompsonSampler,
            LowerBoundSampler,
            ExpectedImprovementSampler,
        ]
    ],
    quantile_arch: Any,
    n_pre_conformal_trials: int = 20,
    searcher_tuning_framework: Optional[str] = None,
    calibration_split_strategy: str = "train_test_split",
) -> List[TunerConfig]:
    """Build tuning configurations for different samplers with a fixed quantile architecture.

    Args:
        samplers: List of sampler instances.
        quantile_arch: Quantile estimator architecture.
        n_pre_conformal_trials: Number of pre-conformal trials.
        searcher_tuning_framework: Value to set in TunerConfig for searcher_tuning_framework.
        calibration_split_strategy: Value to set in QuantileConformalSearcher for calibration_split_strategy.
    Returns:
        List of tuning configuration objects for each sampler.
    """
    configs = []
    for sampler in samplers:
        sampler_copy = deepcopy(sampler)
        searcher = QuantileConformalSearcher(
            quantile_estimator_architecture=quantile_arch,
            sampler=sampler_copy,
            n_pre_conformal_trials=n_pre_conformal_trials,
            n_calibration_folds=5,
            calibration_split_strategy=calibration_split_strategy,
        )
        config_id = create_searcher_config_id(searcher) + (
            f" stf={searcher_tuning_framework}" if searcher_tuning_framework else ""
        )
        configs.append(
            TunerConfig(
                tuner=CCQRModel(backend="ccqr_optimization", searcher=searcher),
                tuner_identifier=config_id,
                searcher_tuning_framework=searcher_tuning_framework,
            )
        )
    return configs


def build_architecture_variation_configurations(
    architectures: List[Any],
    samplers: List[
        Union[
            ThompsonSampler,
            LowerBoundSampler,
            ExpectedImprovementSampler,
        ]
    ],
    n_pre_conformal_trials: int = 20,
    searcher_tuning_framework: Optional[str] = None,
    calibration_split_strategy: str = "train_test_split",
) -> List[TunerConfig]:
    """Build tuning configurations for different quantile architectures and samplers.

    Args:
        architectures: List of quantile estimator architectures.
        samplers: List of sampler instances.
        n_pre_conformal_trials: Number of pre-conformal trials.
        searcher_tuning_framework: Value to set in TunerConfig for searcher_tuning_framework.
        calibration_split_strategy: Value to set in QuantileConformalSearcher for calibration_split_strategy.

    Returns:
        List of tuning configuration objects for each architecture and sampler combination.
    """
    configs = []
    for arch in architectures:
        for sampler in samplers:
            sampler_copy = deepcopy(sampler)
            searcher = QuantileConformalSearcher(
                quantile_estimator_architecture=arch,
                sampler=sampler_copy,
                n_pre_conformal_trials=n_pre_conformal_trials,
                n_calibration_folds=5,
                calibration_split_strategy=calibration_split_strategy,
            )
            config_id = create_searcher_config_id(searcher) + (
                f" stf={searcher_tuning_framework}" if searcher_tuning_framework else ""
            )
            configs.append(
                TunerConfig(
                    tuner=CCQRModel(backend="ccqr_optimization", searcher=searcher),
                    tuner_identifier=config_id,
                    searcher_tuning_framework=searcher_tuning_framework,
                )
            )
    return configs


def get_external_tuning_configurations() -> List[TunerConfig]:
    """Get external (non-ccqr_optimization) tuning configurations for baseline comparison.

    Returns:
        List of external tuning configuration objects (e.g., for optuna, smac).
    """
    return [
        TunerConfig(
            tuner=OptunaModel(backend="optuna", searcher="GP"),
            tuner_identifier="GP-EI",
        ),
        # TunerConfig(
        #     tuner=OptunaModel(backend="optuna", searcher="TPE"),
        #     tuner_identifier="TPE",
        # ),
        # TunerConfig(
        #     tuner=OptunaModel(backend="optuna", searcher="random"),
        #     tuner_identifier="RS",
        # ),
        # TunerConfig(
        #     tuner=SMACModel(backend="smac", searcher="SMAC-EI"),
        #     tuner_identifier="SMAC",
        # ),
    ]
