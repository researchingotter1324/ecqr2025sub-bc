try:
    from ccqr_optimization.selection.acquisition import (
        QuantileConformalSearcher,
    )
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
from hpobench.config.utils import (
    get_external_tuning_configurations,
    build_sampler_variation_configurations,
    build_architecture_variation_configurations,
)
from hpobench.config.config_types import (
    TunerConfig,
    CCQRModel,
)

# 1. Static analysis configurations:
STATIC_ANALYSIS_ESTIMATOR_ARCHITECTURES = [
    "ql",
    "qrf",
    "qgbm",
    "qens5",
]

# 2. Coverage analysis configurations:
COVERAGE_ANALYSIS_CONFIGURATIONS = []
COVERAGE_PLOT_CONFIGURATIONS = []
COVERAGE_INTERVAL_WIDTHS = [0.25, 0.5, 0.75]
ADAPTERS = ["ACI", "DtACI", None]
COVERAGE_ANALYSIS_ARCHITECTURE = "qgbm"

for interval_width in COVERAGE_INTERVAL_WIDTHS:
    for adapter in ADAPTERS:
        split_conformal_sampler = LowerBoundSampler(
            interval_width=interval_width,
            adapter=adapter,
            c=0,
        )
        split_conformal_searcher = QuantileConformalSearcher(
            quantile_estimator_architecture=COVERAGE_ANALYSIS_ARCHITECTURE,
            sampler=split_conformal_sampler,
            n_calibration_folds=5,
            calibration_split_strategy="train_test_split",
        )
        if adapter is None:
            config_identifier = "Split Conformalized"
        elif adapter in ["ACI", "DtACI"]:
            config_identifier = f"Split Conformalized + {adapter}"
        else:
            raise ValueError(f"Unknown adapter: {adapter}")
        split_conformal_config = TunerConfig(
            tuner=CCQRModel(backend="ccqr_optimization", searcher=split_conformal_searcher),
            tuner_identifier=config_identifier,
            searcher_tuning_framework=None,
        )
        COVERAGE_ANALYSIS_CONFIGURATIONS.append(split_conformal_config)

        cv_conformal_sampler = LowerBoundSampler(
            interval_width=interval_width,
            adapter=adapter,
            c=0,
        )
        cv_conformal_searcher = QuantileConformalSearcher(
            quantile_estimator_architecture=COVERAGE_ANALYSIS_ARCHITECTURE,
            sampler=cv_conformal_sampler,
            n_calibration_folds=5,
            calibration_split_strategy="cv",
        )
        if adapter is None:
            config_identifier = "Cross Conformalized"
        elif adapter in ["ACI", "DtACI"]:
            config_identifier = f"Cross Conformalized + {adapter}"
        else:
            raise ValueError(f"Unknown adapter: {adapter}")
        cv_conformal_config = TunerConfig(
            tuner=CCQRModel(backend="ccqr_optimization", searcher=cv_conformal_searcher),
            tuner_identifier=config_identifier,
            searcher_tuning_framework=None,
        )
        COVERAGE_ANALYSIS_CONFIGURATIONS.append(cv_conformal_config)
        COVERAGE_PLOT_CONFIGURATIONS.append(cv_conformal_config)

    # Add the unconformalized configuration for each interval width:
    non_conformal_searcher = QuantileConformalSearcher(
        quantile_estimator_architecture=COVERAGE_ANALYSIS_ARCHITECTURE,
        sampler=LowerBoundSampler(
            interval_width=interval_width,
            adapter=None,
            c=0,
        ),
        n_pre_conformal_trials=10000,
        n_calibration_folds=3,
        calibration_split_strategy="train_test_split",
    )
    non_conformal_config = TunerConfig(
        tuner=CCQRModel(backend="ccqr_optimization", searcher=non_conformal_searcher),
        tuner_identifier="Unconformalized",
        searcher_tuning_framework=None,
    )
    COVERAGE_ANALYSIS_CONFIGURATIONS.append(non_conformal_config)
    COVERAGE_PLOT_CONFIGURATIONS.append(non_conformal_config)


# 3. Create configurations feeding the comparative tuner rank plots:
SAMPLER_VARIATION_N_DEFAULT_QUANTILES = 6
SAMPLER_VARIATION_DEFAULT_ADAPTER = "DtACI"
SAMPLER_VARIATION_CONFIGURATIONS = build_sampler_variation_configurations(
    samplers=[
        ExpectedImprovementSampler(
            n_quantiles=SAMPLER_VARIATION_N_DEFAULT_QUANTILES,
            num_ei_samples=1000,
            adapter=SAMPLER_VARIATION_DEFAULT_ADAPTER,
        ),
        ThompsonSampler(
            n_quantiles=SAMPLER_VARIATION_N_DEFAULT_QUANTILES,
            enable_optimistic_sampling=False,
            adapter=SAMPLER_VARIATION_DEFAULT_ADAPTER,
        ),
        ThompsonSampler(
            n_quantiles=SAMPLER_VARIATION_N_DEFAULT_QUANTILES,
            enable_optimistic_sampling=True,
            adapter=SAMPLER_VARIATION_DEFAULT_ADAPTER,
        ),
    ],
    quantile_arch="qgbm",
    calibration_split_strategy="train_test_split",
)

# 4. Architecture variation configurations:
ARCHITECTURE_VARIATION_ADAPTER = "DtACI"
ARCHITECTURE_VARIATION_N_QUANTILES = 6
ARCHITECTURE_VARIATION_CONFIGURATIONS = build_architecture_variation_configurations(
    architectures=[
        "ql",
        "qrf",
        "qgbm",
        "qens5",
    ],
    samplers=[
        ExpectedImprovementSampler(
            n_quantiles=ARCHITECTURE_VARIATION_N_QUANTILES,
            num_ei_samples=1000,
            adapter=ARCHITECTURE_VARIATION_ADAPTER,
        ),
        ThompsonSampler(
            n_quantiles=ARCHITECTURE_VARIATION_N_QUANTILES,
            enable_optimistic_sampling=False,
            adapter=ARCHITECTURE_VARIATION_ADAPTER,
        ),
        ThompsonSampler(
            n_quantiles=ARCHITECTURE_VARIATION_N_QUANTILES,
            enable_optimistic_sampling=True,
            adapter=ARCHITECTURE_VARIATION_ADAPTER,
        ),
    ],
    calibration_split_strategy="train_test_split",
)

# 5. Limited architecture configurations:
LIMITED_ARCHITECTURE_ADAPTER = "DtACI"
LIMITED_ARCHITECTURE_N_QUANTILES = 6
LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS = (
    build_architecture_variation_configurations(
        architectures=[
            # "qgbm",
            "qens5",
        ],
        samplers=[
             ExpectedImprovementSampler(
                        n_quantiles=LIMITED_ARCHITECTURE_N_QUANTILES,
                        num_ei_samples=1000,
                        adapter=LIMITED_ARCHITECTURE_ADAPTER,
                    ),
        ],
        n_pre_conformal_trials=32,
        searcher_tuning_framework=None,
        calibration_split_strategy="train_test_split",
    )
)

# 6. Pre-conformal comparison configurations:
PRECONFORMAL_ADAPTER = "DtACI"
PRECONFORMAL_N_QUANTILES = 6
PRECONFORMAL_COMPARISON_CONFIGURATIONS = []
for architecture in [
    "qrf",
    "qgbm",
    # "qens5",
]:
    # Simulate normal pre-conformal cutoff vs. unreachable one:
    for pre_conformal_trials in [32, 10000]:
        if pre_conformal_trials == 10000:
            adapter = None
        else:
            adapter = PRECONFORMAL_ADAPTER
        PRECONFORMAL_COMPARISON_CONFIGURATIONS.extend(
            build_architecture_variation_configurations(
                architectures=[architecture],
                samplers=[
                    ExpectedImprovementSampler(
                        n_quantiles=PRECONFORMAL_N_QUANTILES,
                        num_ei_samples=1000,
                        adapter=adapter,
                    ),
                    ThompsonSampler(
                        n_quantiles=PRECONFORMAL_N_QUANTILES,
                        enable_optimistic_sampling=False,
                        adapter=adapter,
                    ),
                    ThompsonSampler(
                        n_quantiles=PRECONFORMAL_N_QUANTILES,
                        enable_optimistic_sampling=True,
                        adapter=adapter,
                    ),
                ],
                n_pre_conformal_trials=pre_conformal_trials,
                calibration_split_strategy="adaptive",
            )
        )


# 7. Quantile count variation configurations:
QUANTILE_COUNT_VARIATION_ADAPTER = "DtACI"
QUANTILE_COUNT_VARIATION_CONFIGURATIONS = []
QUANTILE_COUNT_VALUES = [4, 6, 10]

for n_quantiles in QUANTILE_COUNT_VALUES:
    QUANTILE_COUNT_VARIATION_CONFIGURATIONS.extend(
        build_architecture_variation_configurations(
            architectures=[
                "qgbm",  # NOTE: Use single architecture for this configuration, analysis doesn't support multiple
            ],
            samplers=[
                ExpectedImprovementSampler(
                    n_quantiles=n_quantiles,
                    num_ei_samples=1000,
                    adapter=QUANTILE_COUNT_VARIATION_ADAPTER,
                ),
                ThompsonSampler(
                    n_quantiles=n_quantiles,
                    enable_optimistic_sampling=False,
                    adapter=QUANTILE_COUNT_VARIATION_ADAPTER,
                ),
                ThompsonSampler(
                    n_quantiles=n_quantiles,
                    enable_optimistic_sampling=True,
                    adapter=QUANTILE_COUNT_VARIATION_ADAPTER,
                ),
            ],
            n_pre_conformal_trials=32,
            searcher_tuning_framework=None,
            calibration_split_strategy="train_test_split",
        )
    )


# 8. Search tuning effect configurations:
SEARCH_TUNING_EFFECT_ADAPTER = "DtACI"
SEARCH_TUNING_EFFECT_N_QUANTILES = 6
SEARCH_TUNING_EFFECT_CONFIGURATIONS = []

for searcher_tuning_framework in [None, "fixed"]:
    SEARCH_TUNING_EFFECT_CONFIGURATIONS.extend(
        build_architecture_variation_configurations(
            architectures=[
                "qrf",
                "qgbm",
            ],
            samplers=[
                ThompsonSampler(
                    n_quantiles=SEARCH_TUNING_EFFECT_N_QUANTILES,
                    enable_optimistic_sampling=True,
                    adapter=SEARCH_TUNING_EFFECT_ADAPTER,
                )
            ],
            n_pre_conformal_trials=32,
            searcher_tuning_framework=searcher_tuning_framework,
            calibration_split_strategy="train_test_split",
        )
    )

EXTERNAL_TUNING_CONFIGURATIONS = get_external_tuning_configurations()
