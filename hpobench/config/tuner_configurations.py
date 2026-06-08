try:
    from ccqr_optimization.selection.acquisition import (
        QuantileConformalSearcher,
    )
    from ccqr_optimization.selection.sampling.bound_samplers import (
        LowerBoundSampler,
        PessimisticLowerBoundSampler,
    )
    from ccqr_optimization.selection.sampling.expected_improvement_samplers import (
        ExpectedImprovementSampler,
    )
    from ccqr_optimization.selection.sampling.thompson_samplers import ThompsonSampler
    from ccqr_optimization.selection.sampling.local_search.smac_search import SmacLocalSearch
    from ccqr_optimization.selection.sampling.local_search.mies_search import MiesLocalSearch
except ImportError:
    raise ImportError(
        "ccqr_optimization is a core dependency of this repository, but it is not automatically installed via pyproject.toml, please refer to the README.md for instructions on how to install this separately"
    )
from hpobench.config.utils import (
    get_external_tuning_configurations,
    build_architecture_variation_configurations,
)
from hpobench.config.config_types import (
    TunerConfig,
    CCQRModel,
)
from hpobench.config.constants import DEFAULT_N_CANDIDATES

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
            local_search=None,
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
            local_search=None,
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
            local_search=None,
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

# 4. Architecture variation configurations:
ARCHITECTURE_VARIATION_ADAPTER = "DtACI"
ARCHITECTURE_VARIATION_N_QUANTILES = 6
ARCHITECTURE_VARIATION_CONFIGURATIONS = build_architecture_variation_configurations(
    architectures=[
            "qgbm",
            "qleaf",
            "qknn",
            "ql",
            "qens3",
            # "qens5",
            "qens1",
    ],
    samplers=[
        
        ThompsonSampler(
            n_quantiles=ARCHITECTURE_VARIATION_N_QUANTILES,
            enable_optimistic_sampling=False,
            adapter=ARCHITECTURE_VARIATION_ADAPTER,
        )
    ],
    calibration_split_strategy="train_test_split",
)

# 9. Lower-bound sampler ablations:
LOWERBOUND_ABLATION_ADAPTER = "DtACI"
LOWERBOUND_ABLATION_CONFIGURATIONS = build_architecture_variation_configurations(
    architectures=[
        "ql",
        # "qknn",
        "qleaf",
        "qgbm",
        "qens3",
    ],
    samplers=[
        # --- logarithmic_decay: vary c and interval_width ---
        LowerBoundSampler(interval_width=0.6,  adapter=LOWERBOUND_ABLATION_ADAPTER, beta_decay="logarithmic_decay",         c=0.2, local_search=SmacLocalSearch()),
        LowerBoundSampler(interval_width=0.8,  adapter=LOWERBOUND_ABLATION_ADAPTER, beta_decay="logarithmic_decay",         c=0.2, local_search=SmacLocalSearch()),
        LowerBoundSampler(interval_width=0.8,  adapter=LOWERBOUND_ABLATION_ADAPTER, beta_decay="logarithmic_decay",         c=0.8, local_search=SmacLocalSearch()),
        LowerBoundSampler(interval_width=0.8,  adapter=LOWERBOUND_ABLATION_ADAPTER, beta_decay="logarithmic_decay",         c=2.0, local_search=SmacLocalSearch()),
        LowerBoundSampler(interval_width=0.95, adapter=LOWERBOUND_ABLATION_ADAPTER, beta_decay="logarithmic_decay",         c=2.0, local_search=SmacLocalSearch()),
        # --- inverse_square_root_decay: vary c and interval_width ---
        LowerBoundSampler(interval_width=0.6,  adapter=LOWERBOUND_ABLATION_ADAPTER, beta_decay="inverse_square_root_decay", c=0.2, local_search=SmacLocalSearch()),
        LowerBoundSampler(interval_width=0.8,  adapter=LOWERBOUND_ABLATION_ADAPTER, beta_decay="inverse_square_root_decay", c=0.8, local_search=SmacLocalSearch()),
        LowerBoundSampler(interval_width=0.95, adapter=LOWERBOUND_ABLATION_ADAPTER, beta_decay="inverse_square_root_decay", c=2.0, local_search=SmacLocalSearch()),
        # --- no decay (beta fixed at 1): vary interval_width ---
        LowerBoundSampler(interval_width=0.6,  adapter=LOWERBOUND_ABLATION_ADAPTER, beta_decay=None,                       c=1.0, local_search=SmacLocalSearch()),
        LowerBoundSampler(interval_width=0.95, adapter=LOWERBOUND_ABLATION_ADAPTER, beta_decay=None,                       c=1.0, local_search=SmacLocalSearch()),
        # --- pessimistic baseline (no LCB, pure lower-bound scoring) ---
        PessimisticLowerBoundSampler(interval_width=0.8, adapter=LOWERBOUND_ABLATION_ADAPTER, local_search=SmacLocalSearch()),
    ],
    calibration_split_strategy="train_test_split",
)

# 5. Limited architecture configurations:
LIMITED_ARCHITECTURE_ADAPTER = "DtACI"
LIMITED_ARCHITECTURE_N_QUANTILES = 6
LIMITED_ARCHITECTURE_VARIATION_CONFIGURATIONS = (
    build_architecture_variation_configurations(
        architectures=[
            "qgbm",
            # "qleaf",
            # "qknn",
            "ql",
            "qens3",
            # "qens5",
            # "qens1",
            # "qrf",
        ],
        samplers=[
        ThompsonSampler(
            n_quantiles=LIMITED_ARCHITECTURE_N_QUANTILES,
            enable_optimistic_sampling=False,
            adapter=ARCHITECTURE_VARIATION_ADAPTER,
        ),
        # PessimisticLowerBoundSampler(interval_width=0.8, adapter=LIMITED_ARCHITECTURE_ADAPTER, local_search=SmacLocalSearch()),
    #         ExpectedImprovementSampler(
    #             n_quantiles=LIMITED_ARCHITECTURE_N_QUANTILES,
    #             adapter=LIMITED_ARCHITECTURE_ADAPTER,
    #             ei_mode="soft_quantile_point",
    #             local_search=SmacLocalSearch(
    #     # n_acq_starts=30,
    #     # n_historical_starts=18,
    #     # n_steps_plateau_walk=30,
    #     # num_continuous_neighbors=24
    # ),
    #         ),
    #                     ExpectedImprovementSampler(
    #             n_quantiles=LIMITED_ARCHITECTURE_N_QUANTILES,
    #             adapter=LIMITED_ARCHITECTURE_ADAPTER,
    #             ei_mode="quantile_point",
    #             local_search=SmacLocalSearch(
    #     # n_acq_starts=30,
    #     # n_historical_starts=18,
    #     # n_steps_plateau_walk=30,
    #     # num_continuous_neighbors=24
    # ),
    #         ),
        ],
        n_pre_conformal_trials=32,
        searcher_tuning_framework=None,
        calibration_split_strategy="train_test_split",
    )
)



# 6. Pre-conformal comparison configurations:
PRECONFORMAL_ADAPTER = "DtACI"
PRECONFORMAL_N_QUANTILES = 10
PRECONFORMAL_COMPARISON_CONFIGURATIONS = []
for architecture in [
    # "qknn",
    "ql",
    "qens3",
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
            adapter=PRECONFORMAL_ADAPTER,
            local_search=SmacLocalSearch(),
        ),
        ThompsonSampler(
            n_quantiles=PRECONFORMAL_N_QUANTILES,
            enable_optimistic_sampling=False,
            adapter=PRECONFORMAL_ADAPTER,
        )
                ],
                n_pre_conformal_trials=pre_conformal_trials,
                calibration_split_strategy="adaptive",
            )
        )


# 7. Quantile count variation configurations:
QUANTILE_COUNT_VARIATION_ADAPTER = "DtACI"
QUANTILE_COUNT_VARIATION_CONFIGURATIONS = []
QUANTILE_COUNT_VALUES = [4, 10, 20]

for n_quantiles in QUANTILE_COUNT_VALUES:
    QUANTILE_COUNT_VARIATION_CONFIGURATIONS.extend(
        build_architecture_variation_configurations(
            architectures=[
                "qgbm",
                "qleaf",
                "qknn",
                "ql",
                "qens3",
                "qens1",
            ],
            samplers=[
                ThompsonSampler(
                    n_quantiles=n_quantiles,
                    enable_optimistic_sampling=False,
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
SEARCH_TUNING_EFFECT_N_QUANTILES = 10
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
                    enable_optimistic_sampling=False,
                    adapter=SEARCH_TUNING_EFFECT_ADAPTER,
                )
            ],
            n_pre_conformal_trials=32,
            searcher_tuning_framework=searcher_tuning_framework,
            calibration_split_strategy="train_test_split",
        )
    )

EXTERNAL_TUNING_CONFIGURATIONS = get_external_tuning_configurations()

STATIC_ARCHITECTURES = ["qgbm", "qleaf", "qknn", "ql", "qens3", "qens1"]

# 10. Number of candidates variation configurations:
NUM_CANDIDATES_VARIATION_ADAPTER = "DtACI"
NUM_CANDIDATES_VARIATION_N_QUANTILES = 6
NUM_CANDIDATES_VARIATION_CONFIGURATIONS = []
NUM_CANDIDATES_VALUES = [500, 3000, 10000]

for n_candidates in NUM_CANDIDATES_VALUES:
    NUM_CANDIDATES_VARIATION_CONFIGURATIONS.extend(
        build_architecture_variation_configurations(
            architectures=[
                "qgbm",
                "qleaf",
                "qknn",
                "ql",
                "qens3",
                "qens1",
            ],
            samplers=[
                ThompsonSampler(
                    n_quantiles=NUM_CANDIDATES_VARIATION_N_QUANTILES,
                    enable_optimistic_sampling=False,
                    adapter=NUM_CANDIDATES_VARIATION_ADAPTER,
                ),
            ],
            n_pre_conformal_trials=32,
            searcher_tuning_framework=None,
            calibration_split_strategy="train_test_split",
            n_candidates=n_candidates,
        )
    )
