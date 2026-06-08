from pydantic import BaseModel
from typing import Optional, List, Dict


class ExperimentParameters(BaseModel):
    """Default parameters for hyperparameter optimization experiments.

    Args:
        n_trials: Default number of optimization trials per experiment.
        n_coverage_trials: Number of trials for coverage analysis experiments.
        timeout: Maximum experiment duration in seconds.
        n_warm_starts: Default number of random initialization trials.
        n_coverage_warm_starts: Random trials for coverage analysis.
        default_max_n_instances: Maximum parallel instances for experiments.
        static_data_sizes: Data sizes for static analysis experiments.
        static_tuning_iterations: Tuning iterations for static analysis.
        small_n_repetitions_per_tuner_config: Repetitions for small experiments.
        medium_n_repetitions_per_tuner_config: Repetitions for medium experiments.
        large_n_repetitions_per_tuner_config: Repetitions for large experiments.
    """

    n_trials: Optional[int] = 70
    n_coverage_trials: int = 70

    timeout: Optional[int] = None
    n_warm_starts: int = 15
    n_coverage_warm_starts: int = 15
    default_max_n_instances: int = 10

    static_data_sizes: List[int] = [25, 50, 100, 200, 400]
    static_tuning_iterations: List[int] = [0]

    medium_n_repetitions_per_tuner_config: int = 20
    large_n_repetitions_per_tuner_config: int = medium_n_repetitions_per_tuner_config 


class Aliases(BaseModel):
    """Human-readable aliases for various benchmark components.

    Args: 
        sampler_aliases: Short names for conformal prediction samplers.
        architecture_aliases: Short names for quantile estimator architectures.
        benchmark_aliases: Display names for benchmark suites.
    """

    sampler_aliases: Dict[str, str] = {
        "ThompsonSampler": "TS",
        "ExpectedImprovementSampler": "EI",
        "LowerBoundSampler": "LBS",
        "PessimisticLowerBoundSampler": "PLBS",
    }
    architecture_aliases: Dict[str, str] = {
        "qknn": "QKNN",
        "ql": "QL",
        "qrf": "QRF",
        "qgbm": "QGBM",
        "qens1": "QE1",
        "qens3": "QE3",
        "qens5": "QE5",
        "qleaf": "QLEAF",
    }
    benchmark_aliases: Dict[str, str] = {"jahs201": "JAHS-201", "nas301": "NAS-301"}
