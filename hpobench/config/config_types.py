from pydantic import BaseModel, ConfigDict, model_validator
from typing import Union, Literal, Optional

try:
    from ccqr_optimization.selection.acquisition import (
        QuantileConformalSearcher,
    )
except ImportError:
    raise ImportError(
        "ccqr_optimization is a core dependency of this repository, but it is not automatically installed via pyproject.toml, please refer to the README.md for instructions on how to install this separately"
    )
from hpobench.generation.generate import ObjectiveMetricGenerator


class FloatRange(BaseModel):
    """Configuration for floating-point parameter ranges in hyperparameter search spaces.

    Args:
        lower: Minimum value for the parameter range.
        upper: Maximum value for the parameter range.
        log: Whether to use log scale for sampling. Defaults to False.
    """

    model_config = ConfigDict()

    lower: float
    upper: float
    log: bool = False


class IntRange(BaseModel):
    """Configuration for integer parameter ranges in hyperparameter search spaces.

    Args:
        lower: Minimum value for the parameter range.
        upper: Maximum value for the parameter range.
        log: Whether to use log scale for sampling. Defaults to False.
    """

    model_config = ConfigDict()

    lower: int
    upper: int
    log: bool = False


class CategoricalRange(BaseModel):
    """Configuration for categorical parameter choices in hyperparameter search spaces.

    Args:
        choices: List of possible categorical values (strings, integers, or booleans).
    """

    model_config = ConfigDict()

    choices: list[Union[str, int, bool]]


class TunerModelConfig(BaseModel):
    """Base configuration for hyperparameter optimization tuner models.

    Args:
        backend: Name of the optimization backend framework.
        searcher: Name of the search algorithm within the backend.
    """

    backend: str
    searcher: str


class OptunaModel(TunerModelConfig):
    """Configuration for Optuna-based hyperparameter optimization.

    Args:
        backend: Must be "optuna".
        searcher: Optuna sampler algorithm (TPE, random, CMA-ES, etc.).
    """

    backend: Literal["optuna"]
    searcher: Literal[
        "TPE",
        "random",
        "CMA-ES",
        "GP",
        "NL-GP",
    ]


class SMACModel(TunerModelConfig):
    """Configuration for SMAC-based hyperparameter optimization.

    Args:
        backend: Must be "smac".
        searcher: SMAC acquisition function (EI or TS).
    """

    backend: Literal["smac"]
    searcher: Literal[
        "SMAC-EI",
        "NL-SMAC-EI",
    ]


class CCQRModel(BaseModel):
    """Configuration for conformal prediction-based hyperparameter optimization.

    Args:
        backend: Must be "ccqr_optimization".
        searcher: Quantile conformal searcher instance.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: Literal["ccqr_optimization"]
    searcher: QuantileConformalSearcher


class TunerConfig(BaseModel):
    """Complete configuration for a hyperparameter optimization tuner.

    Args:
        tuner: The tuner model configuration (backend-specific).
        tuner_identifier: Human-readable identifier for the tuner configuration.
        searcher_tuning_framework: Framework for tuning the search algorithm itself.
        n_candidates: Number of candidate configurations to sample during acquisition
            function maximization. Overrides the global default when set.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tuner: Union[
        OptunaModel,
        SMACModel,
        QuantileConformalSearcher,
        CCQRModel,
    ]
    tuner_identifier: str
    searcher_tuning_framework: Optional[Literal["reward_cost", "fixed"]] = None
    n_candidates: Optional[int] = None


class ExperimentConfig(BaseModel):
    """Complete configuration for a hyperparameter optimization experiment.

    Args:
        search_space: Dictionary mapping parameter names to their ranges.
        objective_function: Generator for objective function values.
        tuner_configurations: List of tuner configurations to compare.
        n_warm_starts: Number of random trials before optimization begins.
        benchmark_identifier: Name of the benchmark suite.
        dataset_identifier: Specific dataset within the benchmark.
        metric: Optimization metric name (if applicable).
        n_trials: Maximum number of optimization trials.
        timeout: Maximum experiment duration in seconds.
    """

    search_space: dict[str, Union[IntRange, FloatRange, CategoricalRange]]
    objective_function: ObjectiveMetricGenerator
    tuner_configurations: list[TunerConfig]
    n_warm_starts: int
    benchmark_identifier: str
    dataset_identifier: str
    metric: Optional[str] = None
    n_trials: Optional[int] = None
    timeout: Optional[float] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def check_timeout_or_n_trials(cls, values):
        """Validate that either n_trials or timeout is specified.

        Args:
            values: Dictionary of field values to validate.

        Returns:
            Validated values dictionary.

        Raises:
            ValueError: If neither n_trials nor timeout is specified.
        """
        if isinstance(values, dict):
            if values.get("n_trials") is None and values.get("timeout") is None:
                raise ValueError(
                    "At least one of 'n_trials' or 'timeout' must be specified."
                )
        return values
