import pandas as pd
import optuna
from datetime import datetime, timedelta
from hpobench.config.config_types import TunerConfig
from hpobench.config.config_types import IntRange, FloatRange, CategoricalRange
from typing import Union, Optional, Any, Dict
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler, GPSampler
from hpobench.config.config_types import (
    CCQRModel,
    OptunaModel,
    SMACModel,
)

try:
    from ccqr_optimization.tuning import ConformalTuner
    from hpobench.generation.generate import ObjectiveMetricGenerator
    from ccqr_optimization.selection.sampling.bound_samplers import (
        LowerBoundSampler,
        PessimisticLowerBoundSampler,
    )
    from ccqr_optimization import wrapping as ranges
except ImportError:
    raise ImportError(
        "ccqr_optimization is a core dependency of this repository, but it is not automatically installed via pyproject.toml, please refer to the README.md for instructions on how to install this separately"
    )
from copy import deepcopy
from functools import partial
from ConfigSpace import (
    ConfigurationSpace,
    Configuration,
)
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
)

try:
    from smac.facade.hyperparameter_optimization_facade import (
        HyperparameterOptimizationFacade,
    )
    from smac.scenario import Scenario
    from smac.runhistory.dataclasses import TrialInfo, TrialValue
except ImportError:
    raise ImportError(
        "smac is a core dependency of this repository, but it is not automatically installed via pyproject.toml, please refer to the README.md for instructions on how to install this separately"
    )

N_CANDIDATES = 1000


def create_runtime_tracker() -> list[datetime]:
    """Creates an empty list to track runtime timestamps during optimization.

    Returns:
        Empty list that will store datetime objects for runtime tracking.
    """
    return []


def record_runtime(runtimes: list[datetime]) -> None:
    """Records the current timestamp in the runtime tracking list.

    Args:
        runtimes: List of datetime objects to append the current timestamp to.
    """
    runtimes.append(datetime.now())


def apply_retroactive_timestamps(
    warm_start_configs: Optional[list[tuple[dict, float]]],
    runtimes: list[datetime],
) -> list[datetime]:
    """Applies retroactive timestamps to warm-start configurations.

    Creates timestamps for warm-start configurations by working backwards from the
    earliest objective function runtime, ensuring warm-starts appear to have
    occurred before optimization began.

    Args:
        warm_start_configs: List of (config, loss) tuples for warm-starting.
        runtimes: List of datetime objects from actual objective function calls.

    Returns:
        Combined list of timestamps for warm-start configs followed by objective function runtimes.
    """
    if not warm_start_configs or not runtimes:
        return runtimes

    # Get the smallest timestamp from objective function calls
    min_runtime = min(runtimes).replace(microsecond=0)

    # Assign backwards timestamps to warm-start configs (reverse order)
    warm_start_runtimes = []
    for i in range(len(warm_start_configs)):
        warm_start_runtimes.append(min_runtime - timedelta(seconds=i + 1))

    # Combine warm-start runtimes with objective function runtimes
    return warm_start_runtimes + runtimes


def calculate_breach_status(
    lower_bound: float,
    upper_bound: float,
    realization: float,
) -> int:
    """Calculates whether the true performance breaches the prediction interval.

    Args:
        lower_bound: Lower bound of the conformal prediction interval.
        upper_bound: Upper bound of the conformal prediction interval.
        realization: Actual observed performance value.

    Returns:
        1 if the realization falls outside the prediction interval (breach), 0 otherwise.
    """
    return 1 if (realization < lower_bound or realization > upper_bound) else 0


def calculate_winkler_components(
    lower_bound: float,
    upper_bound: float,
    realization: float,
    alpha: float,
) -> tuple[float, float, float]:
    """Calculates components of the Winkler score for conformal prediction evaluation.

    The Winkler score combines prediction interval width with penalties for miscoverage,
    providing a balanced evaluation metric for uncertainty quantification quality.

    Args:
        lower_bound: Lower bound of the conformal prediction interval.
        upper_bound: Upper bound of the conformal prediction interval.
        realization: Actual observed performance value.
        alpha: Miscoverage rate (1 - confidence level), typically 0.1 for 90% confidence.

    Returns:
        Tuple of (winkler_score, width, miscoverage_penalty) where winkler_score
        is the sum of width and miscoverage_penalty.
    """
    if upper_bound < lower_bound:
        width = 0.0
    else:
        width = upper_bound - lower_bound

    # Calculate miscoverage penalty
    lower_penalty = (
        (2 / alpha) * (lower_bound - realization) if realization <= lower_bound else 0.0
    )
    upper_penalty = (
        (2 / alpha) * (realization - upper_bound) if realization >= upper_bound else 0.0
    )
    miscoverage_penalty = lower_penalty + upper_penalty

    winkler_score = width + miscoverage_penalty

    return winkler_score, width, miscoverage_penalty


def build_history_entry(
    end_time: Optional[Any] = None,
    performance: Optional[Any] = None,
    configurations: Optional[Any] = None,
    iteration: Optional[int] = None,
    estimator_error: Optional[Any] = None,
    searcher_training_time: Optional[Any] = None,
    breach_status: Optional[int] = None,
    winkler_score: Optional[float] = None,
    width: Optional[float] = None,
    miscoverage_penalty: Optional[float] = None,
    tabularized_configuration: Optional[Any] = None,
    acquisition_source: Optional[str] = None,
) -> dict[str, Any]:
    """Creates a standardized dictionary entry for tuning history records.

    Args:
        end_time: Timestamp when the trial completed.
        performance: Observed performance metric value.
        configurations: Dictionary of hyperparameter configuration.
        iteration: Trial iteration number (1-based indexing).
        estimator_error: Prediction error from surrogate model, if applicable.
        searcher_training_time: Time spent training the searcher model.
        breach_status: Binary indicator (0/1) of prediction interval breach.
        winkler_score: Winkler score evaluating prediction interval quality.
        width: Width of the conformal prediction interval.
        miscoverage_penalty: Penalty for prediction interval not containing true value.
        tabularized_configuration: Processed configuration data for analysis.
        acquisition_source: Identifier for the acquisition function used.

    Returns:
        Dictionary containing all trial information with standardized keys.
    """
    return {
        "end_time": end_time,
        "performance": performance,
        "configurations": configurations,
        "iteration": iteration,
        "estimator_error": estimator_error,
        "searcher_training_time": searcher_training_time,
        "breach_status": breach_status,
        "winkler_score": winkler_score,
        "width": width,
        "miscoverage_penalty": miscoverage_penalty,
        "tabularized_configuration": tabularized_configuration,
        "acquisition_source": acquisition_source,
    }


def set_optuna_params(
    trial: optuna.trial.Trial,
    raw_params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
) -> dict[str, Any]:
    """Suggests hyperparameter values for an Optuna trial based on parameter range definitions.

    Args:
        trial: Active Optuna trial object to suggest parameters for.
        raw_params: Dictionary mapping parameter names to their range specifications
            (IntRange, FloatRange, or CategoricalRange).

    Returns:
        Dictionary mapping parameter names to their suggested values for this trial.
    """
    optuna_params: dict[str, Any] = {}
    for name, param in raw_params.items():
        if isinstance(param, IntRange):
            log_flag = getattr(param, "log", False)
            optuna_params[name] = trial.suggest_int(
                name, param.lower, param.upper, log=log_flag
            )
        elif isinstance(param, FloatRange):
            log_flag = getattr(param, "log", False)
            optuna_params[name] = trial.suggest_float(
                name, param.lower, param.upper, log=log_flag
            )
        elif isinstance(param, CategoricalRange):
            optuna_params[name] = trial.suggest_categorical(name, param.choices)
        else:
            raise ValueError(f"Unknown parameter type: {type(param)}")
    return optuna_params


def optuna_artificial_objective(
    trial: optuna.trial.Trial,
    params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    performance_generator: ObjectiveMetricGenerator,
    runtimes: list[datetime],
) -> float:
    """Evaluates a hyperparameter configuration using synthetic performance generation for Optuna.

    Args:
        trial: Optuna trial object containing the hyperparameter suggestions.
        params: Dictionary mapping parameter names to their range specifications.
        performance_generator: Synthetic objective function that generates performance predictions.
        runtimes: List to record timestamps for runtime tracking.

    Returns:
        Predicted performance value for the suggested hyperparameter configuration.
    """
    result = performance_generator.predict(
        configuration=set_optuna_params(trial, params)
    )
    record_runtime(runtimes)

    return result


def build_optuna_distributions(
    raw_params: dict[str, Union[IntRange, FloatRange, CategoricalRange]]
) -> dict[str, optuna.distributions.BaseDistribution]:
    """Creates Optuna distribution objects for parameter spaces to support warm-start functionality.

    Args:
        raw_params: Dictionary mapping parameter names to their range specifications
            (IntRange, FloatRange, or CategoricalRange).

    Returns:
        Dictionary mapping parameter names to corresponding Optuna distribution objects
        for use in warm-start trial creation.
    """
    dists: dict[str, optuna.distributions.BaseDistribution] = {}
    for name, param in raw_params.items():
        if isinstance(param, IntRange):
            log_flag = getattr(param, "log", False)
            if log_flag:
                dists[name] = optuna.distributions.IntLogUniformDistribution(
                    low=param.lower, high=param.upper
                )
            else:
                dists[name] = optuna.distributions.IntUniformDistribution(
                    low=param.lower, high=param.upper
                )
        elif isinstance(param, FloatRange):
            log_flag = getattr(param, "log", False)
            if log_flag:
                dists[name] = optuna.distributions.LogUniformDistribution(
                    low=param.lower, high=param.upper
                )
            else:
                dists[name] = optuna.distributions.UniformDistribution(
                    low=param.lower, high=param.upper
                )
        elif isinstance(param, CategoricalRange):
            dists[name] = optuna.distributions.CategoricalDistribution(
                choices=param.choices
            )
        else:
            raise ValueError(f"Unknown parameter type: {type(param)}")
    return dists


def optuna_tune(
    raw_params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    performance_generator: ObjectiveMetricGenerator,
    tuner_model: OptunaModel,
    warm_start_configs: Optional[list[tuple[dict, float]]] = None,
    random_state: Optional[int] = None,
    n_trials: Optional[int] = None,
    timeout: Optional[float] = None,
) -> pd.DataFrame:
    """Runs hyperparameter optimization using Optuna with a synthetic objective function.

    Args:
        raw_params: Dictionary mapping parameter names to their range specifications
            (IntRange, FloatRange, or CategoricalRange).
        performance_generator: Synthetic objective function for generating performance predictions.
        tuner_model: OptunaModel configuration specifying the search algorithm and parameters.
        warm_start_configs: Optional list of (configuration, loss) tuples for initialization.
        random_state: Optional random seed for reproducible results.
        n_trials: Optional maximum number of optimization trials.
        timeout: Optional time budget in seconds for the optimization process.

    Returns:
        DataFrame containing the complete tuning history with trial results and metadata.
    """
    searcher = tuner_model.searcher

    if searcher == "TPE":
        initialized_sampler = TPESampler(
            seed=random_state, n_startup_trials=0, n_ei_candidates=N_CANDIDATES
        )
    elif searcher == "random":
        initialized_sampler = RandomSampler(seed=random_state)
    elif searcher == "CMA-ES":
        initialized_sampler = CmaEsSampler(seed=random_state, n_startup_trials=0)
    elif searcher == "GP":
        initialized_sampler = GPSampler(seed=random_state, n_startup_trials=0, deterministic_objective=True)
    else:
        raise ValueError(f"Unknown optuna sampler: {searcher}")

    study = optuna.create_study(direction="minimize", sampler=initialized_sampler)
    distributions = build_optuna_distributions(raw_params)

    # Create runtime tracker
    runtimes = create_runtime_tracker()

    if warm_start_configs:
        for config, loss in warm_start_configs:
            trial = optuna.trial.create_trial(
                params=config,
                distributions=distributions,
                value=loss,
                state=optuna.trial.TrialState.COMPLETE,
            )
            study.add_trial(trial)

    if n_trials is not None:
        if warm_start_configs is not None:
            adj_n_trials = n_trials - len(warm_start_configs)
        else:
            adj_n_trials = n_trials
    else:
        adj_n_trials = n_trials

    study.optimize(
        lambda trial: optuna_artificial_objective(
            trial, raw_params, performance_generator, runtimes
        ),
        n_trials=adj_n_trials,
        timeout=timeout,
        n_jobs=1,
    )

    # Apply retroactive timestamps for warm-start configurations
    all_runtimes = apply_retroactive_timestamps(warm_start_configs, runtimes)

    history = [
        build_history_entry(
            end_time=all_runtimes[idx],
            performance=trial.value,
            configurations=trial.params,
            iteration=idx + 1,
            estimator_error=None,
            searcher_training_time=None,
            breach_status=None,
            winkler_score=None,
            width=None,
            miscoverage_penalty=None,
            tabularized_configuration=None,
        )
        for idx, trial in enumerate(study.trials)
    ]
    return pd.DataFrame(history)


def ccqr_optimization_objective_function(
    performance_generator: ObjectiveMetricGenerator,
    runtimes: list[datetime],
) -> Any:
    """Creates a ccqr_optimization-compatible objective function that evaluates hyperparameter configurations.

    Args:
        performance_generator: Synthetic objective function for generating performance predictions.
        runtimes: List to record timestamps for runtime tracking during optimization.

    Returns:
        Callable objective function that takes a configuration dictionary and returns
        the predicted performance value for use with ccqr_optimization tuners.
    """

    def objective(configuration: Dict) -> float:
        result = performance_generator.predict(configuration=configuration)
        record_runtime(runtimes)

        return result

    return objective


def setup_ccqr_optimization_params(
    raw_params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
) -> dict[str, Any]:
    """Converts parameter range specifications to ccqr_optimization-compatible search space definitions.

    Args:
        raw_params: Dictionary mapping parameter names to their range specifications
            (IntRange, FloatRange, or CategoricalRange).

    Returns:
        Dictionary mapping parameter names to ccqr_optimization range objects for search space definition.
    """
    ccqr_optimization_params: dict[str, Any] = {}
    for name, param in raw_params.items():
        if isinstance(param, IntRange):
            log_flag = getattr(param, "log", False)
            ccqr_optimization_params[name] = ranges.IntRange(
                min_value=param.lower, max_value=param.upper, log_scale=log_flag
            )
        elif isinstance(param, FloatRange):
            log_flag = getattr(param, "log", False)
            ccqr_optimization_params[name] = ranges.FloatRange(
                min_value=param.lower, max_value=param.upper, log_scale=log_flag
            )
        elif isinstance(param, CategoricalRange):
            ccqr_optimization_params[name] = ranges.CategoricalRange(choices=param.choices)
        else:
            raise ValueError(f"Unknown parameter type: {type(param)}")
    return ccqr_optimization_params


def ccqr_optimization_tune(
    raw_params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    performance_generator: ObjectiveMetricGenerator,
    tuner_model: CCQRModel,
    warm_start_configs: Optional[list[tuple[dict, float]]] = None,
    random_state: Optional[int] = None,
    n_trials: Optional[int] = None,
    timeout: Optional[float] = None,
    searcher_tuning_framework: Optional[str] = None,
) -> pd.DataFrame:
    """Runs conformal hyperparameter optimization using the ccqr_optimization framework with synthetic objectives.

    Args:
        raw_params: Dictionary mapping parameter names to their range specifications
            (IntRange, FloatRange, or CategoricalRange).
        performance_generator: Synthetic objective function for generating performance predictions.
        tuner_model: ccqr_optimizationModel configuration containing the conformal searcher and parameters.
        warm_start_configs: Optional list of (configuration, loss) tuples for initialization.
        random_state: Optional random seed for reproducible results.
        n_trials: Optional maximum number of optimization trials.
        timeout: Optional time budget in seconds for the optimization process.
        searcher_tuning_framework: Optional framework identifier for searcher training ("decaying" or "fixed").

    Returns:
        DataFrame containing the complete tuning history with conformal prediction intervals and metadata.
    """
    runtimes = create_runtime_tracker()

    objective_fn = ccqr_optimization_objective_function(performance_generator, runtimes)
    ccqr_optimization_params = setup_ccqr_optimization_params(raw_params)
    conformal_tuner = ConformalTuner(
        objective_function=objective_fn,
        search_space=ccqr_optimization_params,
        minimize=True,
        n_candidates=N_CANDIDATES,
        warm_starts=warm_start_configs,
        dynamic_sampling=True,
    )

    adj_n_trials = n_trials
    searcher = tuner_model.searcher

    searcher_copy = deepcopy(searcher)
    # NOTE: We take the original sampler's alpha, to avoid mutation later on:
    if isinstance(searcher.sampler, (LowerBoundSampler, PessimisticLowerBoundSampler)):
        alpha = searcher.sampler.alpha
    # NOTE: Zero random searches because this benchmark repository uses warm-starting:
    conformal_tuner.tune(
        searcher=searcher_copy,
        max_runtime=int(timeout) if timeout is not None else None,
        max_searches=adj_n_trials,
        n_random_searches=0,
        conformal_retraining_frequency=1,
        verbose=False,
        random_state=random_state,
        optimizer_framework=searcher_tuning_framework
        if searcher_tuning_framework in ("decaying", "fixed")
        else None,
    )

    # Apply retroactive timestamps for warm-start configurations
    all_runtimes = apply_retroactive_timestamps(warm_start_configs, runtimes)

    history = []
    for idx, trial in enumerate(conformal_tuner.study.trials):
        # Only extract alpha and calculate metrics if sampler.sampler is LowerBoundSampler or PessimisticLowerBoundSampler
        if (
            isinstance(
                searcher.sampler, (LowerBoundSampler, PessimisticLowerBoundSampler)
            )
            and trial.lower_bound is not None
            and trial.upper_bound is not None
        ):
            breach_status = calculate_breach_status(
                trial.lower_bound, trial.upper_bound, trial.performance
            )
            winkler_score, width, miscoverage_penalty = calculate_winkler_components(
                trial.lower_bound, trial.upper_bound, trial.performance, alpha
            )
        else:
            breach_status = None
            winkler_score = None
            width = None
            miscoverage_penalty = None
        history.append(
            build_history_entry(
                end_time=all_runtimes[idx],
                performance=trial.performance,
                configurations=trial.configuration,
                iteration=idx + 1,
                searcher_training_time=trial.searcher_runtime,
                breach_status=breach_status,
                winkler_score=winkler_score,
                width=width,
                miscoverage_penalty=miscoverage_penalty,
                tabularized_configuration=trial.tabularized_configuration,
            )
        )
    return pd.DataFrame(history)


def setup_smac_configspace(
    raw_params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    random_state: Optional[int] = None,
) -> ConfigurationSpace:
    """Creates SMAC ConfigurationSpace from parameter definitions.

    Args:
        raw_params: Dictionary mapping parameter names to IntRange, FloatRange, or CategoricalRange.
        random_state: Optional random seed.

    Returns:
        SMAC ConfigurationSpace object.
    """
    cs = ConfigurationSpace(seed=random_state)

    for name, param in raw_params.items():
        if isinstance(param, IntRange):
            log_flag = getattr(param, "log", False)
            hp = UniformIntegerHyperparameter(
                name, param.lower, param.upper, log=log_flag
            )
        elif isinstance(param, FloatRange):
            log_flag = getattr(param, "log", False)
            hp = UniformFloatHyperparameter(
                name, param.lower, param.upper, log=log_flag
            )
        elif isinstance(param, CategoricalRange):
            hp = CategoricalHyperparameter(name, param.choices)
        else:
            raise ValueError(f"Unknown parameter type: {type(param)}")
        cs.add_hyperparameter(hp)

    return cs


def smac_objective_function(
    config: Configuration,
    performance_generator: ObjectiveMetricGenerator,
    runtimes: list[datetime],
    seed: int = 0,
) -> float:
    """Objective function for SMAC using a synthetic performance generator.

    Args:
        config: SMAC Configuration object.
        performance_generator: ObjectiveMetricGenerator instance.
        runtimes: List to append runtime timestamps.
        seed: Random seed (required by SMAC interface).

    Returns:
        Predicted performance as float.
    """
    # Convert Configuration to dict for the performance generator
    config_dict = dict(config)
    result = performance_generator.predict(configuration=config_dict)
    record_runtime(runtimes)
    return result


def smac_tune(
    raw_params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    performance_generator: ObjectiveMetricGenerator,
    tuner_model: SMACModel,
    warm_start_configs: Optional[list[tuple[dict, float]]] = None,
    random_state: Optional[int] = None,
    n_trials: Optional[int] = None,
    timeout: Optional[float] = None,
) -> pd.DataFrame:
    """Runs Bayesian optimization using SMAC3 with Random Forest surrogate and acquisition functions.

    Uses vanilla SMAC configuration for fair comparison with other tuners:
    - Each configuration evaluated exactly once (no racing)
    - No random interleaving (always uses acquisition function)
    - No parallelization
    - Single incumbent tracking
    - Deterministic scenario

    Args:
        raw_params: Dictionary mapping parameter names to their range specifications
            (IntRange, FloatRange, or CategoricalRange).
        performance_generator: Synthetic objective function for generating performance predictions.
        tuner_model: SMACModel configuration specifying the acquisition function and parameters.
        warm_start_configs: Optional list of (configuration, loss) tuples for initialization.
        random_state: Optional random seed for reproducible results.
        n_trials: Optional maximum number of optimization trials.
        timeout: Optional time budget in seconds for the optimization process.

    Returns:
        DataFrame containing the complete tuning history with trial results and metadata.
    """
    # Create configuration space
    configspace = setup_smac_configspace(raw_params, random_state)

    scenario = Scenario(
        configspace=configspace,
        deterministic=True,  # Set to deterministic for fair comparison
        n_trials=n_trials if n_trials is not None else 100,
        walltime_limit=timeout,
        seed=random_state,
        n_workers=1,  # No parallelization
    )

    searcher = tuner_model.searcher
    if searcher != "SMAC-EI":
        raise ValueError(f"Unknown SMAC sampler: {searcher}")

    # Setup runtime tracking
    runtimes = create_runtime_tracker()
    objective_fn = partial(
        smac_objective_function,
        performance_generator=performance_generator,
        runtimes=runtimes,
    )

    # Create SMAC facade with fair comparison settings
    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=objective_fn,
        model=HyperparameterOptimizationFacade.get_model(scenario),
        # Disable racing: each configuration evaluated exactly once
        intensifier=HyperparameterOptimizationFacade.get_intensifier(
            scenario, max_config_calls=1
        ),
        # Disable random interleaving: always use acquisition function
        random_design=HyperparameterOptimizationFacade.get_random_design(
            scenario, probability=0.0
        ),
        initial_design=HyperparameterOptimizationFacade.get_initial_design(
            scenario, n_configs=0 if warm_start_configs else None
        ),
        overwrite=True,
    )

    # Handle warm start configurations
    if warm_start_configs:
        for config_dict, cost in warm_start_configs:
            config = Configuration(configspace, config_dict)
            trial_info = TrialInfo(config=config, seed=random_state or 0)
            trial_value = TrialValue(cost=cost)
            smac.tell(trial_info, trial_value)

    # Run optimization
    smac.optimize()

    # Build history from runhistory
    history = []

    # Apply retroactive timestamps for warm-start configurations
    all_runtimes = apply_retroactive_timestamps(warm_start_configs, runtimes)

    for idx, (trial_key, trial_value) in enumerate(smac.runhistory.items()):
        config = smac.runhistory.get_config(trial_key.config_id)
        config_dict = config.get_dictionary()  # Use proper ConfigSpace method

        end_time = all_runtimes[idx]
        history.append(
            build_history_entry(
                end_time=end_time,
                performance=trial_value.cost,
                configurations=config_dict,
                iteration=idx + 1,
                estimator_error=None,
                searcher_training_time=None,
                breach_status=None,
                winkler_score=None,
                width=None,
                miscoverage_penalty=None,
                tabularized_configuration=None,
            )
        )

    return pd.DataFrame(history)


def tune(
    performance_generator: ObjectiveMetricGenerator,
    tuner_config: TunerConfig,
    params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    warm_start_configs: Optional[list[tuple[dict, float]]] = None,
    random_state: Optional[int] = None,
    n_trials: Optional[int] = None,
    timeout: Optional[float] = None,
) -> pd.DataFrame:
    """Unified tuning interface for optuna, ccqr_optimization, and smac.

    Args:
        performance_generator: ObjectiveMetricGenerator instance.
        tuner_config: TunerConfig object specifying tuner and searcher.
        params: Dictionary mapping parameter names to IntRange, FloatRange, or CategoricalRange.
        warm_start_configs: Optional list of (config, loss) tuples for warm start.
        random_state: Optional random seed.
        n_trials: Optional number of trials.
        timeout: Optional time budget in seconds.

    Returns:
        DataFrame with tuning history.
    """
    # Shared arguments for all tuner functions:
    shared_kwargs = {
        "tuner_model": tuner_config.tuner,
        "raw_params": params,
        "performance_generator": performance_generator,
        "warm_start_configs": warm_start_configs,
        "random_state": random_state,
        "n_trials": n_trials,
        "timeout": timeout,
    }

    if tuner_config.tuner.backend == "optuna":
        history = optuna_tune(
            **shared_kwargs,
        )
    elif tuner_config.tuner.backend == "ccqr_optimization":
        history = ccqr_optimization_tune(
            searcher_tuning_framework=tuner_config.searcher_tuning_framework,
            **shared_kwargs,
        )
    elif tuner_config.tuner.backend == "smac":
        history = smac_tune(
            **shared_kwargs,
        )
    else:
        raise ValueError(f"Unknown tuner: {tuner_config.tuner.backend}")

    return history
