from hpobench.config.config_types import IntRange, CategoricalRange, FloatRange
import random
from typing import Optional, Union, TYPE_CHECKING
import pandas as pd
import os
import logging
from datetime import datetime
import optuna

if TYPE_CHECKING:
    from hpobench.generation.generate import ObjectiveMetricGenerator

logger = logging.getLogger(__name__)


def ensure_yahpo_initialized():
    """Initialize YAHPO benchmark configuration and data paths.

    Sets up the YAHPO Gym environment for hyperparameter optimization benchmarking.
    Creates the necessary data directory and initializes the configuration lazily
    to avoid import issues. Includes fallback handling for multiprocessing contexts.
    """
    from yahpo_gym import local_config  # Import here to avoid issues

    yahpo_data_path = "yahpo_bench_data"
    os.makedirs(yahpo_data_path, exist_ok=True)

    try:
        # Only initialize if not already done
        if not hasattr(local_config, "_config") or local_config._config is None:
            local_config.init_config()
        local_config.set_data_path(yahpo_data_path)
    except Exception:
        # Fallback for multiprocessing contexts
        if not hasattr(local_config, "_config") or local_config._config is None:
            local_config._config = {}
        local_config.set_data_path(yahpo_data_path)


class AnalysisPathManager:
    """Simple path manager for organizing analysis outputs by type and purpose."""

    def __init__(self, cache_path: str, run_start_str: str):
        self.cache_path = cache_path
        self.run_start_str = run_start_str
        self.base_path = os.path.join(cache_path, "experiments", run_start_str)

    def get_analysis_path(
        self, analysis_type: str, output_type: str = "data", subfolder: str = None
    ) -> str:
        """Get path for specific analysis type and output type.

        Args:
            analysis_type: e.g., "01_coverage_analysis", "02_sampler_variation"
            output_type: "data" or "plots"
            subfolder: optional subfolder like "statistical_tests", "aggregated_results"
        """
        path = os.path.join(self.base_path, analysis_type, output_type)
        if subfolder:
            path = os.path.join(path, subfolder)
        os.makedirs(path, exist_ok=True)
        return path


def get_group_dict(breakout_col, within_group):
    """Create a dictionary mapping breakout columns to their values within a group.

    Args:
        breakout_col: Column names for grouping, can be None for no grouping.
        within_group: Values corresponding to the breakout columns.

    Returns:
        Dictionary mapping column names to values, or empty dict if no breakout columns.
    """
    if breakout_col is None:
        return {}
    if isinstance(within_group, tuple):
        return dict(zip(breakout_col, within_group))
    return {breakout_col[0]: within_group}


def generate_hyperparameter_combinations(
    params: dict[str, Union[IntRange, FloatRange, CategoricalRange]],
    n_combinations: int,
    random_state: Optional[int] = None,
):
    """Generate random hyperparameter configurations from parameter ranges.

    Args:
        params: Dictionary mapping parameter names to their range specifications.
        n_combinations: Number of random configurations to generate.
        random_state: Optional random seed for reproducible generation.

    Returns:
        List of dictionaries, each containing a random hyperparameter configuration.
    """
    random.seed(random_state)
    combinations = []
    for _ in range(n_combinations):
        combination = {}
        for param_name, param_values in params.items():
            if isinstance(param_values, IntRange):
                combination[param_name] = random.choice(
                    list(range(param_values.lower, param_values.upper + 1))
                )
            elif isinstance(param_values, FloatRange):
                combination[param_name] = random.uniform(
                    param_values.lower, param_values.upper
                )
            elif isinstance(param_values, CategoricalRange):
                combination[param_name] = random.choice(param_values.choices)
            else:
                raise ValueError()
        combinations.append(combination)
    return combinations


def save_analysis_results(
    df: pd.DataFrame,
    cache_path: str,
    run_start_str: str,
    filename: str,
    analysis_type: str = None,
    subfolder: str = None,
):
    """Save analysis results with proper path organization.

    Args:
        df: DataFrame to save
        cache_path: Base cache path
        run_start_str: Run identifier
        filename: Name of the file
        analysis_type: Analysis type (e.g., "01_coverage_analysis")
        subfolder: Optional subfolder (e.g., "statistical_tests")
    """
    if df is not None and not df.empty:
        if analysis_type:
            path_manager = AnalysisPathManager(cache_path, run_start_str)
            analysis_data_path = path_manager.get_analysis_path(
                analysis_type, "data", subfolder
            )
        else:
            # Fallback to old behavior for backward compatibility
            analysis_data_path = os.path.join(cache_path, "data", run_start_str)
            os.makedirs(analysis_data_path, exist_ok=True)

        full_filename = os.path.join(analysis_data_path, filename)
        try:
            df.to_csv(full_filename, index=False)
            logger.info(f"Saved results to {full_filename}")
        except Exception as e:
            logger.error(
                f"Failed to save results to {full_filename}: {e}", exc_info=True
            )
    else:
        logger.warning(f"Skipping save for {filename}: DataFrame is empty or None.")


def add_runtime(
    experiment_log: pd.DataFrame,
    tune_start,
    performance_generator: "ObjectiveMetricGenerator",
    n_warm_starts: int = 0,
):
    """Add runtime predictions to experiment log DataFrame.

    Args:
        experiment_log: DataFrame containing experiment trial data.
        tune_start: Start time of the tuning process.
        performance_generator: Generator for predicting performance and runtime.
        n_warm_starts: Number of warm start configurations at the beginning.

    Returns:
        DataFrame with additional runtime columns.
    """
    experiment_log_copy = experiment_log.copy()

    # Sort observations by end_time from smallest to largest (least to most recent)
    experiment_log_copy = experiment_log_copy.sort_values("end_time").reset_index(
        drop=True
    )

    experiment_log_copy["generator_runtime"] = experiment_log_copy[
        "configurations"
    ].apply(lambda x: performance_generator.predict_runtime(x))

    experiment_log_copy["tuner_runtime"] = 0.0
    # For warm start configurations (first n_warm_starts observations), tuner_runtime = 0
    # For actual optimization trials, calculate based on timing differences
    for i in range(n_warm_starts, len(experiment_log_copy)):
        if i == n_warm_starts:
            # First non-warm start observation: tune_start to end_time
            experiment_log_copy.loc[i, "tuner_runtime"] = (
                experiment_log_copy.loc[i, "end_time"] - tune_start
            ).total_seconds()
        else:
            # Subsequent observations: end_time minus previous row's end_time
            experiment_log_copy.loc[i, "tuner_runtime"] = (
                experiment_log_copy.loc[i, "end_time"]
                - experiment_log_copy.loc[i - 1, "end_time"]
            ).total_seconds()

    # Calculate runtime as sum of generator_runtime and tuner_runtime, then cumsum
    experiment_log_copy["runtime"] = (
        experiment_log_copy["generator_runtime"] + experiment_log_copy["tuner_runtime"]
    ).cumsum()

    return experiment_log_copy


def setup_environment(cache_path: str = "cache/") -> tuple[str, logging.Logger]:
    """Set up the experimental environment with logging and cache directories.

    Args:
        cache_path: Base directory path for storing experimental outputs and logs.

    Returns:
        Tuple of (run_start_str, logger) where run_start_str is a timestamp
        identifier for the current run and logger is the configured logging instance.
    """
    if not os.path.exists(cache_path):
        os.makedirs(cache_path, exist_ok=True)

    run_start = datetime.now()
    run_start_str = run_start.strftime("%Y-%m-%d_%H-%M-%S")

    log_path = os.path.join(cache_path, f"logs/{run_start_str}")
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    log_filename = os.path.join(
        log_path, f"run_{run_start.strftime(format='%m_%d_%Y-%H_%M_%S')}.log"
    )
    logging.basicConfig(
        filename=log_filename,
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logger = logging.getLogger()

    logging.getLogger("hyperopt").setLevel(logging.ERROR)
    logging.getLogger("ccqr_optimization").setLevel(logging.ERROR)
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    logging.getLogger("yahpo").setLevel(logging.WARNING)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return run_start_str, logger
