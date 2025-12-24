import logging
import random
from typing import Optional, Dict, Tuple, get_type_hints, List
from hpobench.gp_opt.wrapping import ParameterRange

import numpy as np
from tqdm import tqdm
from datetime import datetime
import inspect
from hpobench.gp_opt.utils.tracking import (
    Trial,
    Study,
    RuntimeTracker,
    DynamicConfigurationManager,
    StaticConfigurationManager,
    ProgressBarManager,
)
from hpobench.gp_opt.surrogate import GPEstimator
from hpobench.gp_opt.acquisition_functions import (
    BaseAcquisitionFunction,
    optimize_acquisition,
)

logger = logging.getLogger(__name__)


def stop_search(
    n_remaining_configurations: int,
    current_iter: int,
    current_runtime: float,
    max_runtime: Optional[float] = None,
    max_searches: Optional[int] = None,
) -> bool:
    """Determine whether to terminate the hyperparameter search process.

    Evaluates multiple stopping criteria to determine if the optimization should halt.
    The function implements a logical OR of termination conditions: exhausted search space,
    runtime budget exceeded, or iteration limit reached.

    Args:
        n_remaining_configurations: Number of configurations still available for evaluation
        current_iter: Current iteration count in the search process
        current_runtime: Elapsed time since search initiation in seconds
        max_runtime: Maximum allowed runtime in seconds, None for no limit
        max_searches: Maximum allowed iterations, None for no limit

    Returns:
        True if any stopping criterion is met, False otherwise
    """
    if n_remaining_configurations == 0:
        return True

    if max_runtime is not None:
        if current_runtime >= max_runtime:
            return True

    if max_searches is not None:
        if current_iter >= max_searches:
            return True

    return False


class GPTuner:
    def __init__(
        self,
        objective_function: callable,
        search_space: Dict[str, ParameterRange],
        minimize: bool = True,
        n_candidates: int = 3000,
        warm_starts: Optional[List[Tuple[Dict, float]]] = None,
        dynamic_sampling: bool = True,
    ) -> None:
        """Initialize the GP-based hyperparameter tuner.

        Args:
            objective_function: Callable that evaluates hyperparameter configurations.
            search_space: Dictionary mapping parameter names to their range specifications.
            minimize: Whether to minimize (True) or maximize (False) the objective.
            n_candidates: Number of candidate configurations to consider for acquisition optimization.
            warm_starts: Optional list of (configuration, objective_value) tuples for initialization.
            dynamic_sampling: Whether to use dynamic candidate sampling during optimization.
        """
        self.objective_function = objective_function
        self.check_objective_function()

        self.search_space = search_space
        self.minimize = minimize
        self.metric_sign = 1 if minimize else -1
        self.warm_starts = warm_starts
        self.n_candidates = n_candidates
        self.dynamic_sampling = dynamic_sampling
        self.config_manager = None

    def check_objective_function(self) -> None:
        """Validate objective function signature and type annotations.

        Ensures the objective function conforms to the required interface:
        single parameter named 'configuration' of type Dict, returning numeric value.
        This validation prevents runtime errors and ensures compatibility with
        the optimization framework.

        Raises:
            ValueError: If function signature doesn't match requirements
            TypeError: If type annotations are incorrect
        """
        signature = inspect.signature(self.objective_function)
        args = list(signature.parameters.values())

        if len(args) != 1:
            raise ValueError("Objective function must take exactly one argument.")

        first_arg = args[0]
        if first_arg.name != "configuration":
            raise ValueError(
                "The objective function must take exactly one argument named 'configuration'."
            )

        type_hints = get_type_hints(self.objective_function)
        if "configuration" in type_hints and type_hints["configuration"] is not Dict:
            raise TypeError(
                "The 'configuration' argument of the objective must be of type Dict."
            )
        if "return" in type_hints and type_hints["return"] not in [
            int,
            float,
            np.number,
        ]:
            raise TypeError(
                "The return type of the objective function must be numeric (int, float, or np.number)."
            )

    def process_warm_starts(self) -> None:
        """Initialize optimization with pre-evaluated configurations.

        Processes warm start configurations by marking them as searched and creating
        corresponding trial records. This allows the optimization to begin with
        prior knowledge, potentially accelerating convergence by skipping known
        poor configurations and leveraging good starting points.

        The warm start configurations are treated as iteration 0 data and assigned
        the 'warm_start' acquisition source for tracking purposes.
        """
        for idx, (config, performance) in enumerate(self.warm_starts):
            self.config_manager.mark_as_searched(config, performance)
            trial = Trial(
                iteration=idx,
                timestamp=datetime.now(),
                configuration=config.copy(),
                tabularized_configuration=self.config_manager.listify_configs([config])[
                    0
                ],
                performance=performance,
                acquisition_source="warm_start",
            )
            self.study.append_trial(trial)

    def initialize_tuning_resources(self) -> None:
        """Initialize core optimization components and data structures.

        Sets up the study container for trial tracking, configuration manager for
        handling search space sampling, and processes any warm start configurations.
        The configuration manager type (static vs dynamic) determines whether
        the candidate pool is fixed or adaptively resampled during optimization.
        """
        self.study = Study(
            metric_optimization="minimize" if self.minimize else "maximize"
        )

        if self.dynamic_sampling:
            self.config_manager = DynamicConfigurationManager(
                search_space=self.search_space,
                n_candidate_configurations=self.n_candidates,
            )
        else:
            self.config_manager = StaticConfigurationManager(
                search_space=self.search_space,
                n_candidate_configurations=self.n_candidates,
            )

        if self.warm_starts:
            self.process_warm_starts()

    def _evaluate_configuration(self, configuration: Dict) -> Tuple[float, float]:
        """Evaluate a configuration and measure execution time.

        Executes the objective function with the given configuration while tracking
        runtime. This method provides the core evaluation mechanism used throughout
        both random and conformal search phases.

        Args:
            configuration: Parameter configuration dictionary to evaluate

        Returns:
            Tuple of (performance_value, evaluation_runtime)
        """
        runtime_tracker = RuntimeTracker()
        performance = self.objective_function(configuration=configuration)
        runtime = runtime_tracker.return_runtime()

        return performance, runtime

    def random_search(
        self,
        max_random_iter: int,
        max_runtime: Optional[int] = None,
        max_searches: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        """Execute random search phase to initialize optimization with baseline data.

        Performs uniform random sampling of configurations to establish initial
        performance landscape understanding. This phase is crucial for subsequent
        conformal prediction model training, as it provides the foundational
        dataset for uncertainty quantification.

        Args:
            max_random_iter: Maximum number of random configurations to evaluate
            max_runtime: Optional runtime budget in seconds
            max_searches: Optional total iteration limit
            verbose: Whether to display progress information
        """

        available_configs = self.config_manager.get_searchable_configurations()
        adj_n_searches = min(max_random_iter, len(available_configs))
        if adj_n_searches == 0:
            logger.warning("No configurations available for random search")

        search_idxs = np.random.choice(
            len(available_configs), size=adj_n_searches, replace=False
        )
        sampled_configs = [available_configs[idx] for idx in search_idxs]

        progress_iter = (
            tqdm(sampled_configs, desc="Random search: ")
            if verbose
            else sampled_configs
        )

        for config in progress_iter:
            validation_performance, training_time = self._evaluate_configuration(config)

            if np.isnan(validation_performance):
                logger.debug(
                    "Obtained non-numerical performance, skipping configuration."
                )
                self.config_manager.add_to_banned_configurations(config)
                continue

            self.config_manager.mark_as_searched(config, validation_performance)

            trial = Trial(
                iteration=len(self.study.trials),
                timestamp=datetime.now(),
                configuration=config.copy(),
                tabularized_configuration=self.config_manager.listify_configs([config])[
                    0
                ],
                performance=validation_performance,
                acquisition_source="rs",
                target_model_runtime=training_time,
            )
            self.study.append_trial(trial)

            searchable_count = self.config_manager.get_searchable_configurations_count()
            current_runtime = self.search_timer.return_runtime()

            stop = stop_search(
                n_remaining_configurations=searchable_count,
                current_runtime=current_runtime,
                max_runtime=max_runtime,
                current_iter=len(self.study.trials),
                max_searches=max_searches,
            )
            if stop:
                break

    def setup_inferential_search_resources(
        self,
        verbose: bool,
        max_runtime: Optional[int],
        max_searches: Optional[int],
    ) -> Tuple[ProgressBarManager, float]:
        """Initialize progress tracking and iteration limits for conformal search.

        Sets up the progress bar manager for displaying search progress and calculates
        the maximum number of conformal search iterations based on total limits and
        already completed trials from previous phases.

        Args:
            verbose: Whether to display progress information
            max_runtime: Optional maximum runtime in seconds
            max_searches: Optional maximum total iterations

        Returns:
            Tuple of (progress_manager, conformal_max_searches)
        """
        progress_manager = ProgressBarManager(verbose=verbose)
        progress_manager.create_progress_bar(
            max_runtime=max_runtime,
            max_searches=max_searches,
            current_trials=len(self.study.trials),
            description="Conformal search",
        )

        inferential_max_searches = (
            max_searches - len(self.study.trials)
            if max_searches is not None
            else float("inf")
        )

        return progress_manager, inferential_max_searches

    def retrain_searcher(
        self,
        searcher: GPEstimator,
        X: np.array,
        y: np.array,
    ) -> Tuple[float, float]:
        """Train conformal prediction searcher on accumulated data.

        Fits the conformal prediction model using the provided data,
        tracking training time and model performance for adaptive parameter
        optimization. The tuning_count parameter controls internal hyperparameter
        optimization within the searcher.

        Args:
            searcher: Conformal searcher instance to train
            X: Feature matrix (sign-adjusted)
            y: Target values (sign-adjusted)

        Returns:
            Tuple of (training_runtime, estimator_error)
        """
        runtime_tracker = RuntimeTracker()
        searcher.fit(
            X=X,
            y=y,
        )

        training_runtime = runtime_tracker.return_runtime()

        return training_runtime

    def select_next_configuration(
        self,
        searcher: GPEstimator,
        acquisition_func: BaseAcquisitionFunction,
        searchable_configs: List,
        transformed_configs: np.array,
    ) -> Tuple[Dict, int]:
        """Select the most promising configuration using conformal predictions.

        Uses the conformal searcher to predict lower bounds for all available
        configurations and selects the one with the minimum predicted lower bound.
        This implements a pessimistic acquisition strategy that favors configurations
        with high confidence of good performance.

        Args:
            searcher: Trained conformal searcher for predictions
            acquisition_func: Acquisition function to use
            searchable_configs: List of available configuration dictionaries
            transformed_configs: Scaled feature matrix for configurations

        Returns:
            Selected configuration dictionary
        """
        f_best = None
        if len(self.config_manager.searched_performances) > 0:
            # Get performances and apply sign conversion for internal minimization
            performances = (
                np.array(self.config_manager.searched_performances) * self.metric_sign
            )
            # For minimization problems, f_best is the current best (minimum) value
            f_best = np.min(performances)

        next_idx, _ = optimize_acquisition(
            acquisition_func=acquisition_func,
            gp_estimator=searcher,
            candidate_points=transformed_configs,
            f_best=f_best,
        )

        next_config = searchable_configs[next_idx]

        return next_config

    def search(
        self,
        searcher: GPEstimator,
        acquisition_func: BaseAcquisitionFunction,
        retraining_frequency: int,
        verbose: bool,
        max_searches: Optional[int],
        max_runtime: Optional[int],
    ) -> None:
        """Execute conformal prediction-guided hyperparameter search.

        Implements the main conformal search loop that iteratively trains conformal
        prediction models, selects promising configurations based on uncertainty
        quantification, and updates the models with new observations. The method
        supports adaptive parameter tuning through multi-armed bandit optimization.

        Args:
            searcher: Conformal prediction searcher for configuration selection
            conformal_retraining_frequency: Base frequency for model retraining
            verbose: Whether to display search progress
            max_searches: Maximum total iterations including previous phases
            max_runtime: Maximum total runtime budget in seconds
        """
        (
            progress_manager,
            conformal_max_searches,
        ) = self.setup_inferential_search_resources(verbose, max_runtime, max_searches)

        for search_iter in range(conformal_max_searches):
            progress_manager.update_progress(
                current_runtime=(
                    self.search_timer.return_runtime() if max_runtime else None
                ),
                iteration_count=1 if max_searches else 0,
            )

            X = self.config_manager.tabularize_configs(
                self.config_manager.searched_configs
            )
            y = np.array(self.config_manager.searched_performances) * self.metric_sign

            searchable_configs = self.config_manager.get_searchable_configurations()

            X_searchable = self.config_manager.tabularize_configs(searchable_configs)

            # GP retraining phase
            training_runtime = 0
            if search_iter == 0 or search_iter % retraining_frequency == 0:
                training_runtime = self.retrain_searcher(searcher, X, y)

            # Configuration selection phase
            next_config = self.select_next_configuration(
                searcher=searcher,
                acquisition_func=acquisition_func,
                searchable_configs=searchable_configs,
                transformed_configs=X_searchable,
            )

            performance, _ = self._evaluate_configuration(next_config)

            if np.isnan(performance):
                self.config_manager.add_to_banned_configurations(next_config)
                continue

            self.config_manager.mark_as_searched(next_config, performance)
            trial = Trial(
                iteration=len(self.study.trials),
                timestamp=datetime.now(),
                configuration=next_config.copy(),
                tabularized_configuration=self.config_manager.listify_configs(
                    [next_config]
                )[0],
                performance=performance,
                acquisition_source=str(searcher),
                searcher_runtime=training_runtime,
            )
            self.study.append_trial(trial)

            searchable_count = self.config_manager.get_searchable_configurations_count()
            should_stop = stop_search(
                n_remaining_configurations=searchable_count,
                current_runtime=self.search_timer.return_runtime(),
                max_runtime=max_runtime,
                current_iter=len(self.study.trials),
                max_searches=max_searches,
            )

            if should_stop:
                break

        progress_manager.close_progress_bar()

    def tune(
        self,
        acquisition_func: BaseAcquisitionFunction,
        max_searches: Optional[int] = 100,
        max_runtime: Optional[int] = None,
        n_random_searches: int = 15,
        retraining_frequency: int = 1,
        random_state: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        """Execute hyperparameter optimization using conformal prediction surrogate models.

        Performs intelligent hyperparameter search through two phases: random exploration
        for baseline data, then conformal prediction-guided optimization using uncertainty
        quantification to select promising configurations.

        Args:
            max_searches: Maximum total configurations to search (random + conformal searches).
                Default: 100.
            max_runtime: Maximum search time in seconds. Search will terminate after this time,
                regardless of iterations. Default: None (no time limit).
            n_random_searches: Number of random configurations to evaluate before conformal search.
                Provides initial training data for the surrogate model. Default: 15.
            retraining_frequency: How often the conformal surrogate model retrains
                (the model will retrain every retraining_frequency-th search iteration).
                Recommended values are 1 if your target model takes >1 min to train, 2-5 if your
                target model is very small to reduce computational overhead. Default: 1.
            random_state: Random seed for reproducible results. Default: None.
            verbose: Whether to enable progress display. Default: True.

        Example:
            Basic usage::

                from hpobench.gp_opt.tuning import ConformalTuner
                from hpobench.gp_opt.wrapping import IntRange, FloatRange

                def objective(configuration):
                    model = SomeModel(
                        learning_rate=configuration['lr'],
                        hidden_units=configuration['units']
                    )
                    return model.evaluate()

                search_space = {
                    'lr': FloatRange(0.001, 0.1, log_scale=True),
                    'units': IntRange(32, 512)
                }

                tuner = ConformalTuner(
                    objective_function=objective,
                    search_space=search_space,
                    metric_optimization='maximize'
                )

                tuner.tune(n_random_searches=25, max_searches=100)

                best_config = tuner.get_best_params()
                best_score = tuner.get_best_value()
        """

        if random_state is not None:
            random.seed(a=random_state)
            np.random.seed(seed=random_state)

        searcher = GPEstimator()

        self.initialize_tuning_resources()
        self.search_timer = RuntimeTracker()

        n_warm_starts = len(self.warm_starts) if self.warm_starts else 0
        remaining_random_searches = max(0, n_random_searches - n_warm_starts)

        if remaining_random_searches > 0:
            self.random_search(
                max_random_iter=remaining_random_searches,
                max_runtime=max_runtime,
                max_searches=max_searches,
                verbose=verbose,
            )

        self.search(
            searcher=searcher,
            acquisition_func=acquisition_func,
            retraining_frequency=retraining_frequency,
            verbose=verbose,
            max_searches=max_searches,
            max_runtime=max_runtime,
        )

    def get_best_params(self) -> Dict:
        """Retrieve the best configuration found during optimization.

        Returns the parameter configuration that achieved the optimal objective
        function value, according to the specified optimization direction.

        Returns:
            Dictionary containing the optimal parameter configuration
        """
        return self.study.get_best_configuration()

    def get_best_value(self) -> float:
        """Retrieve the best objective function value achieved during optimization.

        Returns the optimal performance value found across all evaluated
        configurations, according to the specified optimization direction.

        Returns:
            Best objective function value achieved
        """
        return self.study.get_best_performance()
