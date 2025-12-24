from __future__ import annotations

import logging
from typing import Any, Callable, Sequence, TYPE_CHECKING
from enum import Enum

import numpy as np
import optuna
from optuna.distributions import BaseDistribution
from optuna.samplers._base import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.trial import FrozenTrial, TrialState

if TYPE_CHECKING:
    import torch

    import optuna._gp.acqf as acqf
    import optuna._gp.gp as gp
    import optuna._gp.optim_mixed as optim_mixed
    import optuna._gp.prior as prior
    import optuna._gp.search_space as gp_search_space
    from optuna.study import Study
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")
    acqf = _LazyImport("optuna._gp.acqf")
    gp_search_space = _LazyImport("optuna._gp.search_space")
    gp = _LazyImport("optuna._gp.gp")
    optim_mixed = _LazyImport("optuna._gp.optim_mixed")
    prior = _LazyImport("optuna._gp.prior")


logger = logging.getLogger(__name__)
EPS = 1e-10


class ExpandedAcquisitionFunction(Enum):
    """Supported ccqr_optimization acquisition functions."""

    EXPECTED_IMPROVEMENT = "expected_improvement"
    LOG_EXPECTED_IMPROVEMENT = "log_expected_improvement"  # More comparable to Optuna
    THOMPSON_SAMPLING = "thompson_sampling"
    OPTIMISTIC_THOMPSON_SAMPLING = (
        "optimistic_thompson_sampling"  # Thompson capped at mean
    )
    CONFIDENCE_BOUND = "confidence_bound"  # UCB/LCB unified


def _standardize_values(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize values for GP modeling."""
    clipped_values = gp.warn_and_convert_inf(values)
    means = np.mean(clipped_values, axis=0)
    stds = np.std(clipped_values, axis=0)
    standardized_values = (clipped_values - means) / np.maximum(EPS, stds)
    return standardized_values, means, stds


class StrippedGPSampler(BaseSampler):
    """
    Gaussian Process sampler with ccqr_optimization acquisition functions.

    This sampler combines Optuna's GP infrastructure with acquisition functions
    inspired by the ccqr_optimization framework. It supports various acquisition strategies
    while maintaining the theoretical foundation of Gaussian process-based
    Bayesian optimization.

    Args:
        acquisition_function: ccqr_optimization acquisition function to use
        n_candidates: Number of candidate configurations to evaluate
        seed: Random seed for reproducibility
        independent_sampler: Sampler for initial trials and conditional parameters
        n_startup_trials: Number of initial random trials
        deterministic_objective: Whether the objective is deterministic
        beta: Exploration parameter for confidence bound methods
        xi: Improvement threshold for expected improvement
        n_samples: Number of samples for Monte Carlo acquisition functions
    """

    def __init__(
        self,
        *,
        acquisition_function: ExpandedAcquisitionFunction = ExpandedAcquisitionFunction.LOG_EXPECTED_IMPROVEMENT,
        n_candidates: int = 2048,
        seed: int | None = None,
        independent_sampler: BaseSampler | None = None,
        n_startup_trials: int = 10,
        deterministic_objective: bool = False,
        maximize: bool = False,
        beta: float = 2.0,
        xi: float = 0.01,
        n_samples: int = 1000,
    ) -> None:
        self._acquisition_function = acquisition_function
        self._n_candidates = n_candidates
        self._rng = LazyRandomState(seed)
        self._independent_sampler = (
            independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        )
        self._intersection_search_space = optuna.search_space.IntersectionSearchSpace()
        self._n_startup_trials = n_startup_trials
        self._log_prior: Callable[
            [gp.KernelParamsTensor], torch.Tensor
        ] = prior.default_log_prior
        self._minimum_noise: float = prior.DEFAULT_MINIMUM_NOISE_VAR
        self._kernel_params_cache_list: list[gp.KernelParamsTensor] | None = None
        self._deterministic = deterministic_objective

        self._maximize = maximize

        self._beta = beta  # For confidence bounds
        self._xi = xi  # For EI
        self._n_samples = n_samples  # For Monte Carlo methods

        self._n_preliminary_samples: int = 2000

        logger.info(
            f"Initialized GPSampler with acquisition function: {acquisition_function.value}"
        )

    def reseed_rng(self) -> None:
        """Reseed the random number generator."""
        self._rng.rng.seed()
        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        """Infer the search space for multivariate sampling."""
        search_space = {}
        for name, distribution in self._intersection_search_space.calculate(
            study
        ).items():
            if distribution.single():
                continue
            search_space[name] = distribution
        return search_space

    def _expected_improvement_acquisition(
        self, mean: torch.Tensor, var: torch.Tensor, f_best: float
    ) -> torch.Tensor:
        """
        Compute Expected Improvement acquisition function.

        This implementation follows the standard EI formula:
        EI(x) = σ(x) * [z * Φ(z) + φ(z)]
        where z = (μ(x) - f_best - ξ) / σ(x)

        Note: Since we transform to maximization internally, we want μ(x) > f_best
        """
        std = torch.sqrt(var)

        # Avoid division by zero
        std_safe = torch.clamp(std, min=EPS)
        # For maximization: we want mean > f_best, so z = (mean - f_best - xi) / std
        z = (mean - f_best - self._xi) / std_safe

        # Compute normal CDF and PDF
        normal_dist = torch.distributions.Normal(0, 1)
        phi = normal_dist.cdf(z)  # CDF
        pdf = torch.exp(normal_dist.log_prob(z))  # PDF

        # Expected Improvement formula
        ei = std * (z * phi + pdf)

        # Handle numerical issues: EI should be non-negative
        ei = torch.clamp(ei, min=0.0)

        return ei

    def _log_expected_improvement_acquisition(
        self, mean: torch.Tensor, var: torch.Tensor, f_best: float
    ) -> torch.Tensor:
        """
        Compute Log Expected Improvement acquisition function.

        This implementation uses Optuna's exact logei function for consistency.
        """
        # Use Optuna's logei implementation directly for exact compatibility
        return self._optuna_logei(mean, var, f_best)

    def _optuna_logei(
        self, mean: torch.Tensor, var: torch.Tensor, f0: float
    ) -> torch.Tensor:
        """
        Optuna's exact logei implementation for numerical consistency.

        This is a direct port of optuna._gp.acqf.logei and standard_logei functions.
        """
        import math

        # Return E_{y ~ N(mean, var)}[max(0, y-f0)]
        sigma = torch.sqrt(var)
        z = (mean - f0) / sigma

        # Switch implementation based on z value for numerical stability
        small = z < -25

        vals = torch.empty_like(z)
        z_small = z[small]
        z_normal = z[~small]
        sqrt_2pi = math.sqrt(2 * math.pi)

        # For normal values (z >= -25)
        if torch.any(~small):
            cdf = 0.5 * torch.special.erfc(-z_normal * math.sqrt(0.5))
            pdf = torch.exp(-0.5 * z_normal**2) * (1 / sqrt_2pi)
            vals[~small] = torch.log(z_normal * cdf + pdf)

        # For small values (z < -25) - use asymptotic expansion
        if torch.any(small):
            r = math.sqrt(0.5 * math.pi) * torch.special.erfcx(
                -z_small * math.sqrt(0.5)
            )
            vals[small] = -0.5 * z_small**2 + torch.log(
                (z_small * r + 1) * (1 / sqrt_2pi)
            )

        # Final logei value
        return torch.log(sigma) + vals

    def _confidence_bound_acquisition(
        self, mean: torch.Tensor, var: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Upper Confidence Bound acquisition function.

        Since we transform to maximization internally, we use UCB:
        UCB(x) = μ(x) + β * σ(x)
        """
        return mean + self._beta * torch.sqrt(var)

    def _thompson_sampling_acquisition(
        self, mean: torch.Tensor, var: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Thompson Sampling acquisition function.

        Samples from the GP posterior distribution N(μ(x), σ²(x)) to balance
        exploration and exploitation probabilistically.
        """
        # Sample from the posterior distribution
        std = torch.sqrt(var)
        # Use the same random seed for reproducibility within a single optimization
        samples = torch.normal(mean, std)
        return samples

    def _optimistic_thompson_sampling_acquisition(
        self, mean: torch.Tensor, var: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Optimistic Thompson Sampling acquisition function.

        This is Thompson sampling floored at the mean (point estimate), following
        the ccqr_optimization implementation.

        Formula: max(sample ~ N(μ(x), σ²(x)), μ(x))
        """
        # Sample from the posterior distribution
        std = torch.sqrt(var)
        samples = torch.normal(mean, std)

        optimistic_samples = torch._foreach_maximum(samples, mean)

        return optimistic_samples

    def _compute_acquisition_function(
        self,
        normalized_params: torch.Tensor,
        kernel_params: gp.KernelParamsTensor,
        search_space: gp_search_space.SearchSpace,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        y_samples: np.ndarray,
    ) -> torch.Tensor:
        """Compute acquisition function values for candidate points."""
        # Get categorical mask
        is_categorical = (
            search_space.scale_types == gp_search_space.ScaleType.CATEGORICAL
        )
        is_categorical_tensor = torch.from_numpy(is_categorical).bool()

        # Compute kernel matrices
        K_train = gp.kernel(is_categorical_tensor, kernel_params, X_train, X_train)
        K_noise = (
            K_train
            + torch.eye(X_train.shape[0], dtype=K_train.dtype, device=K_train.device)
            * kernel_params.noise_var
        )

        # Compute inverse and solve
        try:
            K_inv = torch.linalg.inv(K_noise)
            alpha = torch.linalg.solve(K_noise, Y_train)
        except torch.linalg.LinAlgError:
            # Add jitter for numerical stability
            jitter = 1e-6
            K_noise_jitter = (
                K_noise
                + torch.eye(
                    K_noise.shape[0], dtype=K_noise.dtype, device=K_noise.device
                )
                * jitter
            )
            K_inv = torch.linalg.inv(K_noise_jitter)
            alpha = torch.linalg.solve(K_noise_jitter, Y_train)

        # Compute posterior mean and variance using Optuna's posterior function
        mean, var = gp.posterior(
            kernel_params,
            X_train,
            is_categorical_tensor,
            K_inv,
            alpha,
            normalized_params,
        )

        # Add numerical stability (similar to Optuna's acqf_stabilizing_noise)
        stabilizing_noise = 1e-8  # Small noise for numerical stability
        var = var + stabilizing_noise

        # Compute acquisition function based on selected method
        if (
            self._acquisition_function
            == ExpandedAcquisitionFunction.EXPECTED_IMPROVEMENT
        ):
            f_best = torch.max(Y_train).item() if len(Y_train) > 0 else 0.0
            return self._expected_improvement_acquisition(mean, var, f_best)

        elif (
            self._acquisition_function
            == ExpandedAcquisitionFunction.LOG_EXPECTED_IMPROVEMENT
        ):
            # Use Optuna's approach: f_best is max of standardized values (which are already maximization-oriented)
            f_best = torch.max(Y_train).item() if len(Y_train) > 0 else 0.0
            return self._log_expected_improvement_acquisition(mean, var, f_best)

        elif self._acquisition_function == ExpandedAcquisitionFunction.CONFIDENCE_BOUND:
            return self._confidence_bound_acquisition(mean, var)

        elif (
            self._acquisition_function == ExpandedAcquisitionFunction.THOMPSON_SAMPLING
        ):
            return self._thompson_sampling_acquisition(mean, var)

        elif (
            self._acquisition_function
            == ExpandedAcquisitionFunction.OPTIMISTIC_THOMPSON_SAMPLING
        ):
            return self._optimistic_thompson_sampling_acquisition(mean, var)

        else:
            raise ValueError(
                f"Unknown acquisition function: {self._acquisition_function}"
            )

    def _optimize_acquisition_function(
        self,
        acqf_params: acqf.AcquisitionFunctionParams,
        best_params: np.ndarray | None,
    ) -> np.ndarray:
        """Simplified optimization: uniform sampling + argmax only."""
        # Generate uniform random candidates (no Sobol, no local search)
        n_dims = len(acqf_params.search_space.scale_types)

        # Use uniform random sampling instead of Sobol
        candidates = self._rng.rng.uniform(
            0.0, 1.0, size=(self._n_preliminary_samples, n_dims)
        )

        # Evaluate acquisition function at all candidates using Optuna's method
        acq_values = acqf.eval_acqf_no_grad(acqf_params, candidates)

        # Simply pick the best candidate (no local search)
        best_idx = np.argmax(acq_values)
        best_candidate = candidates[best_idx]

        return best_candidate

    def _optimize_acquisition_function_custom(
        self,
        kernel_params: gp.KernelParamsTensor,
        search_space: gp_search_space.SearchSpace,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        y_samples: np.ndarray,
    ) -> np.ndarray:
        """Custom optimization for non-LOG_EI acquisition functions."""
        # Use uniform sampling for consistency (no Sobol, no local search)
        n_dims = len(search_space.scale_types)

        # Use uniform random sampling
        n_candidates = max(self._n_candidates, 2048)
        candidates = torch.from_numpy(
            self._rng.rng.uniform(0.0, 1.0, size=(n_candidates, n_dims))
        ).double()

        # Compute acquisition function values
        acq_values = self._compute_acquisition_function(
            candidates, kernel_params, search_space, X_train, Y_train, y_samples
        )

        # Find the best candidate
        best_idx = torch.argmax(acq_values)
        best_candidate = candidates[best_idx]

        return best_candidate.numpy()

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        """Sample hyperparameter values using GP-based acquisition function optimization.

        Args:
            study: Optuna study object containing trial history.
            trial: Current frozen trial for conditional parameter sampling.
            search_space: Dictionary mapping parameter names to their distributions.

        Returns:
            Dictionary mapping parameter names to sampled values.
        """
        if search_space == {}:
            return {}

        states = (TrialState.COMPLETE,)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)

        if len(trials) < self._n_startup_trials:
            return {}

        # Convert to internal representation
        (
            internal_search_space,
            normalized_params,
        ) = gp_search_space.get_search_space_and_normalized_params(trials, search_space)

        # Use Optuna's exact approach for value transformation and standardization
        # Optuna transforms based on study direction: maximize -> keep, minimize -> negate
        raw_values = np.array(
            [trial.values[0] for trial in trials]
        )  # Single objective for now

        # Apply Optuna's sign transformation to convert to maximization internally
        # This matches Optuna's GPSampler approach exactly
        _sign = 1.0 if self._maximize else -1.0
        transformed_values = _sign * raw_values

        standardized_values, _, _ = _standardize_values(
            transformed_values.reshape(-1, 1)
        )

        # Clear cache if search space changes
        if self._kernel_params_cache_list is not None and len(
            self._kernel_params_cache_list[0].inverse_squared_lengthscales
        ) != len(internal_search_space.scale_types):
            self._kernel_params_cache_list = None

        # Fit GP model
        is_categorical = (
            internal_search_space.scale_types == gp_search_space.ScaleType.CATEGORICAL
        )
        cache = (
            self._kernel_params_cache_list[0]
            if self._kernel_params_cache_list is not None
            else None
        )

        kernel_params = gp.fit_kernel_params(
            X=normalized_params,
            Y=standardized_values[:, 0],
            is_categorical=is_categorical,
            log_prior=self._log_prior,
            minimum_noise=self._minimum_noise,
            initial_kernel_params=cache,
            deterministic_objective=self._deterministic,
        )

        self._kernel_params_cache_list = [kernel_params]

        # Convert to tensors for acquisition function optimization
        X_train = torch.from_numpy(normalized_params).double()
        Y_train = torch.from_numpy(standardized_values[:, 0]).double()

        # Create acquisition function parameters using Optuna's approach
        if (
            self._acquisition_function
            == ExpandedAcquisitionFunction.LOG_EXPECTED_IMPROVEMENT
        ):
            acqf_params = acqf.create_acqf_params(
                acqf_type=acqf.AcquisitionFunctionType.LOG_EI,
                kernel_params=kernel_params,
                search_space=internal_search_space,
                X=normalized_params,
                Y=standardized_values[:, 0],
            )
            # Use simplified optimization (no warm start needed)
            normalized_param = self._optimize_acquisition_function(acqf_params, None)
        else:
            # For other acquisition functions, fall back to custom implementation
            # (This maintains backward compatibility with ccqr_optimization-specific functions)
            normalized_param = self._optimize_acquisition_function_custom(
                kernel_params,
                internal_search_space,
                X_train,
                Y_train,
                transformed_values,
            )
            return gp_search_space.get_unnormalized_param(
                search_space, normalized_param
            )

        # Convert back to external representation
        return gp_search_space.get_unnormalized_param(search_space, normalized_param)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        """Sample a single parameter independently."""
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        """Called before each trial."""
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        """Called after each trial."""
        self._independent_sampler.after_trial(study, trial, state, values)
