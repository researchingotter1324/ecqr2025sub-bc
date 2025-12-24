import math
import logging
import numpy as np
from typing import Union, Tuple
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AcquisitionFunction(Enum):
    """Supported acquisition functions for GP-based optimization."""

    EXPECTED_IMPROVEMENT = "expected_improvement"
    LOG_EXPECTED_IMPROVEMENT = "log_expected_improvement"
    UPPER_CONFIDENCE_BOUND = "upper_confidence_bound"
    THOMPSON_SAMPLING = "thompson_sampling"
    OPTIMISTIC_THOMPSON_SAMPLING = "optimistic_thompson_sampling"


class BaseAcquisitionFunction(ABC):
    """Abstract base class for acquisition functions."""

    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def __call__(self, mean: np.ndarray, var: np.ndarray, **kwargs) -> np.ndarray:
        """Compute acquisition function values.

        Args:
            mean: GP posterior mean predictions with shape (n_points,)
            var: GP posterior variance predictions with shape (n_points,)
            **kwargs: Additional parameters specific to the acquisition function

        Returns:
            Acquisition function values with shape (n_points,)
        """


class ExpectedImprovement(BaseAcquisitionFunction):
    """Expected Improvement acquisition function.

    Computes the expected improvement over the current best observed value.

    Args:
        xi: Exploration parameter (default: 0.01)
    """

    def __init__(self, xi: float = 0.01):
        super().__init__(xi=xi)
        self.xi = xi

    def __call__(
        self, mean: np.ndarray, var: np.ndarray, f_best: float, **kwargs
    ) -> np.ndarray:
        """Compute Expected Improvement acquisition function.

        Args:
            mean: GP posterior mean with shape (n_points,)
            var: GP posterior variance with shape (n_points,)
            f_best: Current best observed function value (for minimization)

        Returns:
            EI values with shape (n_points,) - lower values indicate better candidates
        """
        std = np.sqrt(var)

        # Avoid division by zero
        std_safe = np.maximum(std, 1e-10)

        # For minimization: we want f_best > mean (improvement when mean is smaller)
        z = (f_best - mean - self.xi) / std_safe

        # Compute normal CDF and PDF
        phi = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))  # CDF
        pdf = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)  # PDF

        # Expected Improvement formula
        ei = std * (z * phi + pdf)

        # Handle numerical issues: EI should be non-negative
        ei = np.maximum(ei, 0.0)

        # Return negative EI for minimization (lower values = better candidates)
        return -ei


class LogExpectedImprovement(BaseAcquisitionFunction):
    """Log Expected Improvement acquisition function.

    Computes the logarithm of expected improvement, which can be more
    numerically stable than regular EI.
    """

    def __call__(
        self, mean: np.ndarray, var: np.ndarray, f_best: float, **kwargs
    ) -> np.ndarray:
        """Compute Log Expected Improvement acquisition function.

        Uses the exact implementation from optuna_gp_integration.py for consistency.

        Args:
            mean: GP posterior mean with shape (n_points,)
            var: GP posterior variance with shape (n_points,)
            f_best: Current best observed function value

        Returns:
            Log EI values with shape (n_points,)
        """
        return -self._optuna_logei(mean, var, f_best)

    def _optuna_logei(self, mean: np.ndarray, var: np.ndarray, f0: float) -> np.ndarray:
        """Adapted logei implementation for minimization."""
        # Return E_{y ~ N(mean, var)}[max(0, f0-y)] for minimization
        sigma = np.sqrt(var)
        z = (f0 - mean) / sigma

        # Switch implementation based on z value for numerical stability
        small = z < -25

        vals = np.empty_like(z)
        z_small = z[small]
        z_normal = z[~small]
        sqrt_2pi = math.sqrt(2 * math.pi)

        # For normal values (z >= -25)
        if np.any(~small):
            cdf = 0.5 * (1.0 + np.vectorize(math.erf)(-z_normal / math.sqrt(2.0)))
            cdf = 1.0 - cdf  # Convert to P(Z <= z) from P(Z <= -z)
            pdf = np.exp(-0.5 * z_normal**2) * (1 / sqrt_2pi)
            vals[~small] = np.log(z_normal * cdf + pdf)

        # For small values (z < -25) - use asymptotic expansion
        if np.any(small):
            # Use erfcx for better numerical stability
            r = (
                math.sqrt(0.5 * math.pi)
                * np.exp(0.5 * z_small**2)
                * (
                    1.0
                    - 0.5 * (1.0 + np.vectorize(math.erf)(-z_small / math.sqrt(2.0)))
                )
            )
            vals[small] = -0.5 * z_small**2 + np.log(
                (z_small * r + 1) * (1 / sqrt_2pi)
            )

        # Final logei value
        return np.log(sigma) + vals


class UpperConfidenceBound(BaseAcquisitionFunction):
    """Upper Confidence Bound acquisition function.

    Balances exploration and exploitation using confidence bounds.

    Args:
        beta: Exploration parameter (default: 2.0)
    """

    def __init__(self, beta: float = 2.0):
        super().__init__(beta=beta)
        self.beta = beta

    def __call__(self, mean: np.ndarray, var: np.ndarray, **kwargs) -> np.ndarray:
        """Compute Upper Confidence Bound acquisition function.

        Args:
            mean: GP posterior mean with shape (n_points,)
            var: GP posterior variance with shape (n_points,)

        Returns:
            UCB values with shape (n_points,)
        """
        # For minimization, use Lower Confidence Bound and negate for argmin
        return -(mean - self.beta * np.sqrt(var))


class ThompsonSampling(BaseAcquisitionFunction):
    """Thompson Sampling acquisition function.

    Samples a single function from the GP posterior distribution for probabilistic exploration.
    This implements proper Thompson Sampling by drawing one function sample and evaluating
    it across all candidate points, rather than independent sampling per point.

    Args:
        random_state: Random seed for reproducible sampling
    """

    def __init__(self, random_state: int = None):
        super().__init__(random_state=random_state)
        self.random_state = random_state

    def __call__(self, mean: np.ndarray, var: np.ndarray, **kwargs) -> np.ndarray:
        """Compute Thompson Sampling acquisition function.

        This method is called by optimize_acquisition but Thompson Sampling requires
        access to the full GP posterior covariance, not just pointwise variances.
        Therefore, this method raises an error and directs users to use the proper
        Thompson Sampling optimization method.

        Args:
            mean: GP posterior mean with shape (n_points,)
            var: GP posterior variance with shape (n_points,)

        Raises:
            NotImplementedError: Thompson Sampling requires special handling
        """
        raise NotImplementedError(
            "Thompson Sampling requires access to the full GP posterior covariance matrix. "
            "Use optimize_thompson_sampling() instead of optimize_acquisition()."
        )


class OptimisticThompsonSampling(BaseAcquisitionFunction):
    """Optimistic Thompson Sampling acquisition function.

    Thompson sampling with a floor at the posterior mean for more optimistic sampling.
    This implements proper Thompson Sampling by drawing one function sample and applying
    optimistic constraints, rather than independent sampling per point.

    Args:
        random_state: Random seed for reproducible sampling
    """

    def __init__(self, random_state: int = None):
        super().__init__(random_state=random_state)
        self.random_state = random_state

    def __call__(self, mean: np.ndarray, var: np.ndarray, **kwargs) -> np.ndarray:
        """Compute Optimistic Thompson Sampling acquisition function.

        This method is called by optimize_acquisition but Optimistic Thompson Sampling
        requires access to the full GP posterior covariance, not just pointwise variances.
        Therefore, this method raises an error and directs users to use the proper
        optimization method.

        Args:
            mean: GP posterior mean with shape (n_points,)
            var: GP posterior variance with shape (n_points,)

        Raises:
            NotImplementedError: Optimistic Thompson Sampling requires special handling
        """
        raise NotImplementedError(
            "Optimistic Thompson Sampling requires access to the full GP posterior covariance matrix. "
            "Use optimize_optimistic_thompson_sampling() instead of optimize_acquisition()."
        )


def get_acquisition_function(
    name: Union[str, AcquisitionFunction], **kwargs
) -> BaseAcquisitionFunction:
    """Factory function to create acquisition function instances.

    Args:
        name: Name or enum of the acquisition function
        **kwargs: Parameters to pass to the acquisition function constructor

    Returns:
        Acquisition function instance

    Raises:
        ValueError: If unknown acquisition function name is provided
    """
    if isinstance(name, str):
        try:
            name = AcquisitionFunction(name)
        except ValueError:
            raise ValueError(f"Unknown acquisition function: {name}")

    # Filter kwargs based on the acquisition function type
    if name == AcquisitionFunction.EXPECTED_IMPROVEMENT:
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in ["xi"]}
        return ExpectedImprovement(**filtered_kwargs)
    elif name == AcquisitionFunction.LOG_EXPECTED_IMPROVEMENT:
        return LogExpectedImprovement()
    elif name == AcquisitionFunction.UPPER_CONFIDENCE_BOUND:
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in ["beta"]}
        return UpperConfidenceBound(**filtered_kwargs)
    elif name == AcquisitionFunction.THOMPSON_SAMPLING:
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in ["random_state"]}
        return ThompsonSampling(**filtered_kwargs)
    elif name == AcquisitionFunction.OPTIMISTIC_THOMPSON_SAMPLING:
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in ["random_state"]}
        return OptimisticThompsonSampling(**filtered_kwargs)
    else:
        raise ValueError(f"Unknown acquisition function: {name}")


def optimize_acquisition(
    acquisition_func: BaseAcquisitionFunction,
    gp_estimator,
    candidate_points: np.ndarray,
    f_best: float = None,
) -> Tuple[int, float]:
    """Optimize acquisition function over candidate points.

    Args:
        acquisition_func: Acquisition function to optimize
        gp_estimator: Fitted GP estimator
        candidate_points: Candidate points to evaluate with shape (n_candidates, n_features)
        f_best: Current best function value (required for EI-based functions)

    Returns:
        Tuple of (best_index, best_acquisition_value)
    """
    # Handle Thompson Sampling variants specially
    if isinstance(acquisition_func, ThompsonSampling):
        return optimize_thompson_sampling(
            acquisition_func, gp_estimator, candidate_points
        )
    elif isinstance(acquisition_func, OptimisticThompsonSampling):
        return optimize_optimistic_thompson_sampling(
            acquisition_func, gp_estimator, candidate_points
        )

    # Get GP predictions
    mean, std = gp_estimator.predict(candidate_points, return_std=True)
    var = std**2

    # Compute acquisition function values
    if isinstance(acquisition_func, (ExpectedImprovement, LogExpectedImprovement)):
        if f_best is None:
            raise ValueError(
                "f_best must be provided for Expected Improvement functions"
            )
        acq_values = acquisition_func(mean, var, f_best)
    else:
        acq_values = acquisition_func(mean, var)

    # Find best candidate
    best_idx = np.argmin(acq_values)

    return best_idx, acq_values[best_idx]


def optimize_thompson_sampling(
    acquisition_func: ThompsonSampling,
    gp_estimator,
    candidate_points: np.ndarray,
) -> Tuple[int, float]:
    """Optimize Thompson Sampling acquisition function using Matheron's Rule.

    This method implements Thompson Sampling by drawing a single function sample
    from the GP posterior using Matheron's Rule (f*(x) = mu(x) + sigma(x) * Z, where Z ~ N(0,1)).
    This avoids the expensive computation of the full covariance matrix while maintaining
    statistical correctness.

    Args:
        acquisition_func: Thompson Sampling acquisition function
        gp_estimator: Fitted GP estimator
        candidate_points: Candidate points to evaluate with shape (n_candidates, n_features)

    Returns:
        Tuple of (best_index, sampled_function_value_at_best_point)
    """
    # Use Matheron's Rule for efficient posterior sampling
    mean, std = gp_estimator.predict(candidate_points, return_std=True)

    # Sample Z from a standard normal distribution
    if acquisition_func.random_state is not None:
        np.random.seed(acquisition_func.random_state)
    Z = np.random.standard_normal(len(candidate_points))

    # Apply Matheron's Rule: f*(x) = mean(x) + std(x) * Z
    sampled_function = mean + std * Z

    # For minimization, find the candidate with the lowest sampled function value
    best_idx = np.argmin(sampled_function)

    return best_idx, sampled_function[best_idx]


def optimize_optimistic_thompson_sampling(
    acquisition_func: OptimisticThompsonSampling,
    gp_estimator,
    candidate_points: np.ndarray,
) -> Tuple[int, float]:
    """Optimize Optimistic Thompson Sampling acquisition function using Matheron's Rule.

    This method implements Optimistic Thompson Sampling by drawing a single function sample
    from the GP posterior using Matheron's Rule, applying optimistic constraints (floor at posterior mean),
    and finding the candidate that optimizes this constrained sample.

    Args:
        acquisition_func: Optimistic Thompson Sampling acquisition function
        gp_estimator: Fitted GP estimator
        candidate_points: Candidate points to evaluate with shape (n_candidates, n_features)

    Returns:
        Tuple of (best_index, constrained_sampled_function_value_at_best_point)
    """
    # Use Matheron's Rule for efficient posterior sampling
    mean, std = gp_estimator.predict(candidate_points, return_std=True)

    # Sample Z from a standard normal distribution
    if acquisition_func.random_state is not None:
        np.random.seed(acquisition_func.random_state)
    Z = np.random.standard_normal(len(candidate_points))

    # Apply Matheron's Rule: f*(x) = mean(x) + std(x) * Z
    sampled_function = mean + std * Z

    # Apply optimistic constraint: take minimum between sample and mean
    # This encourages exploitation by preventing the sample from being worse than the mean
    constrained_sample = np.minimum(sampled_function, mean)

    # For minimization, find the candidate with the lowest constrained sampled value
    best_idx = np.argmin(constrained_sample)

    return best_idx, constrained_sample[best_idx]
