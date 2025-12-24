import logging
import copy
import numpy as np
from typing import Optional, Union, Tuple
from scipy.linalg import solve_triangular, cholesky, LinAlgError
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    ExpSineSquared,
    ConstantKernel as C,
    Kernel,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class GPEstimator:
    """Gaussian Process estimator for hyperparameter optimization.

    This class is adapted from the QuantileGP class but focuses on providing
    posterior mean and variance for acquisition function optimization rather
    than quantile estimation. Features ARD support and robust numerical handling.

    Args:
        kernel: GP kernel specification. Accepts string names ("rbf", "matern",
            "rational_quadratic", "exp_sine_squared") with sensible defaults, or
            custom Kernel objects. Defaults to Matern(nu=2.5).
        noise_variance: Explicit noise variance. If "optimize", will be learned.
            If numeric, uses fixed value. Default is "optimize".
        alpha: Regularization parameter for numerical stability. Range: [1e-12, 1e-6].
        n_restarts_optimizer: Number of restarts for hyperparameter optimization.
        random_state: Seed for reproducible optimization and prediction.
        batch_size: Batch size for prediction to manage memory usage.
        optimize_hyperparameters: Whether to optimize kernel hyperparameters.
        prior_lengthscale_concentration: For future custom optimization (unused).
        prior_lengthscale_rate: For future custom optimization (unused).
        prior_noise_concentration: For future custom optimization (unused).
        prior_noise_rate: For future custom optimization (unused).
    """

    def __init__(
        self,
        kernel: Optional[Union[str, Kernel]] = None,
        noise_variance: Optional[Union[str, float]] = "optimize",
        alpha: float = 1e-10,
        n_restarts_optimizer: int = 5,
        random_state: Optional[int] = None,
        batch_size: Optional[int] = None,
        optimize_hyperparameters: bool = True,
        prior_lengthscale_concentration: float = 2.0,
        prior_lengthscale_rate: float = 1.0,
        prior_noise_concentration: float = 1.1,
        prior_noise_rate: float = 30.0,
    ):
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        self.batch_size = batch_size
        self.optimize_hyperparameters = optimize_hyperparameters
        self.prior_lengthscale_concentration = prior_lengthscale_concentration
        self.prior_lengthscale_rate = prior_lengthscale_rate
        self.prior_noise_concentration = prior_noise_concentration
        self.prior_noise_rate = prior_noise_rate

        # Fitted attributes
        self.X_train_ = None
        self.y_train_ = None
        self.kernel_ = None
        self.noise_variance_ = None
        self.chol_factor_ = None
        self.alpha_ = None
        self.y_train_mean_ = None
        self.y_train_std_ = None
        self.is_fitted_ = False
        # Eigendecomposition fallback attributes
        self.eigenvals_ = None
        self.eigenvecs_ = None
        # Feature normalization attributes
        self.feature_scaler_ = None

    def _get_kernel_object(
        self,
        kernel_spec: Optional[Union[str, Kernel]] = None,
        n_features: Optional[int] = None,
    ) -> Kernel:
        """Convert kernel specification to scikit-learn kernel object with ARD support.

        Creates kernels with per-feature length scales for Automatic Relevance
        Determination (ARD). This allows the model to automatically learn the
        importance of each feature by optimizing individual length scales.

        Args:
            kernel_spec: Kernel specification (string name, kernel object, or None).
            n_features: Number of features for ARD initialization. If None, uses scalar length scale.

        Returns:
            Scikit-learn kernel object with proper ARD bounds for optimization.

        Raises:
            ValueError: If unknown kernel name provided or invalid kernel type.
        """
        # Initialize length scale for ARD
        if n_features is not None and n_features > 1:
            # ARD: one length scale per feature
            length_scale = np.ones(n_features)
            length_scale_bounds = (1e-2, 1e2)
        else:
            # Scalar length scale for single feature or unspecified
            length_scale = 1.0
            length_scale_bounds = (1e-2, 1e2)

        # Default to Matern kernel with ARD
        if kernel_spec is None:
            return C(1.0, (1e-3, 1e3)) * Matern(
                length_scale=length_scale,
                length_scale_bounds=length_scale_bounds,
                nu=2.5,
            )

        # String specifications with ARD support
        elif isinstance(kernel_spec, str):
            kernel_map = {
                "rbf": C(1.0, (1e-3, 1e3))
                * RBF(
                    length_scale=length_scale, length_scale_bounds=length_scale_bounds
                ),
                "matern": C(1.0, (1e-3, 1e3))
                * Matern(
                    length_scale=length_scale,
                    length_scale_bounds=length_scale_bounds,
                    nu=2.5,
                ),
                "rational_quadratic": C(1.0, (1e-3, 1e3))
                * RationalQuadratic(
                    length_scale=length_scale,
                    length_scale_bounds=length_scale_bounds,
                    alpha=1.0,
                    alpha_bounds=(1e-3, 1e3),
                ),
                "exp_sine_squared": C(1.0, (1e-3, 1e3))
                * ExpSineSquared(
                    length_scale=length_scale,
                    length_scale_bounds=length_scale_bounds,
                    periodicity=1.0,
                    periodicity_bounds=(1e-2, 1e2),
                ),
            }

            if kernel_spec not in kernel_map:
                raise ValueError(f"Unknown kernel name: {kernel_spec}")
            return kernel_map[kernel_spec]

        # Kernel object - make a deep copy for safety
        elif isinstance(kernel_spec, Kernel):
            return copy.deepcopy(kernel_spec)

        else:
            raise ValueError(
                f"Kernel must be a string name, Kernel object, or None. Got: {type(kernel_spec)}"
            )

    def _optimize_hyperparameters(self) -> None:
        """Optimize kernel hyperparameters and noise variance using sklearn's optimization."""
        if not self.optimize_hyperparameters:
            return

        # Determine alpha value for optimization
        # If noise_variance is "optimize", use a small alpha and let GP optimize noise
        # If noise_variance is fixed, use it as alpha
        if self.noise_variance == "optimize":
            alpha_for_opt = self.alpha  # Small regularization only
        else:
            alpha_for_opt = self.noise_variance_ + self.alpha

        # Use sklearn's GaussianProcessRegressor for hyperparameter optimization
        # This provides robust optimization with proper parameter mapping
        temp_gp = GaussianProcessRegressor(
            kernel=self.kernel_,
            alpha=alpha_for_opt,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
            normalize_y=False,  # We handle normalization ourselves
        )

        try:
            temp_gp.fit(self.X_train_, self.y_train_)

            # Extract optimized kernel
            self.kernel_ = temp_gp.kernel_

            # Extract optimized noise variance if it was being optimized
            if self.noise_variance == "optimize":
                # sklearn's alpha includes both noise and regularization
                # Extract the optimized noise component
                self.noise_variance_ = max(temp_gp.alpha - self.alpha, 1e-10)

        except Exception as e:
            logging.warning(
                f"Hyperparameter optimization failed: {e}, using default parameters"
            )
            # Keep the original kernel and noise variance if optimization fails

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPEstimator":
        """Fit Gaussian process with proper hyperparameter optimization.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        # Normalize input features
        self.feature_scaler_ = StandardScaler()
        self.X_train_ = self.feature_scaler_.fit_transform(X)

        # Normalize targets
        self.y_train_mean_ = np.mean(y)
        self.y_train_std_ = np.std(y)
        if self.y_train_std_ < 1e-12:
            self.y_train_std_ = 1.0
        self.y_train_ = (y - self.y_train_mean_) / self.y_train_std_

        # Initialize kernel with ARD support
        n_features = X.shape[1]
        self.kernel_ = self._get_kernel_object(self.kernel, n_features)

        # Set noise variance
        if isinstance(self.noise_variance, (int, float)):
            self.noise_variance_ = self.noise_variance
        else:
            self.noise_variance_ = 1e-6  # Default, will be optimized if needed

        # Optimize hyperparameters
        self._optimize_hyperparameters()

        # Fit the model with optimized parameters
        self._fit_gp()

        self.is_fitted_ = True

        return self

    def _fit_gp(self) -> None:
        """Fit GP with current hyperparameters using robust Cholesky decomposition."""
        # Compute kernel matrix
        K = self.kernel_(self.X_train_)

        # Add noise and regularization
        K += (self.noise_variance_ + self.alpha) * np.eye(len(self.X_train_))

        # Robust Cholesky decomposition with progressive regularization
        regularization_levels = [0, 1e-8, 1e-6, 1e-4, 1e-3]

        for reg in regularization_levels:
            try:
                K_reg = K + reg * np.eye(len(self.X_train_)) if reg > 0 else K
                self.chol_factor_ = cholesky(K_reg, lower=True)
                if reg > 0:
                    logging.warning(
                        f"Added regularization {reg} for numerical stability"
                    )
                break
            except LinAlgError:
                if reg == regularization_levels[-1]:
                    # Final fallback: use eigendecomposition for very ill-conditioned matrices
                    logging.warning(
                        "Cholesky failed, using eigendecomposition fallback"
                    )
                    self._fit_gp_eigendecomp(K)
                    return
                continue

        # Solve for alpha using Cholesky decomposition
        self.alpha_ = solve_triangular(self.chol_factor_, self.y_train_, lower=True)

    def _fit_gp_eigendecomp(self, K: np.ndarray) -> None:
        """Fallback GP fitting using eigendecomposition for ill-conditioned matrices."""
        # Eigendecomposition of kernel matrix
        eigenvals, eigenvecs = np.linalg.eigh(K)

        # Clip negative eigenvalues and add regularization
        eigenvals = np.maximum(eigenvals, 1e-12)

        # Reconstruct with regularized eigenvalues
        eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        # Use pseudo-inverse for fitting
        try:
            K_inv = eigenvecs @ np.diag(1.0 / eigenvals) @ eigenvecs.T
            self.alpha_ = K_inv @ self.y_train_
            # Store decomposition for prediction
            self.eigenvals_ = eigenvals
            self.eigenvecs_ = eigenvecs
            self.chol_factor_ = None  # Signal to use eigendecomp in prediction
        except Exception as e:
            raise RuntimeError(f"Both Cholesky and eigendecomposition failed: {e}")

    def predict(
        self, X: np.ndarray, return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Generate predictions using GP posterior.

        Args:
            X: Features for prediction with shape (n_samples, n_features).
            return_std: Whether to return standard deviation along with mean.

        Returns:
            If return_std is False: mean predictions with shape (n_samples,).
            If return_std is True: tuple of (mean, std) with shapes (n_samples,) each.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")

        # Normalize input features using the same scaler from training
        X_scaled = self.feature_scaler_.transform(X)

        if self.batch_size is not None and len(X) > self.batch_size:
            results = []
            for i in range(0, len(X), self.batch_size):
                batch_X = X_scaled[i : i + self.batch_size]
                if return_std:
                    batch_mean, batch_std = self._predict_batch(
                        batch_X, return_std=True
                    )
                    results.append((batch_mean, batch_std))
                else:
                    batch_result = self._predict_batch(batch_X, return_std=False)
                    results.append(batch_result)

            if return_std:
                means, stds = zip(*results)
                result = (np.concatenate(means), np.concatenate(stds))
            else:
                result = np.concatenate(results)
        else:
            result = self._predict_batch(X_scaled, return_std=return_std)

        return result

    def _predict_batch(
        self, X: np.ndarray, return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Compute predictions for a batch of inputs."""
        # Get mean and variance from GP
        y_mean, y_var = self._predict_mean_var(X)

        if return_std:
            y_std = np.sqrt(y_var)
            return y_mean, y_std
        else:
            return y_mean

    def _predict_mean_var(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance using Cholesky or eigendecomposition.

        Args:
            X: Features with shape (n_samples, n_features).

        Returns:
            Tuple of (y_mean, y_var) with shapes (n_samples,) each.
        """
        # Compute kernel between test and training points
        K_star = self.kernel_(X, self.X_train_)

        if self.chol_factor_ is not None:
            # Use Cholesky-based computation
            chol_solve = solve_triangular(self.chol_factor_, K_star.T, lower=True)
            y_mean = chol_solve.T @ self.alpha_

            # Compute variance (in normalized space)
            K_star_star = self.kernel_.diag(X)
            y_var = K_star_star - np.sum(chol_solve**2, axis=0)

        else:
            # Use eigendecomposition fallback
            y_mean = K_star @ self.alpha_

            # Compute variance using eigendecomposition
            K_star_star = self.kernel_.diag(X)
            # K^{-1} = V * Λ^{-1} * V^T
            K_inv_K_star = (
                self.eigenvecs_
                @ (K_star.T / self.eigenvals_.reshape(-1, 1))
                @ self.eigenvecs_.T
            )
            y_var = K_star_star - np.sum(K_star * K_inv_K_star.T, axis=1)

        # Denormalize mean
        y_mean = y_mean * self.y_train_std_ + self.y_train_mean_

        # Ensure non-negative variance before denormalization
        y_var = np.maximum(y_var, 1e-12)

        # Denormalize variance (transforms from normalized to original scale)
        y_var *= self.y_train_std_**2

        # Add noise variance in original scale for total predictive variance
        y_var += self.noise_variance_ * self.y_train_std_**2

        return y_mean, y_var

    def sample_posterior(
        self, X: np.ndarray, n_samples: int = 1, random_state: int = None
    ) -> np.ndarray:
        """Sample functions from the GP posterior distribution.

        Args:
            X: Features with shape (n_samples, n_features).
            n_samples: Number of function samples to draw.
            random_state: Random seed for reproducible sampling.

        Returns:
            Function samples with shape (n_samples, n_points).
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before sampling")

        # Set random state
        if random_state is not None:
            np.random.seed(random_state)

        # Normalize input features using the same scaler from training
        X_scaled = self.feature_scaler_.transform(X)

        # Compute mean and covariance
        y_mean, y_var = self._predict_mean_var(X_scaled)

        # Compute full covariance matrix
        K_star_star = self.kernel_(X_scaled)
        K_star = self.kernel_(X_scaled, self.X_train_)

        if self.chol_factor_ is not None:
            # Use Cholesky-based computation for covariance
            chol_solve = solve_triangular(self.chol_factor_, K_star.T, lower=True)
            cov = K_star_star - chol_solve.T @ chol_solve
        else:
            # Use eigendecomposition fallback
            K_inv_K_star = (
                self.eigenvecs_
                @ (K_star.T / self.eigenvals_.reshape(-1, 1))
                @ self.eigenvecs_.T
            )
            cov = K_star_star - K_star @ K_inv_K_star

        # Add noise variance for total predictive covariance
        cov += (self.noise_variance_ * self.y_train_std_**2) * np.eye(len(X))

        # Ensure positive semi-definite
        cov = (cov + cov.T) / 2  # Make symmetric
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        eigenvals = np.maximum(eigenvals, 1e-10)  # Clip negative eigenvalues
        cov = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        # Sample from multivariate normal
        samples = np.random.multivariate_normal(y_mean, cov, size=n_samples)

        return samples
