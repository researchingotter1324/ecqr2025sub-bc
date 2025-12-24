import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.cluster import KMeans
import statsmodels.api as sm
from stratification_utils import (
    get_benchmark_task_ids,
    sample_benchmark_data,
    validate_dataset,
    select_top_datasets,
    save_stratification,
)
import random

random.seed(42)
np.random.seed(42)

BENCHMARKS = ["rbv2_aknn", "lcbench"]
TOP_COUNT = 5
TOP_PERCENT = None
MAX_PERFECT_ACC_RATIO = 0.05
MIN_RUNTIME = 8


class LightweightGP:
    """Lightweight GP with inducing points for scalability."""

    def __init__(self, n_inducing: int = 100, random_state: int = 42):
        self.n_inducing = n_inducing
        self.random_state = random_state
        self.X_inducing_ = None
        self.y_inducing_ = None
        self.gp_ = None
        self.scaler_X_ = None
        self.scaler_y_ = None

    def _select_inducing_points(self, X: np.ndarray, y: np.ndarray):
        """Select inducing points using K-means clustering."""
        if len(X) <= self.n_inducing:
            return X, y

        np.random.seed(self.random_state)
        kmeans = KMeans(
            n_clusters=self.n_inducing, random_state=self.random_state, n_init=10
        )
        cluster_labels = kmeans.fit_predict(X)

        inducing_indices = []
        for i in range(self.n_inducing):
            cluster_mask = cluster_labels == i
            if np.any(cluster_mask):
                cluster_X = X[cluster_mask]
                centroid = kmeans.cluster_centers_[i]
                distances = np.sum((cluster_X - centroid) ** 2, axis=1)
                closest_idx = np.argmin(distances)
                original_idx = np.where(cluster_mask)[0][closest_idx]
                inducing_indices.append(original_idx)

        return X[inducing_indices], y[inducing_indices]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit lightweight GP with inducing points and ARD Matern kernel."""
        self.scaler_X_ = StandardScaler()
        self.scaler_y_ = StandardScaler()

        X_scaled = self.scaler_X_.fit_transform(X)
        y_scaled = self.scaler_y_.fit_transform(y.reshape(-1, 1)).ravel()

        self.X_inducing_, self.y_inducing_ = self._select_inducing_points(
            X_scaled, y_scaled
        )

        n_features = X.shape[1]
        length_scales = np.ones(n_features)
        kernel = C(1.0, (1e-3, 1e3)) * Matern(
            length_scale=length_scales, length_scale_bounds=(1e-2, 1e2), nu=2.5
        )

        self.gp_ = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            normalize_y=False,
            n_restarts_optimizer=5,
            random_state=self.random_state,
        )

        self.gp_.fit(self.X_inducing_, self.y_inducing_)
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        """Make predictions using the fitted GP."""
        X_scaled = self.scaler_X_.transform(X)

        if return_std:
            y_pred_scaled, y_std_scaled = self.gp_.predict(X_scaled, return_std=True)
            y_pred = self.scaler_y_.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).ravel()
            y_std = y_std_scaled * self.scaler_y_.scale_
            return y_pred, y_std
        else:
            y_pred_scaled = self.gp_.predict(X_scaled)
            y_pred = self.scaler_y_.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).ravel()
            return y_pred


def calculate_heteroscedasticity_score(
    X: np.ndarray, y: np.ndarray, max_samples: int = 200
) -> float:
    """Calculate heteroscedasticity using lightweight GP with inducing points."""
    if len(X) == 0 or len(y) == 0:
        return 0.0

    n_inducing = min(max_samples, len(X))
    gp = LightweightGP(n_inducing=n_inducing, random_state=42)
    gp.fit(X, y)

    predictions, pred_std = gp.predict(X, return_std=True)
    residuals = y - predictions

    if np.any(pred_std > 0):
        standardized_residuals = residuals / (pred_std + 1e-10)
        squared_residuals = standardized_residuals**2
    else:
        squared_residuals = residuals**2

    Z = sm.add_constant(X)
    aux_model = sm.OLS(squared_residuals, Z)
    aux_results = aux_model.fit()

    n = len(squared_residuals)
    k = Z.shape[1] - 1
    r_squared = aux_results.rsquared

    if n > k + 1:
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
    else:
        r_squared_adj = 0.0

    return max(0.0, min(1.0, r_squared_adj))


def create_heteroscedasticity_stratification(
    benchmark_name: str,
    task_ids: list,
    top_count: int = None,
    top_percent: float = None,
    max_perfect_acc_ratio: float = 0.01,
) -> list:
    """Create stratification based on highest heteroscedasticity datasets."""
    scores = {}
    for task_id in task_ids:
        tabularized_configurations, accuracies, runtimes = sample_benchmark_data(
            benchmark_name=benchmark_name, task_id=task_id
        )

        if validate_dataset(
            accuracies=accuracies,
            runtimes=runtimes,
            max_perfect_acc_ratio=max_perfect_acc_ratio,
            min_avg_runtime=MIN_RUNTIME,
        ):
            score = calculate_heteroscedasticity_score(
                X=tabularized_configurations, y=accuracies
            )
            if score > 0:
                scores[task_id] = score

    return select_top_datasets(
        scores=scores, top_count=top_count, top_percent=top_percent
    )


def main():
    for benchmark in BENCHMARKS:
        task_ids = get_benchmark_task_ids(benchmark_name=benchmark)
        top_heteroscedastic = create_heteroscedasticity_stratification(
            benchmark_name=benchmark,
            task_ids=task_ids,
            top_count=TOP_COUNT,
            top_percent=TOP_PERCENT,
            max_perfect_acc_ratio=MAX_PERFECT_ACC_RATIO,
        )

        if top_heteroscedastic:
            output_file = f"top_heteroscedastic_datasets_{benchmark}.json"
            save_stratification(task_ids=top_heteroscedastic, output_file=output_file)


if __name__ == "__main__":
    main()
