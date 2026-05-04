import json
import numpy as np
import os
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import warnings

from yahpo_gym import BenchmarkSet, local_config
from ConfigSpace import Configuration
import ConfigSpace as CS
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

local_config.init_config()
local_config.set_data_path("yahpo_bench_data")


def get_benchmark_task_ids(benchmark_name: str) -> List[str]:
    benchmark_set = BenchmarkSet(benchmark_name)
    return benchmark_set.instances


def get_yahpo_log_info(benchmark: str) -> Set[str]:
    """Return the set of hyperparameter names that use a log scale.

    Args:
        benchmark: YAHPO benchmark scenario name.

    Returns:
        Set of parameter names declared as log=true in config_space.json.
    """
    config_path = os.path.join("yahpo_bench_data", benchmark, "config_space.json")
    log_params = set()
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_data = json.load(f)
        for hp in config_data.get("hyperparameters", []):
            if hp.get("log", False) and hp.get("name"):
                log_params.add(hp["name"])
    return log_params


def get_filtered_configuration(
    configuration: dict,
    config_space: CS.ConfigurationSpace,
    fidelity_space: Dict[str, Any],
    instance_name: str,
    instance_value: Any,
) -> dict:
    config_dict = configuration.copy()

    if fidelity_space:
        config_dict.update(fidelity_space)

    config_dict[instance_name] = instance_value

    cs_config = Configuration(
        config_space,
        values=config_dict,
        allow_inactive_with_values=True,
    )
    active_hyperparameters = config_space.get_active_hyperparameters(cs_config)

    return {
        k: v
        for k, v in config_dict.items()
        if k in active_hyperparameters or k == instance_name
    }


def preprocess_configurations(
    configs: List[Dict],
    config_space: CS.ConfigurationSpace,
    fidelity_space: Dict[str, Any],
    log_params: Set[str],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Featurise configurations into numeric and categorical arrays, dropping constant columns.

    Numeric HPs are extracted in sorted name order and log-transformed where
    applicable. Categorical HPs are extracted as raw strings in sorted name order.
    Any column (numeric or categorical) that is constant across all rows is
    dropped immediately, as it carries no information for either regression or
    distance calculations.

    Args:
        configs: List of hyperparameter configuration dicts.
        config_space: The ConfigurationSpace for the benchmark.
        fidelity_space: Dict of fidelity parameter names to their fixed values.
            These parameters are excluded from the feature matrices.
        log_params: Set of hyperparameter names that should be log-transformed.

    Returns:
        X_num: Float64 matrix of shape (n, n_numeric_active). Log-transformed
            where applicable. Constant columns dropped. Not normalised.
        X_cat_str: Object array of shape (n, n_cat_active) containing raw string
            category values. Constant columns (single unique value) dropped.
        numeric_names: Names of the retained numeric features, same order as
            columns of X_num.
        categorical_names: Names of the retained categorical features, same
            order as columns of X_cat_str.
    """
    if not configs:
        return np.empty((0, 0)), np.empty((0, 0), dtype=object), [], []

    all_hyperparams = {hp.name: hp for hp in config_space.get_hyperparameters()}

    categorical_candidates = []
    numeric_candidates = []

    for param_name, hp in all_hyperparams.items():
        if param_name in fidelity_space:
            continue
        if isinstance(hp, CS.CategoricalHyperparameter):
            categorical_candidates.append(param_name)
        elif isinstance(hp, (CS.UniformFloatHyperparameter, CS.UniformIntegerHyperparameter)):
            numeric_candidates.append(param_name)

    categorical_candidates = sorted(categorical_candidates)
    numeric_candidates = sorted(numeric_candidates)

    X_num_raw = []
    X_cat_raw = []

    for config in configs:
        numeric_row = []
        for param in numeric_candidates:
            value = config[param]
            if isinstance(value, bool):
                value = int(value)
            if param in log_params and value > 0:
                value = np.log(value)
            numeric_row.append(float(value))
        X_num_raw.append(numeric_row)

        X_cat_raw.append([str(config[param]) for param in categorical_candidates])

    X_num_full = np.array(X_num_raw, dtype=np.float64)
    X_cat_full = np.array(X_cat_raw, dtype=object)

    active_num = []
    numeric_names = []
    for i, name in enumerate(numeric_candidates):
        col = X_num_full[:, i]
        if col.max() - col.min() >= 1e-12:
            active_num.append(i)
            numeric_names.append(name)

    active_cat = []
    categorical_names = []
    for i, name in enumerate(categorical_candidates):
        if len(np.unique(X_cat_full[:, i])) > 1:
            active_cat.append(i)
            categorical_names.append(name)

    X_num = X_num_full[:, active_num] if active_num else np.empty((len(configs), 0))
    X_cat_str = X_cat_full[:, active_cat] if active_cat else np.empty((len(configs), 0), dtype=object)

    return X_num, X_cat_str, numeric_names, categorical_names


def gower_distance_matrix(
    X_num: np.ndarray,
    X_cat_str: np.ndarray,
) -> np.ndarray:
    """Compute the symmetric n×n Gower (1971) distance matrix.

    Gower distance is the mean of per-feature partial distances, each in [0, 1]:

        d_Gower(i, j) = (1 / p) * sum_f  d_f(x_i_f, x_j_f)

    where p is the number of features. Partial distances:

        numeric:     d_f = |x_i - x_j| / range_f  (range computed over all n rows)
        categorical: d_f = 0 if x_i == x_j, else 1

    Numeric normalisation is applied internally. Categorical distance requires no
    encoding — pairwise inequality of string values is computed directly.

    Constant numeric columns must be absent (dropped by preprocess_configurations).

    Why not use the gower PyPI package?
    Both available packages (gower v0.1.2 and gower-multiprocessing v0.2.2) contain
    a documented bug: they divide by max rather than range, producing incorrect
    values when min < 0 (as occurs after log-transforming hyperparameters like
    learning rate). This implementation is correct in all cases.

    Args:
        X_num: Numeric feature matrix, shape (n, n_numeric). Values are raw
            (log-transformed, not normalised). No constant columns.
        X_cat_str: Categorical feature matrix, shape (n, n_cat), dtype object.
            Each column contains string category values. No constant columns.

    Returns:
        D: Symmetric Gower distance matrix, shape (n, n), dtype float32,
            with zeros on the diagonal.
    """
    n = X_num.shape[0] if X_num.shape[0] > 0 else X_cat_str.shape[0]
    n_numeric = X_num.shape[1]
    n_cat = X_cat_str.shape[1]
    p = n_numeric + n_cat

    if p == 0 or n == 0:
        return np.zeros((n, n), dtype=np.float32)

    D = np.zeros((n, n), dtype=np.float32)

    for col in range(n_numeric):
        values = X_num[:, col]
        col_range = values.max() - values.min()
        col_norm = (values - values.min()) / col_range
        D += np.abs(col_norm[:, None] - col_norm[None, :]).astype(np.float32)

    for col in range(n_cat):
        cats = X_cat_str[:, col]
        D += (cats[:, None] != cats[None, :]).astype(np.float32)

    D /= p
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def sample_benchmark_data(
    benchmark_name: str,
    task_id: str,
    n_samples: int = 10000,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], np.ndarray, np.ndarray]:
    """Sample configurations from the benchmark surrogate and return feature arrays.

    Args:
        benchmark_name: YAHPO benchmark scenario name (e.g. "lcbench").
        task_id: Instance identifier within the benchmark.
        n_samples: Number of configurations to sample.

    Returns:
        X_num: Numeric feature matrix, shape (n_valid, n_numeric). Log-transformed
            where applicable, constant columns dropped, not normalised.
        X_cat_str: Categorical feature matrix, shape (n_valid, n_cat), dtype object.
            Raw string values, constant columns dropped.
        numeric_names: Names of retained numeric features.
        categorical_names: Names of retained categorical features.
        performances: Observed performance values, shape (n_valid,).
        runtimes: Observed runtime values, shape (n_valid,).
    """
    local_config.init_config()
    local_config.set_data_path("yahpo_bench_data")

    benchmark_set = BenchmarkSet(
        scenario=benchmark_name, instance=task_id, active_session=False, check=False
    )
    config_space = benchmark_set.get_opt_space(drop_fidelity_params=False, seed=42)
    config_space.seed(42)
    configurations = config_space.sample_configuration(n_samples)

    if not isinstance(configurations, list):
        configurations = [configurations]

    log_params = get_yahpo_log_info(benchmark_name)

    fidelity_param_names = benchmark_set.config.fidelity_params
    instance_names = benchmark_set.config.instance_names

    fidelity_space = {}
    for hyperparameter in config_space.get_hyperparameters():
        if hyperparameter.name in fidelity_param_names:
            if hasattr(hyperparameter, "upper"):
                fidelity_space[hyperparameter.name] = hyperparameter.upper
            elif hasattr(hyperparameter, "default_value"):
                fidelity_space[hyperparameter.name] = hyperparameter.default_value

    if benchmark_name == "lcbench":
        fidelity_space["epoch"] = 50

    config_dicts = []
    filtered_configs = []

    for config in configurations:
        config_dict = dict(config)
        config_dicts.append(config_dict)
        filtered_config = get_filtered_configuration(
            configuration=config_dict,
            config_space=config_space,
            fidelity_space=fidelity_space,
            instance_name=instance_names,
            instance_value=task_id,
        )
        filtered_configs.append(filtered_config)

    batch_results = benchmark_set.objective_function(filtered_configs, seed=1234)

    performances = []
    runtimes = []
    valid_indices = []

    for i, result in enumerate(batch_results):
        if benchmark_name.startswith("rbv2"):
            performance = result.get("acc")
        elif benchmark_name == "lcbench":
            performance = result.get("val_accuracy")
        else:
            performance = result.get("auc")

        if "time" in result:
            runtime = result["time"]
        elif "runtime" in result:
            runtime = result["runtime"]
        else:
            runtime = result["timetrain"] + result["timepredict"]

        is_valid = True
        perf_float = None
        runtime_float = None

        if performance is not None:
            perf_float = float(performance)
            if np.isnan(perf_float) or not np.isfinite(perf_float):
                is_valid = False
        else:
            is_valid = False

        if runtime is not None:
            runtime_float = float(runtime)
            if np.isnan(runtime_float) or not np.isfinite(runtime_float) or runtime_float <= 0:
                is_valid = False
        else:
            is_valid = False

        if is_valid:
            performances.append(perf_float)
            runtimes.append(runtime_float)
            valid_indices.append(i)

    valid_config_dicts = [config_dicts[i] for i in valid_indices]

    X_num, X_cat_str, numeric_names, categorical_names = preprocess_configurations(
        configs=valid_config_dicts,
        config_space=config_space,
        fidelity_space=fidelity_space,
        log_params=log_params,
    )

    return (
        X_num,
        X_cat_str,
        numeric_names,
        categorical_names,
        np.array(performances) if performances else np.array([]),
        np.array(runtimes) if runtimes else np.array([]),
    )


def check_perfect_accuracy_ratio(
    accuracies: np.ndarray, max_perfect_acc_ratio: float
) -> bool:
    if len(accuracies) == 0:
        return False

    max_accuracy = max(accuracies)
    is_percentage_scale = max_accuracy > 1
    perfect_threshold = 99.9 if is_percentage_scale else 0.999

    perfect_acc_count = sum(1 for acc in accuracies if acc >= perfect_threshold)
    perfect_acc_ratio = perfect_acc_count / len(accuracies)
    return perfect_acc_ratio > max_perfect_acc_ratio


def save_stratification(task_ids: List[str], output_file: str):
    with open(output_file, "w") as f:
        json.dump(task_ids, f, indent=2)


def validate_dataset(
    accuracies: np.ndarray,
    runtimes: np.ndarray,
    max_perfect_acc_ratio: float = 0.01,
    min_avg_runtime: float = 30,
) -> bool:
    if len(accuracies) == 0 or len(runtimes) == 0:
        return False

    if check_perfect_accuracy_ratio(
        accuracies=accuracies, max_perfect_acc_ratio=max_perfect_acc_ratio
    ):
        return False

    if min_avg_runtime > 0 and np.mean(runtimes) < min_avg_runtime:
        return False

    return True


def select_top_datasets(
    scores: Dict[str, float],
    top_count: Union[int, None] = None,
    top_percent: Union[float, None] = None,
) -> List[str]:
    if top_count is not None and top_percent is not None:
        raise ValueError("Cannot specify both top_count and top_percent")
    if top_count is None and top_percent is None:
        raise ValueError("Must specify either top_count or top_percent")

    if not scores:
        return []

    sorted_tasks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if top_count is not None:
        n_top = min(top_count, len(sorted_tasks))
    else:
        n_top = max(1, int(len(sorted_tasks) * top_percent / 100))

    return [task_id for task_id, _ in sorted_tasks[:n_top]]
