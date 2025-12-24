import json
import numpy as np
import os
from typing import Dict, List, Tuple, Union, Any
import warnings

from yahpo_gym import BenchmarkSet, local_config
from sklearn.preprocessing import OneHotEncoder
from ConfigSpace import Configuration
import ConfigSpace as CS
import logging

np.random.seed(42)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

local_config.init_config()
local_config.set_data_path("yahpo_bench_data")


def get_benchmark_task_ids(benchmark_name: str) -> List[str]:
    """Get all task IDs for a benchmark."""
    benchmark_set = BenchmarkSet(benchmark_name)
    return benchmark_set.instances


def get_yahpo_log_info(benchmark: str) -> Dict[str, bool]:
    """Extract log-scale information from yahpo benchmark JSON config files."""
    config_path = os.path.join("yahpo_bench_data", benchmark, "config_space.json")
    log_info = {}

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_data = json.load(f)

        for hp in config_data.get("hyperparameters", []):
            param_name = hp.get("name")
            log_flag = hp.get("log", False)
            if param_name:
                log_info[param_name] = log_flag

    return log_info


def get_filtered_configuration(
    configuration: dict,
    config_space: CS.ConfigurationSpace,
    fidelity_space: Dict[str, Any],
    instance_name: str,
    instance_value: Any,
) -> dict:
    """Filter configuration to include only active and fidelity parameters."""
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
    log_info: Dict[str, bool],
) -> np.ndarray:
    """Preprocess configurations using proper ConfigSpace-aware encoding."""
    if not configs:
        return np.array([])

    all_hyperparams = {hp.name: hp for hp in config_space.get_hyperparameters()}

    categorical_features = []
    numeric_features = []
    log_scale_features = set()
    categorical_choices = {}

    for param_name, hp in all_hyperparams.items():
        if param_name in fidelity_space:
            continue

        if isinstance(hp, CS.CategoricalHyperparameter):
            categorical_features.append(param_name)
            categorical_choices[param_name] = hp.choices
        elif isinstance(
            hp, (CS.UniformFloatHyperparameter, CS.UniformIntegerHyperparameter)
        ):
            numeric_features.append(param_name)
            if log_info.get(param_name, False):
                log_scale_features.add(param_name)

    categorical_features = sorted(categorical_features)
    numeric_features = sorted(numeric_features)

    X_numeric = []
    X_categorical = []

    for config in configs:
        numeric_row = []
        for param in numeric_features:
            value = config.get(param, 0)
            if isinstance(value, bool):
                value = int(value)

            if param in log_scale_features and value > 0:
                value = np.log(value)

            numeric_row.append(float(value))
        X_numeric.append(numeric_row)

        categorical_row = []
        for param in categorical_features:
            value = config.get(param, categorical_choices[param][0])
            categorical_row.append(str(value))
        X_categorical.append(categorical_row)

    X_numeric = np.array(X_numeric)

    if categorical_features:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        all_categories = [categorical_choices[param] for param in categorical_features]
        encoder.fit(
            [
                [choice for choices in all_categories for choice in choices][
                    : len(categorical_features)
                ]
            ]
        )
        X_categorical = encoder.fit_transform(X_categorical)
        X = np.hstack([X_numeric, X_categorical])
    else:
        X = X_numeric

    return X


def sample_benchmark_data(
    benchmark_name: str,
    task_id: str,
    n_samples: int = 10000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """Sample data from YAHPO Gym benchmark."""
    local_config.init_config()
    local_config.set_data_path("yahpo_bench_data")

    benchmark_set = BenchmarkSet(
        scenario=benchmark_name, instance=task_id, active_session=False, check=False
    )
    config_space = benchmark_set.get_opt_space(drop_fidelity_params=False, seed=42)
    configurations = config_space.sample_configuration(n_samples)

    if not isinstance(configurations, list):
        configurations = [configurations]

    log_info = get_yahpo_log_info(benchmark_name)

    # Extract fidelity parameters and their maximum values
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
    for result in batch_results:
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

        # Collect valid values
        if performance is not None:
            perf_float = float(performance)
            if not np.isnan(perf_float) and np.isfinite(perf_float):
                performances.append(perf_float)

        if runtime is not None:
            runtime_float = float(runtime)
            if (
                not np.isnan(runtime_float)
                and np.isfinite(runtime_float)
                and runtime_float > 0
            ):
                runtimes.append(runtime_float)

    tabularized_configurations = preprocess_configurations(
        configs=config_dicts,
        config_space=config_space,
        fidelity_space=fidelity_space,
        log_info=log_info,
    )

    return (
        tabularized_configurations,
        np.array(performances) if performances else np.array([]),
        np.array(runtimes) if runtimes else np.array([]),
    )


def check_perfect_accuracy_ratio(
    accuracies: np.ndarray, max_perfect_acc_ratio: float
) -> bool:
    """Check if dataset has excessive perfect accuracy configurations."""
    if len(accuracies) == 0:
        return False

    max_accuracy = max(accuracies)
    is_percentage_scale = max_accuracy > 1
    perfect_threshold = 99.9 if is_percentage_scale else 0.999

    perfect_acc_count = sum(1 for acc in accuracies if acc >= perfect_threshold)
    perfect_acc_ratio = perfect_acc_count / len(accuracies)
    return perfect_acc_ratio > max_perfect_acc_ratio


def save_stratification(task_ids: List[str], output_file: str):
    """Save task IDs to JSON file."""
    with open(output_file, "w") as f:
        json.dump(task_ids, f, indent=2)


def validate_dataset(
    accuracies: np.ndarray,
    runtimes: np.ndarray,
    max_perfect_acc_ratio: float = 0.01,
    min_avg_runtime: float = 30,
) -> bool:
    """Validate if a dataset meets the requirements for stratification."""
    # Check for excessive perfect accuracy
    if check_perfect_accuracy_ratio(
        accuracies=accuracies, max_perfect_acc_ratio=max_perfect_acc_ratio
    ):
        return False

    # Check runtime requirements if specified
    if (
        min_avg_runtime > 0
        and len(runtimes) > 0
        and np.mean(runtimes) < min_avg_runtime
    ):
        return False

    return True


def select_top_datasets(
    scores: Dict[str, float],
    top_count: Union[int, None] = None,
    top_percent: Union[float, None] = None,
) -> List[str]:
    """Select top datasets based on scores."""
    if top_count is not None and top_percent is not None:
        raise ValueError("Cannot specify both top_count and top_percent")
    if top_count is None and top_percent is None:
        raise ValueError("Must specify either top_count or top_percent")

    if not scores:
        return []

    # Sort by score (descending)
    sorted_tasks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Determine selection count
    if top_count is not None:
        n_top = min(top_count, len(sorted_tasks))
    else:
        n_top = max(1, int(len(sorted_tasks) * top_percent / 100))

    return [task_id for task_id, _ in sorted_tasks[:n_top]]
