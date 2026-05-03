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
    benchmark_set = BenchmarkSet(benchmark_name)
    return benchmark_set.instances


def get_yahpo_log_info(benchmark: str) -> Dict[str, bool]:
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
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Featurise a list of hyperparameter configurations into a numeric matrix X.

    Categorical hyperparameters are one-hot encoded; numeric hyperparameters
    (float and integer) are kept as raw floats, with log-scale parameters
    log-transformed.  Log-scale parameters that are zero or negative are
    left as-is (the log guard is applied only when value > 0).

    Returns
    -------
    X : np.ndarray, shape (n_configs, n_cols)
        Numeric feature matrix.  Columns consist of:
          - one column per numeric hyperparameter (in sorted name order), then
          - one column per category per categorical hyperparameter (sorted name
            order, categories in the order defined in the ConfigSpace).
    feature_groups : list of dict
        Metadata describing each *original feature* (not column) in the order
        their columns appear in X.  Each entry has:
          "name"     : str   — hyperparameter name
          "type"     : "numeric" | "categorical"
          "col_start": int   — first column index in X for this feature
          "col_end"  : int   — one-past-last column index in X for this feature
        This is used downstream to compute Gower distance correctly (each
        original feature contributes equally regardless of OHE width).
    """
    if not configs:
        return np.array([]), []

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

    # Build feature_groups metadata and assemble X --------------------------
    feature_groups: List[Dict] = []
    col_cursor = 0

    # Numeric features: one column each
    for param in numeric_features:
        feature_groups.append({
            "name":      param,
            "type":      "numeric",
            "col_start": col_cursor,
            "col_end":   col_cursor + 1,
        })
        col_cursor += 1

    if categorical_features:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        all_categories = [
            [str(c) for c in categorical_choices[param]]
            for param in categorical_features
        ]
        encoder.fit([[cats[0] for cats in all_categories]])
        # Fit on all possible category values so every category gets a column
        encoder.fit(
            [[str(c) for c in categorical_choices[param][0:1]]
             for param in categorical_features]
        )
        # Proper fit: supply one row per category combination is not feasible;
        # instead fit with categories= kwarg to guarantee all levels are seen.
        encoder = OneHotEncoder(
            categories=all_categories,
            sparse=False,
            handle_unknown="ignore",
        )
        encoder.fit(X_categorical)
        X_cat_encoded = encoder.transform(X_categorical)

        # Record OHE column ranges for each categorical feature
        ohe_col_start = len(numeric_features)  # numeric columns come first
        for param, cats in zip(categorical_features, all_categories):
            n_cats = len(cats)
            feature_groups.append({
                "name":      param,
                "type":      "categorical",
                "col_start": ohe_col_start,
                "col_end":   ohe_col_start + n_cats,
            })
            ohe_col_start += n_cats

        X = np.hstack([X_numeric, X_cat_encoded])
    else:
        X = X_numeric

    return X, feature_groups


def normalise_for_gp(
    X: np.ndarray,
    feature_groups: List[Dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert the mixed-type feature matrix X into the format expected by the
    Optuna GP (GPRegressor in optuna_gp.py):

      - Numeric columns: min-max normalised to [0, 1] using the observed range
        in X.  (Numeric columns in X are already log-transformed where needed,
        so we just scale the already-transformed values.)
      - Categorical features: collapsed from their OHE block back to a single
        integer-index column (argmax of the OHE block), one column per original
        categorical hyperparameter.

    The output matrix has one column per original hyperparameter (not per OHE
    column), matching the Optuna GP's ARD kernel expectation.

    Returns
    -------
    X_gp : np.ndarray, shape (n, p)  — p = number of original features
    is_categorical : np.ndarray, shape (p,), bool
    """
    n = X.shape[0]
    num_features = len(feature_groups)
    X_gp = np.empty((n, num_features), dtype=np.float64)
    is_categorical = np.zeros(num_features, dtype=bool)

    for col_idx, fg in enumerate(feature_groups):
        cs, ce = fg["col_start"], fg["col_end"]
        block = X[:, cs:ce]

        if fg["type"] == "numeric":
            col = block[:, 0].astype(np.float64)
            col_min, col_max = col.min(), col.max()
            col_range = col_max - col_min
            if col_range < 1e-12:
                X_gp[:, col_idx] = 0.5  # constant feature: place at midpoint
            else:
                X_gp[:, col_idx] = (col - col_min) / col_range
            is_categorical[col_idx] = False

        else:
            # Recover category index from OHE block (argmax of each row)
            X_gp[:, col_idx] = np.argmax(block, axis=1).astype(np.float64)
            is_categorical[col_idx] = True

    return X_gp, is_categorical


def gower_distance_matrix(
    X: np.ndarray,
    feature_groups: List[Dict],
) -> np.ndarray:
    """
    Compute the symmetric n×n Gower (1971) distance matrix for a mixed-type
    feature matrix.

    Gower distance is defined as the mean of per-feature partial distances,
    where each partial distance is normalised to [0, 1]:

        d_Gower(i, j) = (1 / p) * Σ_f  d_f(xᵢ_f, xⱼ_f)

    where p is the number of *original features* (not columns, so OHE does not
    give categorical variables disproportionate weight), and:

        d_f = |xᵢ_f − xⱼ_f| / range_f    for numeric features
        d_f = 0 if same category, 1 otherwise  for categorical features

    The categorical partial distance is computed from the OHE columns as:
        d_f = 0.5 * ||OHE_i_f − OHE_j_f||₁
    which equals 0 when the same category (OHE identical) and 1 when different
    (exactly two positions differ by 1 in the OHE vector).

    Why not use the `gower` PyPI package?
    ---------------------------------------
    The two available packages (`gower` v0.1.2 and `gower-multiprocessing`
    v0.2.2, which share the same core) contain a documented bug in their
    numeric normalisation: they divide by `max` rather than by `range`, i.e.:

        num_ranges[col] = abs(1 - min/max)   # ← wrong when min < 0
        Z_num = Z_num / num_max              # ← should divide by range

    The correct Gower formula is `|xᵢ − xⱼ| / (max − min)`.  The packages
    only recover the correct value when `min = 0`, but our pipeline
    log-transforms hyperparameters with log-scale priors (e.g. learning rate),
    producing columns where `min < 0`.  Using the package would silently
    produce negative partial distances for those columns, corrupting the
    distance matrix.  This implementation uses `(col − col.min()) / col_range`
    which is correct in all cases.

    Parameters
    ----------
    X : np.ndarray, shape (n, n_cols)
        Mixed feature matrix as produced by preprocess_configurations.
    feature_groups : list of dict
        Metadata from preprocess_configurations describing column ranges and
        types for each original feature.

    Returns
    -------
    D : np.ndarray, shape (n, n), dtype float32
        Symmetric Gower distance matrix with zeros on the diagonal.
    """
    n = X.shape[0]
    p = len(feature_groups)
    if p == 0 or n == 0:
        return np.zeros((n, n), dtype=np.float32)

    # Accumulate partial distances; use float32 to save memory for large n
    D = np.zeros((n, n), dtype=np.float32)

    for fg in feature_groups:
        cs, ce = fg["col_start"], fg["col_end"]
        block = X[:, cs:ce]  # shape (n, width)

        if fg["type"] == "numeric":
            # Single column; normalise by observed range
            col = block[:, 0]
            col_range = col.max() - col.min()
            if col_range < 1e-12:
                # Constant feature — contributes 0 distance everywhere
                continue
            col_norm = (col - col.min()) / col_range
            # Pairwise absolute difference: |col_norm_i - col_norm_j|
            diff = np.abs(col_norm[:, None] - col_norm[None, :])  # (n, n)
            D += diff.astype(np.float32)

        else:
            # Categorical: OHE block of width c.
            # d_f(i,j) = 0.5 * L1(OHE_i, OHE_j) ∈ {0, 1}
            # Vectorised: for binary {0,1} OHE, L1 = sum of abs differences.
            # Use float32 arithmetic to avoid large intermediate allocations.
            block_f = block.astype(np.float32)
            # (n, n) via broadcasting: sum over OHE columns
            # To avoid n×n×c tensor, compute as: n² dot-product trick
            # d_f(i,j) = 0 iff OHE_i == OHE_j, i.e. dot(OHE_i, OHE_j) = 1
            # For binary OHE: ||a-b||₁/2 = 1 - a·b  (when exactly one 1 each)
            dot = block_f @ block_f.T  # (n, n)
            D += (1.0 - dot).astype(np.float32)

    D /= p
    # Numerical symmetry guard (floating point can break symmetry slightly)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def sample_benchmark_data(
    benchmark_name: str,
    task_id: str,
    n_samples: int = 10000,
) -> Tuple[np.ndarray, List[Dict], np.ndarray, np.ndarray]:
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

    tabularized_configurations, feature_groups = preprocess_configurations(
        configs=valid_config_dicts,
        config_space=config_space,
        fidelity_space=fidelity_space,
        log_info=log_info,
    )

    return (
        tabularized_configurations,
        feature_groups,
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

    if (
        min_avg_runtime > 0
        and np.mean(runtimes) < min_avg_runtime
    ):
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
