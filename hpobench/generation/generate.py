import numpy as np
from typing import Union, Dict, Any

from ConfigSpace import Configuration

from abc import ABC, abstractmethod

from yahpo_gym import BenchmarkSet

from hpobench.generation.black_box_functions import (
    rastrigin,
    ackley,
    griewank,
    weierstrass,
    shekel,
    hartmann6,
)


def _ensure_yahpo_initialized():
    """Wrapper to avoid circular imports."""
    from hpobench.utils import ensure_yahpo_initialized

    ensure_yahpo_initialized()


class ObjectiveMetricGenerator(ABC):
    """Abstract base class for objective metric generators used in benchmarking.

    Args:
        generator: Name or identifier for the generator (used by subclasses).
    """

    def __init__(self, generator: str):
        self.generator = generator

    @abstractmethod
    def predict(self, configuration: dict[str, Union[str, int, float, bool]]) -> float:
        """Return the objective value for a given configuration.

        Args:
            configuration: Dictionary mapping parameter names to their values.

        Returns:
            The objective value for the given configuration.
        """

    @abstractmethod
    def predict_batch(self, configurations: list[dict]) -> list[float]:
        """Return objective values for multiple configurations in batch.

        Args:
            configurations: List of configuration dictionaries to evaluate.

        Returns:
            List of objective values for the given configurations.
        """

    @abstractmethod
    def predict_runtime(
        self, configuration: dict[str, Union[str, int, float, bool]]
    ) -> float:
        """Return the runtime for a given configuration.

        Args:
            configuration: Dictionary mapping parameter names to their values.

        Returns:
            The runtime for the given configuration.
        """

    @abstractmethod
    def predict_runtime_batch(self, configurations: list[dict]) -> list[float]:
        """Return runtime values for multiple configurations in batch.

        Args:
            configurations: List of configuration dictionaries to evaluate.

        Returns:
            List of runtime values for the given configurations.
        """

    def initialize(self) -> None:
        """Initialize the generator if needed. Default implementation does nothing.

        Returns:
            None
        """


class BlackBoxGenerator(ObjectiveMetricGenerator):
    """Objective metric generator for standard black-box optimization functions.

    Args:
        generator: Name of the black-box function to use (e.g., 'rastrigin', 'ackley').
    """

    def __init__(self, generator: str):
        self.generator = generator

    def _evaluate_function(self, x: np.ndarray) -> float:
        """Evaluate the black-box function for the given parameter vector.

        Args:
            x: Parameter vector as numpy array.

        Returns:
            Function value.
        """
        if self.generator == "rastrigin":
            return rastrigin(x=x)
        elif self.generator == "ackley":
            return ackley(x=x)
        elif self.generator == "griewank":
            return griewank(x=x)
        elif self.generator == "weierstrass":
            return weierstrass(x=x)
        elif self.generator == "shekel":
            return shekel(x=x)
        elif self.generator == "hartmann6":
            return hartmann6(x=x)
        else:
            raise ValueError(f"Unknown generator: {self.generator}")

    def predict(self, configuration: dict[str, Union[str, int, float, bool]]) -> float:
        """Evaluate the black-box function for the given configuration.

        Args:
            configuration: Dictionary mapping parameter names to their values.

        Returns:
            The function value for the given configuration.
        """
        x = np.array(list(configuration.values()), dtype=float)
        return self._evaluate_function(x)

    def predict_batch(self, configurations: list[dict]) -> list[float]:
        """Evaluate multiple configurations in batch.

        Args:
            configurations: List of configuration dictionaries to evaluate.

        Returns:
            List of function values.
        """
        return [self.predict(config) for config in configurations]

    def predict_runtime(
        self, configuration: dict[str, Union[str, int, float, bool]]
    ) -> float:
        """Return the runtime for the given configuration (always 0 for black-box functions).

        Args:
            configuration: Dictionary mapping parameter names to their values.

        Returns:
            Always returns 0.
        """
        return 0

    def predict_runtime_batch(self, configurations: list[dict]) -> list[float]:
        """Return runtime for multiple configurations (always 0 for black-box functions).

        Args:
            configurations: List of configuration dictionaries.

        Returns:
            List of runtime values (all 0 for black-box functions).
        """
        return [0.0] * len(configurations)


class Jahs201Generator(ObjectiveMetricGenerator):
    """Objective metric generator for the JAHS-201 surrogate benchmark.

    Args:
        dataset: Name of the JAHS-201 dataset.
        metrics: List of metric names to use.
        lazy: If True, defer initialization of the generator until first use.
    """

    def __init__(
        self,
        dataset: str,
        metrics: list[str] = ["valid-acc", "runtime"],
        lazy: bool = True,
    ):
        self._dataset = dataset
        self._metrics = metrics
        self._lazy = lazy
        self._initialized = False

        self.default_fidelities = {
            "epoch": 200,
            "W": 16,
            "N": 5,
            "Resolution": 1,
        }

        if not self._lazy:
            self._initialize_generator()
        else:
            self.generator = None

    def _initialize_generator(self) -> None:
        """Initialize the JAHS-201 generator if not already initialized."""
        if not self._initialized:
            from jahs_bench import Benchmark
            self.generator = Benchmark(
                task=self._dataset, lazy=False, metrics=self._metrics
            )
            self._initialized = True

    def initialize(self) -> None:
        """Initialize the generator if needed."""
        self._initialize_generator()

    def _merge_with_fidelities(
        self, configuration: dict[str, Union[str, int, float, bool]]
    ) -> dict[str, Union[str, int, float, bool]]:
        """Merge configuration with default maximum fidelities."""
        merged = configuration.copy()
        merged.update(self.default_fidelities)
        return merged

    def _evaluate_jahs(
        self, configuration: dict[str, Union[str, int, float, bool]]
    ) -> dict:
        """Helper method to evaluate configuration with JAHS-201 benchmark."""
        self._initialize_generator()
        merged_config = self._merge_with_fidelities(configuration)
        return self.generator(merged_config)[self.default_fidelities["epoch"]]

    def predict(self, configuration: dict[str, Union[str, int, float, bool]]) -> float:
        """Return negative validation accuracy for the given configuration."""
        result = self._evaluate_jahs(configuration)
        return -result["valid-acc"]

    def predict_batch(self, configurations: list[dict]) -> list[float]:
        """Evaluate multiple configurations in batch."""
        return [self.predict(config) for config in configurations]

    def predict_runtime(
        self, configuration: dict[str, Union[str, int, float, bool]]
    ) -> float:
        """Return runtime for the given configuration."""
        result = self._evaluate_jahs(configuration)
        return result["runtime"]

    def predict_runtime_batch(self, configurations: list[dict]) -> list[float]:
        """Evaluate runtime for multiple configurations in batch."""
        return [self.predict_runtime(config) for config in configurations]


class YahpoGenerator(ObjectiveMetricGenerator):
    """Objective metric generator for YAHPO Gym surrogate benchmarks.

    Handles instance-specific and fidelity-aware configuration evaluation.
    Automatically uses maximum fidelity values for all fidelity parameters.

    Args:
        dataset: Name of the YAHPO benchmark scenario.
        instance_value: Value of the instance for this experiment.
        instance_name: Name of the instance parameter in the configuration space.
        fidelity_space: Dictionary of fidelity parameter names and their MAXIMUM values.
        config_space: Configuration space object.
    """

    def __init__(
        self,
        dataset: str,
        instance_value: Any,
        instance_name: str,
        fidelity_space: Dict,
        config_space,
    ):
        # Ensure YAHPO is initialized before creating BenchmarkSet
        _ensure_yahpo_initialized()

        self.dataset = dataset
        self.instance_name = instance_name
        self.generator = BenchmarkSet(
            dataset, instance=instance_value, active_session=False
        )

        self.config_space = config_space
        # Store maximum fidelity values (passed from setup functions)
        self.fidelity_space = fidelity_space

    def _get_filtered_configuration(self, configuration: dict) -> dict:
        """Filter the configuration to include only active and fidelity parameters.

        Uses ConfigSpace's built-in get_active_hyperparameters method for robust
        conditional dependency handling.

        Args:
            configuration: Dictionary mapping parameter names to their values.

        Returns:
            Filtered configuration dictionary including only active and fidelity parameters.
        """
        config_dict = configuration.copy()

        # Add fidelity parameters to the configuration for evaluation
        if self.fidelity_space:
            config_dict.update(self.fidelity_space)

        # Add instance parameter for evaluation
        config_dict[self.instance_name] = self.generator.instance

        # Use ConfigSpace's built-in method to get active hyperparameters
        try:
            cs_config = Configuration(
                self.config_space,
                values=config_dict,
                allow_inactive_with_values=True,
            )
            active_hyperparameters = self.config_space.get_active_hyperparameters(
                cs_config
            )

            # Filter configuration to only include active parameters
            filtered_configuration = {
                k: v
                for k, v in config_dict.items()
                if k in active_hyperparameters or k == self.instance_name
            }

        except Exception as e:
            raise ValueError(f"ConfigSpace evaluation failed: {e}")

        return filtered_configuration

    def predict(self, configuration: dict[str, Union[str, int, float, bool]]) -> float:
        """Return the negative primary metric for the given configuration.

        Args:
            configuration: Dictionary mapping parameter names to their values.

        Returns:
            Negative primary metric.
        """
        batch_results = self._batch_evaluate_configurations([configuration])
        return self._extract_performance_metric(batch_results[0])

    def _batch_evaluate_configurations(self, configurations: list[dict]) -> list[dict]:
        """Helper method to filter and evaluate multiple configurations in batch.

        Args:
            configurations: List of configuration dictionaries to evaluate.

        Returns:
            List of batch evaluation results.
        """
        filtered_configs = []
        for config in configurations:
            filtered_config = self._get_filtered_configuration(config)
            filtered_configs.append(filtered_config)

        return self.generator.objective_function(filtered_configs, seed=1234)

    def _extract_performance_metric(self, result: dict) -> float:
        """Extract performance metric from evaluation result.

        Args:
            result: Single evaluation result dictionary.

        Returns:
            Performance value (negated for minimization).
        """
        if "val_accuracy" in result:
            return -result["val_accuracy"]
        elif "acc" in result:
            return -result["acc"]
        elif "auc" in result:
            return -result["auc"]
        else:
            raise ValueError(
                f"No suitable metric found in results: {list(result.keys())}"
            )

    def _extract_runtime_metric(self, result: dict) -> float:
        """Extract runtime metric from evaluation result.

        Args:
            result: Single evaluation result dictionary.

        Returns:
            Runtime value.
        """
        if "time" in result:
            return result["time"]
        elif "runtime" in result:
            return result["runtime"]
        else:
            return result["timetrain"] + result["timepredict"]

    def predict_batch(self, configurations: list[dict]) -> list[float]:
        """Evaluate multiple configurations in batch for improved performance.

        Args:
            configurations: List of configuration dictionaries to evaluate.

        Returns:
            List of performance values (negated for minimization).
        """
        batch_results = self._batch_evaluate_configurations(configurations)
        return [self._extract_performance_metric(result) for result in batch_results]

    def predict_runtime(
        self, configuration: dict[str, Union[str, int, float, bool]]
    ) -> float:
        """Return the runtime for the given configuration.

        Args:
            configuration: Dictionary mapping parameter names to their values.

        Returns:
            Runtime.
        """
        batch_results = self._batch_evaluate_configurations([configuration])
        return self._extract_runtime_metric(batch_results[0])

    def predict_runtime_batch(self, configurations: list[dict]) -> list[float]:
        """Evaluate runtime for multiple configurations in batch.

        Args:
            configurations: List of configuration dictionaries to evaluate.

        Returns:
            List of runtime values.
        """
        batch_results = self._batch_evaluate_configurations(configurations)
        return [self._extract_runtime_metric(result) for result in batch_results]


class NAS301Generator(YahpoGenerator):
    """Specialized objective metric generator for NAS-301 benchmark.

    Extends YahpoGenerator to handle NAS-301 specific parameter name mapping.

    Args:
        instance_value: Value of the instance for this experiment.
        instance_name: Name of the instance parameter in the configuration space.
        fidelity_space: Dictionary of fidelity parameter names and their values.
        config_space: Configuration space object.
    """

    NB301_ATTRIBUTE_NAME_PREFIX = "NetworkSelectorDatasetInfo_COLON_darts_COLON_"

    def __init__(
        self,
        instance_value: Any,
        instance_name: str,
        fidelity_space: Dict,
        config_space,
    ):
        # Initialize with nb301 dataset
        super().__init__(
            dataset="nb301",
            instance_value=instance_value,
            instance_name=instance_name,
            fidelity_space=fidelity_space,
            config_space=config_space,
        )

        # Set default maximum fidelity for NAS-301 (like JAHS-201 generator)
        # NAS-301 uses epoch as fidelity parameter with maximum value of 98
        self.default_fidelities = {
            "epoch": 97,  # Maximum fidelity for NAS-301
        }

        # Initialize parameter name mapping for NAS-301
        self._shortened_keys = set()
        self._initialize_nas301_specifics()

    def _initialize_nas301_specifics(self):
        """Initialize NAS-301 specific parameter name handling."""
        # Create mapping from shortened keys to full YAHPO parameter names
        len_prefix = len(self.NB301_ATTRIBUTE_NAME_PREFIX)

        # Get all parameter names from the YAHPO config space
        yahpo_config_space = self.generator.get_opt_space(drop_fidelity_params=True)

        for param_name in yahpo_config_space.get_hyperparameter_names():
            if param_name.startswith(self.NB301_ATTRIBUTE_NAME_PREFIX):
                shortened_key = param_name[len_prefix:]
                self._shortened_keys.add(shortened_key)

    def _map_configuration_to_yahpo(self, configuration: dict) -> dict:
        """Map shortened parameter names back to full YAHPO parameter names.

        Args:
            configuration: Dictionary with shortened parameter names.

        Returns:
            Dictionary with full YAHPO parameter names.
        """
        mapped_config = {}

        for key, value in configuration.items():
            if key in self._shortened_keys:
                # Map shortened key back to full YAHPO parameter name
                full_key = self.NB301_ATTRIBUTE_NAME_PREFIX + key
                mapped_config[full_key] = value
            else:
                # Keep non-NAS parameters as-is
                mapped_config[key] = value

        return mapped_config

    def _merge_with_fidelities(
        self, configuration: dict[str, Union[str, int, float, bool]]
    ) -> dict[str, Union[str, int, float, bool]]:
        """Merge configuration with default maximum fidelities.

        Args:
            configuration: Dictionary mapping parameter names to their values.

        Returns:
            Configuration merged with default maximum fidelity values.
        """
        merged = configuration.copy()
        merged.update(self.default_fidelities)
        return merged

    def _get_filtered_configuration(self, configuration: dict) -> dict:
        """Override to handle NAS-301 parameter name mapping.

        Args:
            configuration: Dictionary mapping parameter names to their values.

        Returns:
            Filtered configuration dictionary with full YAHPO parameter names.
        """
        # First merge with default maximum fidelities
        merged_config = self._merge_with_fidelities(configuration)

        # Then map the shortened parameter names to full YAHPO names
        mapped_config = self._map_configuration_to_yahpo(merged_config)

        # NOTE: NAS-301 doesn't use an instance parameter in the configuration space
        # The instance is handled at the BenchmarkSet level, not as a hyperparameter

        # Use ConfigSpace's built-in method to get active hyperparameters
        try:
            # Create a temporary config space with the full parameter names for validation
            yahpo_config_space = self.generator.get_opt_space(
                drop_fidelity_params=False
            )

            cs_config = Configuration(
                yahpo_config_space,
                values=mapped_config,
                allow_inactive_with_values=True,
            )
            active_hyperparameters = yahpo_config_space.get_active_hyperparameters(
                cs_config
            )

            # Filter configuration to only include active parameters
            filtered_configuration = {
                k: v for k, v in mapped_config.items() if k in active_hyperparameters
            }

        except Exception as e:
            raise ValueError(f"ConfigSpace evaluation failed for NAS-301: {e}")

        return filtered_configuration
