from pydantic import BaseModel
from typing import Any, Dict, List


class BenchmarkDataSchema(BaseModel):
    """Schema defining column names for benchmark experiment data.

    Args:
        cumulative_coverage_error_col: Column for cumulative coverage error metrics.
        rolling_coverage_error_col: Column for rolling window coverage error.
        rep_col: Column for experiment repetition number.
        perf_col: Column for performance metric values.
        tuner_col: Column for tuner configuration identifier.
        bench_col: Column for benchmark suite name.
        data_col: Column for dataset identifier.
        sampler_col: Column for conformal sampler type.
        confidence_level_col: Column for confidence level values.
        estimator_architecture_col: Column for quantile estimator architecture.
        sampler_n_quantiles_col: Column for number of quantiles used.
        n_candidates_col: Column for number of candidate configurations sampled during acquisition.
        sampler_adapter_col: Column for adaptive conformal method.
        tuner_searcher_tuning_framework_col: Column for searcher tuning framework.
        n_pre_conformal_trials_col: Column for pre-conformal trial count.
        data_size_col: Column for dataset size.
        tuning_iterations_col: Column for tuning iteration count.
        estimator_error_col: Column for quantile estimator error.
        breach_col: Column for coverage breach status.
        extreme_quantile_used_col: Column for binary indicator of whether a trial's
            performance was obtained via the lowest (extreme) quantile bound.
        ei_collapsed_col: Column for binary indicator (0/1) of whether the EI
            acquisition was collapsed (hard-maximization) rather than soft for a trial.
        perc_zero_ei_col: Column for the percentage of candidate configurations
            that had a zero EI value during a trial's acquisition step.
        runtime_unit: Base name for runtime columns.
        iter_unit: Base name for iteration columns.
        norm_runtime_unit: Name for normalized runtime columns.
        norm_iter_unit: Name for normalized iteration columns.
    """

    # Coverage error columns
    cumulative_coverage_error_col: str = "cumulative_coverage_error"
    rolling_coverage_error_col: str = "rolling_coverage_error"
    rep_col: str = "repetition"
    perf_col: str = "performance"
    tuner_col: str = "tuner"
    bench_col: str = "benchmark_identifier"
    data_col: str = "dataset"
    sampler_col: str = "sampler"
    confidence_level_col: str = "confidence_level"
    estimator_architecture_col: str = "estimator_architecture"
    sampler_n_quantiles_col: str = "sampler_n_quantiles"
    n_candidates_col: str = "n_candidates"
    sampler_adapter_col: str = "sampler_adapter"
    tuner_searcher_tuning_framework_col: str = "tuner_searcher_tuning_framework"
    n_pre_conformal_trials_col: str = "n_pre_conformal_trials"
    data_size_col: str = "data_size"
    tuning_iterations_col: str = "tuning_iterations"
    estimator_error_col: str = "mean_pinball_loss"
    breach_col: str = "breach_status"
    extreme_quantile_used_col: str = "extreme_quantile_used"
    ei_collapsed_col: str = "ei_collapsed"
    perc_zero_ei_col: str = "perc_zero_ei"

    runtime_unit: str = "runtime"
    iter_unit: str = "iteration"
    norm_runtime_unit: str = f"normalized_{runtime_unit}"
    norm_iter_unit: str = f"normalized_{iter_unit}"

    def to_list(self) -> List[str]:
        """Convert all schema field values to a list.

        Returns:
            List of all column names and units defined in the schema.
        """
        field_values: Dict[str, Any]
        field_values = self.model_dump()
        return list(field_values.values())
