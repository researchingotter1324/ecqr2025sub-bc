# HPO Benchmarking

## Overview

This repository provides reproducible code for all benchmarking results reported in *"Enhancing Adaptiveness and Sampling Performance in Conformal Hyperparameter Optimization"*.

It is not intended as a general use benchmarking utility and is not coded with the rigor of one.

All **conformal quantile hyperparameter optimization** algorithms are called from a **separate dependency** (instructions later for how to ingest this in your python environment), and can be found at the following standalone, anonymized, repository: [https://github.com/researchingotter1324/ecqr2025sub-c](https://github.com/researchingotter1324/ecqr2025sub-c).

## Installation

Create a python 3.10.9 (or other 3.10 version) environment with your environment manager of choice.

### 1. Repository:
1. Clone repository:
   ```bash
   git clone https://github.com/researchingotter1324/ecqr2025sub-bc
   cd ecqr2025sub-bc
   ```

2. Install as package into your environment:
   ```bash
   pip install .
   ```

### 2. Optuna

`optuna` is not an explicit pypi package dependency of the `hpobench` package, but it is required to run Optuna GP sampler benchmarks.

To resolve this, clone the below anonymized, up-to-date fork of Optuna, which tracks upstream closely and adds targeted customizations to the GP sampler:
   ```bash
   git clone https://github.com/researchingotter1324/ecqr2025sub-optuna
   cd ecqr2025sub-optuna
   ```

And install it in your environment directly by navigating to it while your python environment is active and running:
   ```bash
   pip install .
   ```

### 3. SMAC

`smac` is not an explicit pypi package dependancy of the `hpobench` package, due to incompatibility issues, but it is required to run SMAC benchmarks.

To resolve this, you can clone the below anonymized, single commit copy of a fork with minor edits to SMAC's `ConfigSpace` dependency:
   ```bash
   git clone https://github.com/researchingotter1324/ecqr2025sub-smac
   cd ecqr2025sub-smac
   ```

And install it in your environment directly by navigating to it while your python environment is active and running:
   ```bash
   pip install .
   ```

**NOTE**:
SWIG is required to build the `pyrfr` dependency for SMAC. Install it using conda:
   ```bash
   conda install swig
   ```
SWIG must be installed in the same environment where you're installing the package dependencies.

### 4. Conformal Quantile Optimization Methods

All conformal quantile HPO algorithms are contained in ccqr_optimization. Like SMAC, we need to build this dependency from source. 

First, clone and navigate to the below anonymized repository:

   ```bash
   git clone https://github.com/researchingotter1324/ecqr2025sub-c
   cd ecqr2025sub-c
   ```

Then install it in your environment directly by navigating to it while your python environment is active and running:
   ```bash
   pip install .
   ```

### 5. Benchmark Environment Setup

#### YAHPO Gym

(Currently limited to non-hierarchical benchmarks).

**Setup Instructions**:
1. Install the YAHPO Gym package (included in requirements.txt and pyproject.toml and automatically installed in the earlier main repository `pip install .` step)
2. **Manual Data Setup Required**: Download the YAHPO benchmark data from the forked repository at: https://github.com/slds-lmu/yahpo_data
3. Extract the data folders into the `yahpo_bench_data/` folder at the root of this repository
4. The folder structure should contain subdirectories like:
   - `yahpo_bench_data/rbv2_aknn/`
   - `yahpo_bench_data/lcbench/`
   - `yahpo_bench_data/iaml_*/`

#### JAHS-Bench-201

1. Install JAHS-Bench package (included in requirements.txt and pyproject.toml and automatically installed in the earlier main repository `pip install .` step)
2. Data downloads automatically (takes a while) to `jahs_bench_data/` on first run
3. Or do it manually ahead of time with: `python -m jahs_bench.download --target surrogates`



## Running Experiments

### Entry Point

Execute benchmarks via the main script:

```bash
python run.py
```

NOTE: Ensure you are in the `ecqr2025sub-bc` if running from terminal.

### Experiment Configuration
- Algorithm agnostic parameters can be set in `constants.py`
- Algorithm configurations are set up in `tuner_configurations.py`


### Experiment Components

You can specify which types of analysis to run in `run.py` using the `run_sections` dictionary:

```python
run_sections = {
    "run_coverage_analysis": False,
    "run_sampler_variation_analysis": False,
    "run_architecture_variation_analysis": False,
    "run_external_tuning_analysis": True,
    "run_heteroscedastic_external_tuning_analysis": False,
    "run_skew_external_tuning_analysis": False,
    "run_preconformal_comparison_analysis": False,
    "run_static_analysis": False,
    "run_quantile_count_comparison": False,
    "run_search_tuning_effect_comparison": False,
}
```