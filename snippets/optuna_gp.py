"""
Self-contained port of Optuna's Gaussian Process implementation.

Source: https://github.com/optuna/optuna
Files ported (verbatim logic, adapted imports):
  optuna/_gp/gp.py
  optuna/_gp/prior.py
  optuna/_gp/scipy_blas_thread_patch.py

License
-------
MIT License

Copyright (c) 2018 Preferred Networks, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Modifications from upstream
---------------------------
- Removed all optuna.*  imports; replaced with direct numpy/scipy/torch imports.
- Removed lazy-import wrappers (_LazyImport), experimental decorators, and
  Optuna logging; standard Python logging is used instead.
- `warn_and_convert_inf` prints a warning instead of calling `optuna_warn`.
- The public API is identical to the upstream `optuna._gp.gp.fit_kernel_params`
  and `optuna._gp.gp.GPRegressor`.
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Generator

import numpy as np
import scipy.linalg
import scipy.optimize
import torch
from packaging.version import Version
import scipy

logger = logging.getLogger(__name__)

# ── scipy BLAS thread patch (from optuna/_gp/scipy_blas_thread_patch.py) ─────
# Needed because L-BFGS-B in SciPy >= 1.15 uses OpenBLAS and causes slowdown.

@contextmanager
def _single_blas_thread_if_needed() -> Generator[None, None, None]:
    if Version(scipy.__version__) < Version("1.15.0"):
        yield
    else:
        old_val = os.environ.get("OPENBLAS_NUM_THREADS")
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        try:
            yield
        finally:
            if old_val is None:
                os.environ.pop("OPENBLAS_NUM_THREADS", None)
            else:
                os.environ["OPENBLAS_NUM_THREADS"] = old_val


# ── Kernel (from optuna/_gp/gp.py) ───────────────────────────────────────────

def _warn_and_convert_inf(values: np.ndarray) -> np.ndarray:
    is_finite = np.isfinite(values)
    if np.all(is_finite):
        return values
    warnings.warn("Clipping non-finite values to min/max finite values for GP fitting.")
    is_any_finite = np.any(is_finite, axis=0)
    return np.clip(
        values,
        np.where(is_any_finite, np.min(np.where(is_finite, values, np.inf),  axis=0), 0.0),
        np.where(is_any_finite, np.max(np.where(is_finite, values, -np.inf), axis=0), 0.0),
    )


class _Matern52Kernel(torch.autograd.Function):
    """
    Matern 5/2 kernel: exp(-sqrt(5d)) * (5d/3 + sqrt(5d) + 1).

    Gradient is computed manually to avoid numerical issues at d=0.
    Ported verbatim from optuna/_gp/gp.py::Matern52Kernel.
    """

    @staticmethod
    def forward(ctx: Any, squared_distance: torch.Tensor) -> torch.Tensor:
        sqrt5d = torch.sqrt(5 * squared_distance)
        exp_part = torch.exp(-sqrt5d)
        val = exp_part * ((5 / 3) * squared_distance + sqrt5d + 1)
        deriv = (-5 / 6) * (sqrt5d + 1) * exp_part
        ctx.save_for_backward(deriv)
        return val

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        (deriv,) = ctx.saved_tensors
        return deriv * grad


# ── GP regressor (from optuna/_gp/gp.py) ─────────────────────────────────────

class GPRegressor:
    """
    Gaussian process regressor with Matern-5/2 ARD kernel, MLL-fitted hyperparameters.

    Ported verbatim from optuna/_gp/gp.py::GPRegressor.
    Categorical features use Hamming distance; continuous features use squared
    Euclidean distance, each scaled by per-dimension inverse squared length-scales.
    """

    def __init__(
        self,
        is_categorical: torch.Tensor,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        inverse_squared_lengthscales: torch.Tensor,
        kernel_scale: torch.Tensor,
        noise_var: torch.Tensor,
    ) -> None:
        self._is_categorical = is_categorical
        self._X_train = X_train
        self._y_train = y_train
        self._X_all = X_train
        self._y_all = y_train
        self._squared_X_diff = (X_train.unsqueeze(-2) - X_train.unsqueeze(-3)).square_()
        if self._is_categorical.any():
            self._squared_X_diff[..., self._is_categorical] = (
                self._squared_X_diff[..., self._is_categorical] > 0.0
            ).to(torch.float64)
        self._cov_Y_Y_chol: torch.Tensor | None = None
        self._cov_Y_Y_inv_Y: torch.Tensor | None = None
        self.inverse_squared_lengthscales = inverse_squared_lengthscales
        self.kernel_scale = kernel_scale
        self.noise_var = noise_var

    @property
    def length_scales(self) -> np.ndarray:
        return 1.0 / np.sqrt(self.inverse_squared_lengthscales.detach().cpu().numpy())

    def _cache_matrix(self) -> None:
        assert self._cov_Y_Y_chol is None and self._cov_Y_Y_inv_Y is None
        with torch.no_grad():
            cov_Y_Y = self.kernel().detach().cpu().numpy()
        cov_Y_Y[np.diag_indices(self._X_train.shape[0])] += self.noise_var.item()
        cov_Y_Y_chol = np.linalg.cholesky(cov_Y_Y)
        cov_Y_Y_inv_Y = scipy.linalg.solve_triangular(
            cov_Y_Y_chol.T,
            scipy.linalg.solve_triangular(cov_Y_Y_chol, self._y_train.cpu().numpy(), lower=True),
            lower=False,
        )
        self._cov_Y_Y_chol = torch.from_numpy(cov_Y_Y_chol)
        self._cov_Y_Y_inv_Y = torch.from_numpy(cov_Y_Y_inv_Y)
        self.inverse_squared_lengthscales = self.inverse_squared_lengthscales.detach()
        self.inverse_squared_lengthscales.grad = None
        self.kernel_scale = self.kernel_scale.detach()
        self.kernel_scale.grad = None
        self.noise_var = self.noise_var.detach()
        self.noise_var.grad = None

    def kernel(
        self, X1: torch.Tensor | None = None, X2: torch.Tensor | None = None
    ) -> torch.Tensor:
        if X1 is None:
            assert X2 is None
            sqd = self._squared_X_diff
        else:
            if X2 is None:
                X2 = self._X_train
            sqd = (X1 - X2 if X1.ndim == 1 else X1.unsqueeze(-2) - X2.unsqueeze(-3)).square_()
            if self._is_categorical.any():
                sqd[..., self._is_categorical] = (sqd[..., self._is_categorical] > 0.0).to(
                    torch.float64
                )
        sqdist = sqd.matmul(self.inverse_squared_lengthscales)
        return _Matern52Kernel.apply(sqdist) * self.kernel_scale  # type: ignore

    def posterior(
        self, x: torch.Tensor, joint: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._cov_Y_Y_chol is not None and self._cov_Y_Y_inv_Y is not None
        is_single = x.ndim == 1
        x_ = x.unsqueeze(0) if is_single else x
        mean = torch.linalg.vecdot(
            cov_fx_fX := self.kernel(x_, self._X_all), self._cov_Y_Y_inv_Y
        )
        V = torch.linalg.solve_triangular(
            self._cov_Y_Y_chol,
            torch.linalg.solve_triangular(
                self._cov_Y_Y_chol.T, cov_fx_fX, upper=True, left=False
            ),
            upper=False,
            left=False,
        )
        if joint:
            assert not is_single
            cov_fx_fx = self.kernel(x_, x_)
            var_ = cov_fx_fx - V.matmul(cov_fx_fX.transpose(-1, -2))
            var_.diagonal(dim1=-2, dim2=-1).clamp_min_(0.0)
        else:
            cov_fx_fx = self.kernel_scale
            var_ = cov_fx_fx - torch.linalg.vecdot(cov_fx_fX, V)
            var_.clamp_min_(0.0)
        return (mean.squeeze(0), var_.squeeze(0)) if is_single else (mean, var_)

    def marginal_log_likelihood(self) -> torch.Tensor:
        n = self._X_train.shape[0]
        const = -0.5 * n * math.log(2 * math.pi)
        cov_Y_Y = self.kernel() + self.noise_var * torch.eye(n, dtype=torch.float64)
        L = torch.linalg.cholesky(cov_Y_Y)
        logdet_part = -L.diagonal().log().sum()
        inv_L_y = torch.linalg.solve_triangular(L, self._y_train[:, None], upper=False)[:, 0]
        quad_part = -0.5 * (inv_L_y @ inv_L_y)
        return logdet_part + const + quad_part

    def _fit_kernel_params(
        self,
        log_prior: Callable[[GPRegressor], torch.Tensor],
        minimum_noise: float,
        deterministic_objective: bool,
        gtol: float,
    ) -> GPRegressor:
        n_params = self._X_train.shape[1]
        initial_raw = np.concatenate([
            np.log(self.inverse_squared_lengthscales.detach().cpu().numpy()),
            [
                np.log(self.kernel_scale.item()),
                np.log(self.noise_var.item() - 0.99 * minimum_noise),
            ],
        ])

        def loss_fn(raw: np.ndarray) -> tuple[float, np.ndarray]:
            raw_t = torch.from_numpy(raw).requires_grad_(True)
            with torch.enable_grad():
                self.inverse_squared_lengthscales = torch.exp(raw_t[:n_params])
                self.kernel_scale = torch.exp(raw_t[n_params])
                self.noise_var = (
                    torch.tensor(minimum_noise, dtype=torch.float64)
                    if deterministic_objective
                    else torch.exp(raw_t[n_params + 1]) + minimum_noise
                )
                loss = -self.marginal_log_likelihood() - log_prior(self)
                loss.backward()  # type: ignore
            return loss.item(), raw_t.grad.detach().cpu().numpy()  # type: ignore

        with _single_blas_thread_if_needed():
            res = scipy.optimize.minimize(
                loss_fn, initial_raw, jac=True, method="l-bfgs-b",
                options={"gtol": gtol},
            )
        if not res.success:
            raise RuntimeError(f"Kernel hyperparameter optimisation failed: {res.message}")

        raw_opt = torch.from_numpy(res.x)
        self.inverse_squared_lengthscales = torch.exp(raw_opt[:n_params])
        self.kernel_scale = torch.exp(raw_opt[n_params])
        self.noise_var = (
            torch.tensor(minimum_noise, dtype=torch.float64)
            if deterministic_objective
            else minimum_noise + torch.exp(raw_opt[n_params + 1])
        )
        self._cache_matrix()
        return self


# ── Prior (from optuna/_gp/prior.py) ─────────────────────────────────────────

MINIMUM_NOISE_VAR = 1e-6


def default_log_prior(gpr: GPRegressor) -> torch.Tensor:
    """
    Log prior over GP kernel hyperparameters.
    Ported verbatim from optuna/_gp/prior.py::default_log_prior.
    """

    def gamma_log_prior(x: torch.Tensor, concentration: float, rate: float) -> torch.Tensor:
        return (concentration - 1) * torch.log(x) - rate * x

    return (
        -(0.1 / gpr.inverse_squared_lengthscales + 0.1 * gpr.inverse_squared_lengthscales).sum()
        + gamma_log_prior(gpr.kernel_scale, 2, 1)
        + gamma_log_prior(gpr.noise_var, 1.1, 30)
    )


# ── Public fit function (from optuna/_gp/gp.py) ───────────────────────────────

def fit_gp(
    X: np.ndarray,
    Y: np.ndarray,
    is_categorical: np.ndarray,
    log_prior: Callable[[GPRegressor], torch.Tensor] = default_log_prior,
    minimum_noise: float = MINIMUM_NOISE_VAR,
    deterministic_objective: bool = False,
    gtol: float = 1e-2,
) -> GPRegressor:
    """
    Fit a GP to (X, Y) by maximising the penalised marginal log-likelihood.

    Ported verbatim from optuna/_gp/gp.py::fit_kernel_params.

    Parameters
    ----------
    X : shape (n, d) — training inputs, already normalised to [0, 1]
    Y : shape (n,)   — standardised training targets
    is_categorical : shape (d,) bool — True for categorical dimensions
    log_prior : callable — log prior over kernel hyperparameters
    minimum_noise : float — noise floor (prevents degenerate fits)
    deterministic_objective : bool — fix noise to minimum_noise
    gtol : float — L-BFGS-B gradient tolerance

    Returns
    -------
    GPRegressor — fitted GP, with Cholesky factorisation cached
    """
    default_kernel_params = torch.ones(X.shape[1] + 2, dtype=torch.float64)

    def _make_gpr(params: torch.Tensor) -> GPRegressor:
        return GPRegressor(
            is_categorical=torch.from_numpy(is_categorical),
            X_train=torch.from_numpy(X),
            y_train=torch.from_numpy(Y),
            inverse_squared_lengthscales=params[:-2].clone(),
            kernel_scale=params[-2].clone(),
            noise_var=params[-1].clone(),
        )

    default_gpr = _make_gpr(default_kernel_params)
    error = None
    for gpr_init in [_make_gpr(default_kernel_params), default_gpr]:
        try:
            return GPRegressor(
                is_categorical=torch.from_numpy(is_categorical),
                X_train=torch.from_numpy(X),
                y_train=torch.from_numpy(Y),
                inverse_squared_lengthscales=gpr_init.inverse_squared_lengthscales,
                kernel_scale=gpr_init.kernel_scale,
                noise_var=gpr_init.noise_var,
            )._fit_kernel_params(
                log_prior=log_prior,
                minimum_noise=minimum_noise,
                deterministic_objective=deterministic_objective,
                gtol=gtol,
            )
        except RuntimeError as exc:
            error = exc

    logger.warning(
        f"GP kernel optimisation failed ({error}). Falling back to default hyperparameters."
    )
    fallback = _make_gpr(default_kernel_params)
    fallback._cache_matrix()
    return fallback
