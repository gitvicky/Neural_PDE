#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1D Black-Scholes option pricing — analytic generator + FD reference.

The Black-Scholes PDE for a European option value ``V(S, t)`` is

    V_t + 1/2 sigma^2 S^2 V_SS + r S V_S - r V = 0,

a *backward* parabolic equation with terminal payoff at maturity ``T``.  Under
the log-price change of variables ``x = ln S`` and time-to-maturity
``tau = T - t`` it becomes a **constant-coefficient** advection-diffusion-
reaction equation marched *forward* in ``tau``:

    V_tau = 1/2 sigma^2 V_xx + (r - 1/2 sigma^2) V_x - r V.

This is structurally ConvDiff plus a ``-r V`` reaction term, so the residual is a
single constant 3x3 (time, x) finite-difference stencil and the FFT/Green's-
function pushforward applies directly (see ``Expts/BlackScholes/BlackScholes_Model.py``).

Crucially, the European call/put admits a **closed-form** solution, so the
ground-truth field is exact (no solver error) and the data-free precondition
``R(V_true) ~ 0`` holds up to finite-difference truncation only.  We generate a
family of option-value fields by varying the strike ``K`` across trajectories
while holding ``sigma`` and ``r`` fixed (so the residual operator is constant).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bs_call(S: np.ndarray, K: float, r: float, sigma: float, tau: np.ndarray) -> np.ndarray:
    """Closed-form Black-Scholes European call value.

    ``S`` (spot) and ``tau`` (time to maturity > 0) broadcast against each other.
    """
    S = np.asarray(S, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    sqrt_tau = np.sqrt(np.maximum(tau, 1e-12))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


def generate_call_field(
    *,
    K: float,
    r: float,
    sigma: float,
    x: np.ndarray,
    tau: np.ndarray,
) -> np.ndarray:
    """Analytic call-value field ``V[i_tau, i_x]`` on the log-price grid ``x``.

    ``x = ln S``; ``tau`` is the increasing time-to-maturity grid.  Returned shape
    is ``(len(tau), len(x))`` to match the ConvDiff ``(time, x)`` convention.
    """
    S = np.exp(x)[None, :]            # (1, nx)
    tau_col = np.asarray(tau)[:, None]  # (nt, 1)
    return bs_call(S, K, r, sigma, tau_col)


def generate_dataset(
    *,
    n_traj: int,
    nt: int = 20,
    nx: int = 72,
    r: float = 0.05,
    sigma: float = 0.2,
    tau_min: float = 0.1,
    tau_max: float = 1.0,
    x_min: float = -1.2,
    x_max: float = 1.1,
    strike_range: tuple[float, float] = (0.85, 1.15),
    seed: int = 0,
) -> dict:
    """Generate ``n_traj`` analytic call-value fields with varied strikes.

    ``sigma`` and ``r`` are held fixed so the residual operator is a single
    constant stencil; only the strike ``K`` (hence moneyness) varies across
    trajectories.  Returns a dict with ``u`` of shape ``(n_traj, nt, nx)`` plus
    the grids and parameters.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(x_min, x_max, nx)
    tau = np.linspace(tau_min, tau_max, nt)
    strikes = rng.uniform(strike_range[0], strike_range[1], size=n_traj)

    fields = np.empty((n_traj, nt, nx), dtype=np.float64)
    for i, K in enumerate(strikes):
        fields[i] = generate_call_field(K=K, r=r, sigma=sigma, x=x, tau=tau)

    dt = float(tau[1] - tau[0])
    dx = float(x[1] - x[0])
    return {
        "u": fields,
        "x": x,
        "tau": tau,
        "strikes": strikes,
        "dt": dt,
        "dx": dx,
        "r": r,
        "sigma": sigma,
    }


if __name__ == "__main__":
    data = generate_dataset(n_traj=4)
    print("field shape:", data["u"].shape, "dt", data["dt"], "dx", data["dx"])
    print("value range:", float(data["u"].min()), float(data["u"].max()))
