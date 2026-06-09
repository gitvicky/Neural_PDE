#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Autonomous CartPole dynamics + explicit RK4 integrator.

The classic cart-pole is a 4-state nonlinear ODE for ``s = (x, x_dot, theta,
theta_dot)`` (pole angle ``theta`` measured from upright).  We use the *autonomous*
case (external force ``F = 0``); the pole topples / swings under gravity from a
perturbed initial condition.  A note for later: re-introducing a control law
``F = -K s`` (LQR) only changes ``_accel`` below — the integrator, residual and
conditioned march are unchanged.

``CartPoleStep`` is the exact explicit forward update ``F`` (one RK4 step of size
``dt``).  It is BOTH the data generator and the ``update_fn`` for the conditioned
march, so the one-step solver defect ``R = s^{n+1} - F(s^n)`` is zero on the
generated ground truth and the data-free precondition holds exactly.

State arrays carry the 4 components in the last axis, shape ``(..., 4)``.
"""
from __future__ import annotations

import numpy as np

# Standard Gym cart-pole constants.
GRAVITY = 9.8
MASS_CART = 1.0
MASS_POLE = 0.1
LENGTH = 0.5  # actually half the pole length
TOTAL_MASS = MASS_CART + MASS_POLE
POLEMASS_LENGTH = MASS_POLE * LENGTH


def _accel(state: np.ndarray, force: float = 0.0) -> np.ndarray:
    """Continuous-time derivative ``ds/dt`` for the autonomous cart-pole."""
    x_dot = state[..., 1]
    theta = state[..., 2]
    theta_dot = state[..., 3]
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    temp = (force + POLEMASS_LENGTH * theta_dot**2 * sin_t) / TOTAL_MASS
    theta_acc = (GRAVITY * sin_t - cos_t * temp) / (
        LENGTH * (4.0 / 3.0 - MASS_POLE * cos_t**2 / TOTAL_MASS)
    )
    x_acc = temp - POLEMASS_LENGTH * theta_acc * cos_t / TOTAL_MASS

    deriv = np.empty_like(state)
    deriv[..., 0] = x_dot
    deriv[..., 1] = x_acc
    deriv[..., 2] = theta_dot
    deriv[..., 3] = theta_acc
    return deriv


class CartPoleStep:
    """Exact explicit one-step update ``F`` (RK4 of step ``dt``), force ``F=0``."""

    def __init__(self, dt: float = 0.02):
        self.dt = float(dt)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float64)
        dt = self.dt
        k1 = _accel(s)
        k2 = _accel(s + 0.5 * dt * k1)
        k3 = _accel(s + 0.5 * dt * k2)
        k4 = _accel(s + dt * k3)
        return s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rollout(step: CartPoleStep, s0: np.ndarray, nt: int) -> np.ndarray:
    """March ``F`` from initial states ``s0`` (..., 4) for ``nt`` slices.

    Returns ``(..., nt, 4)`` with slice 0 = ``s0``.
    """
    s0 = np.asarray(s0, dtype=np.float64)
    out = np.empty(s0.shape[:-1] + (nt, 4), dtype=np.float64)
    s = s0.copy()
    out[..., 0, :] = s
    for t in range(1, nt):
        s = step(s)
        out[..., t, :] = s
    return out


def generate_dataset(
    *,
    n_traj: int,
    nt: int = 30,
    dt: float = 0.02,
    theta0_range: float = 0.2,
    thetadot0_range: float = 0.5,
    xdot0_range: float = 0.5,
    seed: int = 0,
) -> dict:
    """Generate ``n_traj`` autonomous cart-pole trajectories from perturbed ICs.

    ICs sample ``theta0, theta_dot0, x_dot0`` uniformly (``x0 = 0``); the pole
    topples under gravity.  Truth is the exact RK4 march, so ``R(truth) = 0``.
    Returns ``u`` of shape ``(n_traj, nt, 4)`` plus the integrator metadata.
    """
    rng = np.random.default_rng(seed)
    s0 = np.zeros((n_traj, 4))
    s0[:, 1] = rng.uniform(-xdot0_range, xdot0_range, n_traj)       # x_dot
    s0[:, 2] = rng.uniform(-theta0_range, theta0_range, n_traj)     # theta
    s0[:, 3] = rng.uniform(-thetadot0_range, thetadot0_range, n_traj)  # theta_dot
    step = CartPoleStep(dt=dt)
    u = rollout(step, s0, nt)
    return {"u": u, "dt": dt, "nt": nt, "labels": ["x", "x_dot", "theta", "theta_dot"]}


if __name__ == "__main__":
    data = generate_dataset(n_traj=4, nt=30)
    u = data["u"]
    step = CartPoleStep(dt=data["dt"])
    # R(truth) defect, should be ~machine zero.
    R = u[:, 1:] - np.stack([step(u[:, t]) for t in range(u.shape[1] - 1)], axis=1)
    print("traj shape:", u.shape)
    print("R(truth) max abs:", float(np.abs(R).max()))
