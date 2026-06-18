#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1D linear acoustics (pressure-velocity) -- exact spectral propagator.

First-order linear hyperbolic system on the periodic box ``[0, L]``::

    d p / dt = -K     d u / dx
    d u / dt = -(1/rho) d p / dx

with bulk modulus ``K`` and density ``rho``; the pressure obeys the wave equation
``p_tt = c^2 p_xx`` with sound speed ``c = sqrt(K/rho)``.  This is the physical
two-field analogue of the (synthetic Re/Im) Schrodinger case: a genuine coupled
*vector* PDE, still linear, so the conditioned pushforward applies in closed form.

``AcousticsStep`` is one exact spectral update: in Fourier space each wavenumber
``k`` evolves under the 2x2 generator ``B_k = [[0, -i k K], [-i k / rho, 0]]`` whose
square is ``-(c k)^2 I``, so ``exp(B_k dt) = cos(omega dt) I + sinc(omega dt) B_k``
with ``omega = c|k|`` -- a unitary (norm-preserving) rotation, the data-generating
"numerical solver" of the workflow.  The field is stored as the last axis ``2`` =
``(p, u)`` so it slots into the per-coordinate split-conformal machinery.
"""
from __future__ import annotations

import numpy as np


class AcousticsStep:
    """One exact spectral update on the periodic box ``[0, L]``.

    Operates on the ``(..., nx, 2)`` real layout ``(p, u)``; ``__call__`` advances one
    snapshot interval ``dt``.
    """

    def __init__(self, nx: int, L: float = 2.0, dt: float = 0.0078125,
                 K: float = 1.0, rho: float = 1.0):
        self.nx = nx
        self.L = L
        self.dt = dt
        self.K = K
        self.rho = rho
        self.c = float(np.sqrt(K / rho))
        x = np.linspace(0.0, L, nx, endpoint=False)
        self.x = x
        self.dx = float(x[1] - x[0])
        k = 2.0 * np.pi * np.fft.fftfreq(nx, d=self.dx)
        self.k = k
        omega = self.c * np.abs(k)
        self.cos_t = np.cos(omega * dt)
        # sinc_t = sin(omega dt)/omega, with the k=0 limit dt (the coupling there
        # carries an extra factor k and vanishes, so the value is immaterial).
        with np.errstate(divide="ignore", invalid="ignore"):
            self.sinc_t = np.where(omega > 0, np.sin(omega * dt) / np.where(omega > 0, omega, 1.0), dt)

    def _step_complex(self, p: np.ndarray, u: np.ndarray):
        P = np.fft.fft(p, axis=-1)
        U = np.fft.fft(u, axis=-1)
        coupling_p = self.sinc_t * (-1j * self.k * self.K)
        coupling_u = self.sinc_t * (-1j * self.k / self.rho)
        Pn = self.cos_t * P + coupling_p * U
        Un = self.cos_t * U + coupling_u * P
        p_new = np.real(np.fft.ifft(Pn, axis=-1))
        u_new = np.real(np.fft.ifft(Un, axis=-1))
        return p_new, u_new

    def __call__(self, field2: np.ndarray) -> np.ndarray:
        field2 = np.asarray(field2, dtype=np.float64)
        p, u = field2[..., 0], field2[..., 1]
        p_new, u_new = self._step_complex(p, u)
        return np.stack([p_new, u_new], axis=-1)


def _periodic_gaussian(x: np.ndarray, x0: float, width: float, L: float) -> np.ndarray:
    """Unit-height Gaussian pulse centred at ``x0`` on the periodic box ``[0, L]``."""
    d = np.abs(x - x0)
    d = np.minimum(d, L - d)
    return np.exp(-(d**2) / (2.0 * width**2))


def rollout(step: AcousticsStep, field0: np.ndarray, nt: int) -> np.ndarray:
    """March ``F`` from initial field ``field0`` (..., nx, 2) for ``nt`` slices."""
    out = np.empty(field0.shape[:-2] + (nt,) + field0.shape[-2:], dtype=np.float64)
    s = field0.copy()
    out[..., 0, :, :] = s
    for t in range(1, nt):
        s = step(s)
        out[..., t, :, :] = s
    return out


def generate_dataset(
    *,
    n_traj: int,
    nt: int = 50,
    nx: int = 128,
    L: float = 2.0,
    K: float = 1.0,
    rho: float = 1.0,
    cfl: float = 0.5,
    n_pulses: int = 2,
    seed: int = 0,
) -> dict:
    """Generate ``n_traj`` acoustic-pulse trajectories.

    Each initial condition is a random superposition of Gaussian pressure pulses with
    zero initial velocity; the pulses split into counter-propagating waves that wrap
    around the periodic box.  Truth is the exact spectral march, so ``R(truth) = 0``
    for the spectral operator (and at the FD-truncation floor for an FD score).
    Returns ``u`` of shape ``(n_traj, nt, nx, 2)`` plus grids/metadata.
    """
    rng = np.random.default_rng(seed)
    c = float(np.sqrt(K / rho))
    dx = L / nx
    dt = cfl * dx / c
    step = AcousticsStep(nx, L=L, dt=dt, K=K, rho=rho)
    x = step.x

    fields0 = np.zeros((n_traj, nx, 2), dtype=np.float64)
    for i in range(n_traj):
        n_p = rng.integers(1, n_pulses + 1)
        p0 = np.zeros(nx)
        for _ in range(int(n_p)):
            x0 = rng.uniform(0.0, L)
            width = rng.uniform(0.04 * L, 0.10 * L)
            amp = rng.uniform(0.5, 1.5) * rng.choice([-1.0, 1.0])
            p0 += amp * _periodic_gaussian(x, x0, width, L)
        fields0[i, :, 0] = p0  # u (velocity) starts at rest

    u = rollout(step, fields0, nt)
    return {"u": u, "x": x, "dt": dt, "dx": step.dx, "nt": nt, "nx": nx,
            "L": L, "K": K, "rho": rho, "c": c, "cfl": cfl}


if __name__ == "__main__":
    data = generate_dataset(n_traj=4, nt=50)
    u = data["u"]
    step = AcousticsStep(data["nx"], L=data["L"], dt=data["dt"],
                         K=data["K"], rho=data["rho"])
    R = u[:, 1:] - np.stack([step(u[:, t]) for t in range(u.shape[1] - 1)], axis=1)
    energy = (0.5 * data["K"] * u[..., 0] ** 2 + 0.5 * data["rho"] * u[..., 1] ** 2).sum(-1) * data["dx"]
    print("field shape:", u.shape, " c =", data["c"], " dt =", data["dt"], " dx =", data["dx"])
    print("R(truth) max abs:", float(np.abs(R).max()))
    print("acoustic energy drift:", float(energy.std() / energy.mean()))
