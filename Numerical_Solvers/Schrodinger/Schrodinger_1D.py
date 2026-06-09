#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1D time-dependent Schrödinger equation — split-step Fourier propagator.

    i d psi/dt = -1/2 d^2 psi/dx^2 + V(x) psi,   V(x) = 1/2 x^2  (harmonic well).

``psi`` is complex.  ``SchrodingerStep`` is one **Strang split-step Fourier**
update (half potential kick, full kinetic drift in Fourier space, half potential
kick) — a unitary, norm-preserving propagator that is the exact explicit discrete
update ``F`` used both to generate the ground truth and to drive the conditioned
march, so the one-step defect ``R = psi^{n+1} - F(psi^n)`` is zero on the truth.

The field is stored as the last axis ``2`` = ``(Re psi, Im psi)`` so it slots into
the real-valued per-coordinate split-conformal machinery (mirroring NS's
``(..., 2)`` velocity layout).
"""
from __future__ import annotations

import numpy as np


def to_complex(field2: np.ndarray) -> np.ndarray:
    """``(..., 2)`` real/imag -> complex ``(...)``."""
    return field2[..., 0] + 1j * field2[..., 1]


def to_real2(psi: np.ndarray) -> np.ndarray:
    """complex ``(...)`` -> ``(..., 2)`` real/imag."""
    return np.stack([psi.real, psi.imag], axis=-1)


class SchrodingerStep:
    """One Strang split-step Fourier update on the periodic box ``[-L, L]``.

    Operates on the real/imag ``(..., nx, 2)`` layout; ``__call__`` advances one
    snapshot interval = ``substeps`` split-steps of size ``dt_solver``.
    """

    def __init__(self, nx: int, L: float = 10.0, dt_solver: float = 0.01,
                 substeps: int = 5):
        self.nx = nx
        self.L = L
        x = np.linspace(-L, L, nx, endpoint=False)
        self.x = x
        self.dx = float(x[1] - x[0])
        self.V = 0.5 * x**2
        k = 2.0 * np.pi * np.fft.fftfreq(nx, d=self.dx)
        self.k2 = k**2
        self.dt = dt_solver
        self.substeps = substeps
        # Pre-baked half-potential and full-kinetic phase factors.
        self._half_pot = np.exp(-1j * self.V * (0.5 * dt_solver))
        self._kin = np.exp(-1j * 0.5 * self.k2 * dt_solver)

    def _step_complex(self, psi: np.ndarray) -> np.ndarray:
        hp, kin = self._half_pot, self._kin
        for _ in range(self.substeps):
            psi = hp * psi
            psi = np.fft.ifft(kin * np.fft.fft(psi, axis=-1), axis=-1)
            psi = hp * psi
        return psi

    def __call__(self, field2: np.ndarray) -> np.ndarray:
        psi = to_complex(np.asarray(field2, dtype=np.float64))
        return to_real2(self._step_complex(psi))


def coherent_state(x: np.ndarray, x0: float, k0: float = 0.0) -> np.ndarray:
    """Normalised Gaussian wavepacket (width 1) centred at ``x0`` with momentum ``k0``."""
    psi = (1.0 / np.pi) ** 0.25 * np.exp(-((x - x0) ** 2) / 2.0 + 1j * k0 * x)
    return psi


def rollout(step: SchrodingerStep, psi0_2: np.ndarray, nt: int) -> np.ndarray:
    """March ``F`` from initial field ``psi0_2`` (..., nx, 2) for ``nt`` slices."""
    out = np.empty(psi0_2.shape[:-2] + (nt,) + psi0_2.shape[-2:], dtype=np.float64)
    s = psi0_2.copy()
    out[..., 0, :, :] = s
    for t in range(1, nt):
        s = step(s)
        out[..., t, :, :] = s
    return out


def generate_dataset(
    *,
    n_traj: int,
    nt: int = 40,
    nx: int = 128,
    L: float = 10.0,
    dt_solver: float = 0.01,
    substeps: int = 5,
    x0_range: float = 2.0,
    k0_range: float = 0.5,
    seed: int = 0,
) -> dict:
    """Generate ``n_traj`` coherent-state trajectories sloshing in the well.

    Each IC is a width-1 Gaussian with displacement ``x0`` and momentum ``k0``
    sampled uniformly; the harmonic well makes it oscillate.  Truth is the exact
    split-step march, so ``R(truth) = 0``.  Returns ``u`` of shape
    ``(n_traj, nt, nx, 2)`` plus grids/metadata.
    """
    rng = np.random.default_rng(seed)
    step = SchrodingerStep(nx, L=L, dt_solver=dt_solver, substeps=substeps)
    x = step.x
    x0 = rng.uniform(-x0_range, x0_range, n_traj)
    k0 = rng.uniform(-k0_range, k0_range, n_traj)
    psi0 = np.stack([coherent_state(x, x0[i], k0[i]) for i in range(n_traj)], axis=0)
    psi0_2 = to_real2(psi0)
    u = rollout(step, psi0_2, nt)
    dt = dt_solver * substeps
    return {"u": u, "x": x, "dt": dt, "dx": step.dx, "nt": nt, "nx": nx,
            "L": L, "dt_solver": dt_solver, "substeps": substeps}


if __name__ == "__main__":
    data = generate_dataset(n_traj=4, nt=40)
    u = data["u"]
    step = SchrodingerStep(data["nx"], L=data["L"], dt_solver=data["dt_solver"],
                           substeps=data["substeps"])
    R = u[:, 1:] - np.stack([step(u[:, t]) for t in range(u.shape[1] - 1)], axis=1)
    # Norm conservation check.
    psi = to_complex(u)
    norm = (np.abs(psi) ** 2).sum(-1) * data["dx"]
    print("field shape:", u.shape)
    print("R(truth) max abs:", float(np.abs(R).max()))
    print("norm drift     :", float(norm.std()))
