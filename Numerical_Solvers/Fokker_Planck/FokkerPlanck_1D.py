#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1D velocity-space Fokker-Planck (Lenard-Bernstein / Dougherty) collision operator.

The Lenard-Bernstein model collision operator -- the standard linear Fokker-Planck
collision operator in gyrokinetic fusion codes (GENE, GS2, ...) -- for the 1D
velocity distribution ``f(v, t)``::

    df/dt = nu d/dv [ (v - u) f + v_th^2 df/dv ],

i.e. a drag (Ornstein-Uhlenbeck restoring drift toward the mean flow ``u``) plus
velocity-space diffusion with thermal speed ``v_th = sqrt(T/m)``.  It is *linear*
(coefficients are fixed background quantities), conserves particle number, and
relaxes any initial distribution to the Maxwellian equilibrium
``f_M(v) ~ exp(-(v-u)^2 / 2 v_th^2)`` (the zero-flux state).

Discretisation: a conservative finite-volume scheme on cell nodes with zero-flux
(reflecting) boundaries -- so density ``sum f dv`` is conserved exactly -- marched
in time by Crank-Nicolson (unconditionally stable; the operator is parabolic, so
an explicit march is CFL-limited).  Each stored snapshot is ``substeps`` CN
micro-steps, so the residual evaluated at the snapshot ``dt`` (used by the
conditioned-pushforward experiment) is *not* exactly the data-generating step --
``R(truth)`` sits at the substep-truncation floor, mirroring the Schrodinger case.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def maxwellian(v: np.ndarray, u: float, v_th: float) -> np.ndarray:
    """Normalised Maxwellian ``exp(-(v-u)^2/2 v_th^2)`` (unit area on the grid)."""
    f = np.exp(-((v - u) ** 2) / (2.0 * v_th**2))
    dv = float(v[1] - v[0])
    return f / (f.sum() * dv)


def build_L(v: np.ndarray, nu: float, v_th: float, u: float) -> sp.csc_matrix:
    """Conservative finite-volume tridiagonal operator ``L`` with ``df/dt = L f``.

    Flux ``Gamma_{i+1/2} = -nu[(v_{i+1/2}-u) f_{i+1/2} + v_th^2 (f_{i+1}-f_i)/dv]``
    with a central face value, zero flux through the domain ends.  The column sums of
    ``L`` vanish, so the scheme conserves ``sum f dv`` exactly and its zero-flux
    steady state is the (discrete) Maxwellian.
    """
    nv = len(v)
    dv = float(v[1] - v[0])
    b = v_th**2 / dv
    coef = nu / dv
    lower = np.zeros(nv)   # lower[i] = L[i, i-1]
    diag = np.zeros(nv)    # diag[i]  = L[i, i]
    upper = np.zeros(nv)   # upper[i] = L[i, i+1]
    for i in range(nv):
        if i == 0:
            ap = (v[0] + dv / 2.0) - u
            diag[0] = coef * (ap / 2.0 - b)
            upper[0] = coef * (ap / 2.0 + b)
        elif i == nv - 1:
            am = (v[-1] - dv / 2.0) - u
            lower[-1] = coef * (b - am / 2.0)
            diag[-1] = -coef * (am / 2.0 + b)
        else:
            ap = (v[i] + dv / 2.0) - u
            am = (v[i] - dv / 2.0) - u
            lower[i] = coef * (b - am / 2.0)
            diag[i] = coef * (dv / 2.0 - 2.0 * b)
            upper[i] = coef * (ap / 2.0 + b)
    return sp.diags([lower[1:], diag, upper[:-1]], [-1, 0, 1], format="csc")


class FokkerPlanckSolver:
    """Crank-Nicolson integrator for the Lenard-Bernstein operator on ``[-V, V]``."""

    def __init__(self, nv: int = 128, V: float = 6.0, nu: float = 1.0,
                 v_th: float = 1.0, u: float = 0.0, dt: float = 0.06,
                 substeps: int = 4):
        self.nv = nv
        self.v = np.linspace(-V, V, nv)
        self.dv = float(self.v[1] - self.v[0])
        self.nu, self.v_th, self.u = nu, v_th, u
        self.dt, self.substeps = dt, substeps
        self.L = build_L(self.v, nu, v_th, u)
        dt_sub = dt / substeps
        eye = sp.identity(nv, format="csc")
        self.A = (eye - 0.5 * dt_sub * self.L).tocsc()
        self.B = (eye + 0.5 * dt_sub * self.L).tocsc()
        self._lu = spla.splu(self.A)

    def step(self, f: np.ndarray) -> np.ndarray:
        """Advance one stored snapshot interval (= ``substeps`` CN micro-steps)."""
        for _ in range(self.substeps):
            f = self._lu.solve(self.B @ f)
        return f

    def rollout(self, f0: np.ndarray, nt: int) -> np.ndarray:
        """March ``f0`` for ``nt`` snapshots -> ``(nt, nv)``."""
        out = np.empty((nt, self.nv), dtype=np.float64)
        out[0] = f0
        f = f0.copy()
        for t in range(1, nt):
            f = self.step(f)
            out[t] = f
        return out


def _sample_ic(rng: np.random.Generator, v: np.ndarray) -> np.ndarray:
    """A random non-equilibrium initial distribution (drifting/bump-on-tail beam)."""
    dv = float(v[1] - v[0])
    # One or two Gaussian beams away from the bulk; relax toward the Maxwellian.
    n_bumps = rng.integers(1, 3)
    f = np.zeros_like(v)
    for _ in range(n_bumps):
        v0 = rng.uniform(-2.5, 2.5)
        w = rng.uniform(0.4, 1.0)
        amp = rng.uniform(0.5, 1.0)
        f += amp * np.exp(-((v - v0) ** 2) / (2.0 * w**2))
    f = np.clip(f, 0.0, None)
    return f / (f.sum() * dv)   # unit density


def generate_dataset(*, n_traj: int, nt: int = 50, nv: int = 128, V: float = 6.0,
                     nu: float = 1.0, v_th: float = 1.0, u: float = 0.0,
                     dt: float = 0.06, substeps: int = 4, seed: int = 0) -> dict:
    """Generate ``n_traj`` relaxation trajectories -> dict with ``u`` of shape (N, nt, nv)."""
    rng = np.random.default_rng(seed)
    solver = FokkerPlanckSolver(nv=nv, V=V, nu=nu, v_th=v_th, u=u, dt=dt, substeps=substeps)
    traj = np.empty((n_traj, nt, nv), dtype=np.float64)
    for k in range(n_traj):
        traj[k] = solver.rollout(_sample_ic(rng, solver.v), nt)
    return {"u": traj, "v": solver.v, "dt": dt, "dv": solver.dv, "nt": nt, "nv": nv,
            "V": V, "nu": nu, "v_th": v_th, "u_drift": u, "substeps": substeps}


if __name__ == "__main__":
    data = generate_dataset(n_traj=8, nt=50)
    traj, v, dv = data["u"], data["v"], data["dv"]
    dens = traj.sum(-1) * dv                          # (N, nt)
    f_eq = maxwellian(v, data["u_drift"], data["v_th"])
    # Entropy S = -int f ln f dv should increase (H-theorem).
    fpos = np.clip(traj, 1e-30, None)
    S = -(fpos * np.log(fpos)).sum(-1) * dv
    final_err = np.abs(traj[:, -1] - f_eq).sum(-1) * dv
    print(f"shape {traj.shape}")
    print(f"density drift (max |rho(t)-rho(0)|) : {np.abs(dens - dens[:, :1]).max():.3e}")
    print(f"entropy non-decreasing fraction      : {(np.diff(S, axis=1) >= -1e-9).mean():.3f}")
    print(f"||f(T) - Maxwellian||_1 (mean)       : {final_err.mean():.3e}")
    print(f"min f over all trajectories          : {traj.min():.3e}")
