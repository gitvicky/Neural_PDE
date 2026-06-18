#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2D velocity-space Fokker-Planck (Lenard-Bernstein) collision operator.

The 2D isotropic Lenard-Bernstein / Dougherty model collision operator on the
velocity plane ``(vx, vy)``::

    df/dt = nu div_v [ (v - u) f + v_th^2 grad_v f ]
          = nu [ d_vx((vx-ux) f) + d_vy((vy-uy) f) + v_th^2 (d2_vx f + d2_vy f) ].

Still linear (fixed background), conserves ``int f dvx dvy``, and relaxes any
distribution to the 2D Maxwellian
``f_M ~ exp(-((vx-ux)^2 + (vy-uy)^2) / 2 v_th^2)``.

The operator separates over the two velocity directions, so the 2D spatial
operator is the **Kronecker sum** of the validated 1D conservative operators
``L_2D = kron(Lx, Iy) + kron(Ix, Ly)`` (C-order flattening, ``vx`` outer / ``vy``
inner).  Conservative finite-volume in each direction + Crank-Nicolson in time
(zero-flux boundaries), so density is conserved and the scheme is unconditionally
stable.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from Neural_PDE.Numerical_Solvers.Fokker_Planck.FokkerPlanck_1D import build_L


def maxwellian_2d(vx: np.ndarray, vy: np.ndarray, u: float, v_th: float) -> np.ndarray:
    """Normalised 2D Maxwellian on the (vx, vy) grid -> (nvx, nvy), unit area."""
    VX, VY = np.meshgrid(vx, vy, indexing="ij")
    f = np.exp(-(((VX - u) ** 2 + (VY - u) ** 2)) / (2.0 * v_th**2))
    dvx = float(vx[1] - vx[0]); dvy = float(vy[1] - vy[0])
    return f / (f.sum() * dvx * dvy)


def build_L2d(vx, vy, nu, v_th, u) -> sp.csc_matrix:
    """Kronecker-sum 2D Lenard-Bernstein operator (C-order: vx outer, vy inner)."""
    Lx = build_L(vx, nu, v_th, u)
    Ly = build_L(vy, nu, v_th, u)
    Ix = sp.identity(len(vx), format="csc")
    Iy = sp.identity(len(vy), format="csc")
    return (sp.kron(Lx, Iy) + sp.kron(Ix, Ly)).tocsc()


class FokkerPlanckSolver2D:
    """Crank-Nicolson integrator for the 2D Lenard-Bernstein operator on [-V,V]^2."""

    def __init__(self, nvx: int = 32, nvy: int = 32, V: float = 5.0, nu: float = 1.0,
                 v_th: float = 1.0, u: float = 0.0, dt: float = 0.1, substeps: int = 4):
        self.nvx, self.nvy = nvx, nvy
        self.vx = np.linspace(-V, V, nvx)
        self.vy = np.linspace(-V, V, nvy)
        self.dvx = float(self.vx[1] - self.vx[0])
        self.dvy = float(self.vy[1] - self.vy[0])
        self.nu, self.v_th, self.u = nu, v_th, u
        self.dt, self.substeps = dt, substeps
        self.N = nvx * nvy
        self.L = build_L2d(self.vx, self.vy, nu, v_th, u)
        dt_sub = dt / substeps
        eye = sp.identity(self.N, format="csc")
        self.A = (eye - 0.5 * dt_sub * self.L).tocsc()
        self.B = (eye + 0.5 * dt_sub * self.L).tocsc()
        self._lu = spla.splu(self.A)

    def step(self, f: np.ndarray) -> np.ndarray:
        """One stored snapshot interval (= substeps CN micro-steps); f flat (N,)."""
        for _ in range(self.substeps):
            f = self._lu.solve(self.B @ f)
        return f

    def rollout(self, f0: np.ndarray, nt: int) -> np.ndarray:
        """March ``f0`` (nvx, nvy) for ``nt`` snapshots -> (nt, nvx, nvy)."""
        out = np.empty((nt, self.nvx, self.nvy), dtype=np.float64)
        out[0] = f0
        f = f0.reshape(-1).copy()
        for t in range(1, nt):
            f = self.step(f)
            out[t] = f.reshape(self.nvx, self.nvy)
        return out


def _sample_ic_2d(rng, vx, vy) -> np.ndarray:
    """Random non-equilibrium 2D initial distribution (drifting Gaussian beams)."""
    VX, VY = np.meshgrid(vx, vy, indexing="ij")
    dvx = float(vx[1] - vx[0]); dvy = float(vy[1] - vy[0])
    f = np.zeros_like(VX)
    for _ in range(rng.integers(1, 3)):
        v0x, v0y = rng.uniform(-2.0, 2.0, size=2)
        w = rng.uniform(0.5, 1.1)
        amp = rng.uniform(0.5, 1.0)
        f += amp * np.exp(-(((VX - v0x) ** 2 + (VY - v0y) ** 2)) / (2.0 * w**2))
    f = np.clip(f, 0.0, None)
    return f / (f.sum() * dvx * dvy)


def generate_dataset(*, n_traj: int, nt: int = 12, nvx: int = 32, nvy: int = 32,
                     V: float = 5.0, nu: float = 1.0, v_th: float = 1.0, u: float = 0.0,
                     dt: float = 0.1, substeps: int = 4, seed: int = 0) -> dict:
    """Generate ``n_traj`` 2D relaxation trajectories -> ``u`` of shape (N, nt, nvx, nvy)."""
    rng = np.random.default_rng(seed)
    solver = FokkerPlanckSolver2D(nvx=nvx, nvy=nvy, V=V, nu=nu, v_th=v_th, u=u,
                                  dt=dt, substeps=substeps)
    traj = np.empty((n_traj, nt, nvx, nvy), dtype=np.float64)
    for k in range(n_traj):
        traj[k] = solver.rollout(_sample_ic_2d(rng, solver.vx, solver.vy), nt)
    return {"u": traj, "vx": solver.vx, "vy": solver.vy, "dt": dt,
            "dvx": solver.dvx, "dvy": solver.dvy, "nt": nt, "nvx": nvx, "nvy": nvy,
            "V": V, "nu": nu, "v_th": v_th, "u_drift": u, "substeps": substeps}


if __name__ == "__main__":
    data = generate_dataset(n_traj=6, nt=12)
    traj, vx, vy = data["u"], data["vx"], data["vy"]
    dvx, dvy = data["dvx"], data["dvy"]
    dens = traj.sum(axis=(-1, -2)) * dvx * dvy            # (N, nt)
    f_eq = maxwellian_2d(vx, vy, data["u_drift"], data["v_th"])
    final_err = np.abs(traj[:, -1] - f_eq).sum(axis=(-1, -2)) * dvx * dvy
    print(f"shape {traj.shape}")
    print(f"density drift (max |rho(t)-rho(0)|) : {np.abs(dens - dens[:, :1]).max():.3e}")
    print(f"||f(T) - Maxwellian||_1 (mean)       : {final_err.mean():.3e}")
    print(f"min f over all trajectories          : {traj.min():.3e}")
