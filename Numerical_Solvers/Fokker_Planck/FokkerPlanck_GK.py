#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gyrokinetic (v_par, v_perp) velocity-space Fokker-Planck (Lenard-Bernstein).

The Lenard-Bernstein / Dougherty model collision operator in the gyrokinetic
velocity coordinates ``(v_parallel, v_perp)``, with the cylindrical perpendicular
metric (gyro-angle integrated out, Jacobian ``2 pi v_perp``)::

    C[f] = nu d_vpar[(v_par - u_par) f + v_th^2 d_vpar f]
         + nu (1/v_perp) d_vperp[ v_perp ( v_perp f + v_th^2 d_vperp f ) ].

Still *linear* (fixed background ``nu``, ``v_th``, ``u_par``), it conserves the
weighted density ``int f * 2 pi v_perp dv_par dv_perp`` and relaxes to the
Maxwellian ``f_M ~ exp(-((v_par-u_par)^2 + v_perp^2)/2 v_th^2)``.

The parallel direction reuses the validated uniform-grid 1D operator
(``FokkerPlanck_1D.build_L``); the perpendicular direction uses a conservative,
Jacobian-weighted finite-volume operator on cell centres ``v_perp,i=(i+1/2) dvp``,
with the inner face at ``v_perp=0`` carrying zero flux automatically (regularity)
and zero flux at the outer boundary.  The full 2D operator is the Kronecker sum
``L_GK = kron(L_par, I_perp) + kron(I_par, L_perp)`` (C-order: v_par outer, v_perp
inner).  Crank-Nicolson in time (unconditionally stable), several micro-steps per
stored snapshot.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from Neural_PDE.Numerical_Solvers.Fokker_Planck.FokkerPlanck_1D import build_L


def perp_grid(n_perp: int, V_perp: float):
    """Cell-centred v_perp grid: returns (centres, faces, cell_volumes W_i)."""
    dvp = V_perp / n_perp
    faces = np.arange(n_perp + 1) * dvp           # vf_0 = 0 ... vf_n = V_perp
    centres = (np.arange(n_perp) + 0.5) * dvp
    W = (faces[1:] ** 2 - faces[:-1] ** 2) / 2.0  # int_cell v_perp dv_perp
    return centres, faces, W, dvp


def build_L_perp(n_perp: int, V_perp: float, nu: float, v_th: float):
    """Conservative cylindrical FV operator for the v_perp part; returns (L_perp, W)."""
    centres, faces, W, dvp = perp_grid(n_perp, V_perp)
    p = v_th**2 / dvp
    lower = np.zeros(n_perp); diag = np.zeros(n_perp); upper = np.zeros(n_perp)
    for i in range(n_perp):
        cL = nu * faces[i]        # inner face flux coeff c_i = nu vf_i
        cR = nu * faces[i + 1]    # outer face flux coeff c_{i+1}
        Wi = W[i]
        # interior-face contributions (Gamma_j = -c_j[(vf_j/2 - p) f_{j-1} + (vf_j/2 + p) f_j])
        if i > 0:  # inner face is interior
            lower[i] = -(1.0 / Wi) * cL * (faces[i] / 2.0 - p)
            diag[i] += -(1.0 / Wi) * cL * (faces[i] / 2.0 + p)
        if i < n_perp - 1:  # outer face is interior
            diag[i] += (1.0 / Wi) * cR * (faces[i + 1] / 2.0 - p)
            upper[i] = (1.0 / Wi) * cR * (faces[i + 1] / 2.0 + p)
        # boundary faces (vf_0 = 0 and vf_n = V_perp) carry zero flux -> nothing added
    L = sp.diags([lower[1:], diag, upper[:-1]], [-1, 0, 1], format="csc")
    return L, W


def build_L_gk(vpar, n_perp, V_perp, nu, v_th, u_par):
    """Kronecker-sum 2D (v_par, v_perp) operator + the velocity-space weight vector."""
    Lpar = build_L(vpar, nu, v_th, u_par)
    Lperp, Wperp = build_L_perp(n_perp, V_perp, nu, v_th)
    Ipar = sp.identity(len(vpar), format="csc")
    Iperp = sp.identity(n_perp, format="csc")
    L = (sp.kron(Lpar, Iperp) + sp.kron(Ipar, Lperp)).tocsc()
    dvpar = float(vpar[1] - vpar[0])
    weight = np.outer(np.full(len(vpar), dvpar), Wperp).reshape(-1)  # dv_par * W_perp
    return L, weight


def maxwellian_gk(vpar, vperp, u_par, v_th):
    """2D (v_par, v_perp) Maxwellian on the grid -> (n_par, n_perp), unit weighted area."""
    VPAR, VPERP = np.meshgrid(vpar, vperp, indexing="ij")
    return np.exp(-(((VPAR - u_par) ** 2 + VPERP**2)) / (2.0 * v_th**2))


class FokkerPlanckSolverGK:
    """Crank-Nicolson integrator for the (v_par, v_perp) Lenard-Bernstein operator."""

    def __init__(self, n_par: int = 48, n_perp: int = 24, V_par: float = 5.0,
                 V_perp: float = 5.0, nu: float = 1.0, v_th: float = 1.0,
                 u_par: float = 0.0, dt: float = 0.1, substeps: int = 4):
        self.n_par, self.n_perp = n_par, n_perp
        self.vpar = np.linspace(-V_par, V_par, n_par)
        self.vperp, self.faces, self.Wperp, self.dvp = perp_grid(n_perp, V_perp)
        self.nu, self.v_th, self.u_par = nu, v_th, u_par
        self.dt, self.substeps = dt, substeps
        self.N = n_par * n_perp
        self.L, self.weight = build_L_gk(self.vpar, n_perp, V_perp, nu, v_th, u_par)
        dt_sub = dt / substeps
        eye = sp.identity(self.N, format="csc")
        self.A = (eye - 0.5 * dt_sub * self.L).tocsc()
        self.B = (eye + 0.5 * dt_sub * self.L).tocsc()
        self._lu = spla.splu(self.A)

    def step(self, f):
        for _ in range(self.substeps):
            f = self._lu.solve(self.B @ f)
        return f

    def rollout(self, f0, nt):
        out = np.empty((nt, self.n_par, self.n_perp), dtype=np.float64)
        out[0] = f0
        f = f0.reshape(-1).copy()
        for t in range(1, nt):
            f = self.step(f)
            out[t] = f.reshape(self.n_par, self.n_perp)
        return out


def _sample_ic_gk(rng, vpar, vperp, weight, shape):
    """Random non-equilibrium (v_par, v_perp) beam(s); normalised to unit weighted mass."""
    VPAR, VPERP = np.meshgrid(vpar, vperp, indexing="ij")
    f = np.zeros_like(VPAR)
    for _ in range(rng.integers(1, 3)):
        v0par = rng.uniform(-2.0, 2.0)
        v0perp = rng.uniform(0.3, 2.5)
        w = rng.uniform(0.5, 1.1)
        amp = rng.uniform(0.5, 1.0)
        f += amp * np.exp(-(((VPAR - v0par) ** 2 + (VPERP - v0perp) ** 2)) / (2.0 * w**2))
    f = np.clip(f, 0.0, None)
    mass = float((f.reshape(-1) * weight).sum())
    return f / mass


def generate_dataset(*, n_traj, nt=12, n_par=48, n_perp=24, V_par=5.0, V_perp=5.0,
                     nu=1.0, v_th=1.0, u_par=0.0, dt=0.1, substeps=4, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    s = FokkerPlanckSolverGK(n_par=n_par, n_perp=n_perp, V_par=V_par, V_perp=V_perp,
                             nu=nu, v_th=v_th, u_par=u_par, dt=dt, substeps=substeps)
    traj = np.empty((n_traj, nt, n_par, n_perp), dtype=np.float64)
    for k in range(n_traj):
        traj[k] = s.rollout(_sample_ic_gk(rng, s.vpar, s.vperp, s.weight, (n_par, n_perp)), nt)
    return {"u": traj, "vpar": s.vpar, "vperp": s.vperp, "weight": s.weight,
            "dt": dt, "n_par": n_par, "n_perp": n_perp, "V_par": V_par, "V_perp": V_perp,
            "nu": nu, "v_th": v_th, "u_par": u_par, "substeps": substeps}


if __name__ == "__main__":
    data = generate_dataset(n_traj=6, nt=12)
    traj, w = data["u"], data["weight"]
    flat = traj.reshape(traj.shape[0], traj.shape[1], -1)
    dens = (flat * w).sum(-1)                                  # weighted density (N, nt)
    f_eq = maxwellian_gk(data["vpar"], data["vperp"], data["u_par"], data["v_th"])
    f_eq = f_eq / float((f_eq.reshape(-1) * w).sum())
    final_err = (np.abs(flat[:, -1] - f_eq.reshape(-1)) * w).sum(-1)
    print(f"shape {traj.shape}")
    print(f"weighted density drift (max)  : {np.abs(dens - dens[:, :1]).max():.3e}")
    print(f"||f(T)-Maxwellian||_w (mean)  : {final_err.mean():.3e}")
    print(f"min f                         : {traj.min():.3e}")
