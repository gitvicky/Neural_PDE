#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1D-1V Vlasov-Fokker-Planck: phase-space transport + collisions, prescribed field.

The kinetic equation for ``f(x, v, t)`` on a periodic configuration-space slab with a
Lenard-Bernstein collision operator and an **externally prescribed** (static, given)
electric field::

    df/dt + v df/dx - a(x) df/dv = nu d/dv [ (v - u) f + v_th^2 df/dv ],

i.e. free streaming ``v df/dx``, acceleration ``-a(x) df/dv`` in the prescribed field
``a(x) = q E(x) / m`` (here ``a(x) = accel_amp * sin(kx x)``), and velocity-space
collisional relaxation toward the Maxwellian.  This is the qualitative step beyond the
pure velocity-space collision cases (``FokkerPlanck_{1D,2D,GK}``): it transports in the
full ``(x, v)`` phase space rather than relaxing in velocity alone.

**Linearity is preserved by holding the field external.**  Because ``a(x)`` is given
(not closed self-consistently through Poisson, which would make this Vlasov-Poisson and
*non*-linear), every term is linear in ``f`` and the whole right-hand side is a single
fixed operator ``df/dt = L f``.  ``L`` is assembled as a Kronecker expression on the
flattened ``(x, v)`` grid (C-order, ``x`` outer / ``v`` inner)::

    L = - kron(D_x, diag(v))        # streaming      -v df/dx
        - kron(diag(a), A_v)        # acceleration   -a(x) df/dv
        + kron(I_x, L_collision)    # collisions     C_LB[f]

so it drops straight into the shared ``space_time_from_L`` builder and the
Crank-Nicolson time-march solver used by the rest of the Fokker-Planck family.

Discretisation: central differences for ``D_x`` (periodic -> skew-symmetric, conservative)
and a conservative central-flux operator ``A_v`` for the velocity advection (zero flux at
the velocity ends, where ``f ~ 0``), plus the validated conservative LB operator
``build_L`` for the collisions.  All three blocks have vanishing column sums, so the
total mass ``sum f dx dv`` is conserved exactly.  Crank-Nicolson in time: the collision
part is parabolic (an explicit march would be CFL-limited) and CN is unconditionally
stable for the whole operator (advection eigenvalues are imaginary, collisions damp), so
``M_U`` stays well conditioned for the conditioned pushforward.  Each stored snapshot is
``substeps`` CN micro-steps, so the residual at the snapshot ``dt`` sits at the
substep-truncation floor (``R(truth) != 0``), mirroring the other FP cases.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from Neural_PDE.Numerical_Solvers.Fokker_Planck.FokkerPlanck_1D import build_L, maxwellian


def _periodic_d1(n: int, h: float) -> sp.csc_matrix:
    """Periodic central first-difference ``(f_{i+1}-f_{i-1})/(2h)`` -> skew, conservative."""
    i = np.arange(n)
    rows = np.concatenate([i, i])
    cols = np.concatenate([(i + 1) % n, (i - 1) % n])
    vals = np.concatenate([np.full(n, 0.5 / h), np.full(n, -0.5 / h)])
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()


def _advect_zeroflux(n: int, h: float) -> sp.csc_matrix:
    """Conservative central-flux first-difference in velocity with zero flux at the ends.

    Flux ``F_{j+1/2} = (f_j + f_{j+1})/2`` (the field factor ``a`` is folded in by the
    caller), ``(A_v f)_j = (F_{j+1/2} - F_{j-1/2})/h``.  Interior rows are the plain
    central difference; the zero-flux end closure (``F_{-1/2}=F_{n-1/2}=0``) gives the
    boundary rows.  Column sums vanish, so the operator conserves ``sum f``.
    """
    rows, cols, vals = [], [], []
    for j in range(n):
        if j == 0:                                   # F_{-1/2} = 0
            rows += [0, 0]; cols += [0, 1]; vals += [0.5 / h, 0.5 / h]
        elif j == n - 1:                             # F_{n-1/2} = 0
            rows += [j, j]; cols += [j - 1, j]; vals += [-0.5 / h, -0.5 / h]
        else:
            rows += [j, j]; cols += [j - 1, j + 1]; vals += [-0.5 / h, 0.5 / h]
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()


def accel_profile(x: np.ndarray, accel_amp: float, kx: float) -> np.ndarray:
    """Prescribed acceleration ``a(x) = accel_amp * sin(kx x)`` (= q E(x) / m)."""
    return accel_amp * np.sin(kx * np.asarray(x, dtype=np.float64))


def build_L_vfp(x: np.ndarray, v: np.ndarray, nu: float, v_th: float, u: float,
                accel: np.ndarray) -> sp.csc_matrix:
    """Kronecker spatial operator ``L`` for the 1D-1V Vlasov-Fokker-Planck step.

    ``df/dt = L f`` on the flattened ``(x, v)`` grid (C-order, ``x`` outer / ``v`` inner)::

        L = -kron(D_x, diag(v)) - kron(diag(a), A_v) + kron(I_x, L_collision),

    with ``D_x`` the periodic central difference, ``A_v`` the conservative velocity
    advection (:func:`_advect_zeroflux`), ``a = accel`` the prescribed field sampled on
    ``x``, and ``L_collision = build_L`` the validated 1D LB operator.  All three blocks
    have zero column sums, so ``L`` conserves total mass exactly.
    """
    nx, nv = len(x), len(v)
    dx = float(x[1] - x[0])
    dv = float(v[1] - v[0])

    Dx = _periodic_d1(nx, dx)
    Vdiag = sp.diags(np.asarray(v, dtype=np.float64), format="csc")
    Adiag = sp.diags(np.asarray(accel, dtype=np.float64), format="csc")
    Av = _advect_zeroflux(nv, dv)
    Ix = sp.identity(nx, format="csc")
    Lcol = build_L(v, nu, v_th, u)

    L = (-sp.kron(Dx, Vdiag) - sp.kron(Adiag, Av) + sp.kron(Ix, Lcol)).tocsc()
    return L


class FokkerPlanckVlasovSolver:
    """Crank-Nicolson integrator for the 1D-1V Vlasov-Fokker-Planck step on ``[0,Lx) x [-V,V]``."""

    def __init__(self, nx: int = 32, nv: int = 48, Lx: float = 2.0 * np.pi, V: float = 5.0,
                 nu: float = 1.0, v_th: float = 1.0, u: float = 0.0,
                 accel_amp: float = 1.0, kx: float | None = None,
                 dt: float = 0.1, substeps: int = 6):
        self.nx, self.nv = nx, nv
        self.x = np.linspace(0.0, Lx, nx, endpoint=False)     # periodic configuration space
        self.v = np.linspace(-V, V, nv)                       # velocity (zero-flux ends)
        self.dx = float(self.x[1] - self.x[0])
        self.dv = float(self.v[1] - self.v[0])
        self.nu, self.v_th, self.u = nu, v_th, u
        self.Lx, self.V = Lx, V
        self.accel_amp = accel_amp
        self.kx = (2.0 * np.pi / Lx) if kx is None else kx    # one wavelength by default
        self.dt, self.substeps = dt, substeps
        self.N = nx * nv

        self.accel = accel_profile(self.x, accel_amp, self.kx)
        self.L = build_L_vfp(self.x, self.v, nu, v_th, u, self.accel)
        dt_sub = dt / substeps
        eye = sp.identity(self.N, format="csc")
        self.A = (eye - 0.5 * dt_sub * self.L).tocsc()
        self.B = (eye + 0.5 * dt_sub * self.L).tocsc()
        self._lu = spla.splu(self.A)

    def step(self, f: np.ndarray) -> np.ndarray:
        """One stored snapshot interval (= ``substeps`` CN micro-steps); ``f`` flat (N,)."""
        for _ in range(self.substeps):
            f = self._lu.solve(self.B @ f)
        return f

    def rollout(self, f0: np.ndarray, nt: int) -> np.ndarray:
        """March ``f0`` (nx, nv) for ``nt`` snapshots -> (nt, nx, nv)."""
        out = np.empty((nt, self.nx, self.nv), dtype=np.float64)
        out[0] = f0
        f = f0.reshape(-1).copy()
        for t in range(1, nt):
            f = self.step(f)
            out[t] = f.reshape(self.nx, self.nv)
        return out


def _sample_ic_vfp(rng, x, v, v_th: float, u: float) -> np.ndarray:
    """Random non-equilibrium phase-space IC: density-modulated Maxwellian + beams.

    A Maxwellian in ``v`` modulated by a random density perturbation ``1 + A cos(kx+phi)``
    in ``x`` (a Landau-damping-style seed), plus one or two random phase-space Gaussian
    blobs so the collisions + streaming have non-trivial structure to relax.  Normalised
    to unit total mass ``sum f dx dv = 1``.
    """
    dx = float(x[1] - x[0]); dv = float(v[1] - v[0])
    X, Vv = np.meshgrid(x, v, indexing="ij")                  # (nx, nv)
    Lx = len(x) * dx

    f_M = np.exp(-((v - u) ** 2) / (2.0 * v_th ** 2))         # (nv,)
    A = rng.uniform(0.1, 0.5)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    k = 2.0 * np.pi / Lx
    rho = 1.0 + A * np.cos(k * x + phi)                       # (nx,)
    f = rho[:, None] * f_M[None, :]

    for _ in range(rng.integers(1, 3)):
        x0 = rng.uniform(0.0, Lx)
        v0 = rng.uniform(-2.0, 2.0)
        wx = rng.uniform(0.4, 1.0) * Lx / (2.0 * np.pi)
        wv = rng.uniform(0.4, 1.0)
        amp = rng.uniform(0.2, 0.6)
        dxp = np.angle(np.exp(1j * (X - x0) * k)) / k         # periodic distance in x
        f += amp * np.exp(-(dxp ** 2) / (2.0 * wx ** 2) - ((Vv - v0) ** 2) / (2.0 * wv ** 2))

    f = np.clip(f, 0.0, None)
    return f / (f.sum() * dx * dv)


def generate_dataset(*, n_traj: int, nt: int = 12, nx: int = 32, nv: int = 48,
                     Lx: float = 2.0 * np.pi, V: float = 5.0, nu: float = 1.0,
                     v_th: float = 1.0, u: float = 0.0, accel_amp: float = 1.0,
                     kx: float | None = None, dt: float = 0.1, substeps: int = 6,
                     seed: int = 0) -> dict:
    """Generate ``n_traj`` phase-space trajectories -> ``u`` of shape (N, nt, nx, nv)."""
    rng = np.random.default_rng(seed)
    solver = FokkerPlanckVlasovSolver(nx=nx, nv=nv, Lx=Lx, V=V, nu=nu, v_th=v_th, u=u,
                                      accel_amp=accel_amp, kx=kx, dt=dt, substeps=substeps)
    traj = np.empty((n_traj, nt, nx, nv), dtype=np.float64)
    for k in range(n_traj):
        traj[k] = solver.rollout(_sample_ic_vfp(rng, solver.x, solver.v, v_th, u), nt)
    return {"u": traj, "x": solver.x, "v": solver.v, "accel": solver.accel,
            "dt": dt, "dx": solver.dx, "dv": solver.dv, "nt": nt, "nx": nx, "nv": nv,
            "Lx": Lx, "V": V, "nu": nu, "v_th": v_th, "u_drift": u,
            "accel_amp": accel_amp, "kx": solver.kx, "substeps": substeps}


if __name__ == "__main__":
    data = generate_dataset(n_traj=6, nt=12)
    traj, x, v = data["u"], data["x"], data["v"]
    dx, dv = data["dx"], data["dv"]
    dens = traj.sum(axis=(-1, -2)) * dx * dv                  # (N, nt) total mass
    print(f"shape {traj.shape}")
    print(f"mass drift (max |rho(t)-rho(0)|) : {np.abs(dens - dens[:, :1]).max():.3e}")
    print(f"min f over all trajectories      : {traj.min():.3e}")
    print(f"accel(x) amplitude               : {np.abs(data['accel']).max():.3e}")
    # streaming CFL at the stored dt (CN is unconditionally stable; this is accuracy only)
    print(f"streaming CFL  v_max dt/dx       : {v.max() * data['dt'] / dx:.3f}")
