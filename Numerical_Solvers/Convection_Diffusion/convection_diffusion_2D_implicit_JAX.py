# %%
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.linalg import solve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# from IPython.display import HTML

class ConvectionDiffusionImplicitSolver:
    """
    Implicit finite difference solver for the 2D convection-diffusion equation
    using the Backward Euler method.

    The equation is: ∂u/∂t + c_x ∂u/∂x + c_y ∂u/∂y = D (∂²u/∂x² + ∂²u/∂y²)
    """

    def __init__(self, Lx, Ly, T, nx, ny, nt, cx=1.0, cy=1.0, D=0.1, bc='periodic'):
        """
        Initialize the 2D convection-diffusion solver.

        Parameters:
        -----------
        Lx, Ly : float
            Domain lengths in x and y directions.
        T : float
            Total simulation time.
        nx, ny : int
            Number of spatial grid points in x and y.
        nt : int
            Number of time steps.
        cx, cy : float
            Convection speeds in x and y.
        D : float
            Diffusion coefficient.
        bc : str
            Boundary condition type: 'periodic'.
        """
        self.Lx, self.Ly = Lx, Ly
        self.T = T
        self.nx, self.ny = nx, ny
        self.nt = nt
        self.cx, self.cy = cx, cy
        self.D = D
        self.bc = bc

        # Grid spacing
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.dt = T / nt

        # Spatial grids
        self.x = jnp.linspace(0, Lx, nx, endpoint=False)
        self.y = jnp.linspace(0, Ly, ny, endpoint=False)
        self.X, self.Y = jnp.meshgrid(self.x, self.y, indexing='ij')

        # Grid size for flattened array
        self.N = nx * ny

        # Precompute the system matrix A for the implicit solver
        self._setup_system_matrix()

    def _setup_system_matrix(self):
        """
        Setup the matrix A for the implicit system A * u^{n+1} = u^n.

        The discretized Backward Euler scheme is:
        (u^{n+1} - u^n)/dt = -c_x * Dx(u^{n+1}) - c_y * Dy(u^{n+1}) + D * (Dxx(u^{n+1}) + Dyy(u^{n+1}))

        Rearranging gives:
        (I + dt * (cx*Dx + cy*Dy) - dt * D * (Dxx + Dyy)) * u^{n+1} = u^n
        A * u^{n+1} = u^n
        """
        # 1D finite difference matrices
        Dx_1d, Dxx_1d = self._get_1d_operators(self.nx, self.dx)
        Dy_1d, Dyy_1d = self._get_1d_operators(self.ny, self.dy)

        # 2D operators using Kronecker products
        Ix, Iy = jnp.eye(self.nx), jnp.eye(self.ny)
        
        # Convection operators
        Dx_2d = jnp.kron(Iy, Dx_1d)
        Dy_2d = jnp.kron(Dy_1d, Ix)
        
        # Diffusion (Laplacian) operators
        Dxx_2d = jnp.kron(Iy, Dxx_1d)
        Dyy_2d = jnp.kron(Dyy_1d, Ix)

        # Full spatial operator matrix
        L_operator = (self.cx * Dx_2d + self.cy * Dy_2d - 
                      self.D * (Dxx_2d + Dyy_2d))

        # System matrix for A * u_new = u_old
        I_total = jnp.eye(self.N)
        self.A = I_total + self.dt * L_operator

    def _get_1d_operators(self, n, delta_x):
        """Creates 1D first and second derivative matrices with periodic BC."""
        # First derivative (central difference)
        D1 = jnp.diag(jnp.ones(n - 1), k=1) - jnp.diag(jnp.ones(n - 1), k=-1)
        if self.bc == 'periodic':
            D1 = D1.at[0, -1].set(-1)
            D1 = D1.at[-1, 0].set(1)
        Dx = D1 / (2 * delta_x)

        # Second derivative (central difference)
        D2 = jnp.diag(-2 * jnp.ones(n)) + jnp.diag(jnp.ones(n-1), k=1) + jnp.diag(jnp.ones(n-1), k=-1)
        if self.bc == 'periodic':
            D2 = D2.at[0, -1].set(1)
            D2 = D2.at[-1, 0].set(1)
        Dxx = D2 / (delta_x**2)
        
        return Dx, Dxx

    @staticmethod
    @jit
    def solve_system(A, u_flat):
        """JIT-compiled function to solve the linear system."""
        return solve(A, u_flat)

    def solve(self, u0_func, aa, bb, cc):
        """
        Solve the convection-diffusion equation with a given initial condition.

        Parameters:
        -----------
        u0_func : callable
            Initial condition u(x, y, 0). Must accept two arguments (X, Y).

        Returns:
        --------
        t : jnp.ndarray
            Time array.
        u_history : jnp.ndarray
            Solution history, shape (nt+1, nx, ny).
        """
        # Time array
        t = jnp.linspace(0, self.T, self.nt + 1)
        
        # History array
        u_history = jnp.zeros((self.nt + 1, self.nx, self.ny))

        # Initial condition
        u0 = u0_func(self.X, self.Y, aa, bb, cc)
        u_history = u_history.at[0].set(u0)
        
        # Flatten the state for the solver
        u_current_flat = u0.flatten()

        # Time integration loop
        for i in range(self.nt):
            # Solve the implicit system A * u_new = u_current
            u_next_flat = self.solve_system(self.A, u_current_flat)
            
            # Update current state
            u_current_flat = u_next_flat
            
            # Store the reshaped result
            u_history = u_history.at[i + 1].set(u_next_flat.reshape(self.nx, self.ny))
            
        return t, u_history

    def animate_solution(self, t, u_history, interval=50, cmap='viridis'):
        """Create and display an animation of the solution."""
        fig, ax = plt.subplots(figsize=(8, 7))
        
        vmax = u_history[0].max()
        vmin = u_history[0].min()

        im = ax.imshow(u_history[0].T, animated=True, cmap=cmap,
                       extent=[0, self.Lx, 0, self.Ly],
                       origin='lower', vmin=vmin, vmax=vmax)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        title = ax.set_title(f'Time: {t[0]:.2f} s')
        fig.colorbar(im, ax=ax)
        
        def animate_frame(frame):
            im.set_array(u_history[frame].T)
            title.set_text(f'Time: {t[frame]:.2f} s')
            return im, title

        anim = FuncAnimation(fig, animate_frame, frames=len(t),
                           interval=interval, blit=True)
        plt.close(fig)
        return anim

    # --- Initial Condition: A Gaussian pulse ---
    def u0_gaussian(X, Y, aa, bb, cc):
        return jnp.exp(-((X - aa)**2 + (Y - bb)**2) / cc)


# # %%
# if __name__ == '__main__':
#     # --- Parameters ---
#     Lx, Ly = 10.0, 10.0   # Domain size
#     T = 5.0               # Total time
#     nx, ny = 64, 64       # Grid resolution
#     nt = 100              # Number of time steps
#     cx, cy = 1.0, 0.5     # Convection velocities
#     D = 0.1              # Diffusion coefficient
#     aa, bb = 2.0, 3.0     # Position of Gaussian 
#     cc = 4.0             # Width of Gaussian


#     # --- Create and run solver ---
#     print("Initializing 2D Convection-Diffusion Solver...")
#     solver = ConvectionDiffusionImplicitSolver(Lx, Ly, T, nx, ny, nt, cx, cy, D, bc='periodic')
    
#     print("Solving the system...")
#     t, u_history = solver.solve(u0_gaussian, aa, bb, cc)
    
#     print("Generating animation...")
#     # To display in a Jupyter environment, simply have `anim` as the last line.
#     anim = solver.animate_solution(t, u_history, interval=40, cmap='inferno')
    
#     # # To save the animation, you might need ffmpeg installed:
#     anim.save('convection_diffusion_2d.mp4', writer='ffmpeg', fps=25)
    
#     # # Display in environments like VSCode or Jupyter
#     # display(HTML(anim.to_jshtml()))
#     # print("Done. If in a Jupyter Notebook, the animation will be displayed below.")

#     # --- Plot initial and final states ---
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))

#     # Initial state
#     im0 = axes[0].imshow(u_history[0].T, extent=[0, Lx, 0, Ly], origin='lower', cmap='inferno')
#     axes[0].set_title(f'Initial State (t = {t[0]:.2f})')
#     axes[0].set_xlabel('x')
#     axes[0].set_ylabel('y')
#     fig.colorbar(im0, ax=axes[0], shrink=0.8)

#     # Final state
#     im1 = axes[1].imshow(u_history[-1].T, extent=[0, Lx, 0, Ly], origin='lower', cmap='inferno')
#     axes[1].set_title(f'Final State (t = {t[-1]:.2f})')
#     axes[1].set_xlabel('x')
#     fig.colorbar(im1, ax=axes[1], shrink=0.8)

#     plt.suptitle('2D Convection-Diffusion using Implicit Backward Euler')
#     plt.tight_layout()
#     plt.show()

# # %%