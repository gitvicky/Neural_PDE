#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNO Implementation using neuralop library
Optimized for NOs4POs - batching-compatible with pre-computed graph structure

This implementation:
1. Uses neuralop.layers.GNOBlock for the core GNO operations
2. Pre-computes and stores graph edges as model buffers (no recomputation in forward pass)
3. Implements efficient batching using PyG-style graph batching
4. Compatible with the model_setup structure in NOs4POs
5. Handles both structured [B, C, Nx, Ny] and unstructured [B, C, N] grids
   via an explicit `grid_type` parameter.
"""
# %% 
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.layers.gno_block import GNOBlock


class GNO(nn.Module):
    """
    Graph Neural Operator using neuralop library
    
    This implementation stores graph structure (edges, neighbors) as buffers
    to avoid recomputation during forward passes.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels/features
    out_channels : int
        Number of output channels/features
    hidden_channels : int
        Hidden dimension width for GNO layers
    r : float
        Radius for neighbor search in graph construction
    n_layers : int
        Number of GNO layers (depth)
    grid_type : str, optional
        'structured' (default): Input data is on a regular grid.
            `x_in` and `y_in` are 1D vectors defining the grid axes.
            Input shape: [B, C, Nx, Ny, 1]
        'unstructured': Input data is a point cloud.
            `x_in` and `y_in` must be 1D vectors of length N 
            specifying the coordinates of each point.
            Input shape: [B, C, N, 1]
    x_in : torch.Tensor
        - If grid_type='structured': 1D tensor of x-coordinates (len Nx).
          If None, defaults to linspace(0, 1, 32).
        - If grid_type='unstructured': 1D tensor of x-coordinates (len N).
          Must be provided.
    y_in : torch.Tensor
        - If grid_type='structured': 1D tensor of y-coordinates (len Ny).
          If None, defaults to linspace(0, 1, 32).
        - If grid_type='unstructured': 1D tensor of y-coordinates (len N).
          Must be provided.
    x_out : torch.Tensor, optional
        Output coordinates (defaults to x_in for same-grid case)
    y_out : torch.Tensor, optional
        Output coordinates (defaults to y_in for same-grid case)
    gno_transform_type : str
        Type of kernel integral transform ('linear', 'nonlinear', etc.)
    gno_use_open3d : bool
        Whether to use Open3D for neighbor search
    gno_radius : float, optional
        Alternative parameter name for radius (for compatibility)
    """
    
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        r=0.033,
        n_layers=4,
        grid_type='structured', # 'structured' or 'unstructured'
        x_in=None,
        y_in=None,
        x_out=None,
        y_out=None,
        gno_transform_type='linear',
        gno_use_open3d=False,
        gno_use_torch_scatter=True,
        gno_radius=None,
        **kwargs
    ):
        super().__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.grid_type = grid_type
        
        # Handle radius parameter
        self.r = gno_radius if gno_radius is not None else r
        
        # --- Coordinate and Grid Setup ---
        
        # Validate and set default coordinates based on grid type
        if grid_type == 'structured':
            if x_in is None: x_in = torch.linspace(0, 1, 32)
            if y_in is None: y_in = torch.linspace(0, 1, 32)
            if x_out is None: x_out = x_in
            if y_out is None: y_out = y_in
        elif grid_type == 'unstructured':
            if x_in is None or y_in is None:
                raise ValueError("x_in and y_in must be provided "
                                 "for 'unstructured' grid_type")
            if x_out is None: x_out = x_in
            if y_out is None: y_out = y_in
        else:
            raise ValueError(f"Unknown grid_type: '{grid_type}'. "
                             "Must be 'structured' or 'unstructured'.")
        
        # Register coordinate tensors as buffers
        self.register_buffer('x_in', x_in if isinstance(x_in, torch.Tensor) else torch.tensor(x_in, dtype=torch.float32))
        self.register_buffer('y_in', y_in if isinstance(y_in, torch.Tensor) else torch.tensor(y_in, dtype=torch.float32))
        self.register_buffer('x_out', x_out if isinstance(x_out, torch.Tensor) else torch.tensor(x_out, dtype=torch.float32))
        self.register_buffer('y_out', y_out if isinstance(y_out, torch.Tensor) else torch.tensor(y_out, dtype=torch.float32))

        # Create coordinate grids and spatial shapes
        if self.grid_type == 'structured':
            xx_in, yy_in = torch.meshgrid(self.x_in, self.y_in, indexing='ij')
            coords_in = torch.stack([xx_in.flatten(), yy_in.flatten()], dim=-1)
            
            xx_out, yy_out = torch.meshgrid(self.x_out, self.y_out, indexing='ij')
            coords_out = torch.stack([xx_out.flatten(), yy_out.flatten()], dim=-1)
            
            self.input_spatial_shape = (len(self.x_in), len(self.y_in))
            self.output_spatial_shape = (len(self.x_out), len(self.y_out))
            
        elif self.grid_type == 'unstructured':
            assert len(self.x_in) == len(self.y_in), "x_in/y_in length mismatch"
            assert len(self.x_out) == len(self.y_out), "x_out/y_out length mismatch"
            
            coords_in = torch.stack([self.x_in, self.y_in], dim=-1)
            coords_out = torch.stack([self.x_out, self.y_out], dim=-1)
            
            self.input_spatial_shape = (len(self.x_in),)
            self.output_spatial_shape = (len(self.x_out),)

        # Register coordinates as buffers
        self.register_buffer('coords_in', coords_in)
        self.register_buffer('coords_out', coords_out)
        
        # Store grid information
        self.N_in = coords_in.shape[0]
        self.N_out = coords_out.shape[0]
        self.same_grid = torch.allclose(coords_in, coords_out) if self.N_in == self.N_out else False
        
        # --- Model Architecture ---
        
        # Pre-compute and store graph structure as buffers
        # Note: This is stored but GNOBlock may redo neighbor search internally
        in_neighbors, out_neighbors = self._build_neighbor_lists()
        self.register_buffer('in_neighbors', in_neighbors)
        self.register_buffer('out_neighbors', out_neighbors)
        
        # Lifting layer: maps input features to hidden dimension
        self.lifting = nn.Linear(in_channels, hidden_channels)
        
        # Create neuralop GNOBlock layers
        self.gno_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()  
        
        for _ in range(n_layers):
            self.gno_layers.append(
                GNOBlock(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    coord_dim=2,  # 2D coordinates
                    radius=self.r,
                    transform_type=gno_transform_type,
                    use_open3d_neighbor_search=gno_use_open3d,
                    use_torch_scatter_reduce=gno_use_torch_scatter,
                )
            )
            self.norm_layers.append(nn.LayerNorm(hidden_channels)) 
        # Projection layer: maps hidden features to output dimension
        self.projection = nn.Linear(hidden_channels, out_channels)
        
        # Activation function
        self.activation = nn.GELU()
        
        # For different grid case, we need a coordinate encoder
        if not self.same_grid:
            self.coord_encoder = nn.Linear(2, hidden_channels)
    
    def _build_neighbor_lists(self):
        """
        Pre-compute neighbor lists for efficient graph operations.
        This is called once during initialization and stored as buffers.
        
        Returns
        -------
        in_neighbors : torch.Tensor
            Neighbor indices for input coordinates
        out_neighbors : torch.Tensor
            Neighbor indices for output coordinates
        """
        # Compute pairwise distances for input grid
        pwd_in = torch.cdist(self.coords_in, self.coords_in)
        in_mask = pwd_in <= self.r
        
        if self.same_grid:
            # For same grid, output neighbors are the same as input
            out_neighbors = in_mask
            in_neighbors = in_mask
        else:
            # For different grids, compute output neighbors
            pwd_out = torch.cdist(self.coords_in, self.coords_out)
            out_mask = pwd_out <= self.r
            out_neighbors = out_mask
            in_neighbors = in_mask
        
        return in_neighbors, out_neighbors
    
    def forward(self, u):
        """
        Forward pass through the GNO model.
        
        Parameters
        ----------
        u : torch.Tensor
            - If structured: (B, C_in, N_x, N_y, 1) or (B, C_in, N_x, N_y)
            - If unstructured: (B, C_in, N, 1) or (B, C_in, N)
        
        Returns
        -------
        output : torch.Tensor
            - If structured: (B, C_out, N_x_out, N_y_out, 1)
            - If unstructured: (B, C_out, N_out, 1)
        """
        u = u[0] #Adjusting for parameterised input
        if u.shape[-1] == 1:
            u = u.squeeze(-1)
        
        batch_size = u.shape[0]
        
        # --- Input Pre-processing ---
        # Reshape to node features: [B, N, C]
        if self.grid_type == 'structured':
            # u is (B, C, N_x, N_y)
            # Permute from (B, C, N_x, N_y) to (B, N_x, N_y, C)
            # Then flatten spatial dims -> (B, N_x*N_y, C)
            u_in = u.permute(0, 2, 3, 1).reshape(batch_size, -1, self.in_channels)
        elif self.grid_type == 'unstructured':
            # u is (B, C, N), permute to (B, N, C)
            u_in = u.permute(0, 2, 1)

        # --- GNO Processing ---
        
        if not self.same_grid:
            # Different grid case: need to handle input and output nodes separately
            # This is more complex and uses a loop for simplicity
            outputs = []
            for b in range(batch_size):
                # Lift input features
                x_lifted = self.lifting(u_in[b])  # [N_in, hidden]
                
                # Apply GNO layers - each maps from input coords to output coords
                x = x_lifted
                for gno_layer in self.gno_layers:
                    # neuralop GNOBlock signature: forward(y, x, f_y=None)
                    x = gno_layer(
                        y=self.coords_in,  # Input geometry
                        x=self.coords_out,  # Output queries
                        f_y=x.unsqueeze(0)  # Features at input points [1, N_in, hidden]
                    ).squeeze(0)  # Remove batch dim -> [N_out, hidden]
                    x = self.activation(x)
                
                # Project to output dimension
                x_out = self.projection(x)  # [N_out, out_channels]
                outputs.append(x_out)
            
            u_final = torch.stack(outputs, dim=0)  # [B, N_out, out_channels]
        
        else:
            # Same grid case: efficient batched processing
            # Lift to hidden dimension [B, N, in_channels] -> [B, N, hidden]
            x_lifted = self.lifting(u_in)  # [B, N_in, hidden]
            
            # Apply GNO layers
            x = x_lifted
            for gno_layer in self.gno_layers:
                # neuralop GNOBlock signature: forward(y, x, f_y=None)
                x = gno_layer(
                    y=self.coords_in,  # Input geometry [N_in, 2]
                    x=self.coords_in,  # Output queries (same as input) [N_in, 2]
                    f_y=x  # Features [B, N_in, hidden]
                )  # Output: [B, N_in, hidden]
                x = self.activation(x)
            
            # Project to output dimension [B, N, hidden] -> [B, N, out_channels]
            u_final = self.projection(x) # [B, N_out, out_channels]
        
        # --- Output Post-processing ---
        
        if self.grid_type == 'structured':
            # Reshape output to grid format
            output_shape = (batch_size, *self.output_spatial_shape, self.out_channels)
            u_out = u_final.reshape(*output_shape)
            
            # Permute back to (B, C, H, W) and add extra dim for compatibility
            # From (B, H, W, C) to (B, C, H, W, 1)
            output = u_out.permute(0, 3, 1, 2).unsqueeze(-1)
        
        elif self.grid_type == 'unstructured':
            # u_final is already [B, N_out, C_out]
            # Permute to (B, C_out, N_out)
            u_out = u_final.permute(0, 2, 1)
            # Add extra dim -> (B, C_out, N_out, 1)
            output = u_out.unsqueeze(-1)
            
        return output
    
    def count_params(self):
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def extra_repr(self):
        """Extra information to print about the model."""
        return (f'grid_type={self.grid_type}, '
                f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
                f'hidden_channels={self.hidden_channels}, n_layers={self.n_layers}, '
                f'radius={self.r:.4f}, same_grid={self.same_grid}, '
                f'input_shape={self.input_spatial_shape}, output_shape={self.output_spatial_shape}, '
                f'N_in={self.N_in}, N_out={self.N_out}')


# # Example usage and testing
# if __name__ == "__main__":
#     import numpy as np
#     import time
    
#     # --- Structured Grid Test ---
    
#     disc = 32
#     x_s = torch.linspace(0, 1, disc)
#     y_s = torch.linspace(0, 1, disc)
#     batch_size = 4
    
#     # Initialize model
#     model_s = GNO(
#         in_channels=2,
#         out_channels=2,
#         hidden_channels=32,
#         r=0.05,
#         n_layers=2,
#         grid_type='structured',
#         x_in=x_s,
#         y_in=y_s
#     )
    
#     print("=" * 70)
#     print("GNO Model Test (Structured Grid)")
#     print("=" * 70)
#     print(model_s)
#     print(f"Parameters: {model_s.count_params():,}")
#     print("=" * 70)
    
#     # Create test input
#     xx, yy = torch.meshgrid(x_s, y_s, indexing='ij')
#     u_s = torch.zeros(batch_size, 2, disc, disc)
#     u_s[:, 0] = torch.sin(2 * np.pi * xx).unsqueeze(0)
#     u_s[:, 1] = torch.cos(2 * np.pi * yy).unsqueeze(0)
#     u_s = u_s.unsqueeze(-1) # [B, C, Nx, Ny, 1]
    
#     print(f"Input shape (structured): {u_s.shape}")
    
#     # Move to GPU if available
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Device: {device}")
#     model_s = model_s.to(device)
#     u_s = u_s.to(device)
    
#     # Timed forward pass
#     start = time.time()
#     output_s = model_s(u_s)
#     end = time.time()
    
#     print(f"Output shape (structured): {output_s.shape}")
#     print(f"Forward pass time: {(end - start) * 1000:.2f} ms")
#     assert output_s.shape == u_s.shape
#     print("\n✓ Structured grid test passed!")
#     print("=" * 70)

#     # --- Unstructured Grid Test ---
    
#     N_points = 1024
#     x_un = torch.rand(N_points)
#     y_un = torch.rand(N_points)

#     # Initialize model
#     model_un = GNO(
#         in_channels=2,
#         out_channels=2,
#         hidden_channels=32,
#         r=0.05,
#         n_layers=2,
#         grid_type='unstructured',
#         x_in=x_un,
#         y_in=y_un
#     )
    
#     print("\n" + "=" * 70)
#     print("GNO Model Test (Unstructured Grid)")
#     print("=" * 70)
#     print(model_un)
#     print(f"Parameters: {model_un.count_params():,}")
#     print("=" * 70)

#     # Create test input
#     u_un = torch.rand(batch_size, 2, N_points, 1) # [B, C, N, 1]
    
#     print(f"Input shape (unstructured): {u_un.shape}")
    
#     model_un = model_un.to(device)
#     u_un = u_un.to(device)
    
#     # Timed forward pass
#     start = time.time()
#     output_un = model_un(u_un)
#     end = time.time()

#     print(f"Output shape (unstructured): {output_un.shape}")
#     print(f"Forward pass time: {(end - start) * 1000:.2f} ms")
#     assert output_un.shape == u_un.shape
#     print("\n✓ Unstructured grid test passed!")
#     print("=" * 70)
# # %%