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
    x_in : torch.Tensor
        Input x-coordinates (1D tensor)
    y_in : torch.Tensor
        Input y-coordinates (1D tensor)
    x_out : torch.Tensor, optional
        Output x-coordinates (defaults to x_in for same-grid case)
    y_out : torch.Tensor, optional
        Output y-coordinates (defaults to y_in for same-grid case)
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
        
        # Handle radius parameter (can be passed as r or gno_radius)
        self.r = gno_radius if gno_radius is not None else r
        
        # Default grid setup
        if x_in is None:
            x_in = torch.linspace(0, 1, 32)
        if y_in is None:
            y_in = torch.linspace(0, 1, 32)
        if x_out is None:
            x_out = x_in
        if y_out is None:
            y_out = y_in
        
        # Register coordinate tensors as buffers
        self.register_buffer('x_in', x_in if isinstance(x_in, torch.Tensor) else torch.tensor(x_in, dtype=torch.float32))
        self.register_buffer('y_in', y_in if isinstance(y_in, torch.Tensor) else torch.tensor(y_in, dtype=torch.float32))
        self.register_buffer('x_out', x_out if isinstance(x_out, torch.Tensor) else torch.tensor(x_out, dtype=torch.float32))
        self.register_buffer('y_out', y_out if isinstance(y_out, torch.Tensor) else torch.tensor(y_out, dtype=torch.float32))
        
        # Create coordinate grids
        xx_in, yy_in = torch.meshgrid(self.x_in, self.y_in, indexing='ij')
        coords_in = torch.stack([xx_in.flatten(), yy_in.flatten()], dim=-1)
        
        xx_out, yy_out = torch.meshgrid(self.x_out, self.y_out, indexing='ij')
        coords_out = torch.stack([xx_out.flatten(), yy_out.flatten()], dim=-1)
        
        # Register coordinates as buffers
        self.register_buffer('coords_in', coords_in)
        self.register_buffer('coords_out', coords_out)
        
        # Store grid information
        self.N_in = coords_in.shape[0]
        self.N_out = coords_out.shape[0]
        self.same_grid = torch.allclose(coords_in, coords_out) if self.N_in == self.N_out else False
        
        self.input_spatial_shape = (len(self.x_in), len(self.y_in))
        self.output_spatial_shape = (len(self.x_out), len(self.y_out))
        
        # Pre-compute and store graph structure as buffers
        # This is the key optimization - graph is built once during initialization
        in_neighbors, out_neighbors = self._build_neighbor_lists()
        self.register_buffer('in_neighbors', in_neighbors)
        self.register_buffer('out_neighbors', out_neighbors)
        
        # Lifting layer: maps input features to hidden dimension
        self.lifting = nn.Linear(in_channels, hidden_channels)
        
        # Create neuralop GNOBlock layers
        self.gno_layers = nn.ModuleList()
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
            Input tensor of shape (batch_size, in_channels, N_x, N_y, 1) or
            (batch_size, in_channels, N_x, N_y)
        
        Returns
        -------
        output : torch.Tensor
            Output tensor of shape (batch_size, out_channels, N_x_out, N_y_out, 1)
        """
        # Handle 5D input by removing last dimension
        if u.dim() == 5:
            u = u[..., 0]
        
        batch_size, channels, N_x, N_y = u.shape
        
        # Reshape to node features: [B, N, C]
        # Permute from (B, C, N_x, N_y) to (B, N_x, N_y, C) then flatten spatial dims
        u_in = u.permute(0, 2, 3, 1).reshape(batch_size, -1, self.in_channels)
        
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
                    # y: input geometry [N_in, coord_dim]
                    # x: output queries [N_out, coord_dim]  
                    # f_y: features at y [N_in, channels]
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
                # For same grid: y=x=coords_in, f_y=features
                # Since we're batching, f_y has shape [B, N, hidden]
                x = gno_layer(
                    y=self.coords_in,  # Input geometry [N_in, 2]
                    x=self.coords_in,  # Output queries (same as input) [N_in, 2]
                    f_y=x  # Features [B, N_in, hidden]
                )  # Output: [B, N_in, hidden]
                x = self.activation(x)
            
            # Project to output dimension [B, N, hidden] -> [B, N, out_channels]
            u_final = self.projection(x)
        
        # Reshape output to grid format
        output_shape = (batch_size, *self.output_spatial_shape, self.out_channels)
        u_out = u_final.reshape(*output_shape)
        
        # Permute back to (B, C, H, W) and add extra dim for compatibility
        # From (B, H, W, C) to (B, C, H, W, 1)
        output = u_out.permute(0, 3, 1, 2).unsqueeze(-1)
        
        return output
    
    def count_params(self):
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def extra_repr(self):
        """Extra information to print about the model."""
        return (f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
                f'hidden_channels={self.hidden_channels}, n_layers={self.n_layers}, '
                f'radius={self.r:.4f}, same_grid={self.same_grid}, '
                f'input_shape={self.input_spatial_shape}, output_shape={self.output_spatial_shape}, '
                f'N_in={self.N_in}, N_out={self.N_out}')


# # Example usage and testing
# if __name__ == "__main__":
#     import numpy as np
    
#     # Create a simple test case
#     disc = 64
#     x = torch.linspace(0, 1, disc)
#     y = torch.linspace(0, 1, disc)
    
#     # Initialize model
#     model = GNO(
#         in_channels=2,
#         out_channels=2,
#         hidden_channels=32,
#         r=0.05,
#         n_layers=2,
#         x_in=x,
#         y_in=y
#     )
    
#     print("=" * 70)
#     print("GNO Model using neuralop library")
#     print("=" * 70)
#     print(f"Input grid shape: {model.input_spatial_shape}")
#     print(f"Output grid shape: {model.output_spatial_shape}")
#     print(f"Total nodes (N_in): {model.N_in:,}")
#     print(f"Radius: {model.r}")
#     print(f"Same grid: {model.same_grid}")
#     print(f"Parameters: {model.count_params():,}")
#     print("=" * 70)
    
#     # Create test input
#     batch_size = 4
#     xx, yy = torch.meshgrid(x, y, indexing='ij')
#     u = torch.zeros(batch_size, 2, disc, disc)
#     u[:, 0] = torch.sin(2 * np.pi * xx).unsqueeze(0)
#     u[:, 1] = torch.cos(2 * np.pi * yy).unsqueeze(0)
#     u = u.unsqueeze(-1)
    
#     print(f"Input shape: {u.shape}")
    
#     # Move to GPU if available
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Device: {device}")
#     model = model.to(device)
#     u = u.to(device)
    
#     # Warm-up pass
#     if device == 'cuda':
#         _ = model(u)
#         torch.cuda.synchronize()
    
#     # Timed forward pass
#     import time
#     start = time.time()
    
#     output = model(u)
    
#     if device == 'cuda':
#         torch.cuda.synchronize()
    
#     end = time.time()
    
#     print(f"Output shape: {output.shape}")
#     print(f"Forward pass time: {(end - start) * 1000:.2f} ms")
#     print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB" if device == 'cuda' else "CPU mode")
#     print("=" * 70)
# # %%
