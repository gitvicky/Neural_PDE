#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GINO Implementation using neuralop library
Optimized for NOs4POs - batching-compatible with pre-computed graph structure

GINO = Geometry-Informed Neural Operator
Architecture: Input GNO → FNO (latent space) → Output GNO + Projection

This implementation:
1. Uses neuralop.layers.GNOBlock for input/output GNO operations
2. Uses neuralop.layers.FNOBlocks for latent space processing
3. Pre-computes and stores graph edges as model buffers (no recomputation in forward pass)
4. Implements efficient batching
5. Compatible with the model_setup structure in NOs4POs
"""
# %% 
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.layers.gno_block import GNOBlock
from neuralop.layers.fno_block import FNOBlocks
from neuralop.layers.channel_mlp import ChannelMLP


class GINO(nn.Module):
    """
    Geometry-Informed Neural Operator using neuralop library
    
    Maps from arbitrary input geometry through a structured latent grid (FNO)
    to arbitrary output geometry.
    
    Architecture:
        Input points → Input GNO → Regular grid (latent) → FNO → Output GNO → Output points
    
    Parameters
    ----------
    in_channels : int
        Number of input channels/features
    out_channels : int
        Number of output channels/features
    hidden_channels : int
        Hidden dimension width for GNO layers
    fno_n_modes : tuple
        Number of Fourier modes for FNO in each dimension (e.g., (16, 16) for 2D)
    fno_n_layers : int
        Number of FNO layers in latent space
    in_gno_radius : float
        Radius for neighbor search in input GNO
    out_gno_radius : float
        Radius for neighbor search in output GNO
    n_gno_layers : int
        Number of layers in input/output GNO blocks
    x_in : torch.Tensor
        Input x-coordinates (1D tensor)
    y_in : torch.Tensor
        Input y-coordinates (1D tensor)
    x_out : torch.Tensor, optional
        Output x-coordinates (defaults to x_in)
    y_out : torch.Tensor, optional
        Output y-coordinates (defaults to y_in)
    latent_grid_size : int or tuple
        Size of latent grid for FNO (e.g., 32 or (32, 32))
    gno_transform_type : str
        Type of kernel integral transform ('linear', 'nonlinear', etc.)
    gno_use_open3d : bool
        Whether to use Open3D for neighbor search
    gno_use_torch_scatter : bool
        Whether to use torch_scatter for reductions
    """
    
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        fno_n_modes=(16, 16),
        fno_n_layers=4,
        in_gno_radius=0.033,
        out_gno_radius=0.033,
        n_gno_layers=2,
        x_in=None,
        y_in=None,
        x_out=None,
        y_out=None,
        latent_grid_size=32,
        gno_transform_type='linear',
        gno_use_open3d=False,
        gno_use_torch_scatter=False,
        projection_channel_ratio=4,
        fno_use_channel_mlp=True,
        fno_channel_mlp_expansion=0.5,
        fno_non_linearity=F.gelu,
        **kwargs
    ):
        super().__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.fno_n_modes = fno_n_modes
        self.fno_n_layers = fno_n_layers
        self.in_gno_radius = in_gno_radius
        self.out_gno_radius = out_gno_radius
        self.n_gno_layers = n_gno_layers
        self.gno_coord_dim = 2  # 2D problems
        
        # Default grid setup
        if x_in is None:
            x_in = torch.linspace(0, 1, 64)
        if y_in is None:
            y_in = torch.linspace(0, 1, 64)
        if x_out is None:
            x_out = x_in
        if y_out is None:
            y_out = y_in
        
        # Register coordinate tensors as buffers
        self.register_buffer('x_in', x_in if isinstance(x_in, torch.Tensor) else torch.tensor(x_in, dtype=torch.float32))
        self.register_buffer('y_in', y_in if isinstance(y_in, torch.Tensor) else torch.tensor(y_in, dtype=torch.float32))
        self.register_buffer('x_out', x_out if isinstance(x_out, torch.Tensor) else torch.tensor(x_out, dtype=torch.float32))
        self.register_buffer('y_out', y_out if isinstance(y_out, torch.Tensor) else torch.tensor(y_out, dtype=torch.float32))
        
        # Create coordinate grids for input/output
        xx_in, yy_in = torch.meshgrid(self.x_in, self.y_in, indexing='ij')
        coords_in = torch.stack([xx_in.flatten(), yy_in.flatten()], dim=-1)
        
        xx_out, yy_out = torch.meshgrid(self.x_out, self.y_out, indexing='ij')
        coords_out = torch.stack([xx_out.flatten(), yy_out.flatten()], dim=-1)
        
        # Register coordinates as buffers
        self.register_buffer('coords_in', coords_in)
        self.register_buffer('coords_out', coords_out)
        
        self.N_in = coords_in.shape[0]
        self.N_out = coords_out.shape[0]
        self.same_grid = torch.allclose(coords_in, coords_out) if self.N_in == self.N_out else False
        
        self.input_spatial_shape = (len(self.x_in), len(self.y_in))
        self.output_spatial_shape = (len(self.x_out), len(self.y_out))
        
        # Create latent grid (structured grid for FNO)
        if isinstance(latent_grid_size, int):
            latent_grid_size = (latent_grid_size, latent_grid_size)
        
        self.latent_grid_size = latent_grid_size
        latent_x = torch.linspace(0, 1, latent_grid_size[0])
        latent_y = torch.linspace(0, 1, latent_grid_size[1])
        latent_xx, latent_yy = torch.meshgrid(latent_x, latent_y, indexing='ij')
        latent_coords = torch.stack([latent_xx.flatten(), latent_yy.flatten()], dim=-1)
        
        self.register_buffer('latent_coords', latent_coords)
        self.N_latent = latent_coords.shape[0]
        
        # =====================================
        # Lifting layer: Maps input features to hidden dimension
        # =====================================
        self.lifting = nn.Linear(in_channels, hidden_channels)
        
        # =====================================
        # Input GNO: Maps from input geometry to latent grid
        # =====================================
        # Note: For 'linear' transform, GNO expects features already in hidden_channels
        self.gno_in = GNOBlock(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            coord_dim=self.gno_coord_dim,
            radius=in_gno_radius,
            transform_type=gno_transform_type,
            use_open3d_neighbor_search=gno_use_open3d,
            use_torch_scatter_reduce=gno_use_torch_scatter,
        )
        
        # =====================================
        # FNO: Processes in latent space on regular grid
        # =====================================
        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=fno_n_modes,
            n_layers=fno_n_layers,
            use_channel_mlp=fno_use_channel_mlp,
            channel_mlp_expansion=fno_channel_mlp_expansion,
            non_linearity=fno_non_linearity,
        )
        
        # =====================================
        # Output GNO: Maps from latent grid to output geometry
        # =====================================
        self.gno_out = GNOBlock(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            coord_dim=self.gno_coord_dim,
            radius=out_gno_radius,
            transform_type=gno_transform_type,
            use_open3d_neighbor_search=gno_use_open3d,
            use_torch_scatter_reduce=gno_use_torch_scatter,
        )
        
        # =====================================
        # Projection: Final mapping to output channels
        # =====================================
        projection_channels = projection_channel_ratio * hidden_channels
        self.projection = ChannelMLP(
            in_channels=hidden_channels,
            out_channels=out_channels,
            hidden_channels=projection_channels,
            n_layers=2,
            n_dim=1,
            non_linearity=fno_non_linearity,
        )
        
        # Activation function
        self.activation = F.gelu
    
    def forward(self, u):
        """
        Forward pass through the GINO model.
        
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
        
        # =====================================
        # Step 0: Lift input features to hidden dimension
        # =====================================
        u_lifted = self.lifting(u_in)  # [B, N_in, hidden_channels]
        
        # =====================================
        # Step 1: Input GNO - Map from input geometry to latent grid
        # =====================================
        # GNOBlock signature: forward(y, x, f_y=None)
        # y: input coords, x: query coords (latent), f_y: features at input
        latent_features = self.gno_in(
            y=self.coords_in,           # Input geometry [N_in, 2]
            x=self.latent_coords,       # Latent grid queries [N_latent, 2]
            f_y=u_lifted                # Features at input [B, N_in, hidden_channels]
        )  # Output: [B, N_latent, hidden_channels]
        
        # =====================================
        # Step 2: Reshape to grid format for FNO
        # =====================================
        # Reshape from [B, N_latent, C] to [B, C, H, W]
        latent_grid = latent_features.reshape(
            batch_size, 
            self.latent_grid_size[0], 
            self.latent_grid_size[1], 
            self.hidden_channels
        )
        # Permute to [B, C, H, W] for FNO
        latent_grid = latent_grid.permute(0, 3, 1, 2)
        
        # =====================================
        # Step 3: FNO - Process in latent space
        # =====================================
        for idx in range(self.fno_n_layers):
            latent_grid = self.fno_blocks(latent_grid, idx)
        
        # =====================================
        # Step 4: Reshape back to point cloud format
        # =====================================
        # Permute from [B, C, H, W] to [B, H, W, C]
        latent_grid = latent_grid.permute(0, 2, 3, 1)
        # Reshape to [B, N_latent, C]
        latent_features = latent_grid.reshape(batch_size, -1, self.hidden_channels)
        
        # =====================================
        # Step 5: Output GNO - Map from latent grid to output geometry
        # =====================================
        output_features = self.gno_out(
            y=self.latent_coords,       # Latent grid geometry [N_latent, 2]
            x=self.coords_out,          # Output queries [N_out, 2]
            f_y=latent_features         # Features at latent grid [B, N_latent, hidden_channels]
        )  # Output: [B, N_out, hidden_channels]
        
        # =====================================
        # Step 6: Project to output channels
        # =====================================
        # ChannelMLP expects [B, C, N] format
        output_features = output_features.permute(0, 2, 1)  # [B, hidden, N_out]
        output_features = self.projection(output_features)   # [B, out_channels, N_out]
        output_features = output_features.permute(0, 2, 1)   # [B, N_out, out_channels]
        
        # =====================================
        # Step 7: Reshape to grid format
        # =====================================
        output_shape = (batch_size, *self.output_spatial_shape, self.out_channels)
        u_out = output_features.reshape(*output_shape)
        
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
                f'hidden_channels={self.hidden_channels}, fno_n_layers={self.fno_n_layers}, '
                f'fno_modes={self.fno_n_modes}, '
                f'in_radius={self.in_gno_radius:.4f}, out_radius={self.out_gno_radius:.4f}, '
                f'same_grid={self.same_grid}, '
                f'input_shape={self.input_spatial_shape}, output_shape={self.output_spatial_shape}, '
                f'latent_shape={self.latent_grid_size}, '
                f'N_in={self.N_in}, N_latent={self.N_latent}, N_out={self.N_out}')


# # Example usage and testing
# if __name__ == "__main__":
#     import numpy as np
#     import time
    
#     # Create a simple test case
#     disc = 32
#     x = torch.linspace(0, 1, disc)
#     y = torch.linspace(0, 1, disc)
    
#     # Initialize model
#     model = GINO(
#         in_channels=2,
#         out_channels=2,
#         hidden_channels=16,
#         fno_n_modes=(8,8),
#         fno_n_layers=2,
#         in_gno_radius=0.05,
#         out_gno_radius=0.05,
#         n_gno_layers=2,
#         x_in=x,
#         y_in=y,
#         latent_grid_size=16
#     )
    
#     print("=" * 70)
#     print("GINO Model using neuralop library")
#     print("=" * 70)
#     print(f"Input grid shape: {model.input_spatial_shape}")
#     print(f"Latent grid shape: {model.latent_grid_size}")
#     print(f"Output grid shape: {model.output_spatial_shape}")
#     print(f"Total nodes - Input: {model.N_in:,}, Latent: {model.N_latent:,}, Output: {model.N_out:,}")
#     print(f"In/Out Radius: {model.in_gno_radius}/{model.out_gno_radius}")
#     print(f"FNO modes: {model.fno_n_modes}")
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
#     start = time.time()
    
#     output = model(u)
    
#     if device == 'cuda':
#         torch.cuda.synchronize()
    
#     end = time.time()
    
#     print(f"Output shape: {output.shape}")
#     print(f"Forward pass time: {(end - start) * 1000:.2f} ms")
#     print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB" if device == 'cuda' else "CPU mode")
#     print("=" * 70)
    
#     # Verify output shape matches input shape (for same grid case)
#     assert output.shape == u.shape, f"Output shape {output.shape} doesn't match input shape {u.shape}"
#     print("✓ Output shape verification passed!")
#     print("=" * 70)
# # %%