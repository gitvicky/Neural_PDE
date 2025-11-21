#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepONet Implementation for NOs4POs
Compatible with grid-based training infrastructure

DeepONet (Deep Operator Network) learns operators mapping between function spaces
using a branch-trunk architecture.

Architecture:
- Branch Net: Encodes input function
- Trunk Net: Encodes query locations
- Output: Inner product of branch and trunk outputs

Original paper: Lu et al., "Learning nonlinear operators via DeepONet" (2021)
Modified for NOs4POs by: @vgopakum

Handles:
1. Grid to coordinate/function conversion (structured grids)
2. Point cloud to function/coordinate conversion (unstructured grids)
3. Batched processing
4. Output reshaping to match original format
5. Compatible interface with model_setup.py
"""
# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from functools import reduce


# ============================================================================
# Core Neural Network Components
# ============================================================================

class FNN(nn.Module):
    """Fully-connected neural network (MLP)."""
    
    def __init__(self, input_size, output_size, width, num_layers, 
                 activation=F.gelu, dropout=0.0):
        super().__init__()
        
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_size, width))
        
        for i in range(1, num_layers - 1):
            self.linears.append(nn.Linear(width, width))
        
        self.linears.append(nn.Linear(width, output_size))
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x):
        for linear in self.linears[:-1]:
            x = linear(x)
            x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout(x)
        x = self.linears[-1](x)
        return x


class DeepONetBase(nn.Module):
    """
    Base DeepONet architecture.
    
    Branch network: Encodes the input function
    Trunk network: Encodes the query locations
    Output: Inner product + bias
    """
    
    def __init__(
        self,
        branch_input_size,
        trunk_input_size,
        branch_width,
        trunk_width,
        branch_depth,
        trunk_depth,
        output_size,
        activation=F.gelu,
        branch_dropout=0.0,
        trunk_dropout=0.0
    ):
        super().__init__()
        
        self.output_size = output_size
        
        # Branch network: encodes input function
        self.branch = FNN(
            input_size=branch_input_size,
            output_size=output_size,
            width=branch_width,
            num_layers=branch_depth,
            activation=activation,
            dropout=branch_dropout
        )
        
        # Trunk network: encodes query locations
        self.trunk = FNN(
            input_size=trunk_input_size,
            output_size=output_size,
            width=trunk_width,
            num_layers=trunk_depth,
            activation=activation,
            dropout=trunk_dropout
        )
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, branch_input, trunk_input):
        """
        Forward pass.
        
        Parameters
        ----------
        branch_input : torch.Tensor
            Input function features [batch_size, branch_input_size]
        trunk_input : torch.Tensor
            Query locations [num_queries, trunk_input_size]
        
        Returns
        -------
        output : torch.Tensor
            Predictions at query locations [batch_size, num_queries]
        """
        # Encode input function
        branch_output = self.branch(branch_input)  # [B, output_size]
        
        # Encode query locations
        trunk_output = self.trunk(trunk_input)  # [N, output_size]
        
        # Compute inner product: [B, output_size] * [N, output_size] -> [B, N]
        # Use einsum for efficient batched inner product
        output = torch.einsum('bi,ni->bn', branch_output, trunk_output)
        
        # Add bias
        output = output + self.bias
        
        return output


# ============================================================================
# Wrapper for Grid-based Training
# ============================================================================

class DeepONetWrapper(nn.Module):
    """
    Wrapper that makes DeepONet compatible with grid-based training.
    
    Handles:
    - Structured (grid) and Unstructured (point cloud) inputs.
    - Grid to function/coordinate conversion.
    - Batched processing.
    - Multi-channel input/output.
    - Output reshaping.
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        branch_width,
        trunk_width,
        branch_depth,
        trunk_depth,
        x_in,
        y_in,
        basis_size,
        grid_type='structured', # 'structured' or 'unstructured'
        activation=F.gelu,
        branch_dropout=0.0,
        trunk_dropout=0.0
    ):
        """
        Parameters
        ----------
        ...
        x_in : torch.Tensor
            - If grid_type='structured': 1D tensor of x-coordinates
            - If grid_type='unstructured': 1D tensor of x-coordinates (len N)
        y_in : torch.Tensor
            - If grid_type='structured': 1D tensor of y-coordinates
            - If grid_type='unstructured': 1D tensor of y-coordinates (len N)
        grid_type : str, optional
            Type of grid. 'structured' for [B, C, Nx, Ny] or 'unstructured' 
            for [B, C, N]. Default is 'structured'.
        ...
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.basis_size = basis_size
        self.grid_type = grid_type

        # Register coordinate tensors as buffers
        self.register_buffer('x_in', x_in if isinstance(x_in, torch.Tensor) 
                           else torch.tensor(x_in, dtype=torch.float32))
        self.register_buffer('y_in', y_in if isinstance(y_in, torch.Tensor) 
                           else torch.tensor(y_in, dtype=torch.float32))
        
        if self.grid_type == 'structured':
            # Create structured coordinate grid
            xx, yy = torch.meshgrid(self.x_in, self.y_in, indexing='ij')
            coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
            
            self.N_x = len(self.x_in)
            self.N_y = len(self.y_in)
            self.N = self.N_x * self.N_y
            
            self.input_spatial_shape = (self.N_x, self.N_y)
            self.output_spatial_shape = (self.N_x, self.N_y)
            
        elif self.grid_type == 'unstructured':
            # Create unstructured coordinate grid
            assert len(self.x_in) == len(self.y_in), \
                "x_in and y_in must have the same length for unstructured grid"
            coords = torch.stack([self.x_in, self.y_in], dim=-1)
            
            self.N = len(self.x_in)
            self.N_x = None
            self.N_y = None
            
            self.input_spatial_shape = (self.N,)
            self.output_spatial_shape = (self.N,)
            
        else:
            raise ValueError(f"Unknown grid_type: '{self.grid_type}'. "
                             "Must be 'structured' or 'unstructured'.")

        self.register_buffer('coords', coords) # Shape [N, 2]
        
        # Create DeepONets for each output channel
        self.deeponets = nn.ModuleList([
            DeepONetBase(
                branch_input_size=self.N * in_channels,  # Flattened input function
                trunk_input_size=2,  # (x, y) coordinates
                branch_width=branch_width,
                trunk_width=trunk_width,
                branch_depth=branch_depth,
                trunk_depth=trunk_depth,
                output_size=basis_size,
                activation=activation,
                branch_dropout=branch_dropout,
                trunk_dropout=trunk_dropout
            )
            for _ in range(out_channels)
        ])
    
    def forward(self, u):
        """
        Forward pass converting grid/point format to function format and back.
        
        Parameters
        ----------
        u : torch.Tensor
            - If structured: (B, C_in, N_x, N_y, 1) or (B, C_in, N_x, N_y)
            - If unstructured: (B, C_in, N, 1) or (B, C_in, N)
        
        Returns
        -------
        output : torch.Tensor
            - If structured: (B, C_out, N_x, N_y, 1)
            - If unstructured: (B, C_out, N, 1)
        """
        # Handle trailing time dimension
        if u.shape[-1] == 1:
            u = u.squeeze(-1)
        # u is now [B, C, Nx, Ny] or [B, C, N]
        
        batch_size = u.shape[0]
        
        # Flatten input function for branch network:
        # [B, C, Nx, Ny] -> [B, C*Nx*Ny]
        # [B, C, N]      -> [B, C*N]
        branch_input = u.reshape(batch_size, -1)
        
        # Trunk input is the coordinate grid: [N, 2]
        trunk_input = self.coords
        
        # Process each output channel
        outputs = []
        for deeponet in self.deeponets:
            # Forward through DeepONet: [B, N]
            out = deeponet(branch_input, trunk_input)
            outputs.append(out)
        
        # Stack outputs: [out_channels, B, N] -> [B, out_channels, N]
        output = torch.stack(outputs, dim=1)
        
        # Reshape to grid
        if self.grid_type == 'structured':
            # [B, C_out, N] -> [B, C_out, Nx, Ny]
            output = output.reshape(batch_size, self.out_channels, self.N_x, self.N_y)
        elif self.grid_type == 'unstructured':
            # Output is already [B, C_out, N], which is the correct shape
            pass 
        
        # Add time dimension for compatibility
        output = output.unsqueeze(-1)
        
        return output
    
    def count_params(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# User-facing Model
# ============================================================================

class DeepONet(nn.Module):
    """
    DeepONet for grid-based PDE problems.
    
    Compatible with NOs4POs training infrastructure.
    Can handle both structured 2D grids and unstructured 2D point clouds.
    
    ...
    
    Parameters
    ----------
    ...
    grid_type : str, optional
        'structured' (default): Input data is on a regular grid.
            `x_in` and `y_in` are 1D vectors defining the grid axes.
            Input shape: [B, C, Nx, Ny, 1]
        'unstructured': Input data is a point cloud.
            `x_in` and `y_in` must be 1D vectors of length N 
            specifying the coordinates of each point.
            Input shape: [B, C, N, 1]
    x_in : torch.Tensor or None
        - If grid_type='structured': 1D tensor of x-coordinates (len Nx).
          If None, defaults to linspace(0, 1, 64).
        - If grid_type='unstructured': 1D tensor of x-coordinates (len N).
          Must be provided.
    y_in : torch.Tensor or None
        - If grid_type='structured': 1D tensor of y-coordinates (len Ny).
          If None, defaults to linspace(0, 1, 64).
        - If grid_type='unstructured': 1D tensor of y-coordinates (len N).
          Must be provided.
    ...
    """
    
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        branch_width=256,
        trunk_width=256,
        branch_depth=4,
        trunk_depth=4,
        grid_type='structured',
        x_in=None,
        y_in=None,
        basis_size=100,
        activation=F.gelu,
        branch_dropout=0.0,
        trunk_dropout=0.0
    ):
        super().__init__()
        
        # Validate and set default coordinates based on grid type
        if grid_type == 'structured':
            if x_in is None:
                x_in = torch.linspace(0, 1, 64)
            if y_in is None:
                y_in = torch.linspace(0, 1, 64)
        elif grid_type == 'unstructured':
            if x_in is None or y_in is None:
                raise ValueError("x_in and y_in must be provided "
                                 "for 'unstructured' grid_type")
        else:
            raise ValueError(f"Unknown grid_type: '{grid_type}'. "
                             "Must be 'structured' or 'unstructured'.")
        
        # Wrap for grid compatibility
        self.model = DeepONetWrapper(
            in_channels=in_channels,
            out_channels=out_channels,
            branch_width=branch_width,
            trunk_width=trunk_width,
            branch_depth=branch_depth,
            trunk_depth=trunk_depth,
            grid_type=grid_type,
            x_in=x_in,
            y_in=y_in,
            basis_size=basis_size,
            activation=activation,
            branch_dropout=branch_dropout,
            trunk_dropout=trunk_dropout
        )
    
    def forward(self, u):
        u = u[0]
        return self.model(u)
    
    def count_params(self):
        return self.model.count_params()
    
    @property
    def input_spatial_shape(self):
        return self.model.input_spatial_shape
    
    @property
    def output_spatial_shape(self):
        return self.model.output_spatial_shape


# # ============================================================================
# # Testing and Examples
# # ============================================================================

# if __name__ == "__main__":
#     print("=" * 70)
#     print("Testing DeepONet for NOs4POs")
#     print("=" * 70)
    
#     # Setup
#     disc = 64
#     x = torch.linspace(0, 1, disc)
#     y = torch.linspace(0, 1, disc)
#     batch_size = 4
    
#     # Test DeepONet (Structured)
#     print("\n" + "=" * 70)
#     print("DeepONet Model (Structured Grid)")
#     print("=" * 70)
#     deeponet = DeepONet(
#         in_channels=2,
#         out_channels=2,
#         branch_width=128,
#         trunk_width=128,
#         branch_depth=4,
#         trunk_depth=4,
#         grid_type='structured',
#         x_in=x,
#         y_in=y,
#         basis_size=100
#     )
    
#     print(f"Grid type: {deeponet.model.grid_type}")
#     print(f"Input grid shape: {deeponet.input_spatial_shape}")
#     print(f"Output grid shape: {deeponet.output_spatial_shape}")
#     print(f"Parameters: {deeponet.count_params():,}")
#     print(f"Basis size: 100")
    
#     # Create test input
#     xx, yy = torch.meshgrid(x, y, indexing='ij')
#     u = torch.zeros(batch_size, 2, disc, disc)
#     u[:, 0] = torch.sin(2 * np.pi * xx).unsqueeze(0)
#     u[:, 1] = torch.cos(2 * np.pi * yy).unsqueeze(0)
#     u = u.unsqueeze(-1)
    
#     print(f"Input shape: {u.shape}")
    
#     # Forward pass
#     import time
#     start = time.time()
#     output = deeponet(u)
#     end = time.time()
    
#     print(f"Output shape: {output.shape}")
#     print(f"Forward pass time: {(end - start) * 1000:.2f} ms")
    
#     # Verify shapes match
#     assert output.shape == u.shape, f"Shape mismatch: {output.shape} vs {u.shape}"
#     print("\n✓ Structured grid test passed! DeepONet is compatible.")
#     print("=" * 70)
    
#     # Test DeepONet (Unstructured)
#     print("\n" + "=" * 70)
#     print("DeepONet Model (Unstructured Grid / Point Cloud)")
#     print("=" * 70)
    
#     N_points = 1024
#     x_un = torch.rand(N_points)
#     y_un = torch.rand(N_points)
    
#     deeponet_un = DeepONet(
#         in_channels=2,
#         out_channels=2,
#         branch_width=128,
#         trunk_width=128,
#         branch_depth=4,
#         trunk_depth=4,
#         grid_type='unstructured',
#         x_in=x_un,
#         y_in=y_un,
#         basis_size=100
#     )
    
#     print(f"Grid type: {deeponet_un.model.grid_type}")
#     print(f"Input 'grid' shape (N,): {deeponet_un.input_spatial_shape}")
#     print(f"Output 'grid' shape (N,): {deeponet_un.output_spatial_shape}")
#     print(f"Total points (N): {deeponet_un.model.N}")
#     print(f"Parameters: {deeponet_un.count_params():,}")
    
#     # Create test input
#     u_un = torch.rand(batch_size, 2, N_points, 1)
#     print(f"Input shape: {u_un.shape}")
    
#     # Forward pass
#     start = time.time()
#     output_un = deeponet_un(u_un)
#     end = time.time()
    
#     print(f"Output shape: {output_un.shape}")
#     print(f"Forward pass time: {(end - start) * 1000:.2f} ms")
    
#     # Verify shapes match
#     expected_shape = (batch_size, 2, N_points, 1)
#     assert output_un.shape == expected_shape, \
#         f"Shape mismatch: {output_un.shape} vs {expected_shape}"
#     print("\n✓ Unstructured grid test passed! DeepONet is compatible.")
#     print("=" * 70)
# # %%