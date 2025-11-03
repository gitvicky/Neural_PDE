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
1. Grid to coordinate/function conversion
2. Batched processing
3. Output reshaping to match grid format
4. Compatible interface with model_setup.py
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
    - Grid to function/coordinate conversion
    - Batched processing
    - Multi-channel input/output
    - Output reshaping
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
        activation=F.gelu,
        branch_dropout=0.0,
        trunk_dropout=0.0
    ):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels/variables
        out_channels : int
            Number of output channels/variables
        branch_width : int
            Width of branch network
        trunk_width : int
            Width of trunk network
        branch_depth : int
            Depth of branch network
        trunk_depth : int
            Depth of trunk network
        x_in : torch.Tensor
            x-coordinates of the grid
        y_in : torch.Tensor
            y-coordinates of the grid
        basis_size : int
            Size of the basis (output_size in DeepONet)
        activation : callable
            Activation function
        branch_dropout : float
            Dropout rate for branch network
        trunk_dropout : float
            Dropout rate for trunk network
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.basis_size = basis_size
        
        # Register coordinate tensors as buffers
        self.register_buffer('x_in', x_in if isinstance(x_in, torch.Tensor) 
                           else torch.tensor(x_in, dtype=torch.float32))
        self.register_buffer('y_in', y_in if isinstance(y_in, torch.Tensor) 
                           else torch.tensor(y_in, dtype=torch.float32))
        
        # Create coordinate grid
        xx, yy = torch.meshgrid(self.x_in, self.y_in, indexing='ij')
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        self.register_buffer('coords', coords)
        
        self.N_x = len(self.x_in)
        self.N_y = len(self.y_in)
        self.N = self.N_x * self.N_y
        
        self.input_spatial_shape = (self.N_x, self.N_y)
        self.output_spatial_shape = (self.N_x, self.N_y)
        
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
        Forward pass converting grid format to function format and back.
        
        Parameters
        ----------
        u : torch.Tensor
            Input of shape (B, in_channels, N_x, N_y, 1) or (B, in_channels, N_x, N_y)
        
        Returns
        -------
        output : torch.Tensor
            Output of shape (B, out_channels, N_x, N_y, 1)
        """
        # Handle 5D input
        if u.dim() == 5:
            u = u[..., 0]  # Remove time dimension
        
        batch_size, channels, N_x, N_y = u.shape
        
        # Flatten input function for branch network: [B, C, H, W] -> [B, C*H*W]
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
        
        # Reshape to grid: [B, out_channels, N] -> [B, out_channels, H, W]
        output = output.reshape(batch_size, self.out_channels, self.N_x, self.N_y)
        
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
    
    The DeepONet learns operators using a branch-trunk architecture:
    - Branch network: Encodes the input function (learns from data)
    - Trunk network: Encodes query locations (learns coordinate patterns)
    - Output: Inner product of branch and trunk + bias
    
    Parameters
    ----------
    in_channels : int
        Number of input variables/channels
    out_channels : int
        Number of output variables/channels
    branch_width : int
        Width of branch network hidden layers
    trunk_width : int
        Width of trunk network hidden layers
    branch_depth : int
        Number of layers in branch network
    trunk_depth : int
        Number of layers in trunk network
    x_in : torch.Tensor
        x-coordinates of the grid
    y_in : torch.Tensor
        y-coordinates of the grid
    basis_size : int
        Size of basis functions (output dimension of branch/trunk)
        Higher = more expressive but more parameters
    activation : callable
        Activation function (default: F.gelu)
    branch_dropout : float
        Dropout rate for branch network
    trunk_dropout : float
        Dropout rate for trunk network
    """
    
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        branch_width=256,
        trunk_width=256,
        branch_depth=4,
        trunk_depth=4,
        x_in=None,
        y_in=None,
        basis_size=100,
        activation=F.gelu,
        branch_dropout=0.0,
        trunk_dropout=0.0
    ):
        super().__init__()
        
        # Default grid
        if x_in is None:
            x_in = torch.linspace(0, 1, 64)
        if y_in is None:
            y_in = torch.linspace(0, 1, 64)
        
        # Wrap for grid compatibility
        self.model = DeepONetWrapper(
            in_channels=in_channels,
            out_channels=out_channels,
            branch_width=branch_width,
            trunk_width=trunk_width,
            branch_depth=branch_depth,
            trunk_depth=trunk_depth,
            x_in=x_in,
            y_in=y_in,
            basis_size=basis_size,
            activation=activation,
            branch_dropout=branch_dropout,
            trunk_dropout=trunk_dropout
        )
    
    def forward(self, u):
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
    
#     # Test DeepONet
#     print("\n" + "=" * 70)
#     print("DeepONet Model")
#     print("=" * 70)
#     deeponet = DeepONet(
#         in_channels=2,
#         out_channels=2,
#         branch_width=128,
#         trunk_width=128,
#         branch_depth=4,
#         trunk_depth=4,
#         x_in=x,
#         y_in=y,
#         basis_size=100
#     )
    
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
#     print("\n✓ All tests passed! DeepONet is compatible with NOs4POs")
#     print("=" * 70)
    
#     # Parameter breakdown
#     print("\n" + "=" * 70)
#     print("Parameter Breakdown")
#     print("=" * 70)
#     branch_params = sum(p.numel() for p in deeponet.model.deeponets[0].branch.parameters())
#     trunk_params = sum(p.numel() for p in deeponet.model.deeponets[0].trunk.parameters())
#     print(f"Branch network (per channel): {branch_params:,}")
#     print(f"Trunk network (per channel): {trunk_params:,}")
#     print(f"Total (2 output channels): {deeponet.count_params():,}")
#     print("=" * 70)
# # %%
