#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implicit Neural Representations (INRs) for NOs4POs
Compatible with grid-based training infrastructure

Models:
- SIREN: Sinusoidal Representation Networks
- FourierNet: Multiplicative Fourier Networks

Original authors: @vsitzmann (SIREN), @Fathony et al. (FourierNet)
Modified for NOs4POs by: @vgopakum

These models automatically handle:
1. Grid to coordinate conversion
2. Batched coordinate generation
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
# SIREN Components
# ============================================================================

class SineLayer(nn.Module):
    """
    Single layer with sine activation.
    
    See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for 
    discussion of omega_0.
    """
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                           1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                           np.sqrt(6 / self.in_features) / self.omega_0)
    
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class SirenBase(nn.Module):
    """
    Base SIREN network (coordinate MLP).
    """
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, 
                 outermost_linear=True, first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                            np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        """coords: [B*N, coord_dim]"""
        return self.net(coords)


# ============================================================================
# FourierNet Components
# ============================================================================

class FourierLayer(nn.Module):
    """Sine filter as used in FourierNet."""
    
    def __init__(self, in_features, out_features, weight_scale):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.data *= weight_scale
        self.linear.bias.data.uniform_(-np.pi, np.pi)
    
    def forward(self, x):
        return torch.sin(self.linear(x))


class MFNBase(nn.Module):
    """
    Multiplicative Filter Network base class.
    """
    def __init__(self, hidden_size, out_size, n_layers, weight_scale, 
                 bias=True, output_act=False):
        super().__init__()
        
        self.linear = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias) for _ in range(n_layers)]
        )
        self.output_linear = nn.Linear(hidden_size, out_size)
        self.output_act = output_act
        
        for lin in self.linear:
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / hidden_size),
                np.sqrt(weight_scale / hidden_size),
            )
    
    def forward(self, x):
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linear[i - 1](out)
        out = self.output_linear(out)
        
        if self.output_act:
            out = torch.sin(out)
        
        return out


class FourierNetBase(MFNBase):
    """Base Fourier Network (coordinate MLP)."""
    
    def __init__(self, in_size, hidden_size, out_size, n_layers=3,
                 input_scale=256.0, weight_scale=1.0, bias=True, output_act=False):
        super().__init__(hidden_size, out_size, n_layers, weight_scale, bias, output_act)
        self.filters = nn.ModuleList(
            [
                FourierLayer(in_size, hidden_size, input_scale / np.sqrt(n_layers + 1))
                for _ in range(n_layers + 1)
            ]
        )


# ============================================================================
# Wrapper for Grid-based Training
# ============================================================================

class INRWrapper(nn.Module):
    """
    Wrapper that makes coordinate-based MLPs compatible with grid-based training.
    
    Handles:
    - Grid to coordinate conversion
    - Batched processing
    - Output reshaping
    """
    
    def __init__(self, coordinate_mlp, in_channels, out_channels, x_in, y_in):
        """
        Parameters
        ----------
        coordinate_mlp : nn.Module
            The coordinate-based MLP (SIREN or FourierNet)
        in_channels : int
            Number of input channels/variables
        out_channels : int
            Number of output channels/variables
        x_in : torch.Tensor
            x-coordinates of the grid
        y_in : torch.Tensor
            y-coordinates of the grid
        """
        super().__init__()
        
        self.mlp = coordinate_mlp
        self.in_channels = in_channels
        self.out_channels = out_channels
        
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
    
    def forward(self, u):
        """
        Forward pass converting grid format to coordinate format and back.
        
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
        
        # Convert grid to point cloud: [B, C, H, W] -> [B, H, W, C] -> [B, N, C]
        u_points = u.permute(0, 2, 3, 1).reshape(batch_size, -1, self.in_channels)
        
        # Prepare coordinates for each batch: [N, 2] -> [B, N, 2+C]
        # Expand coordinates for batch
        coords_batch = self.coords.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate coordinates with features: [B, N, 2+C]
        mlp_input = torch.cat([coords_batch, u_points], dim=-1)
        
        # Reshape for MLP: [B, N, 2+C] -> [B*N, 2+C]
        mlp_input_flat = mlp_input.reshape(-1, 2 + self.in_channels)
        
        # Forward through coordinate MLP: [B*N, 2+C] -> [B*N, out_channels]
        output_flat = self.mlp(mlp_input_flat)
        
        # Reshape back: [B*N, out_channels] -> [B, N, out_channels]
        output_points = output_flat.reshape(batch_size, self.N, self.out_channels)
        
        # Convert back to grid: [B, N, C] -> [B, H, W, C] -> [B, C, H, W]
        output_grid = output_points.reshape(batch_size, self.N_x, self.N_y, self.out_channels)
        output = output_grid.permute(0, 3, 1, 2)
        
        # Add time dimension for compatibility
        output = output.unsqueeze(-1)
        
        return output
    
    def count_params(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# User-facing Models
# ============================================================================

class SIREN(nn.Module):
    """
    SIREN for grid-based PDE problems.
    
    Compatible with NOs4POs training infrastructure.
    
    Parameters
    ----------
    in_channels : int
        Number of input variables/channels
    out_channels : int
        Number of output variables/channels
    hidden_features : int
        Width of hidden layers
    hidden_layers : int
        Number of hidden layers
    x_in : torch.Tensor
        x-coordinates of the grid
    y_in : torch.Tensor
        y-coordinates of the grid
    first_omega_0 : float
        Frequency of first layer
    hidden_omega_0 : float
        Frequency of hidden layers
    outermost_linear : bool
        Whether to use linear activation on output
    """
    
    def __init__(self, in_channels=1, out_channels=1, hidden_features=256, 
                 hidden_layers=5, x_in=None, y_in=None,
                 first_omega_0=30, hidden_omega_0=30., outermost_linear=True):
        super().__init__()
        
        # Default grid
        if x_in is None:
            x_in = torch.linspace(0, 1, 64)
        if y_in is None:
            y_in = torch.linspace(0, 1, 64)
        
        # Create coordinate MLP
        # Input: [x, y, feature_1, ..., feature_C]
        coord_mlp = SirenBase(
            in_features=2 + in_channels,  # coordinates + features
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_channels,
            outermost_linear=outermost_linear,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0
        )
        
        # Wrap for grid compatibility
        self.model = INRWrapper(coord_mlp, in_channels, out_channels, x_in, y_in)
    
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


class FourierNet(nn.Module):
    """
    Multiplicative Fourier Network for grid-based PDE problems.
    
    Compatible with NOs4POs training infrastructure.
    
    Parameters
    ----------
    in_channels : int
        Number of input variables/channels
    out_channels : int
        Number of output variables/channels
    hidden_size : int
        Width of hidden layers
    n_layers : int
        Number of hidden layers
    x_in : torch.Tensor
        x-coordinates of the grid
    y_in : torch.Tensor
        y-coordinates of the grid
    input_scale : float
        Frequency scale for Fourier features
    weight_scale : float
        Weight initialization scale
    bias : bool
        Whether to use bias in linear layers
    output_act : bool
        Whether to use sine activation on output
    """
    
    def __init__(self, in_channels=1, out_channels=1, hidden_size=256, 
                 n_layers=3, x_in=None, y_in=None,
                 input_scale=256.0, weight_scale=1.0, bias=True, output_act=False):
        super().__init__()
        
        # Default grid
        if x_in is None:
            x_in = torch.linspace(0, 1, 64)
        if y_in is None:
            y_in = torch.linspace(0, 1, 64)
        
        # Create coordinate MLP
        # Input: [x, y, feature_1, ..., feature_C]
        coord_mlp = FourierNetBase(
            in_size=2 + in_channels,  # coordinates + features
            hidden_size=hidden_size,
            out_size=out_channels,
            n_layers=n_layers,
            input_scale=input_scale,
            weight_scale=weight_scale,
            bias=bias,
            output_act=output_act
        )
        
        # Wrap for grid compatibility
        self.model = INRWrapper(coord_mlp, in_channels, out_channels, x_in, y_in)
    
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


# ============================================================================
# Testing and Examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing INR Models for NOs4POs")
    print("=" * 70)
    
    # Setup
    disc = 64
    x = torch.linspace(0, 1, disc)
    y = torch.linspace(0, 1, disc)
    batch_size = 4
    
    # Test SIREN
    print("\n" + "=" * 70)
    print("SIREN Model")
    print("=" * 70)
    siren = SIREN(
        in_channels=2,
        out_channels=2,
        hidden_features=128,
        hidden_layers=4,
        x_in=x,
        y_in=y,
        first_omega_0=30,
        hidden_omega_0=30.
    )
    
    print(f"Input grid shape: {siren.input_spatial_shape}")
    print(f"Output grid shape: {siren.output_spatial_shape}")
    print(f"Parameters: {siren.count_params():,}")
    
    # Create test input
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    u = torch.zeros(batch_size, 2, disc, disc)
    u[:, 0] = torch.sin(2 * np.pi * xx).unsqueeze(0)
    u[:, 1] = torch.cos(2 * np.pi * yy).unsqueeze(0)
    u = u.unsqueeze(-1)
    
    print(f"Input shape: {u.shape}")
    
    # Forward pass
    import time
    start = time.time()
    output = siren(u)
    end = time.time()
    
    print(f"Output shape: {output.shape}")
    print(f"Forward pass time: {(end - start) * 1000:.2f} ms")
    
    # Test FourierNet
    print("\n" + "=" * 70)
    print("FourierNet Model")
    print("=" * 70)
    fourier = FourierNet(
        in_channels=2,
        out_channels=2,
        hidden_size=128,
        n_layers=3,
        x_in=x,
        y_in=y,
        input_scale=256.0
    )
    
    print(f"Input grid shape: {fourier.input_spatial_shape}")
    print(f"Output grid shape: {fourier.output_spatial_shape}")
    print(f"Parameters: {fourier.count_params():,}")
    
    # Forward pass
    start = time.time()
    output = fourier(u)
    end = time.time()
    
    print(f"Output shape: {output.shape}")
    print(f"Forward pass time: {(end - start) * 1000:.2f} ms")
    
    # Verify shapes match
    assert output.shape == u.shape, f"Shape mismatch: {output.shape} vs {u.shape}"
    print("\n✓ All tests passed! Models are compatible with NOs4POs")
    print("=" * 70)
# %%
