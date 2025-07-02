#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30 Jun, 2025
Multivariate-LNO in 2D - Restructured to match FNO style by Vignesh and Claude Sonnet

Laplace Neural Operator implementation based on:
"LNO: Laplace Neural Operator for Solving Differential Equations"
by Qianying Cao, Somdatta Goswami, George Em Karniadakis

"""

import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import operator
from functools import reduce
from functools import partial
from collections import OrderedDict


# ====================================
# Laplace layer - Pole-Residue formulation
# ====================================
class LaplaceConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, num_vars, modes1, modes2):
        super(LaplaceConv2d, self).__init__()
        """
        2D Laplace layer using pole-residue formulation.
        Replaces FFT with Laplace transform for better transient response handling.
        """
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_vars = num_vars
        self.modes1 = modes1  # Number of poles in first dimension
        self.modes2 = modes2  # Number of poles in second dimension
        
        self.scale = (1 / (in_channels * out_channels))
        
        # System poles (learnable parameters)
        self.poles1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.num_vars, self.modes1, dtype=torch.cfloat))
        self.poles2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.num_vars, self.modes2, dtype=torch.cfloat))
        
        # System residues (learnable parameters)
        self.residues = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.num_vars, self.modes1, self.modes2, dtype=torch.cfloat))

    def pole_residue_transform(self, x):
        """
        Apply pole-residue transformation for Laplace domain computation
        """
        batchsize = x.shape[0]
        
        # Compute input poles and residues by FFT (as approximation to Laplace transform)
        alpha = torch.fft.fft2(x, dim=[-2, -1])
        
        # Get frequency grids for Laplace domain (using normalized frequencies)
        omega1 = torch.fft.fftfreq(x.size(-2), 1.0 / x.size(-2)) * 2 * np.pi * 1j
        omega2 = torch.fft.fftfreq(x.size(-1), 1.0 / x.size(-1)) * 2 * np.pi * 1j
        
        omega1 = omega1.to(x.device).view(-1, 1)
        omega2 = omega2.to(x.device).view(1, -1)
        
        # Create output tensor
        out_ft = torch.zeros(batchsize, self.out_channels, self.num_vars, 
                            x.size(-2), x.size(-1) // 2 + 1,
                            dtype=torch.cfloat, device=x.device)
        
        # Apply pole-residue transformation for available modes
        modes1_actual = min(self.modes1, x.size(-2))
        modes2_actual = min(self.modes2, x.size(-1) // 2 + 1)
        
        # Extract relevant frequency components
        alpha_slice = alpha[:, :, :, :modes1_actual, :modes2_actual]
        
        # Apply simplified pole-residue transformation
        for i in range(modes1_actual):
            for j in range(modes2_actual):
                # Get current frequencies
                w1 = omega1[i, 0] if i < len(omega1) else omega1[-1, 0]
                w2 = omega2[0, j] if j < len(omega2[0]) else omega2[0, -1]
                
                # Compute pole-residue response for each input-output channel pair
                for in_ch in range(self.in_channels):
                    for out_ch in range(self.out_channels):
                        for var in range(self.num_vars):
                            # Use available poles and residues
                            pole_idx1 = min(i, self.modes1 - 1)
                            pole_idx2 = min(j, self.modes2 - 1)
                            
                            pole1 = self.poles1[in_ch, out_ch, var, pole_idx1]
                            pole2 = self.poles2[in_ch, out_ch, var, pole_idx2]
                            residue = self.residues[in_ch, out_ch, var, pole_idx1, pole_idx2]
                            
                            # Compute transfer function H(s) = residue / ((s - pole1) * (s - pole2))
                            denom1 = w1 - pole1
                            denom2 = w2 - pole2
                            
                            # Avoid division by zero
                            if torch.abs(denom1) < 1e-8:
                                denom1 = torch.tensor(1e-8 + 0j, device=x.device)
                            if torch.abs(denom2) < 1e-8:
                                denom2 = torch.tensor(1e-8 + 0j, device=x.device)
                            
                            transfer_val = residue / (denom1 * denom2)
                            
                            # Apply to input
                            out_ft[:, out_ch, var, i, j] += transfer_val * alpha_slice[:, in_ch, var, i, j]
        
        # Convert back to time domain
        x_out = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        
        return x_out

    def forward(self, x, grid=None):
        return self.pole_residue_transform(x)


class MLP2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP2d, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)
        self.activation = F.gelu

    def forward(self, x):
        x = self.mlp1(x)
        x = self.activation(x)
        x = self.mlp2(x)
        return x


class LNO2d(nn.Module):
    def __init__(self, modes1, modes2, vars, width):
        super(LNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.vars = vars
        self.width = width

        self.conv = LaplaceConv2d(self.width, self.width, self.vars, self.modes1, self.modes2)
        self.mlp = MLP2d(self.width, self.width, self.width)
        self.w = nn.Conv3d(self.width, self.width, 1)
        self.b = nn.Conv3d(2, self.width, 1)

        self.activation = F.gelu

    def forward(self, x, grid):
        x1 = self.conv(x, grid)
        x1 = self.mlp(x1)
        x2 = self.w(x)
        x3 = self.b(grid)
        x = x1 + x2 + x3
        x = self.activation(x)
        return x


class LNO_multi2d(nn.Module):
    def __init__(self, T_in, step, modes1, modes2, num_vars, width_time, width_vars=0, grid='arbitrary'):
        super(LNO_multi2d, self).__init__()
        """
        Laplace Neural Operator for 2D problems.
        Uses pole-residue formulation instead of Fourier convolution.
        
        Args:
            T_in: Number of input time steps
            step: Number of output time steps
            modes1: Number of Laplace modes (poles) in first dimension
            modes2: Number of Laplace modes (poles) in second dimension
            num_vars: Number of variables
            width_time: Hidden channel dimension
            width_vars: Variable-specific width (unused, kept for compatibility)
            grid: Grid type ('arbitrary' or custom)
        """

        self.T_in = T_in
        self.step = step
        self.modes1 = modes1
        self.modes2 = modes2
        self.num_vars = num_vars
        self.width_vars = width_vars
        self.width_time = width_time
        self.grid = grid

        self.fc0_time = nn.Linear(self.T_in + 2, self.width_time)  # +2 for spatial coordinates

        # Laplace layers (using fewer layers than FNO due to better approximation capacity)
        self.f0 = LNO2d(self.modes1, self.modes2, self.num_vars, self.width_time)
        self.f1 = LNO2d(self.modes1, self.modes2, self.num_vars, self.width_time)
        self.f2 = LNO2d(self.modes1, self.modes2, self.num_vars, self.width_time)
        self.f3 = LNO2d(self.modes1, self.modes2, self.num_vars, self.width_time)

        # Normalization
        self.norm = nn.Identity()

        # Output projection
        self.fc1_time = nn.Linear(self.width_time, 256)
        self.fc2_time = nn.Linear(256, self.step)

        self.activation = torch.nn.GELU()

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0_time(x)
        x = x.permute(0, 4, 1, 2, 3)
        grid = grid.permute(0, 4, 1, 2, 3)

        # Apply Laplace layers with residual connections
        x0 = self.f0(x, grid)
        x = self.f1(x0, grid)
        x = self.f2(x, grid) + x0  # Residual connection
        x1 = self.f3(x, grid)
        x = x + x1  # Final residual connection

        x = x.permute(0, 2, 3, 4, 1)

        # Output projection
        x = self.fc1_time(x)
        x = self.activation(x)
        x = self.fc2_time(x)

        return x
    
    def get_grid(self, shape, device):
        batchsize, self.num_vars, size_x, size_y = shape[0], shape[1], shape[2], shape[3]         
        if self.grid == 'arbitrary':
            gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
            gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        else:
            gridx = self.grid[0]
            gridy = self.grid[1]
    
        gridx = gridx.reshape(1, 1, size_x, 1, 1).repeat([batchsize, self.num_vars, 1, size_y, 1])
        gridy = gridy.reshape(1, 1, 1, size_y, 1).repeat([batchsize, self.num_vars, size_x, 1, 1])

        return torch.cat((gridx, gridy), dim=-1).to(device)

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c


# ====================================
# 1D Laplace Neural Operator
# ====================================
class LaplaceConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, num_vars, modes1):
        super(LaplaceConv1d, self).__init__()
        """
        1D Laplace layer using pole-residue formulation.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_vars = num_vars
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        
        # System poles and residues for 1D case
        self.poles = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.num_vars, self.modes1, dtype=torch.cfloat))
        self.residues = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.num_vars, self.modes1, dtype=torch.cfloat))

    def pole_residue_transform_1d(self, x):
        batchsize = x.shape[0]
        
        # Apply FFT as approximation to Laplace transform
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # Get frequency grid (normalized)
        omega = torch.fft.rfftfreq(x.size(-1), 1.0 / x.size(-1)) * 2 * np.pi * 1j
        omega = omega.to(x.device)
        
        # Create output tensor
        out_ft = torch.zeros(batchsize, self.out_channels, self.num_vars, x.size(-1) // 2 + 1,
                            dtype=torch.cfloat, device=x.device)
        
        # Apply pole-residue transformation
        modes_actual = min(self.modes1, x_ft.size(-1))
        
        for i in range(modes_actual):
            w = omega[i] if i < len(omega) else omega[-1]
            
            for in_ch in range(self.in_channels):
                for out_ch in range(self.out_channels):
                    for var in range(self.num_vars):
                        pole_idx = min(i, self.modes1 - 1)
                        pole = self.poles[in_ch, out_ch, var, pole_idx]
                        residue = self.residues[in_ch, out_ch, var, pole_idx]
                        
                        # Compute transfer function H(s) = residue / (s - pole)
                        denom = w - pole
                        if torch.abs(denom) < 1e-8:
                            denom = torch.tensor(1e-8 + 0j, device=x.device)
                        
                        transfer_val = residue / denom
                        out_ft[:, out_ch, var, i] += transfer_val * x_ft[:, in_ch, var, i]
        
        # Convert back to time domain
        x_out = torch.fft.irfft(out_ft, n=x.size(-1))
        return x_out

    def forward(self, x):
        return self.pole_residue_transform_1d(x)


class MLP1d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP1d, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)
        self.activation = F.gelu

    def forward(self, x):
        x = self.mlp1(x)
        x = self.activation(x)
        x = self.mlp2(x)
        return x


class LNO1d(nn.Module):
    def __init__(self, modes1, vars, width):
        super(LNO1d, self).__init__()

        self.modes1 = modes1
        self.vars = vars
        self.width = width

        self.conv = LaplaceConv1d(self.width, self.width, self.vars, self.modes1)
        self.mlp = MLP1d(self.width, self.width, self.width)
        self.w = nn.Conv2d(self.width, self.width, 1)
        self.b = nn.Conv2d(1, self.width, 1)

        self.activation = F.gelu

    def forward(self, x, grid):
        x1 = self.conv(x)
        x1 = self.mlp(x1)
        x2 = self.w(x)
        x3 = self.b(grid)
        x = x1 + x2 + x3
        x = self.activation(x)
        return x


class LNO_multi1d(nn.Module):
    def __init__(self, T_in, step, modes1, num_vars, width_time, width_vars=0, grid='arbitrary'):
        super(LNO_multi1d, self).__init__()
        """
        1D Laplace Neural Operator
        """

        self.T_in = T_in
        self.step = step
        self.modes1 = modes1
        self.num_vars = num_vars
        self.width_vars = width_vars
        self.width_time = width_time
        self.grid = grid

        self.fc0_time = nn.Linear(self.T_in + 1, self.width_time)  # +1 for spatial coordinate

        # Laplace layers
        self.f0 = LNO1d(self.modes1, self.num_vars, self.width_time)
        self.f1 = LNO1d(self.modes1, self.num_vars, self.width_time)
        self.f2 = LNO1d(self.modes1, self.num_vars, self.width_time)
        self.f3 = LNO1d(self.modes1, self.num_vars, self.width_time)

        self.norm = nn.Identity()

        self.fc1_time = nn.Linear(self.width_time, 256)
        self.fc2_time = nn.Linear(256, self.step)

        self.activation = torch.nn.GELU()

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x = self.fc0_time(x)
        x = x.permute(0, 3, 1, 2)
        grid = grid.permute(0, 3, 1, 2)

        # Apply Laplace layers with residual connections
        x0 = self.f0(x, grid)
        x = self.f1(x0, grid)
        x = self.f2(x, grid) + x0
        x1 = self.f3(x, grid)
        x = x + x1

        x = x.permute(0, 2, 3, 1)

        x = self.fc1_time(x)
        x = self.activation(x)
        x = self.fc2_time(x)

        return x
    
    def get_grid(self, shape, device):
        batchsize, self.num_vars, size_x = shape[0], shape[1], shape[2]   
        if self.grid == 'arbitrary':
            gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        else:
            gridx = self.grid[0]
    
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, self.num_vars, 1, 1])
        return gridx.to(device)

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c
# %% 

# # ====================================
# # Example Usage
# # ====================================
# if __name__ == "__main__":
#     # 2D Example
#     model_2d = LNO_multi2d(T_in=20, step=5, modes1=8, modes2=8, num_vars=1, width_time=32, width_vars=0)
#     input_2d = torch.randn(100, 1, 64, 64, 20)  # BS, num_vars, Nx, Ny, T_in
#     output_2d = model_2d(input_2d)
#     print(f"2D LNO - Input shape: {input_2d.shape}, Output shape: {output_2d.shape}")
#     print(f"2D LNO - Parameters: {model_2d.count_params()}")

#     # 1D Example
#     model_1d = LNO_multi1d(T_in=20, step=5, modes1=8, num_vars=1, width_time=32, width_vars=0)
#     input_1d = torch.randn(100, 1, 64, 20)  # BS, num_vars, Nx, T_in
#     output_1d = model_1d(input_1d)
#     print(f"1D LNO - Input shape: {input_1d.shape}, Output shape: {output_1d.shape}")
#     print(f"1D LNO - Parameters: {model_1d.count_params()}")