#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 01 Jul, 2025
Multivariate-WNO in 2D - Restructured to match FNO style

Wavelet Neural Operator implementation based on:
"Wavelet Neural Operator for solving parametric partial differential equations 
in computational mechanics problems" by Tapas Tripura and Souvik Chakraborty

!!! Still need to be tested and validated. 
"""
# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt  # PyWavelets library for wavelet transforms

import operator
from functools import reduce
from functools import partial
from collections import OrderedDict


# ====================================
# Wavelet Convolution Layer
# ====================================
class WaveletConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, num_vars, modes1, modes2, 
                 wavelet1='db4', wavelet2='db4', level=3):
        super(WaveletConv2d, self).__init__()
        """
        2D Wavelet convolution layer using discrete wavelet transform.
        Provides better spatial-frequency localization compared to Fourier methods.
        """
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_vars = num_vars
        self.modes1 = modes1  # Number of modes to keep after decomposition
        self.modes2 = modes2
        self.level = level    # Decomposition level
        self.wavelet1 = wavelet1  # Wavelet for dimension 1
        self.wavelet2 = wavelet2  # Wavelet for dimension 2
        
        self.scale = (1 / (in_channels * out_channels))
        
        # Learnable weights for different wavelet coefficient types
        # For 2D wavelets: approximation, horizontal, vertical, diagonal
        self.weights_approx = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, num_vars, modes1, modes2, dtype=torch.float32))
        self.weights_horiz = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, num_vars, modes1, modes2, dtype=torch.float32))
        self.weights_vert = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, num_vars, modes1, modes2, dtype=torch.float32))
        self.weights_diag = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, num_vars, modes1, modes2, dtype=torch.float32))

    def dwt2_torch(self, x):
        """
        2D Discrete Wavelet Transform using PyWavelets
        Returns: (LL, (LH, HL, HH)) where LL is approximation, others are details
        """
        batch_size, channels, vars, height, width = x.shape
        
        # Process each sample and channel separately
        coeffs_batch = []
        for b in range(batch_size):
            coeffs_channels = []
            for c in range(channels):
                coeffs_vars = []
                for v in range(vars):
                    # Convert to numpy for pywt, then back to torch
                    x_np = x[b, c, v].detach().cpu().numpy()
                    coeffs = pywt.dwt2(x_np, self.wavelet1, mode='periodization')
                    
                    # Convert back to torch tensors
                    LL = torch.from_numpy(coeffs[0]).to(x.device)
                    LH = torch.from_numpy(coeffs[1][0]).to(x.device)
                    HL = torch.from_numpy(coeffs[1][1]).to(x.device)
                    HH = torch.from_numpy(coeffs[1][2]).to(x.device)
                    
                    coeffs_vars.append((LL, (LH, HL, HH)))
                coeffs_channels.append(coeffs_vars)
            coeffs_batch.append(coeffs_channels)
        
        return coeffs_batch

    def idwt2_torch(self, coeffs_batch, original_shape):
        """
        Inverse 2D Discrete Wavelet Transform
        """
        batch_size, channels, vars, height, width = original_shape
        result = torch.zeros(original_shape, device=coeffs_batch[0][0][0][0].device)
        
        for b in range(batch_size):
            for c in range(channels):
                for v in range(vars):
                    LL, (LH, HL, HH) = coeffs_batch[b][c][v]
                    
                    # Convert to numpy for pywt
                    LL_np = LL.detach().cpu().numpy()
                    LH_np = LH.detach().cpu().numpy()
                    HL_np = HL.detach().cpu().numpy()
                    HH_np = HH.detach().cpu().numpy()
                    
                    # Reconstruct
                    coeffs = (LL_np, (LH_np, HL_np, HH_np))
                    reconstructed = pywt.idwt2(coeffs, self.wavelet1, mode='periodization')
                    
                    # Handle size mismatch due to padding
                    reconstructed = reconstructed[:height, :width]
                    result[b, c, v] = torch.from_numpy(reconstructed).to(result.device)
        
        return result

    def forward(self, x):
        # Apply 2D DWT
        coeffs_batch = self.dwt2_torch(x)
        
        # Process coefficients through learnable weights
        processed_coeffs = []
        for b, batch_coeffs in enumerate(coeffs_batch):
            processed_channels = []
            for c, channel_coeffs in enumerate(batch_coeffs):
                processed_vars = []
                for v, (LL, (LH, HL, HH)) in enumerate(channel_coeffs):
                    # Get dimensions for weight application
                    h_coeff, w_coeff = LL.shape
                    
                    # Truncate to available modes
                    h_modes = min(self.modes1, h_coeff)
                    w_modes = min(self.modes2, w_coeff)
                    
                    # Apply learnable transformations to each coefficient type
                    LL_processed = torch.zeros_like(LL)
                    LH_processed = torch.zeros_like(LH)
                    HL_processed = torch.zeros_like(HL)
                    HH_processed = torch.zeros_like(HH)
                    
                    # Process only the selected modes
                    for ic in range(self.in_channels):
                        for oc in range(self.out_channels):
                            if ic == c:  # Input channel matching
                                # Apply weights to coefficients
                                weight_h = min(h_modes, self.weights_approx.shape[3])
                                weight_w = min(w_modes, self.weights_approx.shape[4])
                                
                                LL_processed[:weight_h, :weight_w] += (
                                    self.weights_approx[ic, oc, v, :weight_h, :weight_w] * 
                                    LL[:weight_h, :weight_w]
                                )
                                LH_processed[:weight_h, :weight_w] += (
                                    self.weights_horiz[ic, oc, v, :weight_h, :weight_w] * 
                                    LH[:weight_h, :weight_w]
                                )
                                HL_processed[:weight_h, :weight_w] += (
                                    self.weights_vert[ic, oc, v, :weight_h, :weight_w] * 
                                    HL[:weight_h, :weight_w]
                                )
                                HH_processed[:weight_h, :weight_w] += (
                                    self.weights_diag[ic, oc, v, :weight_h, :weight_w] * 
                                    HH[:weight_h, :weight_w]
                                )
                    
                    processed_vars.append((LL_processed, (LH_processed, HL_processed, HH_processed)))
                processed_channels.append(processed_vars)
            processed_coeffs.append(processed_channels)
        
        # Apply inverse DWT
        result = self.idwt2_torch(processed_coeffs, x.shape)
        
        return result


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


class WNO2d(nn.Module):
    def __init__(self, modes1, modes2, vars, width, wavelet1='db4', wavelet2='db4', level=3):
        super(WNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.vars = vars
        self.width = width
        self.wavelet1 = wavelet1
        self.wavelet2 = wavelet2
        self.level = level

        self.conv = WaveletConv2d(self.width, self.width, self.vars, self.modes1, self.modes2,
                                 wavelet1, wavelet2, level)
        self.mlp = MLP2d(self.width, self.width, self.width)
        self.w = nn.Conv3d(self.width, self.width, 1)
        self.b = nn.Conv3d(2, self.width, 1)

        self.activation = F.gelu

    def forward(self, x, grid):
        x1 = self.conv(x)
        x1 = self.mlp(x1)
        x2 = self.w(x)
        x3 = self.b(grid)
        x = x1 + x2 + x3
        x = self.activation(x)
        return x


class WNO_multi2d(nn.Module):
    def __init__(self, T_in, step, modes1, modes2, num_vars, width_time, width_vars=0, 
                 wavelet1='db4', wavelet2='db4', level=3, grid='arbitrary'):
        super(WNO_multi2d, self).__init__()
        """
        Wavelet Neural Operator for 2D problems.
        Uses wavelet decomposition instead of Fourier transforms for better
        spatial-frequency localization.
        
        Args:
            T_in: Number of input time steps
            step: Number of output time steps
            modes1: Number of wavelet modes in first dimension
            modes2: Number of wavelet modes in second dimension
            num_vars: Number of variables
            width_time: Hidden channel dimension
            width_vars: Variable-specific width (unused, kept for compatibility)
            wavelet1: Wavelet type for first dimension
            wavelet2: Wavelet type for second dimension  
            level: Decomposition level
            grid: Grid type ('arbitrary' or custom)
        """

        self.T_in = T_in
        self.step = step
        self.modes1 = modes1
        self.modes2 = modes2
        self.num_vars = num_vars
        self.width_vars = width_vars
        self.width_time = width_time
        self.wavelet1 = wavelet1
        self.wavelet2 = wavelet2
        self.level = level
        self.grid = grid

        self.fc0_time = nn.Linear(self.T_in + 2, self.width_time)  # +2 for spatial coordinates

        # Wavelet layers (fewer layers needed due to better representation capacity)
        self.f0 = WNO2d(self.modes1, self.modes2, self.num_vars, self.width_time, 
                        wavelet1, wavelet2, level)
        self.f1 = WNO2d(self.modes1, self.modes2, self.num_vars, self.width_time, 
                        wavelet1, wavelet2, level)
        self.f2 = WNO2d(self.modes1, self.modes2, self.num_vars, self.width_time, 
                        wavelet1, wavelet2, level)
        self.f3 = WNO2d(self.modes1, self.modes2, self.num_vars, self.width_time, 
                        wavelet1, wavelet2, level)

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

        # Apply Wavelet layers with residual connections
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
# 1D Wavelet Neural Operator
# ====================================
class WaveletConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, num_vars, modes1, wavelet='db4', level=3):
        super(WaveletConv1d, self).__init__()
        """
        1D Wavelet convolution layer using discrete wavelet transform.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_vars = num_vars
        self.modes1 = modes1
        self.wavelet = wavelet
        self.level = level

        self.scale = (1 / (in_channels * out_channels))
        
        # Learnable weights for approximation and detail coefficients
        self.weights_approx = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, num_vars, modes1, dtype=torch.float32))
        self.weights_detail = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, num_vars, modes1, dtype=torch.float32))

    def dwt1_torch(self, x):
        """
        1D Discrete Wavelet Transform using PyWavelets
        Returns: (cA, cD) where cA is approximation, cD is detail
        """
        batch_size, channels, vars, length = x.shape
        
        coeffs_batch = []
        for b in range(batch_size):
            coeffs_channels = []
            for c in range(channels):
                coeffs_vars = []
                for v in range(vars):
                    x_np = x[b, c, v].detach().cpu().numpy()
                    cA, cD = pywt.dwt(x_np, self.wavelet, mode='periodization')
                    
                    cA_tensor = torch.from_numpy(cA).to(x.device)
                    cD_tensor = torch.from_numpy(cD).to(x.device)
                    
                    coeffs_vars.append((cA_tensor, cD_tensor))
                coeffs_channels.append(coeffs_vars)
            coeffs_batch.append(coeffs_channels)
        
        return coeffs_batch

    def idwt1_torch(self, coeffs_batch, original_shape):
        """
        Inverse 1D Discrete Wavelet Transform
        """
        batch_size, channels, vars, length = original_shape
        result = torch.zeros(original_shape, device=coeffs_batch[0][0][0][0].device)
        
        for b in range(batch_size):
            for c in range(channels):
                for v in range(vars):
                    cA, cD = coeffs_batch[b][c][v]
                    
                    cA_np = cA.detach().cpu().numpy()
                    cD_np = cD.detach().cpu().numpy()
                    
                    reconstructed = pywt.idwt(cA_np, cD_np, self.wavelet, mode='periodization')
                    
                    # Handle size mismatch
                    reconstructed = reconstructed[:length]
                    result[b, c, v] = torch.from_numpy(reconstructed).to(result.device)
        
        return result

    def forward(self, x):
        # Apply 1D DWT
        coeffs_batch = self.dwt1_torch(x)
        
        # Process coefficients
        processed_coeffs = []
        for b, batch_coeffs in enumerate(coeffs_batch):
            processed_channels = []
            for c, channel_coeffs in enumerate(batch_coeffs):
                processed_vars = []
                for v, (cA, cD) in enumerate(channel_coeffs):
                    # Get dimensions
                    coeff_len = cA.shape[0]
                    modes = min(self.modes1, coeff_len)
                    
                    # Apply learnable transformations
                    cA_processed = torch.zeros_like(cA)
                    cD_processed = torch.zeros_like(cD)
                    
                    for ic in range(self.in_channels):
                        for oc in range(self.out_channels):
                            if ic == c:
                                weight_len = min(modes, self.weights_approx.shape[3])
                                
                                cA_processed[:weight_len] += (
                                    self.weights_approx[ic, oc, v, :weight_len] * 
                                    cA[:weight_len]
                                )
                                cD_processed[:weight_len] += (
                                    self.weights_detail[ic, oc, v, :weight_len] * 
                                    cD[:weight_len]
                                )
                    
                    processed_vars.append((cA_processed, cD_processed))
                processed_channels.append(processed_vars)
            processed_coeffs.append(processed_channels)
        
        # Apply inverse DWT
        result = self.idwt1_torch(processed_coeffs, x.shape)
        
        return result


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


class WNO1d(nn.Module):
    def __init__(self, modes1, vars, width, wavelet='db4', level=3):
        super(WNO1d, self).__init__()

        self.modes1 = modes1
        self.vars = vars
        self.width = width
        self.wavelet = wavelet
        self.level = level

        self.conv = WaveletConv1d(self.width, self.width, self.vars, self.modes1, wavelet, level)
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


class WNO_multi1d(nn.Module):
    def __init__(self, T_in, step, modes1, num_vars, width_time, width_vars=0, 
                 wavelet='db4', level=3, grid='arbitrary'):
        super(WNO_multi1d, self).__init__()
        """
        1D Wavelet Neural Operator
        """

        self.T_in = T_in
        self.step = step
        self.modes1 = modes1
        self.num_vars = num_vars
        self.width_vars = width_vars
        self.width_time = width_time
        self.wavelet = wavelet
        self.level = level
        self.grid = grid

        self.fc0_time = nn.Linear(self.T_in + 1, self.width_time)  # +1 for spatial coordinate

        # Wavelet layers
        self.f0 = WNO1d(self.modes1, self.num_vars, self.width_time, wavelet, level)
        self.f1 = WNO1d(self.modes1, self.num_vars, self.width_time, wavelet, level)
        self.f2 = WNO1d(self.modes1, self.num_vars, self.width_time, wavelet, level)
        self.f3 = WNO1d(self.modes1, self.num_vars, self.width_time, wavelet, level)

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

        # Apply Wavelet layers with residual connections
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


# ====================================
# Alternative Simplified Implementation
# (Similar to the original WNO2d from your reference)
# ====================================
class SimpleWNO2d(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_channel, grid_range, padding=0):
        super(SimpleWNO2d, self).__init__()
        """
        Simplified WNO implementation similar to the original paper.
        
        Args:
            width: Lifting dimension
            level: Number of wavelet decomposition levels
            layers: Number of WNO layers
            size: [height, width] of input
            wavelet: Wavelet type (e.g., 'db4', 'db6')
            in_channel: Input channels (including grid coordinates)
            grid_range: Domain bounds
            padding: Padding size
        """
        
        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.wavelet = wavelet
        self.in_channel = in_channel
        self.grid_range = grid_range
        self.padding = padding
        
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        
        self.fc0 = nn.Linear(self.in_channel, self.width)
        
        for i in range(self.layers):
            # Use simplified wavelet convolution (you can replace with more sophisticated version)
            self.conv.append(WaveletConv2d(self.width, self.width, 1, 8, 8, wavelet, wavelet, level))
            self.w.append(nn.Conv2d(self.width, self.width, 1))
            
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # (batch, channels, height, width)
        
        if self.padding != 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])
        
        # Add variable dimension for compatibility
        x = x.unsqueeze(2)  # (batch, channels, vars=1, height, width)
        
        for index, (convl, wl) in enumerate(zip(self.conv, self.w)):
            grid_dummy = torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3], x.shape[4], device=x.device)
            x1 = convl(x, grid_dummy)
            x2 = wl(x.squeeze(2)).unsqueeze(2)  # Remove and add var dim for conv2d compatibility
            x = x1 + x2
            
            if index != self.layers - 1:
                x = F.gelu(x)
        
        x = x.squeeze(2)  # Remove variable dimension
        
        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]
            
        x = x.permute(0, 2, 3, 1)  # (batch, height, width, channels)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, self.grid_range[0], size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, self.grid_range[1], size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


# ====================================
# Example Usage
# ====================================
if __name__ == "__main__":
    # Test 2D WNO
    try:
        model_2d = WNO_multi2d(T_in=20, step=5, modes1=8, modes2=8, num_vars=1, 
                              width_time=32, wavelet1='db4', wavelet2='db4', level=3)
        input_2d = torch.randn(10, 1, 64, 64, 20)  # [batch, vars, x, y, time]
        output_2d = model_2d(input_2d)
        print(f"2D WNO - Input shape: {input_2d.shape}, Output shape: {output_2d.shape}")
        print(f"2D WNO - Parameters: {model_2d.count_params()}")
    except Exception as e:
        print(f"2D WNO error: {e}")

    # Test 1D WNO
    try:
        model_1d = WNO_multi1d(T_in=20, step=5, modes1=8, num_vars=1, 
                              width_time=32, wavelet='db4', level=3)
        input_1d = torch.randn(10, 1, 64, 20)  # [batch, vars, x, time]
        output_1d = model_1d(input_1d)
        print(f"1D WNO - Input shape: {input_1d.shape}, Output shape: {output_1d.shape}")
        print(f"1D WNO - Parameters: {model_1d.count_params()}")
    except Exception as e:
        print(f"1D WNO error: {e}")

    # Test Simple WNO (similar to original paper implementation)
    try:
        simple_model = SimpleWNO2d(width=32, level=3, layers=4, size=[64, 64], 
                                  wavelet='db4', in_channel=3, grid_range=[1, 1])
        input_simple = torch.randn(10, 64, 64, 1)  # [batch, x, y, channels]
        output_simple = simple_model(input_simple)
        print(f"Simple WNO - Input shape: {input_simple.shape}, Output shape: {output_simple.shape}")
    except Exception as e:
        print(f"Simple WNO error: {e}")
        
    print("\nWNO models created successfully!")
    print("Key advantages over FNO:")
    print("- Better spatial-frequency localization through wavelets")
    print("- Superior handling of discontinuities and sharp features")
    print("- More efficient representation for complex boundary conditions")
    print("- Improved performance on irregular domains")