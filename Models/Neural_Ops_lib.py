
# File: Neural_Ops_lib.py
"""
This file contains custom neural operator models that extend the functionality of existing libraries.
It includes implementations of FNO2d and UNO models adapted for multi-dimensional inputs.
""" 
# %%
import torch
import torch.nn as nn
from neuralop.models import FNO2d
from neuralop.models import UNO
from neuralop.models import TFNO

class UNO_multi2d(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.uno = UNO(
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        hidden_channels=hidden_channels,
                        projection_channels=256,
                        n_layers=5,
                        uno_out_channels=[32,64,64,64,32],
                        uno_n_modes=[[16,16],[8,8],[8,8],[8,8],[16,16]],
                        uno_scalings=[[1.0,1.0],[0.5,0.5],[1,1],[2,2],[1,1]],
                        horizontal_skips_map=None,
                        channel_mlp_skip="linear",
                        domain_padding=0.2,
                        norm='group_norm' #Testing this
                        )

    def forward(self, x):
        x = x[...,0]
        x = self.uno(x)
        x = torch.unsqueeze(x, -1)
        return x 

    def count_params(self):
        """
        Count the number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.uno.parameters() if p.requires_grad)


class FNO_multi2d(nn.Module):
    def __init__(self, in_channels, out_channels, width, n_modes_height, n_modes_width, n_layers):
        super().__init__()
        self.fno = FNO2d(
                        n_modes_height=n_modes_height,       # Number of Fourier modes to keep along height dimension
                        n_modes_width=n_modes_width,        # Number of Fourier modes to keep along width dimension
                        hidden_channels=width,      # Width of the FNO (number of channels)
                        in_channels=in_channels,           # Number of input channels
                        out_channels=out_channels,          # Number of output channels
                        lifting_channels=width,    # Channels in the lifting block (lifting_channel_ratio * hidden_channels)
                        projection_channels=256, # Channels in the projection block
                        n_layers=n_layers,              # Number of Fourier layers
        )

    def forward(self, x):
        x = x[...,0]
        x = self.fno(x)
        x = torch.unsqueeze(x, -1)
        return x
    
    def count_params(self):
        """
        Count the number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.fno.parameters() if p.requires_grad)


class TFNO_multi2d(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, n_modes_height, n_modes_width, rank=0.05):
        super().__init__()
        self.tfno = TFNO(
                        n_modes=(n_modes_height, n_modes_width),  # Number of Fourier modes to keep along height and width dimensions
                        hidden_channels=hidden_channels,      # Width of the FNO (number of channels)
                        in_channels=in_channels,           # Number of input channels
                        out_channels=out_channels,          # Number of output channels
                        factorization='tucker',             # Factorization type
                        implementation='factorized',        # Implementation type
                        rank=rank                            # Rank for the factorization
        )

    def forward(self, x):
        x = x[...,0]
        x = self.tfno(x)
        x = torch.unsqueeze(x, -1)
        return x

    def count_params(self):
        """
        Count the number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.tfno.parameters() if p.requires_grad)

# %%
# #Example Usage
# uno = UNO_multi2d(in_channels=2, out_channels=2, hidden_channels=32)
# ins = torch.randn(20,2,100,100,1) #BS, num_vars, Nx, Ny, T_in
# outs = uno(ins)
# # %%
# fno = FNO_multi2d(in_channels=2, out_channels=2, width=32, n_modes_height=16, n_modes_width=16, n_layers=4)
# ins = torch.randn(20,2,100,100,1) #BS, num_vars, Nx, Ny, T_in
# outs = fno(ins)

# tfno = TFNO_multi2d(in_channels=2, out_channels=2, hidden_channels=32, n_modes_height=16, n_modes_width=16, rank=0.05)
# ins = torch.randn(20,2,100,100,1) #BS, num_vars, Nx, Ny, T_in
# outs = tfno(ins)
# %%
