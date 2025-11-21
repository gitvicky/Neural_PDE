# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22 Apr, 2024
@author: vgopakum

U-Net in 2D

"""      
# %% 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.functional as F 

import operator
from functools import reduce
from functools import partial
from collections import OrderedDict

# %%
class UNet1d(nn.Module):

    def __init__(self, in_channels=20, out_channels=10, init_features=32):
        super(UNet1d, self).__init__()

        features = init_features
        self.encoder1 = UNet1d._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = UNet1d._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder3 = UNet1d._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder4 = UNet1d._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = UNet1d._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose1d(
            features * 16, features * 8, kernel_size=3, stride=2
        )
        self.decoder4 = UNet1d._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose1d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet1d._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose1d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet1d._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose1d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet1d._block(features * 2, features, name="dec1")

        self.conv = nn.Conv1d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        enc1 = self.encoder1(x)
        print(enc1.shape)
        enc2 = self.encoder2(self.pool1(enc1))
        print(enc2.shape)
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        out = self.decoder1(dec1)
        out = out.permute(0, 2, 1)
        return out


    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm1d(num_features=features)),
                    (name + "tanh1", nn.Tanh()),
                    (
                        name + "conv2",
                        nn.Conv1d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm1d(num_features=features)),
                    (name + "tanh2", nn.Tanh()),
                ]
            )
        )
    
    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c
    
# %% 
class UNet2d(nn.Module):

    def __init__(self, in_channels=20, out_channels=5, init_features=32, in_vars=1, out_vars=1, dropout=False):
        super(UNet2d, self).__init__()

        features = init_features
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.dropout = dropout

        self.encoder1 = UNet2d._block(in_channels*in_vars, features, name="enc1", dropout=dropout)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet2d._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottleneck = UNet2d._block(features * 2, features * 4, name="bottleneck")

        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2, padding=0
        )
        self.decoder2 = UNet2d._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2, padding=0
        )
        self.decoder1 = UNet2d._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels*out_vars, kernel_size=1
        )

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        x = x.view(x.shape[0], self.in_channels * self.in_vars, x.shape[3], x.shape[4])

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))

        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        out = self.conv(dec1)

        out = out.view(out.shape[0], self.out_vars, self.out_channels, out.shape[2], out.shape[3])
        out = out.permute(0, 1, 3, 4, 2)
        return out
    

    @staticmethod
    def _block(in_channels, features, name, dropout=False):
        layers = [
            (name + "conv1", nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False)),
            (name + "norm1", nn.BatchNorm2d(num_features=features)),
            (name + "tanh1", nn.Tanh())
        ]
        if dropout:
            layers.append((name + "dropout1", nn.Dropout(0.5)))
        layers.extend([
            (name + "conv2", nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)),
            (name + "norm2", nn.BatchNorm2d(num_features=features)),
            (name + "tanh2", nn.Tanh())
        ])
        if dropout:
            layers.append((name + "dropout2", nn.Dropout(0.5)))
        return nn.Sequential(OrderedDict(layers))

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c
# %%
# # #Example Usage
# model = UNet2d(in_channels=5, out_channels=10, init_features=32, in_vars=1, out_vars=2, dropout=False)
# print(model.count_params())
# ins = torch.randn(20,1,100,100,5) #BS, num_vars, Nx, Ny, T_in
# outs = model(ins)
# %% 