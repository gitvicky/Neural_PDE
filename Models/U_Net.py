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
class UNet2d(nn.Module):

    def __init__(self, in_channels=20, out_channels=5, init_features=32, num_vars=1, dropout=False):
        super(UNet2d, self).__init__()

        features = init_features
        self.num_vars = num_vars
        self.dropout = dropout

        self.encoder1 = UNet2d._block(in_channels, features, name="enc1", dropout=dropout)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet2d._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottleneck = UNet2d._block(features * 2, features * 4, name="bottleneck")

        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet2d._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=3, stride=2
        )
        self.decoder1 = UNet2d._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        x = x.view(int(x.shape[0]*self.num_vars), x.shape[2], x.shape[3], x.shape[4])

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

        out = out.view(out.shape[0], self.num_vars, out.shape[1], out.shape[2], out.shape[3])
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
