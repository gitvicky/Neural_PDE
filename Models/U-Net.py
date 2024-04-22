# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22 Apr, 2024
@author: vgopakum

U-Net in 2D

"""      
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

    def __init__(self, in_channels=20, out_channels=5, init_features=32):
        super(UNet2d, self).__init__()

        features = init_features
        self.encoder1 = UNet2d._block(in_channels, features, name="enc1")
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
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))


        bottleneck = self.bottleneck(self.pool2(enc2))

        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "tanh1", nn.Tanh()),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "tanh2", nn.Tanh()),
                ]
            )
        )
    

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c
    

class UNet2d_dropout(nn.Module):

    def __init__(self, in_channels=20, out_channels=5, init_features=32, dropout_rate=0.1):
        super(UNet2d_dropout, self).__init__()

        features = init_features
        self.encoder1 = UNet2d._block(in_channels, features, name="enc1")
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


        # self.dropout_layers = nn.ModuleList()
        # for ii in range(4):
        #     self.dropout_layers.append(nn.Dropout(p=dropout_rate))

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        enc1 = self.encoder1(x)
        # enc1 = self.dropout_layers[0](enc1)
        enc1 = self.dropout(enc1) #Dropout
        enc2 = self.encoder2(self.pool1(enc1))
        # enc2 = self.dropout_ layers[1](enc2)
        enc2 = self.dropout(enc2) #Dropout


        bottleneck = self.bottleneck(self.pool2(enc2))

        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        # dec2 = self.dropout_layers[2](dec2)
        dec2 = self.dropout(dec2) #Dropout
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        # dec1 = self.dropout_layers[3](dec1)
        dec1 = self.dropout(dec1) #Dropout
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "tanh1", nn.Tanh()),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "tanh2", nn.Tanh()),
                ]
            )
        )
    

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c
    
    def enable_dropout(self):
            """Function to enable the dropout layers during test-time"""
            self.dropout.train()
            # for m in self.dropout_layers:
            #     if m.__class__.__name__.startswith("Dropout"):
            #         m.train() 


#Estimating the output uncertainties using Dropout. 
def Dropout_eval(net, x, step, Nrepeat=10):
    net.eval()
    net.enable_dropout()
    preds = torch.zeros(Nrepeat, x.shape[0], step, x.shape[-1], x.shape[-1])
    for i in range(Nrepeat):
        preds[i] = net(x)
    return torch.mean(preds, axis=0), torch.std(preds, axis=0)
