# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22 Apr, 2024
@author: DeepXde
@modified: vgopakum

DeepOnet

"""      
import numpy as np 
import torch 
import torch.nn as nn 
import torch.functional as F 

# %%
################################################################
# DeepOnet - Code modified from DeepXDE
################################################################

class FNN(torch.nn.Module):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes, activation=torch.nn.Tanh):
        super().__init__()

        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                torch.nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i], dtype=torch.float32
                )
            )

    def forward(self, inputs):
        x = inputs
        for j, linear in enumerate(self.linears[:-1]):
            x = (
                self.activation(linear(x))
            )
        x = self.linears[-1](x)
        return x


class DeepONet(torch.nn.Module):
    """Deep operator network.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
    """

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation=torch.nn.Tanh,
        ):
        super().__init__()
        # Fully connected network
        self.activation = activation
        self.branch = FNN(layer_sizes_branch, activation)
        self.trunk = FNN(layer_sizes_trunk, activation)
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))

def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        # Branch net to encode the input function
        x_func = self.branch(x_func)
        # Trunk net to encode the domain of the output function
        x_loc = self.activation(self.trunk(x_loc))
        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = torch.einsum("bi,bi->b", x_func, x_loc)
        x = torch.unsqueeze(x, 1)
        # Add bias
        x += self.b
        return x