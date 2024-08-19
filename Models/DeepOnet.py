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

import operator
from functools import reduce
from functools import partial
from collections import OrderedDict

# %%
################################################################
# DeepOnet - Code modified from DeepXDE
################################################################

class FNN(torch.nn.Module):
    """Fully-connected neural network."""

    def __init__(self, input_size, output_size, width, num_layers, activation=torch.tanh):
        super().__init__()

        self.linears = torch.nn.ModuleList()
        self.linears.append(torch.nn.Linear(input_size, width, dtype=torch.float32))
        for i in range(1, num_layers-1):
            self.linears.append(
                torch.nn.Linear(
                    width, width, dtype=torch.float32
                )
            )
        self.linears.append(torch.nn.Linear(width, output_size, dtype=torch.float32))
        self.activation = activation 

    def forward(self, inputs):
        x = inputs
        for j, linear in enumerate(self.linears[:-1]):
            x = self.activation(linear(x))
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
        in_branch,
        width_branch,
        layers_branch, 
        out_branch,
        in_trunk,
        width_trunk,
        layers_trunk, 
        out_trunk,
        activation=torch.tanh
        ):
        super(DeepONet, self).__init__()


        # Fully connected network
        self.activation = activation
        self.branch = FNN(in_branch, out_branch, width_branch, layers_branch)
        self.trunk = FNN(in_trunk, out_trunk, width_trunk, layers_trunk)
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

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c
# %% 
#Example Usage
model = DeepONet(in_branch=100,
        width_branch=256,
        layers_branch=4, 
        out_branch=100,
        in_trunk=2,
        width_trunk=256,
        layers_trunk=4, 
        out_trunk=100)

trunk_in = torch.randn(20, 100)
branch_in = torch.randn(1, 2)
output = model([trunk_in, branch_in])
print(output.shape)
# %%
