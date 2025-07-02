"""

gated MLPs (gMLP) modified to fit for spatial data - not using patching but using spatial projection along x and y axes. 
---
Original Code https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/gmlp/__init__.py
---
Looking for efficiencies from using permutation and elementwise operations. 

"""

# %% 
from typing import Optional

import torch
from torch import nn


class gMLPBlock(nn.Module):
    r"""
    ## gMLP Block

    Each block does the following transformations to input embeddings
    $X \in \mathbb{R}^{n \times d}$ where $n$ is the sequence length
    and $d$ is the dimensionality of the embeddings:

    \begin{align}
    Z &= \sigma(XU) \\
    \tilde{Z} &= s(Z) \\
    Y &= \tilde{Z}V \\
    \end{align}

    where $V$ and $U$ are learnable projection weights.
    $s(\cdot)$ is the Spacial Gating Unit defined below.
    Output dimensionality of $s(\cdot)$ will be half of $Z$.
    $\sigma$ is an activation function such as
    [GeLU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html).

    """
    def __init__(self, d_in: int, d_ffn: int, Nx: int, Ny: int, norm='LayerNorm'):
        """
        * d_in is the dimensionality ($d$) of $X$
        * d_ffn is the dimensionality of $Z$
        * Nx is the length of discretisation along x-axis
        * Ny is the length of discretisation along y-axis

        """
        super().__init__()
        # Normalization layer fro Pre-Norm
        if norm == 'LayerNorm':
            self.norm = nn.LayerNorm([d_in])
        else:
            self.norm = nn.Identity()
        # Activation function $\sigma$
        self.activation = nn.GELU()
        # Projection layer for $Z = \sigma(XU)$
        self.proj1 = nn.Linear(d_in, d_ffn)
        # Spacial Gating Unit $s(\cdot)$
        self.sgu = SpacialGatingUnit(d_ffn, Nx, Ny)
        # Projection layer for $Y = \tilde{Z}V$
        self.proj2 = nn.Linear(d_ffn // 2, d_in)
        # Embedding size (required by [Encoder](../models.html#Encoder).
        # We use the encoder module from transformer architecture and plug
        # *gMLP* block as a replacement for the [Transformer Layer](../models.html#Encoder).
        self.size = d_in

    def forward(self, x: torch.Tensor):
        """
        * x is the input embedding tensor $X$ of shape [seq_len, batch_size, d_model]
        * mask is a boolean mask of shape [seq_len, seq_len, 1] that controls the visibility of tokens
         among each other.
        """
        # Keep a copy for shortcut connection
        shortcut = x
        # Normalize $X$
        x = self.norm(x)
        # Projection and activation $Z = \sigma(XU)$
        z = self.activation(self.proj1(x))
        # Spacial Gating Unit $\tilde{Z} = s(Z)$
        z = self.sgu(z)
        # Final projection $Y = \tilde{Z}V$
        z = self.proj2(z)

        # Add the shortcut connection
        return z + shortcut


class SpacialGatingUnit(nn.Module):
    r"""
    ## Spatial Gating Unit

    $$s(Z) = Z_1 \odot f_{W,b}(Z_2)$$

    where $f_{W,b}(Z) = W Z + b$ is a linear transformation along the sequence dimension,
    and $\odot$ is element-wise multiplication.
    $Z$ is split into to parts of equal size $Z_1$ and $Z_2$ along the channel dimension (embedding dimension).
    """
    def __init__(self, d_z: int, Nx: int, Ny: int, norm='LayerNorm'):
        """
        * d_z is the dimensionality of $Z$
        * seq_len is the sequence length
        """
        super().__init__()
        if norm == 'LayerNorm':
            self.norm = nn.LayerNorm([d_z // 2])
        else:
            self.norm = nn.Identity()

        # #Using einsum 
        # self.weight_x = nn.Parameter(torch.zeros(Nx, Nx).uniform_(-0.01, 0.01), requires_grad=True)
        # self.bias_x = nn.Parameter(torch.ones(Nx), requires_grad=True)
        # self.weight_y = nn.Parameter(torch.zeros(Ny, Ny).uniform_(-0.01, 0.01), requires_grad=True)
        # self.bias_y = nn.Parameter(torch.ones(Ny), requires_grad=True)

        #Using Permute
        self.linear_x = nn.Linear(Nx, Nx)
        self.linear_y = nn.Linear(Ny, Ny)


    def forward(self, z: torch.Tensor):

        # Get sequence length
        seq_len = z.shape[0]
        # Split $Z$ into $Z_1$ and $Z_2$
        z1, z2 = torch.chunk(z, 2, dim=-1)

        # Normalize $Z_2$ before $f_{W,b}(\cdot)$
        z2 = self.norm(z2)

        # # Get the weight matrix; truncate if larger than seq_len
        # weight_x = self.weight_x[:seq_len, :seq_len]
        # weight_y = self.weight_y[:seq_len, :seq_len]

        # # $f_{W,b}(Z_2) = W Z_2 + b$
        # z2_x = torch.einsum('ij,jkbd->ikbd', weight_x, z2) + self.bias_x[:seq_len, None, None]
        # z2_y = torch.einsum('ik,jkbd->ijbd', weight_y, z2) + self.bias_y[:seq_len, None, None]

        z2_x = self.linear_x(z2.permute(1, 2, 3, 0)).permute(3, 0, 1, 2)
        z2_y = self.linear_y(z2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # $Z_1 \odot f_{W,b}(Z_2)$
        return z1 * z2_x * z2_y
    

class gMLP(nn.Module):
    def __init__(self, n_blocks: int, d_in: int, d_ffn: int, Nx: int, Ny: int):
        """
        * n_blocks is the number of gmLP blocks
        * d_in is the dimensionality ($d$) of $X$
        * d_ffn is the dimensionality of $Z$
        * Nx is the length of discretisation along x-axis
        * Ny is the length of discretisation along y-axis

        """
        super().__init__()


        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(
                gMLPBlock(d_in=d_in, d_ffn=d_ffn, Nx=Nx, Ny=Ny)                
            )


    def forward(self, x):
        x = x.permute(2, 3, 0, 1)

        for block in self.blocks:
            x = block(x)
        x = x.permute(2, 3, 0, 1)
        return x 

    def count_params(self):
        nparams = 0

        for param in self.parameters():
            nparams += param.numel()
        return nparams

# %% 
# #Example Usage

# X = torch.ones(1, 2, 64, 64) #BS, ndim, Nx, Ny
# model = gMLP(n_blocks = 8, d_in=2, d_ffn=32, Nx=64, Ny=64)
# Y = model(X)

# print(f"Input shape: {X.shape}")
# print(f"Output shape: {Y.shape}")
# print(f"Paramters: {model.count_params()}")
# %%