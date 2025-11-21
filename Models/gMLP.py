"""
---
Code is from https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/gmlp/__init__.py
---

# Pay Attention to MLPs (gMLP)

This is a [PyTorch](https://pytorch.org) implementation of the paper
[Pay Attention to MLPs](https://arxiv.org/abs/2105.08050).

This paper introduces a Multilayer Perceptron (MLP) based architecture with gating,
which they name **gMLP**. It consists of a stack of $L$ *gMLP* blocks.

"""

# %% 
from typing import Optional

import torch
from torch import nn


class GMLPBlock(nn.Module):
    """
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

    def __init__(self, d_model: int, d_ffn: int, seq_len: int):
        """
        * `d_model` is the dimensionality ($d$) of $X$
        * `d_ffn` is the dimensionality of $Z$
        * `seq_len` is the length of the token sequence ($n$)
        """
        super().__init__()
        # Normalization layer fro Pre-Norm
        self.norm = nn.LayerNorm([d_model])
        # Activation function $\sigma$
        self.activation = nn.GELU()
        # Projection layer for $Z = \sigma(XU)$
        self.proj1 = nn.Linear(d_model, d_ffn)
        # Spacial Gating Unit $s(\cdot)$
        self.sgu = SpacialGatingUnit(d_ffn, seq_len)
        # Projection layer for $Y = \tilde{Z}V$
        self.proj2 = nn.Linear(d_ffn // 2, d_model)
        # Embedding size (required by [Encoder](../models.html#Encoder).
        # We use the encoder module from transformer architecture and plug
        # *gMLP* block as a replacement for the [Transformer Layer](../models.html#Encoder).
        self.size = d_model

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        * `x` is the input embedding tensor $X$ of shape `[seq_len, batch_size, d_model]`
        * `mask` is a boolean mask of shape `[seq_len, seq_len, 1]` that controls the visibility of tokens
         among each other.
        """
        # Keep a copy for shortcut connection
        print(f'Input: {x.shape}')
        shortcut = x
        # Normalize $X$
        x = self.norm(x)
        # Projection and activation $Z = \sigma(XU)$
        z = self.activation(self.proj1(x))
        print(f'Proj1: {z.shape}')
        # Spacial Gating Unit $\tilde{Z} = s(Z)$
        z = self.sgu(z, mask)
        # Final projection $Y = \tilde{Z}V$
        z = self.proj2(z)

        # Add the shortcut connection
        return z + shortcut


    def count_params(self):
        nparams = 0

        for param in self.parameters():
            nparams += param.numel()
        return nparams
    
    
class SpacialGatingUnit(nn.Module):
    """
    ## Spatial Gating Unit

    $$s(Z) = Z_1 \odot f_{W,b}(Z_2)$$

    where $f_{W,b}(Z) = W Z + b$ is a linear transformation along the sequence dimension,
    and $\odot$ is element-wise multiplication.
    $Z$ is split into to parts of equal size $Z_1$ and $Z_2$ along the channel dimension (embedding dimension).
    """
    def __init__(self, d_z: int, seq_len: int):
        """
        * `d_z` is the dimensionality of $Z$
        * `seq_len` is the sequence length
        """
        super().__init__()
        # Normalization layer before applying $f_{W,b}(\cdot)$
        self.norm = nn.LayerNorm([d_z // 2])
        # Weight $W$ in $f_{W,b}(\cdot)$.
        #
        # The paper notes that it's important to initialize weights to small values and the bias to $1$,
        # so that during the initial training $s(\cdot)$ is close to identity (apart from the split).
        self.weight = nn.Parameter(torch.zeros(seq_len, seq_len).uniform_(-0.01, 0.01), requires_grad=True)
        # Weight $b$ in $f_{W,b}(\cdot)$
        #
        # The paper notes that it's important to initialize bias to $1$.
        self.bias = nn.Parameter(torch.ones(seq_len), requires_grad=True)

    def forward(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        * `z` is the input $Z$ of shape `[seq_len, batch_size, d_z]`
        * `mask` is is a boolean mask of shape `[seq_len, seq_len, 1]` that controls the visibility of tokens
         among each other. The last dimension of size `1` is the batch, which we have in other transformer
         implementations and was left for compatibility.
        """

        # Get sequence length
        seq_len = z.shape[0]
        # Split $Z$ into $Z_1$ and $Z_2$
        z1, z2 = torch.chunk(z, 2, dim=-1)
        print(f'Channel_split1: {z1.shape}, Channel_split2: {z2.shape}') #Split along the Channels


        # Normalize $Z_2$ before $f_{W,b}(\cdot)$
        z2 = self.norm(z2)
        # Get the weight matrix; truncate if larger than `seq_len`
        weight = self.weight[:seq_len, :seq_len]

        # $f_{W,b}(Z_2) = W Z_2 + b$
        z2 = torch.einsum('ij,jbd->ibd', weight, z2) + self.bias[:seq_len, None, None]
        print(f'spatial_proj: {z2.shape}')

        # $Z_1 \odot f_{W,b}(Z_2)$
        return z1 * z2
    


# #Example Usage

# X = torch.ones(64, 1, 2) # seq_len, BS, ndim
# model = GMLPBlock(d_model=2, d_ffn=32, seq_len=64)
# out = model(X)
# %%
