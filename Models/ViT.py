# %%
import torch.nn as nn
import torch
import math

from einops import rearrange
from einops.layers.torch import Rearrange

# Inspired from AI for Science Lecture series at ETH Zurich

# %%
class PositionalEncoding(nn.Module):
    """
    Positional Encoding module supporting multiple encoding types.
    
    Parameters
    ----------
    encoding_type : str
        Type of positional encoding: 'learnable', 'sinusoidal', or 'none'
    num_patches : int
        Number of patches (sequence length)
    embed_dim : int
        Embedding dimension
    dropout : float
        Dropout rate
    """
    def __init__(self, encoding_type='learnable', num_patches=None, embed_dim=None, dropout=0.):
        super().__init__()
        self.encoding_type = encoding_type
        self.dropout = nn.Dropout(dropout)
        
        if encoding_type == 'learnable':
            # Learnable positional embeddings (default, as in original code)
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
            
        elif encoding_type == 'sinusoidal':
            # Sinusoidal positional embeddings (fixed, not learned)
            pe = torch.zeros(num_patches, embed_dim)
            position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # Add batch dimension
            
            # Register as buffer (not a parameter, won't be updated during training)
            self.register_buffer('pos_embedding', pe)
            
        elif encoding_type == 'none':
            # No positional encoding
            self.pos_embedding = None
            
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}. Choose from 'learnable', 'sinusoidal', or 'none'.")
    
    def forward(self, x):
        """
        Add positional encoding to input.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, num_patches, embed_dim)
            
        Returns
        -------
        torch.Tensor
            Output with positional encoding added
        """
        if self.pos_embedding is not None:
            _, n, _ = x.shape
            x = x + self.pos_embedding[:, :n]
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                AttentionBlock(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViT(nn.Module):
    def __init__(self,
                image_size,
                patch_size,
                embed_dim,
                depth,
                n_heads,
                mlp_dim = 256,
                in_channels = 1,      # Input channels (previously 'channels')
                out_channels = 1,     # Output channels (new parameter)
                dim_head = 32,
                emb_dropout = 0.,
                pos_encoding_type = 'sinusoidal'):  # New parameter for positional encoding type
        super().__init__()
        image_height, image_width = image_size[0], image_size[1]
        patch_height, patch_width = patch_size[0], patch_size[1]

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        
        # Input patch dimension based on input channels
        input_patch_dim = in_channels * patch_height * patch_width
        
        # Output patch dimension based on output channels
        output_patch_dim = out_channels * patch_height * patch_width
        
        # Store for later use
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_height = image_height
        self.image_width = image_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(input_patch_dim),
            nn.Linear(input_patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.patch_to_image = nn.Sequential(
            nn.Linear(embed_dim, output_patch_dim),
            nn.LayerNorm(output_patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                     p1 = patch_height, p2 = patch_width, 
                     h = image_height // patch_height,
                     c = out_channels)
        )
        
        # Use the new PositionalEncoding module
        self.pos_encoding = PositionalEncoding(
            encoding_type=pos_encoding_type,
            num_patches=num_patches,
            embed_dim=embed_dim,
            dropout=emb_dropout
        )

        self.transformer = TransformerBlock(embed_dim, depth, n_heads, dim_head, mlp_dim)

        # Final convolution layer that maps from output channels to output channels
        self.conv_last = torch.nn.Conv2d(in_channels = out_channels,
                                          out_channels= out_channels,
                                          kernel_size = 3,
                                          padding     = 1)

    def forward(self, img):
        img = img[...,0]
        x = self.to_patch_embedding(img)
        x = self.pos_encoding(x)  # Apply positional encoding
        x = self.transformer(x)
        x = self.patch_to_image(x)
        x = self.conv_last(x)
        x = torch.unsqueeze(x, dim =-1)
        return x

    def count_params(self):
        nparams = 0
        for param in self.parameters():
            nparams += param.numel()
        return nparams
    
# # %% Example usage with different input/output channels
# image_size = (64, 64)
# patch_size = (16, 16)
# embed_dim = 128
# depth = 4
# n_heads = 4
# dim_head = 32
# emb_dropout = 0.0

# # Example 1: 3 input channels (RGB), 1 output channel (grayscale)
# model1 = ViT(image_size = image_size,
#             patch_size = patch_size,
#             embed_dim = embed_dim,
#             depth = depth,
#             n_heads = n_heads,
#             mlp_dim = 256,
#             in_channels = 3,      # RGB input
#             out_channels = 1,     # Grayscale output
#             dim_head = dim_head,
#             emb_dropout = emb_dropout)

# # Example 2: 1 input channel, 5 output channels (multi-task prediction)
# model2 = ViT(image_size = image_size,
#             patch_size = patch_size,
#             embed_dim = embed_dim,
#             depth = depth,
#             n_heads = n_heads,
#             mlp_dim = 256,
#             in_channels = 1,      # Single channel input
#             out_channels = 5,     # 5 output channels
#             dim_head = dim_head,
#             emb_dropout = emb_dropout)

# # Example 3: 4 input channels, 2 output channels
# model3 = ViT(image_size = image_size,
#             patch_size = patch_size,
#             embed_dim = embed_dim,
#             depth = depth,
#             n_heads = n_heads,
#             mlp_dim = 256,
#             in_channels = 4,      # 4 input channels
#             out_channels = 2,     # 2 output channels
#             dim_head = dim_head,
#             emb_dropout = emb_dropout)

# # Test the models
# print("Example 1: RGB to Grayscale")
# X1 = torch.rand(16, 3, 64, 64, 1)  # 3 input channels
# Y1 = model1(X1)
# print(f"Input shape: {X1.shape}")
# print(f"Output shape: {Y1.shape}")
# print(f"Parameters: {model1.count_params()}")

# print("\nExample 2: Single channel to 5 channels")
# X2 = torch.rand(16, 1, 64, 64, 1)  # 1 input channel
# Y2 = model2(X2)
# print(f"Input shape: {X2.shape}")
# print(f"Output shape: {Y2.shape}")
# print(f"Parameters: {model2.count_params()}")

# print("\nExample 3: 4 channels to 2 channels")
# X3 = torch.rand(16, 4, 64, 64, 1)  # 4 input channels
# Y3 = model3(X3)
# print(f"Input shape: {X3.shape}")
# print(f"Output shape: {Y3.shape}")
# print(f"Parameters: {model3.count_params()}")
# %%
