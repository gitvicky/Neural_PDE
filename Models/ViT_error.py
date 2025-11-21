# %% 
#Currently there is an issue with the model. Both attention and MLP are doing sequence mixing and nothing is doing along the token dimensions as I understand this. 
import torch
import torch.nn as nn
import operator
from functools import reduce


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H', W')
        print(f'Project: {x.shape}')
        x = x.flatten(2)  # (B, embed_dim, H'*W')
        x = x.transpose(1, 2)  # (B, H'*W', embed_dim)
        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        print(f'q: {q.shape}, k^t: {k.transpose(-2, -1).shape}, v: {v.shape}')
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x     

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        print(f'attn: {x.shape}')
        x = x + self.mlp(self.norm2(x))
        print(f'mlp: {x.shape}')
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, out_channels=3, embed_dim=768, depth=12,
                 n_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_channels = out_channels

    def forward(self, x):
        x = self.patch_embed(x)
        print(f'patch embed: {x.shape}')
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)
        x = self.linear(x)
        # Reshape to image
        B = x.shape[0]
        x = x.view(B, self.patch_embed.grid_size, self.patch_embed.grid_size, self.patch_size, self.patch_size, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.out_channels, self.img_size, self.img_size)

        return x
    
    
    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

# Example usage
img_size = 128
patch_size = 16
in_channels = 2
out_channels = 2
embed_dim = 768
depth = 2
n_heads = 4

model = VisionTransformer(
    img_size=img_size,
    patch_size=patch_size,
    in_channels=in_channels,
    out_channels=out_channels,
    embed_dim=embed_dim,
    depth=depth,
    n_heads=n_heads
)

# Generate a random input tensor
x = torch.randn(1, in_channels, img_size, img_size)

# Forward pass
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Paramters: {model.count_params()}")
# %%



# %%
#Including both the variables and the time -- Batch, Variables, Nt, Nx, Ny

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, time_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (time_channels // patch_size[0], img_size[0] // patch_size[1], img_size[1] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        x = x.flatten(2)  # (B, embed_dim, T'*H'*W')
        x = x.transpose(1, 2)  # (B, T'*H'*W', embed_dim)
        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(4, 16, 16), in_channels=3, time_channels=16, out_channels=3, embed_dim=768, depth=12,
                 n_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, time_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, patch_size[0] * patch_size[1] * patch_size[2] * out_channels)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.time_channels = time_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)
        
        x = self.head(x)
        
        # Reshape to original dimensions
        B = x.shape[0]
        x = x.view(B, self.patch_embed.grid_size[0], self.patch_embed.grid_size[1], self.patch_embed.grid_size[2],
                   self.patch_size[0], self.patch_size[1], self.patch_size[2], self.out_channels)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.view(B, self.out_channels, self.time_channels, self.img_size[0], self.img_size[1])
        x = x.permute(0, 1, 3, 4, 2)

        return x
    
    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c
    
# # Example usage
# img_size = (128, 128)
# patch_size = (1, 16, 16) #Should be a factor of img_size. 
# in_channels = 2
# time_channels = 1
# out_channels = 2
# embed_dim = 128
# depth = 1
# n_heads = 1

# model = VisionTransformer(
#     img_size=img_size,
#     patch_size=patch_size,
#     in_channels=in_channels,
#     time_channels=time_channels,
#     out_channels=out_channels,
#     embed_dim=embed_dim,
#     depth=depth,
#     n_heads=n_heads
# )

# # Generate a random input tensor
# x = torch.randn(1, in_channels, img_size[0], img_size[1], time_channels)

# # Forward pass
# output = model(x)
# print(f"Input shape: {x.shape}")
# print(f"Output shape: {output.shape}")
# print(f"Paramters: {model.count_params()}")

# %%
