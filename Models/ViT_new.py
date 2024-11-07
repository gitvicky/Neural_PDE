# %%
import torch.nn as nn
import torch

from einops import rearrange
from einops.layers.torch import Rearrange

#Inspired from AI for Science Lecture series at ETH Zurich
# %% 

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

# %% 

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

# %% 

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

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# %% 


class ViT(nn.Module):
    def __init__(self,
                image_size,
                patch_size,
                embed_dim,
                depth,
                n_heads,
                mlp_dim = 256,
                channels = 1,
                dim_head = 32,
                emb_dropout = 0.,):
        super().__init__()
        image_height, image_width = image_size[0], image_size[1]
        patch_height, patch_width = patch_size[0], patch_size[1]

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.patch_to_image = nn.Sequential(
            nn.Linear(embed_dim, patch_dim),
            nn.LayerNorm(patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height, p2 = patch_width, h = image_height // patch_height)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerBlock(embed_dim, depth, n_heads, dim_head, mlp_dim)

        self.conv_last = torch.nn.Conv2d(in_channels = channels,
                                          out_channels= channels,
                                          kernel_size = 3,
                                          padding     = 1)

    def forward(self, img):
        img = img[...,0]
        x = self.to_patch_embedding(img)
        _, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.patch_to_image(x)
        x = self.conv_last(x)
        x = torch.unsqueeze(x, -1)
        return x


    def count_params(self):
        nparams = 0

        for param in self.parameters():
            nparams += param.numel()
        return nparams
    
# %% 

# #Example usage
# image_size = (64, 64)
# patch_size = (16, 16)
# embed_dim = 128
# depth = 4
# n_heads = 4
# dim_head = 32
# emb_dropout = 0.0

# model = ViT(image_size = image_size,
#             patch_size = patch_size,
#             embed_dim = embed_dim,
#             depth = depth,
#             n_heads = n_heads,
#             mlp_dim = 256,
#             channels = 1,
#             dim_head = dim_head,
#             emb_dropout = emb_dropout)

# #Bs, N_vars, Nx, Ny, Nt
# X = torch.rand(16,1,64,64,1)
# Y = model(X)

# print(f"Input shape: {X.shape}")
# print(f"Output shape: {Y.shape}")
# print(f"Paramters: {model.count_params()}")
# %%
