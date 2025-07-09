"""
Simplified from MPP Code base for fixed history training.
https://github.com/PolymathicAI/the_well/tree/master/the_well/benchmark/models/avit
"""
# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath


class hMLP_stem(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        dim_in: int,
        hidden_dim: int,
        groups: int = 12,
        n_spatial_dims: int = 2,
    ):
        super().__init__()
        if n_spatial_dims == 1:
            conv = nn.Conv1d
        elif n_spatial_dims == 2:
            conv = nn.Conv2d
        elif n_spatial_dims == 3:
            conv = nn.Conv3d
        self.n_spatial_dims = n_spatial_dims

        self.in_projs = torch.nn.Sequential(
            *[
                conv(dim_in, hidden_dim // 4, kernel_size=4, stride=4, bias=False),
                nn.GroupNorm(groups, hidden_dim // 4, affine=True),
                nn.GELU(),
                conv(
                    hidden_dim // 4,
                    hidden_dim // 4,
                    kernel_size=2,
                    stride=2,
                    bias=False,
                ),
                nn.GroupNorm(groups, hidden_dim // 4, affine=True),
                nn.GELU(),
                conv(hidden_dim // 4, hidden_dim, kernel_size=2, stride=2, bias=False),
                nn.GroupNorm(groups, hidden_dim, affine=True),
            ]
        )

    def forward(self, x):
        x = self.in_projs(x)
        return x


class hMLP_output(nn.Module):
    """Patch to Image De-bedding"""

    def __init__(
        self,
        dim_out: int,
        hidden_dim: int = 768,
        groups: int = 12,
        n_spatial_dims: int = 2,
    ):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        if n_spatial_dims == 1:
            conv = nn.ConvTranspose1d
            self.conv_func = F.conv_transpose1d
        elif n_spatial_dims == 2:
            conv = nn.ConvTranspose2d
            self.conv_func = F.conv_transpose2d
        elif n_spatial_dims == 3:
            conv = nn.ConvTranspose3d
            self.conv_func = F.conv_transpose3d
        else:
            conv = nn.Linear
        self.out_proj = nn.Sequential(
            *[
                conv(hidden_dim, hidden_dim // 4, kernel_size=2, stride=2, bias=False),
                nn.GroupNorm(groups, hidden_dim // 4, affine=True),
                nn.GELU(),
                conv(
                    hidden_dim // 4,
                    hidden_dim // 4,
                    kernel_size=2,
                    stride=2,
                    bias=False,
                ),
                nn.GroupNorm(groups, hidden_dim // 4, affine=True),
                nn.GELU(),
                conv(hidden_dim // 4, dim_out, kernel_size=4, stride=4, bias=False),
            ]
        )

    def forward(self, x):
        x = self.out_proj(x)
        return x


class AxialAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim=768,
        num_heads=12,
        n_spatial_dims=2,
        drop_path=0,
        layer_scale_init_value=1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.n_spatial_dims = n_spatial_dims
        # Regularization
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((hidden_dim)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # Just doing plain MHA but with paralell MLP
        self.norm = nn.LayerNorm(hidden_dim, bias=False)
        self.fused_heads = [hidden_dim, hidden_dim, hidden_dim, 4 * hidden_dim]
        self.fused_projection = nn.Linear(hidden_dim, sum(self.fused_heads))
        self.qnorm = nn.LayerNorm(hidden_dim // num_heads, bias=False)
        self.knorm = nn.LayerNorm(hidden_dim // num_heads, bias=False)
        self.output_head = nn.Linear(hidden_dim, hidden_dim)

        self.mlp_remaining = nn.Sequential(
            nn.GELU(), nn.Linear(4 * hidden_dim, hidden_dim)
        )

        # Make rearrange strings for different spatial dims
        if n_spatial_dims == 2:
            self.head_split = "b h w (he c) -> b h w he c"
            self.spatial_permutations = [
                ("b h w he c -> b h he w c", "b h he w c -> b h w (he c)"),
                ("b h he w c -> b h w he c", "b h w he c -> b h w (he c)"),
            ]

        elif n_spatial_dims == 3:
            self.head_split = "b h w d (he c) -> b h w d he c"
            self.spatial_permutations = [
                ("b h w d he c -> b h w he d c", "b h w he d c -> b h w d (he c)"),
                ("b h w he d c -> b h d he w c", "b h d he w c -> b h w d (he c)"),
                ("b h d he w c -> b w d he h c", "b w d he h c -> b h w d (he c)"),
            ]

    def forward(self, x):
        # input is t x b x c x h x w
        input = x.clone()
        x = self.norm(x)
        # Spatial forward
        q, k, v, ff = self.fused_projection(x).split(self.fused_heads, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, self.head_split, he=self.num_heads), (q, k, v)
        )
        q, k = self.qnorm(q), self.knorm(k)

        out = torch.zeros_like(x)
        for in_permutation, out_permutation in self.spatial_permutations:
            q, k, v = map(lambda t: rearrange(t, in_permutation), (q, k, v))
            ax_out = F.scaled_dot_product_attention(q, k, v)
            ax_out = rearrange(ax_out, out_permutation)
            out = out + ax_out
        # Recombine
        x = self.output_head(out) + self.mlp_remaining(ff)
        return input + self.drop_path(self.gamma * x)


class AViT(nn.Module):
    """
    Uses axial attention to predict forward dynamics. This simplified version
    just stacks time in channels.

    Args:
        dim_in (int): Number of input channels
        dim_out (int): Number of output channels
        n_spatial_dims (int): Number of spatial dimensions (1, 2, or 3)
        spatial_resolution (tuple): Size of spatial dimensions
        hidden_dim (int): Dimension of the embedding
        num_heads (int): Number of attention heads
        processor_blocks (int): Number of blocks (consisting of spatial mixing - temporal attention)
        drop_path (float): Drop path rate
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution: tuple[int, ...],
        hidden_dim: int = 768,
        num_heads: int = 12,
        processor_blocks: int = 8,
        drop_path: float = 0.0,
    ):
        super().__init__()
        
        # Store parameters as instance variables
        self.n_spatial_dims = n_spatial_dims
        self.spatial_resolution = spatial_resolution
        self.drop_path = drop_path
        self.dp = np.linspace(0, drop_path, processor_blocks)

        # Patch size hardcoded at 16 in this implementation
        self.patch_size = 16
        # Embedding
        pe_size = tuple(int(k / self.patch_size) for k in self.spatial_resolution) + (
            hidden_dim,
        )
        self.absolute_pe = nn.Parameter(0.02 * torch.randn(*pe_size))
        self.embed = hMLP_stem(
            dim_in=dim_in,
            hidden_dim=hidden_dim,
            n_spatial_dims=self.n_spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [
                AxialAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    n_spatial_dims=self.n_spatial_dims,
                    drop_path=self.dp[i],
                )
                for i in range(processor_blocks)
            ]
        )
        self.debed = hMLP_output(
            hidden_dim=hidden_dim,
            dim_out=dim_out,
            n_spatial_dims=self.n_spatial_dims,
        )

        if self.n_spatial_dims == 2:
            self.embed_reshapes = ["b h w c -> b c h w", "b c h w -> b h w c"]
        elif self.n_spatial_dims == 3:
            self.embed_reshapes = ["b h w d c -> b c h w d", "b c h w d -> b h w d c"]

    def forward(self, x):
        # Input B, ..., C
        # Encode
        x = rearrange(x, self.embed_reshapes[0])
        x = self.embed(x)
        x = rearrange(x, self.embed_reshapes[1])
        # Process
        x = x + self.absolute_pe
        for blk in self.blocks:
            x = blk(x)
        # Decode
        x = rearrange(x, self.embed_reshapes[0])
        x = self.debed(x)
        x = rearrange(x, self.embed_reshapes[1])
        return x


# %%
# Example usage:
if __name__ == "__main__":
    import torch
    
    # Example configuration for 2D
    dim_in = 3  # input channels
    dim_out = 1  # output channels
    n_spatial_dims = 2  # 2D case
    spatial_resolution = (128, 128)  # height, width (must be divisible by 16)
    hidden_dim = 768
    num_heads = 12
    processor_blocks = 8
    
    # Create the model
    model = AViT(
        dim_in=dim_in,
        dim_out=dim_out,
        n_spatial_dims=n_spatial_dims,
        spatial_resolution=spatial_resolution,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        processor_blocks=processor_blocks,
        drop_path=0.1
    )
    
    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, *spatial_resolution, dim_in)
    
    # Forward pass
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model created successfully!")
    
    # # Example for 3D
    # print("\n3D Example:")
    # model_3d = AViT(
    #     dim_in=1,
    #     dim_out=1,
    #     n_spatial_dims=3,
    #     spatial_resolution=(64, 64, 64),  # depth, height, width
    #     hidden_dim=384,
    #     num_heads=6,
    #     processor_blocks=4,
    # )
    
    # dummy_input_3d = torch.randn(1, 64, 64, 64, 1)
    # output_3d = model_3d(dummy_input_3d)
    # print(f"3D Input shape: {dummy_input_3d.shape}")
    # print(f"3D Output shape: {output_3d.shape}")

# %%
