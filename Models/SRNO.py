################################################################
# SRNO - Code based on "Super-Resolution Neural Operator"
# (Wei et al., CVPR 2023) and formatted like the FNO code.
################################################################
# %% 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import operator
from functools import reduce
from functools import partial
from collections import OrderedDict

# %% 
################################################################
# 1. Encoder (E_psi)
# The paper uses EDSR-baseline or RDN[cite: 188].
# This is a simple CNN placeholder for E_psi.
################################################################
class SimpleEncoder(nn.Module):
    def __init__(self, in_channels=3, d_e=64):
        super(SimpleEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, d_e, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(d_e, d_e, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(d_e, d_e, kernel_size=3, padding=1)
        self.activation = F.relu

    def forward(self, x):
        # Input x: (B, 3, H_c, W_c)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        # Output: (B, d_e, H_c, W_c)
        return x

# %% 
################################################################
# 2. Lifting (L)
# Implements the "interpolation-free" lifting from Fig. 2c [cite: 110, 152, 155]
################################################################
class Lifting(nn.Module):
    def __init__(self, d_e, width):
        super(Lifting, self).__init__()
        self.d_e = d_e
        self.width = width
        # Input dim: 4 * (d_e + 2) + 2 [cite: 156]
        # 4 neighbors, each with d_e features + 2 relative coords
        # + 2 for scale encoding
        self.mlp = nn.Linear(4 * (d_e + 2) + 2, width)

    def forward(self, lr_feat, hr_coord_grid, scale):
        """
        lr_feat: (B, d_e, H_c, W_c) - Features from encoder
        hr_coord_grid: (B, H_f, W_f, 2) - Normalized coords [0, 1] for HR grid
        scale: (float, float) - (scale_x, scale_y)
        """
        B, C, H_c, W_c = lr_feat.shape
        B, H_f, W_f, _ = hr_coord_grid.shape
        N = H_f * W_f
        
        # --- 1. Get neighbor coordinates and bilinear weights ---
        # Flatten HR coordinates and scale to LR grid
        hr_coords = hr_coord_grid.view(B, N, 2) # (B, N, 2)
        lr_coords = hr_coords * torch.tensor([W_c - 1, H_c - 1], device=lr_feat.device)
        
        x_lr, y_lr = lr_coords[..., 0], lr_coords[..., 1]
        
        # Get 4 neighbor integer coords
        x0 = torch.floor(x_lr)
        x1 = x0 + 1
        y0 = torch.floor(y_lr)
        y1 = y0 + 1
        
        # Calculate bilinear weights (s_l in paper [cite: 152])
        s_x1 = x_lr - x0
        s_x0 = 1 - s_x1
        s_y1 = y_lr - y0
        s_y0 = 1 - s_y1
        
        s_00 = (s_x0 * s_y0).unsqueeze(-1) # (B, N, 1)
        s_10 = (s_x1 * s_y0).unsqueeze(-1)
        s_01 = (s_x0 * s_y1).unsqueeze(-1)
        s_11 = (s_x1 * s_y1).unsqueeze(-1)

        # Clamp coordinates to be within grid bounds
        x0_c = x0.long().clamp(0, W_c - 1)
        x1_c = x1.long().clamp(0, W_c - 1)
        y0_c = y0.long().clamp(0, H_c - 1)
        y1_c = y1.long().clamp(0, H_c - 1)

        # --- 2. Gather neighbor features ---
        # Flatten feature map for gathering
        lr_feat_flat = lr_feat.view(B, C, H_c * W_c)
        
        # Get 1D indices for 4 neighbors
        idx_00 = (y0_c * W_c + x0_c).unsqueeze(1).expand(-1, C, -1) # (B, C, N)
        idx_10 = (y0_c * W_c + x1_c).unsqueeze(1).expand(-1, C, -1)
        idx_01 = (y1_c * W_c + x0_c).unsqueeze(1).expand(-1, C, -1)
        idx_11 = (y1_c * W_c + x1_c).unsqueeze(1).expand(-1, C, -1)
        
        # Gather features
        feat_00 = torch.gather(lr_feat_flat, 2, idx_00).permute(0, 2, 1) # (B, N, C)
        feat_10 = torch.gather(lr_feat_flat, 2, idx_10).permute(0, 2, 1)
        feat_01 = torch.gather(lr_feat_flat, 2, idx_01).permute(0, 2, 1)
        feat_11 = torch.gather(lr_feat_flat, 2, idx_11).permute(0, 2, 1)

        # --- 3. Calculate relative coordinates (delta_l in paper [cite: 155]) ---
        delta_00 = torch.stack([x_lr - x0, y_lr - y0], dim=-1) # (B, N, 2)
        delta_10 = torch.stack([x_lr - x1, y_lr - y0], dim=-1)
        delta_01 = torch.stack([x_lr - x0, y_lr - y1], dim=-1)
        delta_11 = torch.stack([x_lr - x1, y_lr - y1], dim=-1)

        # --- 4. Construct concatenated input for MLP ---
        # Apply bilinear weights to features
        weighted_feat_00 = feat_00 * s_00
        weighted_feat_10 = feat_10 * s_10
        weighted_feat_01 = feat_01 * s_01
        weighted_feat_11 = feat_11 * s_11
        
        # Concatenate weighted features and deltas
        inp_00 = torch.cat([weighted_feat_00, delta_00], dim=-1) # (B, N, C+2)
        inp_10 = torch.cat([weighted_feat_10, delta_10], dim=-1)
        inp_01 = torch.cat([weighted_feat_01, delta_01], dim=-1)
        inp_11 = torch.cat([weighted_feat_11, delta_11], dim=-1)
        
        # Create scale encoding (c in paper [cite: 155])
        r_x, r_y = scale
        scale_code = torch.tensor([2/r_x, 2/r_y], device=lr_feat.device)
        scale_code = scale_code.view(1, 1, 2).expand(B, N, 2)
        
        # Concatenate all inputs
        full_inp = torch.cat([inp_00, inp_10, inp_01, inp_11, scale_code], dim=-1)
        
        # --- 5. Apply MLP ---
        z0 = self.mlp(full_inp) # (B, N, width)
        return z0

# %% 
################################################################
# 3. Kernel Integral (K)
# Implements the Galerkin-type attention block from Fig. 2b [cite: 78, 168]
################################################################
class KernelIntegral(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super(KernelIntegral, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.head_dim = embed_dim // n_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.ln_k = nn.LayerNorm(embed_dim)
        self.ln_v = nn.LayerNorm(embed_dim)
        
        # FFN (O block in Fig. 2b [cite: 185, 187])
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.activation = F.gelu

    def forward(self, z):
        # z: (B, N, C) where C = embed_dim, N = H_f * W_f
        B, N, C = z.shape
        z_res1 = z
        
        # --- 1. Compute Q, K, V ---
        q = self.q_proj(z) # (B, N, C)
        k = self.k_proj(z)
        v = self.v_proj(z)
        
        # --- 2. Apply LayerNorm to K, V [cite: 87, 95, 181] ---
        k_norm = self.ln_k(k)
        v_norm = self.ln_v(v)
        
        # --- 3. Reshape for Multi-Head Attention [cite: 189] ---
        q = q.view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, N, D_h)
        # Transpose k for kernel dot product
        k_norm = k_norm.view(B, N, self.n_heads, self.head_dim).permute(0, 2, 3, 1) # (B, H, D_h, N)
        v_norm = v_norm.view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, N, D_h)

        # --- 4. Galerkin-type Attention (Eq. 12 ) ---
        # (B, H, D_h, N) @ (B, H, N, D_h) -> (B, H, D_h, D_h)
        kernel = torch.matmul(k_norm, v_norm) / N # Normalize by number of points
        
        # (B, H, N, D_h) @ (B, H, D_h, D_h) -> (B, H, N, D_h)
        attn_out = torch.matmul(q, kernel)
        
        # Reshape back to (B, N, C)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, C)
        
        # --- 5. Residuals and FFN (Eq. 13  & Fig. 2b [cite: 88, 90]) ---
        # First residual connection
        z_res2 = z_res1 + attn_out
        
        # FFN block
        ffn_out = self.ffn(z_res2)
        
        # Second residual connection
        z_out = z_res1 + ffn_out
        
        return z_out

# %% ################################################################
# 4. SRNO (Main Module)
# Combines L, K, and P
################################################################
class SRNO(nn.Module):
    def __init__(self, in_channels=3, d_e=64, width=256, n_layers=2, n_heads=16):
        super(SRNO, self).__init__()
        
        self.d_e = d_e
        self.width = width
        self.n_layers = n_layers # T=2 in paper
        self.n_heads = n_heads   # n_heads=16 in paper
        
        # 1. Encoder (E_psi)
        self.encoder = SimpleEncoder(in_channels=in_channels, d_e=d_e)
        
        # 2. Lifting (L)
        self.lifting = Lifting(d_e=d_e, width=width)
        
        # 3. Kernel Integrals (K)
        self.kernel_layers = nn.ModuleList(
            [KernelIntegral(embed_dim=width, n_heads=n_heads) for _ in range(n_layers)]
        )
        
        # 4. Projection (P)
        # Maps final latent representation back to RGB
        self.projection = nn.Linear(width, 3)

    def forward(self, lr_image, hr_coord_grid, scale):
        """
        lr_image: (B, 3, H_c, W_c) - Low-resolution input image
        hr_coord_grid: (B, H_f, W_f, 2) - Normalized coords [0, 1] for HR grid
        scale: (float, float) - (scale_x, scale_y)
        """
        
        # --- THIS IS THE CORRECTED LINE ---
        B, H_f, W_f, _ = hr_coord_grid.shape
        
        # 1. Encoder
        # (B, 3, H_c, W_c) -> (B, d_e, H_c, W_c)
        lr_feat = self.encoder(lr_image)
        
        # 2. Lifting
        # (B, d_e, H_c, W_c) -> (B, N, width) where N = H_f * W_f
        z = self.lifting(lr_feat, hr_coord_grid, scale)
        
        # 3. Kernel Integrals
        # (B, N, width) -> (B, N, width)
        for layer in self.kernel_layers:
            z = layer(z)
            
        # 4. Projection
        # (B, N, width) -> (B, N, 3)
        out = self.projection(z)
        
        # Reshape to image format
        # (B, N, 3) -> (B, H_f, W_f, 3) -> (B, 3, H_f, W_f)
        out = out.view(B, H_f, W_f, 3).permute(0, 3, 1, 2)
        
        return out

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c

# %%
if __name__ == '__main__':
    # --- Example Usage ---
    
    # Model parameters from paper [cite: 188, 189, 190]
    D_E = 64
    WIDTH = 256
    N_LAYERS = 2
    N_HEADS = 16
    
    model = SRNO(d_e=D_E, width=WIDTH, n_layers=N_LAYERS, n_heads=N_HEADS)
    print(f"SRNO parameter count: {model.count_params():,}")
    
    # --- Dummy Data ---
    BATCH_SIZE = 4
    
    # Low-Res Input
    H_c, W_c = 48, 48
    lr_img = torch.randn(BATCH_SIZE, 3, H_c, W_c)
    
    # High-Res Output Grid & Scale
    SCALE = 8.5
    H_f, W_f = int(H_c * SCALE), int(W_c * SCALE)
    
    # Create normalized [0, 1] coordinate grid for HR output
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, H_f), 
        torch.linspace(0, 1, W_f),
        indexing='ij'
    )
    hr_grid = torch.stack([grid_x, grid_y], dim=-1) # (H_f, W_f, 2)
    hr_grid_batch = hr_grid.unsqueeze(0).repeat(BATCH_SIZE, 1, 1, 1) # (B, H_f, W_f, 2)

    # --- Forward Pass ---
    # Note: 'scale' is passed as a tuple (scale_x, scale_y)
    hr_img_out = model(lr_img, hr_grid_batch, scale=(SCALE, SCALE))
    
    print(f"\nInput LR image shape:  {lr_img.shape}")
    print(f"Input HR grid shape:   {hr_grid_batch.shape}")
    print(f"Output HR image shape: {hr_img_out.shape}")
    
    # Verify output shape
    assert hr_img_out.shape == (BATCH_SIZE, 3, H_f, W_f)
    print("\nForward pass successful!")