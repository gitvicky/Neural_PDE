import torch
import torch.nn as nn
import math

# We assume the GINO code provided earlier is installed at this path
from neuralop.models.gino import GINO

class GINO2DTimeSolver(nn.Module):
    """
    Wrapper around the GINO architecture to solve a 2D PDE.
    Handles reshaping of 4D tensors and generation of the latent grid.
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 coord_dim=2, 
                 fno_modes=(8,8),
                 fno_hidden_channels=16,
                 latent_resolution=(32, 32), # Grid size for FNO
                 radius=0.2):
        super().__init__()
        
        self.latent_resolution = latent_resolution
        self.coord_dim = coord_dim
        
        # Initialize GINO model
        # We set fno_n_modes to half the resolution (Nyquist)
        fno_modes = (latent_resolution[0]//2, latent_resolution[1]//2)
        
        self.gino = GINO(
            in_channels=in_channels,
            out_channels=out_channels,
            gno_coord_dim=coord_dim,
            in_gno_radius=radius,
            out_gno_radius=radius,
            fno_n_modes=fno_modes,
            fno_hidden_channels=fno_hidden_channels,
            fno_in_channels=in_channels, # Fix: Match GNO output dim to input dim
            gno_use_open3d=False
        )

        # Create latent grid (buffer so it moves with device)
        self.register_buffer('latent_queries', self._create_latent_grid(latent_resolution))

    def _create_latent_grid(self, resolution):
        """Generates a regular grid on [0,1]^2 of shape (1, res_x, res_y, 2)"""
        x = torch.linspace(0, 1, resolution[0])
        y = torch.linspace(0, 1, resolution[1])
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1)
        return grid.unsqueeze(0) # (1, res_x, res_y, 2)

    def forward(self, x_in, x_out, xx):
        """
        x_in: (Batch, N_points, 2) -> Input coordinates
        x_out: (Batch, N_points, 2) -> Output coordinates
        xx: (Batch, num_vars, N_points, 1) -> Input features
        
        Returns:
        out: (Batch, num_vars, N_points, 1)
        """
        
        # 1. Reshape Input Features: [B, C, N, 1] -> [B, N, C]
        if xx.ndim == 4:
            xx = xx.squeeze(-1).permute(0, 2, 1)
        
        # 2. Extract Geometry
        # GINO expects geometry to be (1, N, Dim) and shared across batch
        # We take the first element of the batch.
        if x_in.ndim == 3:
            input_geom = x_in[0].unsqueeze(0)
        else:
            input_geom = x_in.unsqueeze(0)

        # GINO's output GNO expects 2D queries (N, Dim) to use native search correctly
        if x_out.ndim == 3:
            output_queries = x_out[0] # (N, 2)
        else:
            output_queries = x_out

        # 3. Forward Pass through GINO
        out = self.gino(
            input_geom=input_geom,
            latent_queries=self.latent_queries,
            output_queries=output_queries,
            x=xx
        )
        # out shape: [Batch, N, out_channels]

        # 4. Reshape Output: [B, N, C] -> [B, C, N, 1]
        out = out.permute(0, 2, 1).unsqueeze(-1)

        return out

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# # ==========================================
# #  Synthetic Data & Main
# # ==========================================

# def generate_synthetic_data(num_samples=100, num_points=200):
#     """
#     Generates synthetic data with 2 input channels (Variables).
#     Returns inputs/targets in shape [Batch, 2, N, 1]
#     """
#     # 1. Base Mesh: Random 2D coordinates in [0,1]
#     base_coords = torch.rand(num_points, 2)
#     x_c = base_coords[:, 0]
#     y_c = base_coords[:, 1]

#     inputs_list = []
#     targets_list = []
    
#     # Create batch of coords (repeating the same mesh)
#     coords_batch = base_coords.unsqueeze(0).repeat(num_samples, 1, 1)

#     for _ in range(num_samples):
#         phi_x = torch.rand(1).item() * 2 * math.pi
#         phi_y = torch.rand(1).item() * 2 * math.pi

#         # --- Variable 1 (e.g., u velocity) ---
#         u_t = torch.sin(2 * math.pi * x_c + phi_x) * torch.cos(2 * math.pi * y_c + phi_y)
#         u_tp1 = torch.sin(2 * math.pi * (x_c + 0.1) + phi_x) * torch.cos(2 * math.pi * (y_c + 0.1) + phi_y)
        
#         # --- Variable 2 (e.g., v velocity) - different freq/phase ---
#         v_t = torch.cos(3 * math.pi * x_c + phi_x) * torch.sin(3 * math.pi * y_c + phi_y)
#         v_tp1 = torch.cos(3 * math.pi * (x_c + 0.1) + phi_x) * torch.sin(3 * math.pi * (y_c + 0.1) + phi_y)
        
#         # Stack variables [2, N]
#         input_vars = torch.stack([u_t, v_t])
#         target_vars = torch.stack([u_tp1, v_tp1])
        
#         inputs_list.append(input_vars)
#         targets_list.append(target_vars)
    
#     # Stack batch: [Batch, 2, N]
#     inputs = torch.stack(inputs_list)
#     targets = torch.stack(targets_list)
    
#     # Expand last dim for time: [Batch, 2, N, 1]
#     inputs = inputs.unsqueeze(-1)
#     targets = targets.unsqueeze(-1)

#     return coords_batch, inputs, targets

# def main():
#     BATCH_SIZE = 16
#     EPOCHS = 20
#     LR = 1e-3
#     NUM_POINTS = 300 
#     RADIUS = 0.2     
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Running on {device}...")

#     print("Generating synthetic PDE data (2 Variables)...")
#     coords, inputs, targets = generate_synthetic_data(num_samples=50, num_points=NUM_POINTS)
    
#     print(f"Input Shape: {inputs.shape}")   # Should be [50, 2, 300, 1]
#     print(f"Target Shape: {targets.shape}") # Should be [50, 2, 300, 1]

#     coords = coords.to(device)
#     inputs = inputs.to(device)
#     targets = targets.to(device)

#     # Initialize GINO Solver
#     model = GINO2DTimeSolver(
#         in_channels=2,    # 2 Input variables
#         out_channels=2,   # 2 Output variables
#         coord_dim=2,      
#         latent_resolution=(32, 32),
#         radius=RADIUS
#     ).to(device)

#     print(f"Model parameters: {model.count_params()}")

#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#     loss_fn = nn.MSELoss()

#     print("Starting training...")
#     model.train()
    
#     for epoch in range(EPOCHS):
#         epoch_loss = 0
        
#         for i in range(0, len(inputs), BATCH_SIZE):
#             batch_coords = coords[i:i+BATCH_SIZE]
#             batch_in = inputs[i:i+BATCH_SIZE]
#             batch_target = targets[i:i+BATCH_SIZE]

#             optimizer.zero_grad()
            
#             pred = model(x_in=batch_coords, x_out=batch_coords, xx=batch_in)
            
#             loss = loss_fn(pred, batch_target)
#             loss.backward()
#             optimizer.step()
            
#             epoch_loss += loss.item()

#         if (epoch+1) % 5 == 0:
#             print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss/(len(inputs)/BATCH_SIZE):.6f}")

#     print("\nEvaluation on one sample:")
#     model.eval()
#     with torch.no_grad():
#         test_coords = coords[0:1]
#         test_in = inputs[0:1]
#         test_target = targets[0:1]
        
#         prediction = model(test_coords, test_coords, test_in)
#         final_loss = loss_fn(prediction, test_target)
        
#         print(f"Target Shape: {test_target.shape}")
#         print(f"Output Shape: {prediction.shape}")
#         print(f"MSE Error: {final_loss.item():.6f}")

# if __name__ == "__main__":
#     main()