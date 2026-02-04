# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# We import GNOBlock. 
# Note: The code you provided IS the source code for neuralop.layers.GNOBlock.
# Assuming you have the library installed, we import from there. 
# If you are using your local file, replace this with: from your_file import GNOBlock
from neuralop.layers.gno_block import GNOBlock

class GNO2DTimeSolver(nn.Module):
    """
    A Neural Operator architecture using GNO Blocks to solve a 2D PDE in time.
    
    Architecture:
    1. Lift: Projects physical input (u, v, etc) to latent channels.
    2. Process: Layers of GNOBlocks to integrate information over the mesh.
    3. Project: Projects latent channels back to physical output.
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 coord_dim=2, 
                 latent_channels=32, 
                 num_layers=3, 
                 radius=0.2):
        super().__init__()

        # 1. Lifting Layer
        # Maps input function values (e.g., velocity at t) to latent space
        self.lifting = nn.Linear(in_channels, latent_channels)

        # 2. Processing Layers (The GNO Blocks)
        self.layers = nn.ModuleList()
        
        for _ in range(num_layers):
            # We use the GNOBlock as defined in your prompt/neuralop library
            gno_layer = GNOBlock(
                in_channels=latent_channels,
                out_channels=latent_channels,
                coord_dim=coord_dim,
                radius=radius,
                transform_type='linear', # (b) Integral transform
                use_open3d_neighbor_search=False # Set False for simple 2D PyTorch fallback
            )
            self.layers.append(gno_layer)

        # 3. Projection Layer
        # Maps latent space back to physical values (e.g., velocity at t+1)
        self.projection = nn.Sequential(
            nn.Linear(latent_channels, latent_channels * 2),
            nn.GELU(),
            nn.Linear(latent_channels * 2, out_channels)
        )

    def forward(self, y, x, f_y):
        """
        y: (Batch, N_points, coord_dim) -> Input coordinates
        x: (Batch, N_points, coord_dim) -> Output coordinates (usually same as y for autoregressive)
        f_y: (Batch, N_points, in_channels) -> Input features (physics state at time t)
        """
        
        # Lift input features to latent space
        # f_y shape: [B, N, in_channels] -> [B, N, latent_channels]
        h = self.lifting(f_y)
        print(h.shape)

        # Apply GNO Layers
        # GNOBlock expects specific shapes. If the code provided is exact, 
        # it expects y and x to be [N, coord_dim] (not batched) if the geometry is constant.
        # If geometry varies per batch, we usually iterate or use batch-supported search.
        # Here we assume constant geometry for simplicity (Standard for PDE solvers).
        
        # Extract single geometry sample (assuming all items in batch share the mesh)
        mesh_y = y[0] 
        mesh_x = x[0] 

        for layer in self.layers:
            # GNOBlock Forward: (y, x, f_y)
            # Note: The output of GNO is often the integral result. 
            # Standard ResNet logic: h_new = Activation(GNO(h)) + h
            
            h_out = layer(y=mesh_y, x=mesh_x, f_y=h)
            h = F.gelu(h_out) + h # Skip connection
            print(h.shape)

        # Project back to output
        out = self.projection(h)
        print(out.shape)
        return out

# # ==========================================
# #  Synthetic Data & Training Loop
# # ==========================================

# def generate_synthetic_data(num_samples=100, num_points=200):
#     """
#     Generates synthetic wave data on a 2D random mesh.
#     Input: State at t
#     Output: State at t+1 (simply shifted)
#     """
#     # Random 2D coordinates in [0,1]
#     coords = torch.rand(num_samples, num_points, 2)
    
#     # Generate a wave function: sin(2pi * x) * cos(2pi * y)
#     # We will shift the phase to simulate time evolution
#     x_c = coords[..., 0]
#     y_c = coords[..., 1]
    
#     # Input: Time t
#     u_t = torch.sin(2 * math.pi * x_c) * torch.cos(2 * math.pi * y_c)
    
#     # Target: Time t+dt (Phase shift)
#     u_t_plus_1 = torch.sin(2 * math.pi * (x_c + 0.1)) * torch.cos(2 * math.pi * (y_c + 0.1))
    
#     # Add channel dimension: (Batch, Points, 1)
#     return coords, u_t.unsqueeze(-1), u_t_plus_1.unsqueeze(-1)

# # %% 
# # Hyperparameters
# BATCH_SIZE = 5
# EPOCHS = 1
# LR = 1e-3
# NUM_POINTS = 300 # Size of the mesh
# RADIUS = 0.1     # Neighborhood search radius

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Running on {device}...")

# # 1. Prepare Data
# print("Generating synthetic PDE data...")
# coords, inputs, targets = generate_synthetic_data(num_samples=100, num_points=NUM_POINTS)

# # Move to device
# coords = coords.to(device)
# inputs = inputs.to(device)
# targets = targets.to(device)

# # 2. Initialize Model
# model = GNO2DTimeSolver(
#     in_channels=1,    # Scalar field (e.g. Pressure)
#     out_channels=1,   # Scalar field
#     coord_dim=2,      # 2D Mesh
#     radius=RADIUS
# ).to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# loss_fn = nn.MSELoss()

# # 3. Training Loop
# print("Starting training...")
# model.train()

# for epoch in range(EPOCHS):
#     epoch_loss = 0
    
#     # Simple batching loop
#     for i in range(0, len(coords), BATCH_SIZE):
#         batch_coords = coords[i:i+BATCH_SIZE]
#         batch_in = inputs[i:i+BATCH_SIZE]
#         batch_target = targets[i:i+BATCH_SIZE]

#         optimizer.zero_grad()
        
#         # Forward pass: Predict t+1 based on t and coordinates
#         pred = model(y=batch_coords, x=batch_coords, f_y=batch_in)
        
#         loss = loss_fn(pred, batch_target)
#         loss.backward()
#         optimizer.step()
        
#         epoch_loss += loss.item()

#     if (epoch+1) % 10 == 0:
#         print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss/len(coords):.6f}")

# # 4. Evaluation
# print("\nEvaluation on one sample:")
# model.eval()
# with torch.no_grad():
#     test_coords = coords[0:1]
#     test_in = inputs[0:1]
#     test_target = targets[0:1]
    
#     prediction = model(test_coords, test_coords, test_in)
#     final_loss = loss_fn(prediction, test_target)
    
#     print(f"Target Value (Sample): {test_target[0, 0, 0].item():.4f}")
#     print(f"Predicted Value (Sample): {prediction[0, 0, 0].item():.4f}")
#     print(f"MSE Error: {final_loss.item():.6f}")

# # %% 