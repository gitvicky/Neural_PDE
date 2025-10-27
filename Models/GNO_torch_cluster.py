# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
# Removed unused torch_geometric.utils
# Added torch_cluster imports
from torch_cluster import radius_graph, radius
import numpy as np


class GNOLayer(MessagePassing):
    """
    Single GNO (Graph Neural Operator) Layer
    
    MODIFICATIONS:
    - message() uses torch.einsum
    - forward() accepts pre-computed edge_weights
    """
    def __init__(self, in_channels, out_channels, edge_dim, aggr='mean'):
        super().__init__(aggr=aggr)
        
        # Edge network (MLP that processes edge features to create edge weights)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_dim * 2),
            nn.GELU(),
            nn.Linear(edge_dim * 2, in_channels * out_channels)
        )
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def forward(self, x, edge_index, edge_weights):
        """
        x: Node features [N, in_channels]
        edge_index: Edge indices [2, E]
        edge_weights: Precomputed edge weights [E, in_channels, out_channels]
        """
        # Propagate messages
        # We now expect edge_weights to be passed in
        return self.propagate(edge_index, x=x, edge_weights=edge_weights)
    
    def message(self, x_j, edge_weights):
        """
        x_j: Features of source nodes [E, in_channels]
        edge_weights: Learned weights for each edge [E, in_channels, out_channels]
        
        Optimization: Using torch.einsum for the batched matrix-vector product.
        'bi,bio->bo' means:
        - b: batch dimension (number of edges, E)
        - i: input channels (C_in)
        - o: output channels (C_out)
        Multiply [E, C_in] with [E, C_in, C_out] and sum over C_in -> [E, C_out]
        """
        return torch.einsum('bi,bio->bo', x_j, edge_weights)


class GNOBlock(nn.Module):
    """
    Complete GNO block with multiple layers
    
    MODIFICATIONS:
    - forward() pre-computes edge_weights once and passes them to each layer.
    """
    def __init__(self, in_channels, out_channels, hidden_channels, edge_dim, n_layers=4):
        super().__init__()
        
        self.lifting = nn.Linear(in_channels, hidden_channels)
        
        self.gno_layers = nn.ModuleList([
            GNOLayer(hidden_channels, hidden_channels, edge_dim)
            for _ in range(n_layers)
        ])
        
        self.projection = nn.Linear(hidden_channels, out_channels)
        self.activation = nn.GELU()
        
    def forward(self, x, edge_index, edge_attr):
        """
        x: Node features [B*N, in_channels]
        edge_index: Edge indices [2, B*E]
        edge_attr: Edge features [B*E, edge_dim]
        """
        # Lift to hidden dimension
        x = self.lifting(x)
        
        # Optimization: Pre-compute all edge weights ONCE
        # We pass the same edge_attr to each layer's MLP
        edge_weights_list = []
        for layer in self.gno_layers:
            weights = layer.edge_mlp(edge_attr)
            weights = weights.view(-1, layer.in_channels, layer.out_channels)
            edge_weights_list.append(weights)
            
        # Apply GNO layers
        for i, layer in enumerate(self.gno_layers):
            # Pass pre-computed weights
            x = layer(x, edge_index, edge_weights=edge_weights_list[i])
            x = self.activation(x)
        
        # Project to output dimension
        x = self.projection(x)
        
        return x


class GNO(nn.Module):
    """
    Full GNO model
    
    MODIFICATIONS:
    - forward() implements true batching for the `same_grid` case.
    - _build_graph() uses torch_cluster for memory-efficient graph creation.
    
    NOTE: The original _build_graph() method is UNCHANGED.
    """
    def __init__(self, in_channels=1, out_channels=1, hidden_channels=32, n_layers=4, r=0.1,
                 x_in=None, y_in=None, x_out=None, y_out=None):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.r = r
        
        # Set up grids
        if x_in is None:
            x_in = torch.linspace(0, 1, 32)
        if y_in is None:
            y_in = torch.linspace(0, 1, 32)
        if x_out is None:
            x_out = x_in
        if y_out is None:
            y_out = y_in
        
        self.register_buffer('x_in', x_in if isinstance(x_in, torch.Tensor) else torch.tensor(x_in, dtype=torch.float32))
        self.register_buffer('y_in', y_in if isinstance(y_in, torch.Tensor) else torch.tensor(y_in, dtype=torch.float32))
        self.register_buffer('x_out', x_out if isinstance(x_out, torch.Tensor) else torch.tensor(x_out, dtype=torch.float32))
        self.register_buffer('y_out', y_out if isinstance(y_out, torch.Tensor) else torch.tensor(y_out, dtype=torch.float32))
        
        # Create coordinates
        xx_in, yy_in = torch.meshgrid(self.x_in, self.y_in, indexing='ij')
        coords_in = torch.stack([xx_in.flatten(), yy_in.flatten()], dim=-1)
        
        xx_out, yy_out = torch.meshgrid(self.x_out, self.y_out, indexing='ij')
        coords_out = torch.stack([xx_out.flatten(), yy_out.flatten()], dim=-1)
        
        self.register_buffer('coords_in', coords_in)
        self.register_buffer('coords_out', coords_out)
        
        self.N_in = coords_in.shape[0]
        self.N_out = coords_out.shape[0]
        self.same_grid = torch.allclose(coords_in, coords_out) if self.N_in == self.N_out else False
        
        self.input_spatial_shape = (len(self.x_in), len(self.y_in))
        self.output_spatial_shape = (len(self.x_out), len(self.y_out))
        
        # Build graph
        edge_index, edge_attr = self._build_graph()
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_attr', edge_attr)
        
        # GNO block - this is the key component from neural operator perspective
        self.gno_block = GNOBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            edge_dim=4,  # 2D coordinates concatenated
            n_layers=n_layers
        )
        
        # Coordinate encoder for output nodes (if different grid)
        if not self.same_grid:
            self.coord_encoder = nn.Linear(2, hidden_channels)
    
    def _build_graph(self):
        """
        MODIFIED: Uses torch_cluster.radius_graph and torch_cluster.radius
        for efficient, sparse graph construction without creating a
        dense N x N distance matrix.
        """
        if self.same_grid:
            # Use radius_graph for a simple graph
            # loop=True to include self-loops, matching cdist(x,x) <= r
            edge_index = radius_graph(self.coords_in, r=self.r, loop=True)
            edge_attr = torch.cat([self.coords_in[edge_index[0]], self.coords_in[edge_index[1]]], dim=-1)
        else:
            # Use radius for a bipartite graph (in -> out)
            # radius(x, y, r) returns [y_idx, x_idx] (target, source)
            # We want [source, target]
            edge_index = radius(x=self.coords_in, y=self.coords_out, r=self.r)
            
            # Flip to [source_idx (from x), target_idx (from y)]
            edge_index = edge_index.flip(0)
            
            # Create edge attributes
            edge_attr = torch.cat([self.coords_in[edge_index[0]], self.coords_out[edge_index[1]]], dim=-1)
            
            # Offset the target node indices to their position in the combined graph
            edge_index[1, :] = edge_index[1, :] + self.N_in
        
        return edge_index, edge_attr
    
    def forward(self, u):
        """
        u: (batch_size, in_channels, N_x, N_y, 1) or (batch_size, in_channels, N_x, N_y)
        
        MODIFICATION: Implemented batched forward pass for `same_grid` case.
        """
        if u.dim() == 5:
            u = u[..., 0]
        
        batch_size, channels, N_x, N_y = u.shape
        
        # Reshape to node features [B, N, C]
        u_in = u.permute(0, 2, 3, 1).reshape(batch_size, -1, self.in_channels)
        
        if not self.same_grid:
            # Create output node features (just coordinates)
            coord_features = self.coord_encoder(self.coords_out.unsqueeze(0).expand(batch_size, -1, -1))
            
            # Process each batch (Original slow loop)
            outputs = []
            for b in range(batch_size):
                # Combine input and output nodes
                x_combined = torch.cat([u_in[b], coord_features[b]], dim=0)
                
                # Apply GNO block
                x_out = self.gno_block(x_combined, self.edge_index, self.edge_attr)
                
                # Extract output nodes
                outputs.append(x_out[self.N_in:])
            
            u_final = torch.stack(outputs, dim=0)
        else:
            # --- OPTIMIZED BATCHED FORWARD PASS ---
            
            # 1. Reshape x to [B*N, C]
            x = u_in.reshape(-1, self.in_channels) # [B*N_in, C_in]
            
            # 2. Create batched edge_index
            # Create B offsets: [0, N, 2*N, 3*N, ...]
            offsets = torch.arange(batch_size, device=x.device) * self.N_in
            # Add offsets to edge_index
            # [2, E] + [B, 1, 1] -> broadcast to [B, 2, E] -> reshape to [2, B*E]
            batched_edge_index = self.edge_index.unsqueeze(0) + offsets.view(-1, 1, 1)
            batched_edge_index = batched_edge_index.reshape(2, -1)
            
            # 3. Create batched edge_attr
            # Repeat edge_attr B times
            batched_edge_attr = self.edge_attr.repeat(batch_size, 1)
            
            # 4. Call GNO block ONCE
            x_out = self.gno_block(x, batched_edge_index, batched_edge_attr)
            
            # 5. Reshape output
            # x_out is [B*N_out, C_out]
            u_final = x_out.view(batch_size, self.N_out, self.out_channels)
        
        # Reshape to output format
        output_shape = (batch_size, *self.output_spatial_shape, self.out_channels)
        u_out = u_final.view(*output_shape)
        
        # Permute back to (B, C, H, W) and add extra dim
        return u_out.permute(0, 3, 1, 2).unsqueeze(-1)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())

# # %% 
# # Example usage
# if __name__ == "__main__":
#     disc = 100
#     x = torch.linspace(0, 1, disc)
#     y = torch.linspace(0, 1, disc)
    
#     # Create GNO model using PyG's MessagePassing as the core
#     model = GNO(
#         in_channels=2,
#         out_channels=2,
#         hidden_channels=16,
#         n_layers=2,
#         r=0.1, # <-- Kept your original radius
#         x_in=x,
#         y_in=y
#     )
    
#     print(f"Model input grid: {model.input_spatial_shape}")
#     print(f"Model output grid: {model.output_spatial_shape}")
#     print(f"Parameters: {model.count_params():,}")
#     print(f"Nodes (N): {model.N_in:,}")
#     print(f"Edges (E) per sample: {model.edge_index.shape[1]:,}")
    
#     if model.edge_index.shape[1] > 1_000_000:
#         print("\n" + "*"*50)
#         print("WARNING: You have > 1 million edges per sample.")
#         print(f"Total batched edges = {model.edge_index.shape[1] * 4:,}")
#         print("This will consume a very large amount of memory.")
#         print("Test with a smaller `disc` (e.g., 32) if you get an OOM error.")
#         print("*"*50 + "\n")
    
#     # Test
#     batch_size = 4
#     xx, yy = torch.meshgrid(x, y, indexing='ij')
#     u = torch.zeros(batch_size, 2, disc, disc)
#     u[:, 0] = torch.sin(2 * np.pi * xx).unsqueeze(0)
#     u[:, 1] = torch.cos(2 * np.pi * yy).unsqueeze(0)
#     u = u.unsqueeze(-1)
    
#     # Move to GPU if available to see real speedup
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Moving model and data to {device}")
#     model.to(device)
#     u = u.to(device)
    
#     print(f"Input shape: {u.shape}")
    
#     try:
#         # Time the forward pass
#         import time
#         # Warm-up (if on GPU)
#         if device == 'cuda':
#             _ = model(u)
#             torch.cuda.synchronize() # Wait for GPU to finish
        
#         start = time.time()
        
#         out = model(u)
        
#         if device == 'cuda':
#             torch.cuda.synchronize() # Wait for GPU to finish
#         end = time.time()
        
#         print(f"Output shape: {out.shape}")
#         print(f"Batched forward pass time: {(end - start) * 1000:.2f} ms")
    
#     except RuntimeError as e:
#         if "out of memory" in str(e):
#             print("\n" + "="*50)
#             print("ERROR: Ran out of memory (OOM) as predicted.")
#             print("This is due to the dense radius graph (E) and batching (B).")
#             print("The batched edge weight tensor is too large.")
#             print("Reduce `disc` or, better yet, switch to k-NN.")
#             print("="*50)
#         else:
#             raise e