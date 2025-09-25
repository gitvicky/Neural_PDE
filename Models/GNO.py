import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import numpy as np


class GNOLayer(MessagePassing):
    """
    Single GNO (Graph Neural Operator) Layer using PyTorch Geometric's MessagePassing
    This is the core building block similar to what you'd find in neural operator libraries
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
        
    def forward(self, x, edge_index, edge_attr):
        """
        x: Node features [N, in_channels]
        edge_index: Edge indices [2, E]
        edge_attr: Edge features [E, edge_dim]
        """
        # Generate edge weights from edge features
        edge_weights = self.edge_mlp(edge_attr)  # [E, in_channels * out_channels]
        edge_weights = edge_weights.view(-1, self.in_channels, self.out_channels)  # [E, in_channels, out_channels]
        
        # Propagate messages
        return self.propagate(edge_index, x=x, edge_weights=edge_weights)
    
    def message(self, x_j, edge_weights):
        """
        x_j: Features of source nodes [E, in_channels]
        edge_weights: Learned weights for each edge [E, in_channels, out_channels]
        """
        # Apply edge-specific transformation: [E, in_channels] @ [E, in_channels, out_channels] -> [E, out_channels]
        out = torch.bmm(x_j.unsqueeze(1), edge_weights).squeeze(1)
        return out


class GNOBlock(nn.Module):
    """
    Complete GNO block with multiple layers
    This is what you'd use as a drop-in replacement in neural operator architectures
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
        x: Node features [N, in_channels] or [B*N, in_channels]
        edge_index: Edge indices [2, E]
        edge_attr: Edge features [E, edge_dim]
        """
        # Lift to hidden dimension
        x = self.lifting(x)
        
        # Apply GNO layers
        for layer in self.gno_layers:
            x = layer(x, edge_index, edge_attr)
            x = self.activation(x)
        
        # Project to output dimension
        x = self.projection(x)
        
        return x


class GNO(nn.Module):
    """
    Full GNO model matching your original interface but using PyG's MessagePassing
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
        if self.same_grid:
            pwd = torch.cdist(self.coords_in, self.coords_in)
            edge_index = torch.stack(torch.where(pwd <= self.r))
            edge_attr = torch.cat([self.coords_in[edge_index[0]], self.coords_in[edge_index[1]]], dim=-1)
        else:
            pwd = torch.cdist(self.coords_in, self.coords_out)
            edge_index = torch.stack(torch.where(pwd <= self.r))
            edge_attr = torch.cat([self.coords_in[edge_index[0]], self.coords_out[edge_index[1]]], dim=-1)
            edge_index[1, :] = edge_index[1, :] + self.N_in
        
        return edge_index, edge_attr
    
    def forward(self, u):
        """
        u: (batch_size, in_channels, N_x, N_y, 1) or (batch_size, in_channels, N_x, N_y)
        """
        if u.dim() == 5:
            u = u[..., 0]
        
        batch_size, channels, N_x, N_y = u.shape
        
        # Reshape to node features
        u_in = u.permute(0, 2, 3, 1).reshape(batch_size, -1, self.in_channels)
        
        if not self.same_grid:
            # Create output node features (just coordinates)
            coord_features = self.coord_encoder(self.coords_out.unsqueeze(0).expand(batch_size, -1, -1))
            
            # Process each batch
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
            # Process each batch
            outputs = []
            for b in range(batch_size):
                x_out = self.gno_block(u_in[b], self.edge_index, self.edge_attr)
                outputs.append(x_out)
            
            u_final = torch.stack(outputs, dim=0)
        
        # Reshape to output format
        output_shape = (batch_size, self.out_channels, *self.output_spatial_shape)
        u_out = u_final.reshape(*output_shape)
        
        return u_out.unsqueeze(-1)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


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
#         r=0.1,
#         x_in=x,
#         y_in=y
#     )
    
#     print(f"Model input grid: {model.input_spatial_shape}")
#     print(f"Model output grid: {model.output_spatial_shape}")
#     print(f"Parameters: {model.count_params():,}")
    
#     # Test
#     batch_size = 4
#     xx, yy = torch.meshgrid(x, y, indexing='ij')
#     u = torch.zeros(batch_size, 2, disc, disc)
#     u[:, 0] = torch.sin(2 * np.pi * xx).unsqueeze(0)
#     u[:, 1] = torch.cos(2 * np.pi * yy).unsqueeze(0)
#     u = u.unsqueeze(-1)
    
#     print(f"\nInput shape: {u.shape}")
#     out = model(u)
#     print(f"Output shape: {out.shape}")
    
#     print("\nGNOBlock can be imported and used as a standalone module:")
#     print("from this_module import GNOBlock")
#     print("gno_layer = GNOBlock(in_channels=32, out_channels=32, hidden_channels=64, edge_dim=4)")