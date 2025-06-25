# %% 

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Import the kernel integration and neighborhood search modules
# Simplified versions based on the provided code

class NeighborSearch(nn.Module):
    """Find neighbors within a certain radius"""
    def __init__(self):
        super().__init__()
    
    def forward(self, data, queries, radius):
        """
        Parameters:
        -----------
        data: torch.Tensor of shape [n, d]
            The source points
        queries: torch.Tensor of shape [m, d]
            The target points to find neighbors for
        radius: float
            Search radius
            
        Returns:
        --------
        Dictionary with neighbors_index and neighbors_row_splits
        """
        # Compute pairwise distances
        dists = torch.cdist(queries, data)
        
        # Find points within the radius
        in_nbr = torch.where(dists <= radius, 1., 0.)
        nbr_indices = in_nbr.nonzero()[:,1].reshape(-1)
        
        # Count neighbors per query point
        nbrhd_sizes = torch.sum(in_nbr, dim=1)
        
        # Create row splits (cumulative sum of neighborhood sizes)
        splits = torch.zeros(queries.shape[0] + 1, dtype=torch.long)
        splits[1:] = torch.cumsum(nbrhd_sizes, dim=0)
        
        return {
            'neighbors_index': nbr_indices.long(),
            'neighbors_row_splits': splits.long()
        }

class MLPLinear(nn.Module):
    """Simple MLP"""
    def __init__(self, layers, non_linearity=F.gelu):
        super().__init__()
        self.n_layers = len(layers) - 1
        self.fcs = nn.ModuleList()
        self.non_linearity = non_linearity
        
        for j in range(self.n_layers):
            self.fcs.append(nn.Linear(layers[j], layers[j + 1]))
    
    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
        return x

class IntegralTransform(nn.Module):
    """Kernel integral transform for mapping between grids"""
    def __init__(self, mlp_layers):
        super().__init__()
        self.mlp = MLPLinear(layers=mlp_layers)
    
    def forward(self, y, neighbors, x=None, f_y=None):
        """
        Parameters:
        -----------
        y: torch.Tensor [n, d1]
            Source points
        neighbors: dict
            Neighborhood information
        x: torch.Tensor [m, d2]
            Target points (where function will be defined)
        f_y: torch.Tensor [n, d3]
            Function values at source points
            
        Returns:
        --------
        torch.Tensor [m, d3]
            Function values mapped to target points
        """
        if x is None:
            x = y
            
        # Get features at all neighbor points
        rep_features = y[neighbors["neighbors_index"]]
        
        # Get number of neighbors per target point
        num_reps = (
            neighbors["neighbors_row_splits"][1:] - 
            neighbors["neighbors_row_splits"][:-1]
        )
        
        # Repeat each target point according to its number of neighbors
        self_features = torch.repeat_interleave(x, num_reps, dim=0)
        
        # Concatenate source and target points
        agg_features = torch.cat([rep_features, self_features], dim=-1)
        
        # Apply kernel function
        kernel_values = self.mlp(agg_features)
        
        if f_y is not None:
            # Get function values at source points
            in_features = f_y[neighbors["neighbors_index"]]
            # Multiply by kernel
            kernel_values = kernel_values * in_features
        
        # Aggregate results using mean reduction
        # This is a simplified version of segment_csr
        out_features = torch.zeros((x.shape[0], kernel_values.shape[1]), 
                                  device=kernel_values.device)
        
        # Manually implement the segment reduction
        for i in range(x.shape[0]):
            start_idx = neighbors["neighbors_row_splits"][i]
            end_idx = neighbors["neighbors_row_splits"][i + 1]
            
            if end_idx > start_idx:  # Only if there are neighbors
                # Compute mean over all neighbors
                out_features[i] = kernel_values[start_idx:end_idx].mean(dim=0)
                
        return out_features

class MeshEncoder(nn.Module):
    """
    Encodes data from an unstructured mesh to a structured grid
    
    Parameters:
    -----------
    kernel_mlp_layers: list
        MLP layer sizes for the kernel network
    radius: float
        Neighborhood search radius
    """
    def __init__(self, kernel_mlp_layers, radius):
        super().__init__()
        self.neighbor_search = NeighborSearch()
        self.integral_transform = IntegralTransform(mlp_layers=kernel_mlp_layers)
        self.radius = radius
        
    def forward(self, unstructured_points, unstructured_values, structured_points):
        """
        Parameters:
        -----------
        unstructured_points: torch.Tensor [n, d_pos]
            Coordinates in unstructured mesh
        unstructured_values: torch.Tensor [n, d_val]
            Function values on unstructured mesh
        structured_points: torch.Tensor [m, d_pos]
            Coordinates in structured grid
            
        Returns:
        --------
        torch.Tensor [m, d_val]
            Function values mapped to structured grid
        """
        # Find neighborhoods in the unstructured grid for each structured point
        neighbors = self.neighbor_search(
            unstructured_points, structured_points, self.radius
        )
        
        # Apply integral transform to map values
        structured_values = self.integral_transform(
            unstructured_points, neighbors, structured_points, unstructured_values
        )
        
        return structured_values

# # Now let's create a minimal example

# def generate_unstructured_grid(n_points=200, seed=42):
#     """Generate a random unstructured 2D grid"""
#     np.random.seed(seed)
#     points = np.random.rand(n_points, 2)
#     return torch.tensor(points, dtype=torch.float32)

# def generate_function_values(points, function_type='sine'):
#     """Generate function values on the points"""
#     if function_type == 'sine':
#         # A simple sine wave function
#         values = torch.sin(2 * np.pi * points[:, 0]) * torch.cos(2 * np.pi * points[:, 1])
#         return values.reshape(-1, 1)
#     elif function_type == 'gaussian':
#         # A 2D Gaussian
#         center = torch.tensor([0.5, 0.5])
#         dist = torch.sum((points - center)**2, dim=1)
#         values = torch.exp(-dist / 0.05)
#         return values.reshape(-1, 1)

# def generate_structured_grid(n_points=20):
#     """Generate a regular structured 2D grid"""
#     x = np.linspace(0, 1, n_points)
#     y = np.linspace(0, 1, n_points)
#     xx, yy = np.meshgrid(x, y)
#     points = np.stack([xx.flatten(), yy.flatten()], axis=1)
#     return torch.tensor(points, dtype=torch.float32), (n_points, n_points)

# def main():
#     # Generate unstructured grid
#     unstructured_points = generate_unstructured_grid(n_points=500)
    
#     # Generate function values on unstructured grid (sine wave pattern)
#     unstructured_values = generate_function_values(unstructured_points, function_type='gaussian')
    
#     # Generate structured grid
#     structured_points, grid_shape = generate_structured_grid(n_points=20)
    
#     # Create mesh encoder
#     mesh_encoder = MeshEncoder(
#         kernel_mlp_layers=[4, 16, 16, 1],  # Input dim is 4 (2+2 for concatenated positions)
#         radius=0.1  # Search radius - adjust based on point density
#     )
    
#     # Encode from unstructured to structured grid
#     structured_values = mesh_encoder(
#         unstructured_points, unstructured_values, structured_points
#     )
    
#     # Plot the results
#     plt.figure(figsize=(15, 5))
    
#     # Plot unstructured grid
#     plt.subplot(121)
#     plt.scatter(
#         unstructured_points[:, 0].numpy(), 
#         unstructured_points[:, 1].numpy(), 
#         c=unstructured_values.numpy(), 
#         s=20, 
#         cmap='viridis'
#     )
#     plt.colorbar(label='Function Value')
#     plt.title('Unstructured Grid')
#     plt.xlabel('x')
#     plt.ylabel('y')
    
#     # Plot structured grid
#     plt.subplot(122)
#     structured_values_reshaped = structured_values.detach().numpy().reshape(grid_shape)
#     plt.imshow(
#         structured_values_reshaped, 
#         origin='lower', 
#         extent=[0, 1, 0, 1], 
#         cmap='viridis'
#     )
#     plt.colorbar(label='Function Value')
#     plt.title('Structured Grid (after encoding)')
#     plt.xlabel('x')
#     plt.ylabel('y')
    
#     plt.tight_layout()
#     plt.savefig('mesh_encoding_result.png')
#     plt.close()
    
#     # Create 3D visualization
#     fig = plt.figure(figsize=(15, 10))
    
#     # Unstructured data in 3D
#     ax1 = fig.add_subplot(121, projection='3d')
#     ax1.scatter(
#         unstructured_points[:, 0].numpy(),
#         unstructured_points[:, 1].numpy(),
#         unstructured_values.numpy(),
#         c=unstructured_values.numpy(),
#         cmap='viridis'
#     )
#     ax1.set_title('Unstructured Grid (3D)')
#     ax1.set_xlabel('x')
#     ax1.set_ylabel('y')
#     ax1.set_zlabel('Value')
    
#     # Structured data in 3D
#     ax2 = fig.add_subplot(122, projection='3d')
#     xx, yy = np.meshgrid(
#         np.linspace(0, 1, grid_shape[0]),
#         np.linspace(0, 1, grid_shape[1])
#     )
#     ax2.plot_surface(
#         xx, 
#         yy, 
#         structured_values_reshaped,
#         cmap='viridis',
#         alpha=0.8
#     )
#     ax2.set_title('Structured Grid (3D)')
#     ax2.set_xlabel('x')
#     ax2.set_ylabel('y')
#     ax2.set_zlabel('Value')
    
#     plt.tight_layout()
#     plt.savefig('mesh_encoding_3d.png')

# if __name__ == "__main__":
#     main()
#     print("Processing complete. Check the output images.")
# # %%
