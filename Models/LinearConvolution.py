# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys 

sys.path.append('..')
sys.path.append('../..')

class BoundaryManager:
    """Manages explicit boundary conditions for convolution operations."""
    
    def __init__(self, kernel_size):
        if isinstance(kernel_size, int):
            self.kernel_h = kernel_size
            self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size
            
        self.pad_l = self.kernel_w // 2
        self.pad_r = self.kernel_w // 2
        self.pad_t = self.kernel_h // 2
        self.pad_b = self.kernel_h // 2
        
        self.boundary_types = {
            'left': 'periodic', 'right': 'periodic', 
            'top': 'periodic', 'bottom': 'periodic'
        }
        self.boundary_values = {
            'left': 0.0, 'right': 0.0, 'top': 0.0, 'bottom': 0.0
        }
    
    def set_all_boundaries(self, bc_type, value=0.0):
        for side in ['left', 'right', 'top', 'bottom']:
            self.boundary_types[side] = bc_type
            self.boundary_values[side] = value

    def set_boundary_type(self, side, bc_type, value=0.0):
        self.boundary_types[side] = bc_type
        self.boundary_values[side] = value
    
    def pad_signal(self, x):
        """Applies explicit padding based on configured boundary conditions."""
        
        # Optimize: If all boundaries are the same, use fast torch padding
        if len(set(self.boundary_types.values())) == 1:
            bc_type = self.boundary_types['left']
            pad = (self.pad_l, self.pad_r, self.pad_t, self.pad_b)
            
            if bc_type == 'periodic':
                return F.pad(x, pad, mode='circular')
            elif bc_type == 'symmetric':
                return F.pad(x, pad, mode='reflect')
            elif bc_type in ['neumann', 'replicate']:
                return F.pad(x, pad, mode='replicate')
            elif bc_type in ['dirichlet', 'constant']:
                val = self.boundary_values['left']
                return F.pad(x, pad, mode='constant', value=val)

        # Fallback: Manual padding for mixed boundary conditions
        result = x
        if self.pad_l > 0 or self.pad_r > 0:
            result = self._pad_dim(result, -1, self.pad_l, self.pad_r, 'left', 'right')
        if self.pad_t > 0 or self.pad_b > 0:
            result = self._pad_dim(result, -2, self.pad_t, self.pad_b, 'top', 'bottom')
            
        return result

    def _pad_dim(self, x, dim, pad_pre, pad_post, side_pre, side_post):
        parts = []
        # Pre-padding
        if pad_pre > 0:
            bc = self.boundary_types[side_pre]
            if bc == 'periodic':
                parts.append(x.index_select(dim, torch.arange(x.shape[dim]-pad_pre, x.shape[dim], device=x.device)))
            elif bc == 'symmetric':
                parts.append(x.index_select(dim, torch.arange(pad_pre, 0, -1, device=x.device)))
            elif bc in ['neumann', 'replicate']:
                parts.append(x.index_select(dim, torch.zeros(pad_pre, dtype=torch.long, device=x.device)))
            else: # Dirichlet/Constant
                shape = list(x.shape)
                shape[dim] = pad_pre
                parts.append(torch.full(shape, self.boundary_values[side_pre], device=x.device))
        
        parts.append(x)
        
        # Post-padding
        if pad_post > 0:
            bc = self.boundary_types[side_post]
            if bc == 'periodic':
                parts.append(x.index_select(dim, torch.arange(0, pad_post, device=x.device)))
            elif bc == 'symmetric':
                idx = torch.arange(x.shape[dim]-2, x.shape[dim]-2-pad_post, -1, device=x.device)
                parts.append(x.index_select(dim, idx))
            elif bc in ['neumann', 'replicate']:
                idx = torch.full((pad_post,), x.shape[dim]-1, dtype=torch.long, device=x.device)
                parts.append(x.index_select(dim, idx))
            else: # Dirichlet/Constant
                shape = list(x.shape)
                shape[dim] = pad_post
                parts.append(torch.full(shape, self.boundary_values[side_post], device=x.device))
                
        return torch.cat(parts, dim=dim)


class LinearConvolution2d(nn.Module):
    """
    A single linear convolution layer with explicit boundary handling.
    Includes toggleable bias term.
    """
    def __init__(self, kernel_size=3, in_features=1, out_features=1, 
                 init_type='xavier', boundary_type='periodic', 
                 use_bias=False): # NEW: Added use_bias toggle
        super(LinearConvolution2d, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        
        # 1. Learnable Kernel
        self.kernel = nn.Parameter(torch.empty(out_features, in_features, 
                                              kernel_size, kernel_size))
        self._initialize_kernel(init_type)

        # 2. Learnable Bias (Optional)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            # Registering as None ensures F.conv2d ignores it cleanly
            self.register_parameter('bias', None)
        
        # 3. Boundary Manager
        self.boundary_manager = BoundaryManager(kernel_size)
        self.boundary_manager.set_all_boundaries(boundary_type)
        
    def _initialize_kernel(self, init_type):
        with torch.no_grad():
            if init_type == 'xavier' or init_type == 'random':
                nn.init.xavier_uniform_(self.kernel)
            elif init_type == 'identity':
                nn.init.zeros_(self.kernel)
                center = self.kernel_size // 2
                min_feat = min(self.in_features, self.out_features)
                for i in range(min_feat):
                    self.kernel[i, i, center, center] = 1.0
            elif init_type == 'zeros':
                nn.init.zeros_(self.kernel)

    def forward(self, x):
        # Normalize Input Shape
        x_in, original_shape = self._check_shape(x)
        
        # Adjust Channels
        if x_in.shape[1] != self.in_features and x_in.shape[1] == 1:
            x_in = x_in.expand(-1, self.in_features, -1, -1)
            
        # Apply Explicit Boundary Padding
        x_padded = self.boundary_manager.pad_signal(x_in)
        
        # Linear Convolution (With or Without Bias)
        # F.conv2d handles self.bias being None automatically
        out = F.conv2d(x_padded, self.kernel, bias=self.bias)
        
        # Restore Output Shape
        return self._restore_shape(out, original_shape)

    def _check_shape(self, x):
        shape = x.shape
        if len(shape) == 5 and shape[-1] == 1: 
            return x.squeeze(-1), shape
        elif len(shape) == 3: 
            return x.unsqueeze(1), shape
        elif len(shape) == 2: 
            return x.unsqueeze(0).unsqueeze(0), shape
        return x, shape

    def _restore_shape(self, x, original_shape):
        if len(original_shape) == 5:
            return x.unsqueeze(-1)
        elif len(original_shape) == 3:
            return x.squeeze(1)
        elif len(original_shape) == 2:
            return x.squeeze(0).squeeze(0)
        return x


class LinearConvolutionalModel(nn.Module):
    """
    A deep LINEAR convolutional network.
    Structure: [Pad -> Conv (+Bias)] -> [Pad -> Conv (+Bias)] ...
    """
    def __init__(self, in_features=1, hidden_features=16, out_features=1,
                 num_layers=3, kernel_size=3, boundary_type='periodic',
                 init_type='xavier', use_bias=False): # NEW: Added use_bias toggle
        super(LinearConvolutionalModel, self).__init__()
        
        self.layers = nn.ModuleList()
        
        if isinstance(hidden_features, int):
            hidden_feats = [hidden_features] * (num_layers - 1)
        else:
            hidden_feats = hidden_features
            
        dims = [in_features] + hidden_feats + [out_features]
        
        for i in range(num_layers):
            self.layers.append(
                LinearConvolution2d(
                    kernel_size=kernel_size,
                    in_features=dims[i],
                    out_features=dims[i+1],
                    boundary_type=boundary_type,
                    init_type=init_type,
                    use_bias=use_bias # Pass the toggle down
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# # =============================================================================
# # EXAMPLE USAGE
# # =============================================================================
# if __name__ == "__main__":
#     # Create a completely linear model with Periodic boundaries
#     linear_model = LinearConvolutionalModel(
#         in_features=2,
#         hidden_features=8,
#         out_features=2,
#         num_layers=3,
#         kernel_size=3,
#         boundary_type='periodic',
#         init_type='xavier',
#         use_bias=False
#     )
    
#     print(f"Linear Model Parameters: {linear_model.count_params()}")
    
#     # Input: (Batch, Channels, Height, Width)
#     x = torch.randn(4, 2, 32, 32)
    
#     with torch.no_grad():
#         output = linear_model(x)
        
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {output.shape}")
#     print("Linear convolution completed successfully.")


# %%
