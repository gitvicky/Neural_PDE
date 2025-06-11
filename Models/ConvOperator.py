# %% 
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)

from Neural_PDE.Models.boundary_conditions import OptimizedBoundaryManager

class Convolution2d(nn.Module):
    def __init__(self, kernel_size=3, in_features=1, out_features=1, init_type='random', boundary_type='periodic'):
        """
        Create a learnable convolution with configurable kernel and boundary conditions.
        
        Args:
            kernel_size (int): The size of the convolution kernel
            in_features (int): Number of input feature channels
            out_features (int): Number of output feature channels
            init_type (str): Initialization strategy - 'random', 'identity', 'zeros', or 'ones'
            boundary_type (str): Type of boundary condition - 'dirichlet', 'neumann', 'periodic', 'symmetric'
        """
        super(Convolution2d, self).__init__()
        
        # Create a parameter tensor for convolutional kernel with new dimensions
        # Conv2d expects (out_channels, in_channels, kernel_h, kernel_w)
        self.kernel = nn.Parameter(torch.empty(out_features, in_features, kernel_size, kernel_size))
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        
        # Initialize the kernel based on the specified strategy
        if init_type == 'random':
            # Xavier/Glorot initialization
            nn.init.xavier_uniform_(self.kernel)
        elif init_type == 'identity':
            # Initialize to approximate identity operation
            # This only makes sense when in_features == out_features
            nn.init.zeros_(self.kernel)
            if in_features == out_features:
                # Set center pixel to 1 for corresponding input-output channels
                for i in range(in_features):
                    self.kernel[i, i, kernel_size//2, kernel_size//2] = 1.0
            else:
                # For different input/output features, use identity where possible, zeros elsewhere
                min_features = min(in_features, out_features)
                for i in range(min_features):
                    self.kernel[i, i, kernel_size//2, kernel_size//2] = 1.0
        elif init_type == 'zeros':
            # Initialize with zeros
            nn.init.zeros_(self.kernel)
        elif init_type == 'ones':
            # Initialize with ones
            nn.init.ones_(self.kernel)
        else:
            raise ValueError(f"Unknown initialization type: {init_type}")
        
        # Set up boundary manager using the optimized version
        self.boundary_manager = OptimizedBoundaryManager(kernel_size)
        self.boundary_manager.set_all_boundaries(boundary_type)
    
    def apply_boundary_padding(self, x):
        """
        Apply padding to the input tensor based on boundary conditions.
        Optimized version that processes batches more efficiently.
        """
        # Fast path: if all boundaries are the same type, we can process the entire batch at once
        if len(set(self.boundary_manager.boundary_types.values())) == 1:
            bc_type = next(iter(self.boundary_manager.boundary_types.values()))
            
            # For symmetric, constant, or replicate, we can use a single F.pad call on the entire batch
            if bc_type in ['symmetric', 'dirichlet', 'neumann', 'outflow']:
                # Get padding values for all sides
                pad = (
                    self.boundary_manager.pad_left, 
                    self.boundary_manager.pad_right, 
                    self.boundary_manager.pad_top, 
                    self.boundary_manager.pad_bottom
                )
                
                if bc_type == 'symmetric':
                    return F.pad(x, pad, mode='reflect')
                elif bc_type == 'dirichlet':
                    bc_value = next(iter(self.boundary_manager.boundary_values.values()))
                    return F.pad(x, pad, mode='constant', value=bc_value)
                elif bc_type in ['neumann', 'outflow']:
                    return F.pad(x, pad, mode='replicate')
            
            # For periodic, we need to handle it specially but can still be optimized
            elif bc_type == 'periodic':
                pad_left = self.boundary_manager.pad_left
                pad_right = self.boundary_manager.pad_right
                pad_top = self.boundary_manager.pad_top
                pad_bottom = self.boundary_manager.pad_bottom
                
                # First pad horizontally for all batches and channels at once
                if pad_left > 0 or pad_right > 0:
                    left_pad = x[..., -pad_left:] if pad_left > 0 else torch.tensor([], device=x.device)
                    right_pad = x[..., :pad_right] if pad_right > 0 else torch.tensor([], device=x.device)
                    
                    if left_pad.numel() > 0 and right_pad.numel() > 0:
                        x = torch.cat([left_pad, x, right_pad], dim=-1)
                    elif left_pad.numel() > 0:
                        x = torch.cat([left_pad, x], dim=-1)
                    elif right_pad.numel() > 0:
                        x = torch.cat([x, right_pad], dim=-1)
                
                # Then pad vertically for all batches and channels at once
                if pad_top > 0 or pad_bottom > 0:
                    top_pad = x[..., -pad_top:, :] if pad_top > 0 else torch.tensor([], device=x.device)
                    bottom_pad = x[..., :pad_bottom, :] if pad_bottom > 0 else torch.tensor([], device=x.device)
                    
                    if top_pad.numel() > 0 and bottom_pad.numel() > 0:
                        x = torch.cat([top_pad, x, bottom_pad], dim=-2)
                    elif top_pad.numel() > 0:
                        x = torch.cat([top_pad, x], dim=-2)
                    elif bottom_pad.numel() > 0:
                        x = torch.cat([x, bottom_pad], dim=-2)
                
                return x
        
        # Fall back to slower but more flexible approach for mixed boundary types
        batch_size = x.shape[0]
        channels = x.shape[1]
        
        # Process each batch and channel separately for proper boundary handling
        result = []
        for b in range(batch_size):
            channels_result = []
            for c in range(channels):
                # Extract 2D slice and apply boundary padding
                x_slice = x[b, c]  # (height, width)
                padded_slice = self.boundary_manager.pad_signal(x_slice)
                channels_result.append(padded_slice.unsqueeze(0))
            
            # Stack channels back together
            batch_result = torch.cat(channels_result, dim=0).unsqueeze(0)
            result.append(batch_result)
        
        # Stack batches back together
        return torch.cat(result, dim=0)
    
    def get_kernel(self):
        """Get the current value of the convolution kernel."""
        return self.kernel.detach()
    
    def count_params(self):
        """Count the number of parameters in the kernel."""
        return self.kernel.numel()
    
    def set_boundary_type(self, boundary_type, value=0.0):
        """Set the boundary condition type for all sides."""
        self.boundary_manager.set_all_boundaries(boundary_type, value)
    
    def set_specific_boundary(self, side, bc_type, value=0.0):
        """Set boundary condition for a specific side."""
        self.boundary_manager.set_boundary_type(side, bc_type, value)

    def forward(self, x=None):
        """
        Forward pass to apply convolution with proper boundary conditions.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, features, Nx, Ny, 1)
                              If None, return the current kernel
        
        Returns:
            torch.Tensor: Result of convolution with shape adjusted for output features
        """
        if x is None:
            return self.kernel
        
        # Input shape validation and save original shape
        original_shape = x.shape
        
        # Handle 5D input [batch_size, features, Nx, Ny, 1]
        if len(original_shape) == 5:
            # Remove the last dimension if it's 1
            if original_shape[4] == 1:
                x = x.squeeze(-1)  # -> [batch_size, features, Nx, Ny]
            else:
                raise ValueError(f"Expected last dimension to be 1, got {original_shape[4]}")
        elif len(original_shape) == 4:
            # Already in the right format [batch_size, features, Nx, Ny]
            pass
        elif len(original_shape) == 3:  # [batch, Nx, Ny]
            x = x.unsqueeze(1)  # -> [batch, 1, Nx, Ny]
        elif len(original_shape) == 2:  # [Nx, Ny]
            x = x.unsqueeze(0).unsqueeze(0)  # -> [1, 1, Nx, Ny]
        else:
            raise ValueError(f"Unexpected input shape: {original_shape}")
        
        batch_size = x.shape[0]
        x_channels = x.shape[1]
        
        # Handle the case where input channels don't match kernel in_features
        if x_channels != self.in_features:
            # Optimize channel copying for the common case of expanding a single channel
            if x_channels == 1 and self.in_features > 1:
                # Broadcasting is much faster than a loop for this case
                x = x.expand(-1, self.in_features, -1, -1)
            else:
                # Create a new tensor with the correct number of channels
                new_x = torch.zeros(batch_size, self.in_features, x.shape[2], x.shape[3], device=x.device)
                # Copy data from the input tensor, handling both cases (more or fewer channels)
                for i in range(min(x_channels, self.in_features)):
                    new_x[:, i] = x[:, i % x_channels]
                x = new_x
        
        # Apply padding based on boundary conditions
        padded_x = self.apply_boundary_padding(x)
        
        # Now perform convolution using F.conv2d
        output = F.conv2d(padded_x, self.kernel)
        
        # Restore original shape format based on the input
        if len(original_shape) == 5:  # Input was [batch_size, features, Nx, Ny, 1]
            output = output.unsqueeze(-1)  # -> [batch_size, out_features, Nx, Ny, 1]
        elif len(original_shape) == 2:  # Input was [Nx, Ny]
            if self.out_features == 1:
                output = output.squeeze(0).squeeze(0)  # -> [Nx, Ny]
            else:
                output = output.squeeze(0)  # -> [out_features, Nx, Ny]
        elif len(original_shape) == 3:  # Input was [batch, Nx, Ny]
            if self.out_features == 1:
                output = output.squeeze(1)  # -> [batch, Nx, Ny]
            # else: keep as [batch, out_features, Nx, Ny]
        
        return output


class ConvolutionalModel(nn.Module):
    """
    A multi-layer convolutional model using optimized boundary conditions.
    """
    def __init__(self, 
                 in_features=1, 
                 hidden_features=None, 
                 out_features=1, 
                 num_layers=3, 
                 kernel_size=3,
                 boundary_type='periodic', 
                 activation='relu',
                 init_type='random',
                 final_activation=None):
        """
        Initialize a multi-layer convolutional model.
        
        Args:
            in_features (int): Number of input features/channels
            hidden_features (int or list): Number of features in hidden layers
                              If int, all hidden layers have the same number of features
                              If list, specifies features for each hidden layer
            out_features (int): Number of output features/channels
            num_layers (int): Total number of convolutional layers
            kernel_size (int or list): Size of convolution kernels
                        If int, all layers use same kernel size
                        If list, specifies kernel size for each layer
            boundary_type (str or list): Boundary condition type
                          If str, all layers use same boundary type
                          If list, specifies boundary type for each layer
            activation (str or None): Activation function to use after each layer (except last)
                        Must be one of: 'gelu', 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', 'none'
            init_type (str or list): Kernel initialization strategy
                       If str, all layers use same initialization
                       If list, specifies initialization for each layer
            final_activation (str or None): Activation function after the final layer
        """
        super(ConvolutionalModel, self).__init__()
        
        # Map activation function names to their classes
        self.activation_map = {
            'gelu': nn.GELU,
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'leaky_relu': lambda: nn.LeakyReLU(0.2),
            'elu': nn.ELU,
            'selu': nn.SELU,
            'none': None
        }
        
        # Validate and prepare hidden_features
        if hidden_features is None:
            hidden_features = max(in_features, out_features)
        
        if isinstance(hidden_features, int):
            hidden_features = [hidden_features] * (num_layers - 1)
        elif len(hidden_features) != num_layers - 1:
            raise ValueError(f"If hidden_features is a list, it must have {num_layers-1} elements")
        
        # Prepare layer-specific parameters
        # Convert single values to lists if needed
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * num_layers
        elif len(kernel_size) != num_layers:
            raise ValueError(f"If kernel_size is a list, it must have {num_layers} elements")
            
        if isinstance(boundary_type, str):
            boundary_type = [boundary_type] * num_layers
        elif len(boundary_type) != num_layers:
            raise ValueError(f"If boundary_type is a list, it must have {num_layers} elements")
            
        if isinstance(init_type, str):
            init_type = [init_type] * num_layers
        elif len(init_type) != num_layers:
            raise ValueError(f"If init_type is a list, it must have {num_layers} elements")
        
        # Build feature dimensions for each layer
        layer_features = [in_features] + hidden_features + [out_features]
        
        # Create the layers
        self.conv_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        
        for i in range(num_layers):
            # Create convolutional layer
            conv = Convolution2d(
                kernel_size=kernel_size[i],
                in_features=layer_features[i],
                out_features=layer_features[i+1],
                init_type=init_type[i],
                boundary_type=boundary_type[i]
            )
            self.conv_layers.append(conv)
            
            # Add activation function (except for the last layer)
            if i < num_layers - 1:
                act_fn = self._get_activation(activation)
                self.activation_layers.append(act_fn)
            else:
                # For the final layer, use the specified final_activation (if any)
                act_fn = self._get_activation(final_activation)
                self.activation_layers.append(act_fn)
    
    def _get_activation(self, activation_name):
        """Helper method to get activation function module"""
        if activation_name is None or activation_name.lower() == 'none':
            return None
            
        if activation_name.lower() not in self.activation_map:
            raise ValueError(f"Unsupported activation: {activation_name}")
            
        activation_fn = self.activation_map[activation_name.lower()]
        if activation_fn is None:
            return None
        elif callable(activation_fn) and not isinstance(activation_fn, type):
            # If it's a factory function like lambda: nn.LeakyReLU(0.2)
            return activation_fn()
        else:
            # If it's a class like nn.ReLU
            return activation_fn()
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor with shape compatible with first layer
            
        Returns:
            torch.Tensor: Output tensor
        """
        for i, (conv, act) in enumerate(zip(self.conv_layers, self.activation_layers)):
            x = conv(x)
            if act is not None:
                x = act(x)
        return x

    def count_parameters(self):
        """
        Count the total number of trainable parameters in the model.
        
        Returns:
            int: Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
# %% 
# #Example Usage
# # Basic usage with default settings (3 layers, all periodic boundaries)
# model = ConvolutionalModel(
#     in_features=2, 
#     hidden_features=[8,16,8],
#     out_features=1, 
#     num_layers=4,
#     activation='gelu',
#     final_activation='none',
#     init_type='random'
# )

# # Advanced usage with layer-specific settings
# model = ConvolutionalModel(
#     in_features=3,
#     hidden_features=[8, 16, 8],  # Different sizes for each hidden layer
#     out_features=2,
#     num_layers=5,
#     kernel_size=[3, 5, 7, 5, 3],  # Different kernel sizes
#     boundary_type=['periodic', 'symmetric', 'periodic', 'symmetric', 'periodic'],
#     activation='gelu',
#     init_type=['random', 'xavier', 'random', 'zeros', 'identity'],
#     final_activation='tanh'
# )

# # Process a batch of data
# batch_size = 16
# x = torch.randn(batch_size, 3, 64, 64, 1)  # 5D input tensor
# output = model(x)  # Shape: [batch_size, 2, 64, 64, 1]
# %%
