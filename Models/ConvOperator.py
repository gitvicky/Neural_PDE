# %% 
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)

from boundary_conditions import OptimizedBoundaryManager


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
        
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        
        # Create a parameter tensor for convolutional kernel
        # Conv2d expects (out_channels, in_channels, kernel_h, kernel_w)
        self.kernel = nn.Parameter(torch.empty(out_features, in_features, kernel_size, kernel_size))
        
        # Initialize the kernel based on the specified strategy
        self._initialize_kernel(init_type)
        
        # Set up boundary manager using the optimized version
        self.boundary_manager = OptimizedBoundaryManager(kernel_size)
        self.boundary_manager.set_all_boundaries(boundary_type)
    
    def _initialize_kernel(self, init_type):
        """Initialize the kernel using the specified strategy."""
        with torch.no_grad():  # Disable gradient tracking during initialization
            if init_type == 'random' or init_type == 'xavier':
                # Xavier/Glorot initialization
                nn.init.xavier_uniform_(self.kernel)
            elif init_type == 'identity':
                # Initialize to approximate identity operation
                nn.init.zeros_(self.kernel)
                if self.in_features == self.out_features:
                    # Set center pixel to 1 for corresponding input-output channels
                    for i in range(self.in_features):
                        self.kernel[i, i, self.kernel_size//2, self.kernel_size//2] = 1.0
                else:
                    # For different input/output features, use identity where possible
                    min_features = min(self.in_features, self.out_features)
                    for i in range(min_features):
                        self.kernel[i, i, self.kernel_size//2, self.kernel_size//2] = 1.0
            elif init_type == 'zeros':
                nn.init.zeros_(self.kernel)
            elif init_type == 'ones':
                nn.init.ones_(self.kernel)
            else:
                raise ValueError(f"Unknown initialization type: {init_type}")
    
    def apply_boundary_padding(self, x):
        """
        Apply padding to the input tensor based on boundary conditions.
        Optimized version that processes batches more efficiently.
        """
        # Fast path: if all boundaries are the same type, process entire batch at once
        if len(set(self.boundary_manager.boundary_types.values())) == 1:
            bc_type = next(iter(self.boundary_manager.boundary_types.values()))
            
            # For symmetric, constant, or replicate, use single F.pad call on entire batch
            if bc_type in ['symmetric', 'dirichlet', 'neumann', 'outflow']:
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
            
            # For periodic boundary conditions
            elif bc_type == 'periodic':
                return self._apply_periodic_padding(x)
        
        # Fall back to slower approach for mixed boundary types
        return self._apply_mixed_boundary_padding(x)
    
    def _apply_periodic_padding(self, x):
        """Apply periodic boundary padding efficiently."""
        pad_left = self.boundary_manager.pad_left
        pad_right = self.boundary_manager.pad_right
        pad_top = self.boundary_manager.pad_top
        pad_bottom = self.boundary_manager.pad_bottom
        
        # Pad horizontally
        if pad_left > 0 or pad_right > 0:
            pads = []
            if pad_left > 0:
                pads.append(x[..., -pad_left:])
            pads.append(x)
            if pad_right > 0:
                pads.append(x[..., :pad_right])
            x = torch.cat(pads, dim=-1)
        
        # Pad vertically
        if pad_top > 0 or pad_bottom > 0:
            pads = []
            if pad_top > 0:
                pads.append(x[..., -pad_top:, :])
            pads.append(x)
            if pad_bottom > 0:
                pads.append(x[..., :pad_bottom, :])
            x = torch.cat(pads, dim=-2)
        
        return x
    
    def _apply_mixed_boundary_padding(self, x):
        """Apply mixed boundary conditions (slower but more flexible)."""
        batch_size, channels = x.shape[0], x.shape[1]
        
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
        
        # Input shape validation and normalization
        x, original_shape = self._normalize_input_shape(x)
        
        # Handle channel dimension mismatch
        x = self._adjust_input_channels(x)
        
        # Apply padding based on boundary conditions
        padded_x = self.apply_boundary_padding(x)
        
        # Perform convolution
        output = F.conv2d(padded_x, self.kernel)
        
        # Restore original shape format
        return self._restore_output_shape(output, original_shape)
    
    def _normalize_input_shape(self, x):
        """Normalize input to 4D tensor [batch, channels, height, width]."""
        original_shape = x.shape
        
        if len(original_shape) == 5:
            # Remove the last dimension if it's 1
            if original_shape[4] == 1:
                x = x.squeeze(-1)  # -> [batch_size, features, Nx, Ny]
            else:
                raise ValueError(f"Expected last dimension to be 1, got {original_shape[4]}")
        elif len(original_shape) == 4:
            # Already in the right format
            pass
        elif len(original_shape) == 3:  # [batch, Nx, Ny]
            x = x.unsqueeze(1)  # -> [batch, 1, Nx, Ny]
        elif len(original_shape) == 2:  # [Nx, Ny]
            x = x.unsqueeze(0).unsqueeze(0)  # -> [1, 1, Nx, Ny]
        else:
            raise ValueError(f"Unexpected input shape: {original_shape}")
        
        return x, original_shape
    
    def _adjust_input_channels(self, x):
        """Adjust input channels to match kernel requirements."""
        batch_size, x_channels = x.shape[0], x.shape[1]
        
        if x_channels != self.in_features:
            if x_channels == 1 and self.in_features > 1:
                # Broadcasting is much faster for expanding single channel
                x = x.expand(-1, self.in_features, -1, -1)
            else:
                # Handle general case
                new_x = torch.zeros(batch_size, self.in_features, x.shape[2], x.shape[3], 
                                  device=x.device, dtype=x.dtype)
                for i in range(self.in_features):
                    new_x[:, i] = x[:, i % x_channels]
                x = new_x
        
        return x
    
    def _restore_output_shape(self, output, original_shape):
        """Restore output to match original input shape format."""
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
        
        return output


class ConvolutionalModel(nn.Module):
    """
    A multi-layer convolutional model using optimized boundary conditions.
    """
    
    ACTIVATION_MAP = {
        'gelu': nn.GELU,
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'leaky_relu': lambda: nn.LeakyReLU(0.2),
        'elu': nn.ELU,
        'selu': nn.SELU,
        'none': None
    }
    
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
            out_features (int): Number of output features/channels
            num_layers (int): Total number of convolutional layers
            kernel_size (int or list): Size of convolution kernels
            boundary_type (str or list): Boundary condition type
            activation (str): Activation function ('gelu', 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', 'none')
            init_type (str or list): Kernel initialization strategy
            final_activation (str or None): Activation function after the final layer
        """
        super(ConvolutionalModel, self).__init__()
        
        # Validate inputs
        self._validate_inputs(num_layers, activation, final_activation)
        
        # Prepare layer-specific parameters
        layer_params = self._prepare_layer_params(
            num_layers, in_features, hidden_features, out_features,
            kernel_size, boundary_type, init_type
        )
        
        # Create the layers
        self.conv_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        
        self._build_layers(layer_params, activation, final_activation)
    
    def _validate_inputs(self, num_layers, activation, final_activation):
        """Validate input parameters."""
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        
        if activation and activation.lower() not in self.ACTIVATION_MAP:
            raise ValueError(f"Unsupported activation: {activation}")
        
        if final_activation and final_activation.lower() not in self.ACTIVATION_MAP:
            raise ValueError(f"Unsupported final_activation: {final_activation}")
    
    def _prepare_layer_params(self, num_layers, in_features, hidden_features, out_features,
                             kernel_size, boundary_type, init_type):
        """Prepare parameters for each layer."""
        # Handle hidden_features
        if hidden_features is None:
            hidden_features = max(in_features, out_features)
        
        if isinstance(hidden_features, int):
            hidden_features = [hidden_features] * (num_layers - 1)
        elif len(hidden_features) != num_layers - 1:
            raise ValueError(f"hidden_features list must have {num_layers-1} elements")
        
        # Convert single values to lists
        def _to_list(param, name):
            if isinstance(param, (int, str)):
                return [param] * num_layers
            elif len(param) != num_layers:
                raise ValueError(f"{name} list must have {num_layers} elements")
            return param
        
        kernel_sizes = _to_list(kernel_size, "kernel_size")
        boundary_types = _to_list(boundary_type, "boundary_type")
        init_types = _to_list(init_type, "init_type")
        
        # Build feature dimensions for each layer
        layer_features = [in_features] + hidden_features + [out_features]
        
        return {
            'layer_features': layer_features,
            'kernel_sizes': kernel_sizes,
            'boundary_types': boundary_types,
            'init_types': init_types
        }
    
    def _build_layers(self, layer_params, activation, final_activation):
        """Build the convolutional and activation layers."""
        num_layers = len(layer_params['kernel_sizes'])
        
        for i in range(num_layers):
            # Create convolutional layer
            conv = Convolution2d(
                kernel_size=layer_params['kernel_sizes'][i],
                in_features=layer_params['layer_features'][i],
                out_features=layer_params['layer_features'][i+1],
                init_type=layer_params['init_types'][i],
                boundary_type=layer_params['boundary_types'][i]
            )
            self.conv_layers.append(conv)
            
            # Add activation function
            if i < num_layers - 1:
                # Hidden layers use the main activation
                act_fn = self._get_activation(activation)
            else:
                # Final layer uses final_activation
                act_fn = self._get_activation(final_activation)
            
            self.activation_layers.append(act_fn)
    
    def _get_activation(self, activation_name):
        """Get activation function module."""
        if activation_name is None or activation_name.lower() == 'none':
            return None
        
        activation_fn = self.ACTIVATION_MAP[activation_name.lower()]
        if activation_fn is None:
            return None
        elif callable(activation_fn) and not isinstance(activation_fn, type):
            # Factory function like lambda: nn.LeakyReLU(0.2)
            return activation_fn()
        else:
            # Class like nn.ReLU
            return activation_fn()
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        for conv, act in zip(self.conv_layers, self.activation_layers):
            x = conv(x)
            if act is not None:
                x = act(x)
        return x

    def count_parameters(self):
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Example usage
if __name__ == "__main__":
    # Basic usage with default settings
    model1 = ConvolutionalModel(
        in_features=2, 
        hidden_features=[8, 16, 8],
        out_features=1, 
        num_layers=4,
        activation='gelu',
        final_activation='none',
        init_type='random'
    )
    
    # Advanced usage with layer-specific settings
    model2 = ConvolutionalModel(
        in_features=3,
        hidden_features=[8, 16, 16, 8],  # Different sizes for each hidden layer
        out_features=2,
        num_layers=5,
        kernel_size=[3, 5, 7, 5, 3],  # Different kernel sizes
        boundary_type=['periodic', 'symmetric', 'periodic', 'symmetric', 'periodic'],
        activation='gelu',
        init_type=['random', 'xavier', 'random', 'zeros', 'identity'],
        final_activation='tanh'
    )
    
    print(f"Model 1 parameters: {model1.count_parameters()}")
    print(f"Model 2 parameters: {model2.count_parameters()}")
    
    # Test with sample data
    batch_size = 16
    x = torch.randn(batch_size, 3, 64, 64, 1)
    
    try:
        output = model2(x)
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Error: {e}")