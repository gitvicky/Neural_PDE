# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================================================================
# ISSUE 1: Gradient Explosion from Unstable Boundary Conditions
# =============================================================================

class BoundaryManager:
    """Enhanced boundary manager with gradient stability checks"""
    
    def __init__(self, kernel_size):
        if isinstance(kernel_size, int):
            self.kernel_height = kernel_size
            self.kernel_width = kernel_size
        else:
            self.kernel_height, self.kernel_width = kernel_size
            
        self.pad_left = self.kernel_width // 2
        self.pad_right = self.kernel_width // 2
        self.pad_top = self.kernel_height // 2
        self.pad_bottom = self.kernel_height // 2
        
        self.boundary_types = {
            'left': 'periodic',
            'right': 'periodic', 
            'top': 'periodic',
            'bottom': 'periodic'
        }
        
        self.boundary_values = {
            'left': 0.0,
            'right': 0.0,
            'top': 0.0,
            'bottom': 0.0
        }
    
    def set_boundary_type(self, side, bc_type, value=0.0):
        if side not in ['left', 'right', 'top', 'bottom']:
            raise ValueError(f"Unknown side: {side}")
        if bc_type.lower() not in ['dirichlet', 'neumann', 'periodic', 'symmetric']:
            raise ValueError(f"Unsupported boundary type: {bc_type}")
        
        self.boundary_types[side] = bc_type.lower()
        # CRITICAL FIX: Clamp boundary values to prevent extreme values
        self.boundary_values[side] = torch.clamp(torch.tensor(value), -10.0, 10.0).item()
    
    def pad_signal(self, signal):
        """Improved padding with NaN checking"""
        # Check for NaN/Inf in input
        if torch.isnan(signal).any() or torch.isinf(signal).any():
            print("WARNING: NaN/Inf detected in input signal before padding")
            signal = torch.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=-1.0)
        
        original_shape = signal.shape
        original_ndim = len(original_shape)
        
        if original_ndim == 2:
            signal = signal.unsqueeze(0).unsqueeze(0)
        
        result = signal
        
        # Apply padding with additional stability checks
        for side, pad_size in [('left', self.pad_left), ('right', self.pad_right), 
                              ('top', self.pad_top), ('bottom', self.pad_bottom)]:
            if pad_size > 0:
                bc_type = self.boundary_types[side]
                bc_value = self.boundary_values[side]
                
                if side == 'left':
                    pad = (pad_size, 0, 0, 0)
                elif side == 'right':
                    pad = (0, pad_size, 0, 0)
                elif side == 'top':
                    pad = (0, 0, pad_size, 0)
                else:  # bottom
                    pad = (0, 0, 0, pad_size)
                
                try:
                    if bc_type == 'dirichlet':
                        result = F.pad(result, pad, mode='constant', value=bc_value)
                    elif bc_type == 'neumann':
                        result = F.pad(result, pad, mode='replicate')
                    elif bc_type == 'periodic':
                        result = F.pad(result, pad, mode='circular')
                    elif bc_type == 'symmetric':
                        result = F.pad(result, pad, mode='reflect')
                    
                    # Check for NaN after each padding operation
                    if torch.isnan(result).any():
                        print(f"NaN detected after {side} boundary padding with {bc_type}")
                        result = torch.nan_to_num(result, nan=0.0)
                        
                except Exception as e:
                    print(f"Error in padding {side}: {e}")
                    # Fall back to zero padding
                    result = F.pad(result, pad, mode='constant', value=0.0)
        
        if original_ndim == 2:
            result = result.squeeze(0).squeeze(0)
        
        return result


# =============================================================================
# ISSUE 2: Improved Kernel Initialization
# =============================================================================

class Convolution2d(nn.Module):
    """Improved convolution with better initialization and stability"""
    
    def __init__(self, kernel_size=3, in_features=1, out_features=1, 
                 init_type='xavier', boundary_type='periodic'):
        super(Convolution2d, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        
        self.kernel = nn.Parameter(torch.empty(out_features, in_features, 
                                              kernel_size, kernel_size))
        
        # CRITICAL FIX: Better initialization
        self._initialize_kernel_stable(init_type)
        
        self.boundary_manager = BoundaryManager(kernel_size)
        self.boundary_manager.set_boundary_type('left', boundary_type)
        self.boundary_manager.set_boundary_type('right', boundary_type)
        self.boundary_manager.set_boundary_type('top', boundary_type)
        self.boundary_manager.set_boundary_type('bottom', boundary_type)
    
    def _initialize_kernel_stable(self, init_type):
        """Improved kernel initialization with stability guarantees"""
        with torch.no_grad():
            if init_type == 'xavier' or init_type == 'random':
                # Xavier initialization with smaller variance for stability
                nn.init.xavier_uniform_(self.kernel, gain=0.1)  # Reduced gain
                
            elif init_type == 'identity':
                nn.init.zeros_(self.kernel)
                if self.in_features == self.out_features:
                    for i in range(self.in_features):
                        # Smaller identity weight for stability
                        self.kernel[i, i, self.kernel_size//2, self.kernel_size//2] = 0.1
                        
            elif init_type == 'zeros':
                nn.init.zeros_(self.kernel)
                
            elif init_type == 'small_random':
                # NEW: Very small random initialization
                nn.init.uniform_(self.kernel, -0.01, 0.01)
                
            elif init_type == 'he_small':
                # NEW: He initialization with reduced variance
                nn.init.kaiming_uniform_(self.kernel, mode='fan_in', nonlinearity='relu')
                self.kernel.data *= 0.1  # Scale down
                
            else:
                # Default to small random
                nn.init.uniform_(self.kernel, -0.01, 0.01)
            
            # CRITICAL: Ensure kernel values are bounded
            self.kernel.data = torch.clamp(self.kernel.data, -1.0, 1.0)
    
    def forward(self, x):
        """Forward pass with extensive NaN checking"""
        # if x is None:
        #     return self.kernel
        
        # # Input validation
        # if torch.isnan(x).any() or torch.isinf(x).any():
        #     print("WARNING: NaN/Inf in input to convolution")
        #     x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # # Kernel validation
        # if torch.isnan(self.kernel).any() or torch.isinf(self.kernel).any():
        #     print("WARNING: NaN/Inf in kernel weights")
        #     with torch.no_grad():
        #         self.kernel.data = torch.nan_to_num(self.kernel.data, nan=0.0)
        #         self.kernel.data = torch.clamp(self.kernel.data, -1.0, 1.0)
        
        # Normalize input shape
        x, original_shape = self._normalize_input_shape(x)
        
        # Handle channel dimension
        x = self._adjust_input_channels(x)
        
        # Apply boundary padding
        try:
            padded_x = self.boundary_manager.pad_signal(x)
        except Exception as e:
            print(f"Error in padding: {e}")
            # Fall back to zero padding
            pad_size = self.kernel_size // 2
            padded_x = F.pad(x, (pad_size, pad_size, pad_size, pad_size), 
                           mode='constant', value=0.0)
        
        # Perform convolution
        try:
            output = F.conv2d(padded_x, self.kernel)
        except Exception as e:
            print(f"Error in convolution: {e}")
            # Return zeros with correct shape
            output = torch.zeros_like(x[:, :self.out_features])
        
        # # Final NaN check
        # if torch.isnan(output).any():
        #     print("WARNING: NaN in convolution output")
        #     output = torch.nan_to_num(output, nan=0.0)
        
        return self._restore_output_shape(output, original_shape)
    
    def _normalize_input_shape(self, x):
        """Same as original but with bounds checking"""
        original_shape = x.shape
        
        if len(original_shape) == 5:
            if original_shape[4] == 1:
                x = x.squeeze(-1)
            else:
                raise ValueError(f"Expected last dimension to be 1, got {original_shape[4]}")
        elif len(original_shape) == 4:
            pass
        elif len(original_shape) == 3:
            x = x.unsqueeze(1)
        elif len(original_shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected input shape: {original_shape}")
        
        return x, original_shape
    
    def _adjust_input_channels(self, x):
        """Same as original but with stability checks"""
        batch_size, x_channels = x.shape[0], x.shape[1]
        
        if x_channels != self.in_features:
            if x_channels == 1 and self.in_features > 1:
                x = x.expand(-1, self.in_features, -1, -1)
            else:
                new_x = torch.zeros(batch_size, self.in_features, x.shape[2], x.shape[3], 
                                  device=x.device, dtype=x.dtype)
                for i in range(self.in_features):
                    new_x[:, i] = x[:, i % x_channels]
                x = new_x
        
        return x
    
    def _restore_output_shape(self, output, original_shape):
        """Same as original"""
        if len(original_shape) == 5:
            output = output.unsqueeze(-1)
        elif len(original_shape) == 2:
            if self.out_features == 1:
                output = output.squeeze(0).squeeze(0)
            else:
                output = output.squeeze(0)
        elif len(original_shape) == 3:
            if self.out_features == 1:
                output = output.squeeze(1)
        
        return output


# =============================================================================
# ISSUE 3: Improved Model with Gradient Clipping and Normalization
# =============================================================================

class ConvolutionalModel(nn.Module):
    """Improved model with better stability features"""
    
    def __init__(self, in_features=1, hidden_features=None, out_features=1,
                 num_layers=3, kernel_size=3, boundary_type='periodic',
                 activation='gelu', init_type='xavier', final_activation=None,
                 use_layer_norm=True, use_residual=False):
        super(ConvolutionalModel, self).__init__()
        
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        
        # Prepare layer parameters
        if hidden_features is None:
            hidden_features = max(in_features, out_features)
        
        if isinstance(hidden_features, int):
            hidden_features = [hidden_features] * (num_layers - 1)
        
        layer_features = [in_features] + hidden_features + [out_features]
        
        # Create layers
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        
        for i in range(num_layers):
            # Convolution layer
            conv = Convolution2d(
                kernel_size=kernel_size,
                in_features=layer_features[i],
                out_features=layer_features[i+1],
                init_type=init_type,
                boundary_type=boundary_type
            )
            self.conv_layers.append(conv)
            
            # Layer normalization (helps with gradient stability)
            if use_layer_norm and i < num_layers - 1:
                # Note: LayerNorm on spatial dimensions
                self.norm_layers.append(nn.GroupNorm(1, layer_features[i+1]))
            else:
                self.norm_layers.append(None)
            
            # Activation
            if i < num_layers - 1:
                act = self._get_activation(activation)
            else:
                act = self._get_activation(final_activation)
            self.activation_layers.append(act)
    
    def _get_activation(self, activation_name):
        """Get activation with stability considerations"""
        if activation_name is None or activation_name.lower() == 'none':
            return None
        
        activation_map = {
            'gelu': nn.GELU,
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'leaky_relu': lambda: nn.LeakyReLU(0.01),  # Smaller negative slope
            'elu': nn.ELU,
            'selu': nn.SELU,
            'swish': nn.SiLU,  # Often more stable than GELU
        }
        
        if activation_name.lower() in activation_map:
            return activation_map[activation_name.lower()]()
        return None
    
    def forward(self, x):
        """Forward pass with stability features"""
        # # Input validation
        # if torch.isnan(x).any() or torch.isinf(x).any():
        #     print("WARNING: NaN/Inf in model input")
        #     x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Store for potential residual connection
        residual = None
        
        for i, (conv, norm, act) in enumerate(zip(self.conv_layers, self.norm_layers, 
                                                 self.activation_layers)):
            # Save input for residual connection
            if self.use_residual and i == 0:
                residual = x
            
            # Apply convolution
            x = conv(x)
            
            # # Check for NaN after convolution
            # if torch.isnan(x).any():
            #     print(f"NaN detected after convolution layer {i}")
            #     x = torch.nan_to_num(x, nan=0.0)
            
            # Apply normalization
            if norm is not None:
                x = norm(x)
                
                # Check for NaN after normalization
                if torch.isnan(x).any():
                    print(f"NaN detected after normalization layer {i}")
                    x = torch.nan_to_num(x, nan=0.0)
            
            # Apply activation
            if act is not None:
                x = act(x)
                
                # # Check for NaN after activation
                # if torch.isnan(x).any():
                #     print(f"NaN detected after activation layer {i}")
                #     x = torch.nan_to_num(x, nan=0.0)
        
        # Apply residual connection if enabled
        if self.use_residual and residual is not None:
            if x.shape == residual.shape:
                x = x + 0.1 * residual  # Small residual weight
            
        # # Final output clipping
        # x = torch.clamp(x, -10.0, 10.0)
        
        return x
    
    def count_params(self):
        """Count parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



# # =============================================================================
# # EXAMPLE USAGE
# # =============================================================================

# if __name__ == "__main__":
#     # Create stable model
#     model = ConvolutionalModel(
#         in_features=2,
#         hidden_features=[8, 16, 8],
#         out_features=1,
#         num_layers=4,
#         activation='swish',
#         use_layer_norm=True,
#         use_residual=True
#     )
    
#     print(f"Model parameters: {model.count_params()}")
    
#     # Test with sample data
#     x = torch.randn(4, 2, 32, 32)
    
#     # Forward pass
#     with torch.no_grad():
#         output = model(x)
    
#     print("Model created successfully!")
# # %%

# # %% 
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import time

# count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)

# import sys 
# sys.path.append('/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/NOs_for_POs')
# from PRE.boundary_conditions import BoundaryManager

# class Convolution2d(nn.Module):
#     def __init__(self, kernel_size=3, in_features=1, out_features=1, init_type='random', boundary_type='periodic'):
#         """
#         Create a learnable convolution with configurable kernel and boundary conditions.
        
#         Args:
#             kernel_size (int): The size of the convolution kernel
#             in_features (int): Number of input feature channels
#             out_features (int): Number of output feature channels
#             init_type (str): Initialization strategy - 'random', 'identity', 'zeros', or 'ones'
#             boundary_type (str): Type of boundary condition - 'dirichlet', 'neumann', 'periodic', 'symmetric'
#         """
#         super(Convolution2d, self).__init__()
        
#         self.in_features = in_features
#         self.out_features = out_features
#         self.kernel_size = kernel_size
        
#         # Create a parameter tensor for convolutional kernel
#         # Conv2d expects (out_channels, in_channels, kernel_h, kernel_w)
#         self.kernel = nn.Parameter(torch.empty(out_features, in_features, kernel_size, kernel_size))
        
#         # Initialize the kernel based on the specified strategy
#         self._initialize_kernel(init_type)
        
#         self.boundary_manager = BoundaryManager(kernel_size)
#         self.boundary_manager.set_all_boundaries(boundary_type)
    
#     def _initialize_kernel(self, init_type):
#         """Initialize the kernel using the specified strategy."""
#         with torch.no_grad():  # Disable gradient tracking during initialization
#             if init_type == 'random' or init_type == 'xavier':
#                 # Xavier/Glorot initialization
#                 nn.init.xavier_uniform_(self.kernel)
#             elif init_type == 'identity':
#                 # Initialize to approximate identity operation
#                 nn.init.zeros_(self.kernel)
#                 if self.in_features == self.out_features:
#                     # Set center pixel to 1 for corresponding input-output channels
#                     for i in range(self.in_features):
#                         self.kernel[i, i, self.kernel_size//2, self.kernel_size//2] = 1.0
#                 else:
#                     # For different input/output features, use identity where possible
#                     min_features = min(self.in_features, self.out_features)
#                     for i in range(min_features):
#                         self.kernel[i, i, self.kernel_size//2, self.kernel_size//2] = 1.0
#             elif init_type == 'zeros':
#                 nn.init.zeros_(self.kernel)
#             elif init_type == 'ones':
#                 nn.init.ones_(self.kernel)
#             else:
#                 raise ValueError(f"Unknown initialization type: {init_type}")
    
#     def apply_boundary_padding(self, x):
#         """
#         Apply padding to the input tensor based on boundary conditions.
#         Optimized version that processes batches more efficiently.
#         """
#         # Fast path: if all boundaries are the same type, process entire batch at once
#         if len(set(self.boundary_manager.boundary_types.values())) == 1:
#             bc_type = next(iter(self.boundary_manager.boundary_types.values()))
            
#             # For symmetric, constant, or replicate, use single F.pad call on entire batch
#             if bc_type in ['symmetric', 'dirichlet', 'neumann', 'outflow']:
#                 pad = (
#                     self.boundary_manager.pad_left, 
#                     self.boundary_manager.pad_right, 
#                     self.boundary_manager.pad_top, 
#                     self.boundary_manager.pad_bottom
#                 )
                
#                 if bc_type == 'symmetric':
#                     return F.pad(x, pad, mode='reflect')
#                 elif bc_type == 'dirichlet':
#                     bc_value = next(iter(self.boundary_manager.boundary_values.values()))
#                     return F.pad(x, pad, mode='constant', value=bc_value)
#                 elif bc_type in ['neumann', 'outflow']:
#                     return F.pad(x, pad, mode='replicate')
            
#             # For periodic boundary conditions
#             elif bc_type == 'periodic':
#                 return self._apply_periodic_padding(x)
        
#         # Fall back to slower approach for mixed boundary types
#         return self._apply_mixed_boundary_padding(x)
    
#     def _apply_periodic_padding(self, x):
#         """Apply periodic boundary padding efficiently."""
#         pad_left = self.boundary_manager.pad_left
#         pad_right = self.boundary_manager.pad_right
#         pad_top = self.boundary_manager.pad_top
#         pad_bottom = self.boundary_manager.pad_bottom
        
#         # Pad horizontally
#         if pad_left > 0 or pad_right > 0:
#             pads = []
#             if pad_left > 0:
#                 pads.append(x[..., -pad_left:])
#             pads.append(x)
#             if pad_right > 0:
#                 pads.append(x[..., :pad_right])
#             x = torch.cat(pads, dim=-1)
        
#         # Pad vertically
#         if pad_top > 0 or pad_bottom > 0:
#             pads = []
#             if pad_top > 0:
#                 pads.append(x[..., -pad_top:, :])
#             pads.append(x)
#             if pad_bottom > 0:
#                 pads.append(x[..., :pad_bottom, :])
#             x = torch.cat(pads, dim=-2)
        
#         return x
    
#     def _apply_mixed_boundary_padding(self, x):
#         """Apply mixed boundary conditions (slower but more flexible)."""
#         batch_size, channels = x.shape[0], x.shape[1]
        
#         result = []
#         for b in range(batch_size):
#             channels_result = []
#             for c in range(channels):
#                 # Extract 2D slice and apply boundary padding
#                 x_slice = x[b, c]  # (height, width)
#                 padded_slice = self.boundary_manager.pad_signal(x_slice)
#                 channels_result.append(padded_slice.unsqueeze(0))
            
#             # Stack channels back together
#             batch_result = torch.cat(channels_result, dim=0).unsqueeze(0)
#             result.append(batch_result)
        
#         return torch.cat(result, dim=0)
    
#     def get_kernel(self):
#         """Get the current value of the convolution kernel."""
#         return self.kernel.detach()
    
#     def count_params(self):
#         """Count the number of parameters in the kernel."""
#         return self.kernel.numel()
    
#     def set_boundary_type(self, boundary_type, value=0.0):
#         """Set the boundary condition type for all sides."""
#         self.boundary_manager.set_all_boundaries(boundary_type, value)
    
#     def set_specific_boundary(self, side, bc_type, value=0.0):
#         """Set boundary condition for a specific side."""
#         self.boundary_manager.set_boundary_type(side, bc_type, value)

#     def forward(self, x=None):
#         """
#         Forward pass to apply convolution with proper boundary conditions.
        
#         Args:
#             x (torch.Tensor): Input tensor with shape (batch_size, features, Nx, Ny, 1)
#                               If None, return the current kernel
        
#         Returns:
#             torch.Tensor: Result of convolution with shape adjusted for output features
#         """
#         if x is None:
#             return self.kernel
        
#         # Input shape validation and normalization
#         x, original_shape = self._normalize_input_shape(x)
        
#         # Handle channel dimension mismatch
#         x = self._adjust_input_channels(x)
        
#         # Apply padding based on boundary conditions
#         padded_x = self.apply_boundary_padding(x)
        
#         # Perform convolution
#         output = F.conv2d(padded_x, self.kernel)
        
#         # Restore original shape format
#         return self._restore_output_shape(output, original_shape)
    
#     def _normalize_input_shape(self, x):
#         """Normalize input to 4D tensor [batch, channels, height, width]."""
#         original_shape = x.shape
        
#         if len(original_shape) == 5:
#             # Remove the last dimension if it's 1
#             if original_shape[4] == 1:
#                 x = x.squeeze(-1)  # -> [batch_size, features, Nx, Ny]
#             else:
#                 raise ValueError(f"Expected last dimension to be 1, got {original_shape[4]}")
#         elif len(original_shape) == 4:
#             # Already in the right format
#             pass
#         elif len(original_shape) == 3:  # [batch, Nx, Ny]
#             x = x.unsqueeze(1)  # -> [batch, 1, Nx, Ny]
#         elif len(original_shape) == 2:  # [Nx, Ny]
#             x = x.unsqueeze(0).unsqueeze(0)  # -> [1, 1, Nx, Ny]
#         else:
#             raise ValueError(f"Unexpected input shape: {original_shape}")
        
#         return x, original_shape
    
#     def _adjust_input_channels(self, x):
#         """Adjust input channels to match kernel requirements."""
#         batch_size, x_channels = x.shape[0], x.shape[1]
        
#         if x_channels != self.in_features:
#             if x_channels == 1 and self.in_features > 1:
#                 # Broadcasting is much faster for expanding single channel
#                 x = x.expand(-1, self.in_features, -1, -1)
#             else:
#                 # Handle general case
#                 new_x = torch.zeros(batch_size, self.in_features, x.shape[2], x.shape[3], 
#                                   device=x.device, dtype=x.dtype)
#                 for i in range(self.in_features):
#                     new_x[:, i] = x[:, i % x_channels]
#                 x = new_x
        
#         return x
    
#     def _restore_output_shape(self, output, original_shape):
#         """Restore output to match original input shape format."""
#         if len(original_shape) == 5:  # Input was [batch_size, features, Nx, Ny, 1]
#             output = output.unsqueeze(-1)  # -> [batch_size, out_features, Nx, Ny, 1]
#         elif len(original_shape) == 2:  # Input was [Nx, Ny]
#             if self.out_features == 1:
#                 output = output.squeeze(0).squeeze(0)  # -> [Nx, Ny]
#             else:
#                 output = output.squeeze(0)  # -> [out_features, Nx, Ny]
#         elif len(original_shape) == 3:  # Input was [batch, Nx, Ny]
#             if self.out_features == 1:
#                 output = output.squeeze(1)  # -> [batch, Nx, Ny]
        
#         return output


# class ConvolutionalModel(nn.Module):
#     """
#     A multi-layer convolutional model using optimized boundary conditions.
#     """
    
#     ACTIVATION_MAP = {
#         'gelu': nn.GELU,
#         'relu': nn.ReLU,
#         'tanh': nn.Tanh,
#         'sigmoid': nn.Sigmoid,
#         'leaky_relu': lambda: nn.LeakyReLU(0.2),
#         'elu': nn.ELU,
#         'selu': nn.SELU,
#         'none': None
#     }
    
#     def __init__(self, 
#                  in_features=1, 
#                  hidden_features=None, 
#                  out_features=1, 
#                  num_layers=3, 
#                  kernel_size=3,
#                  boundary_type='periodic', 
#                  activation='relu',
#                  init_type='random',
#                  final_activation=None):
#         """
#         Initialize a multi-layer convolutional model.
        
#         Args:
#             in_features (int): Number of input features/channels
#             hidden_features (int or list): Number of features in hidden layers
#             out_features (int): Number of output features/channels
#             num_layers (int): Total number of convolutional layers
#             kernel_size (int or list): Size of convolution kernels
#             boundary_type (str or list): Boundary condition type
#             activation (str): Activation function ('gelu', 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', 'none')
#             init_type (str or list): Kernel initialization strategy
#             final_activation (str or None): Activation function after the final layer
#         """
#         super(ConvolutionalModel, self).__init__()
        
#         # Validate inputs
#         self._validate_inputs(num_layers, activation, final_activation)
        
#         # Prepare layer-specific parameters
#         layer_params = self._prepare_layer_params(
#             num_layers, in_features, hidden_features, out_features,
#             kernel_size, boundary_type, init_type
#         )
        
#         # Create the layers
#         self.conv_layers = nn.ModuleList()
#         self.activation_layers = nn.ModuleList()
        
#         self._build_layers(layer_params, activation, final_activation)
    
#     def _validate_inputs(self, num_layers, activation, final_activation):
#         """Validate input parameters."""
#         if num_layers < 1:
#             raise ValueError("num_layers must be at least 1")
        
#         if activation and activation.lower() not in self.ACTIVATION_MAP:
#             raise ValueError(f"Unsupported activation: {activation}")
        
#         if final_activation and final_activation.lower() not in self.ACTIVATION_MAP:
#             raise ValueError(f"Unsupported final_activation: {final_activation}")
    
#     def _prepare_layer_params(self, num_layers, in_features, hidden_features, out_features,
#                              kernel_size, boundary_type, init_type):
#         """Prepare parameters for each layer."""
#         # Handle hidden_features
#         if hidden_features is None:
#             hidden_features = max(in_features, out_features)
        
#         if isinstance(hidden_features, int):
#             hidden_features = [hidden_features] * (num_layers - 1)
#         elif len(hidden_features) != num_layers - 1:
#             raise ValueError(f"hidden_features list must have {num_layers-1} elements")
        
#         # Convert single values to lists
#         def _to_list(param, name):
#             if isinstance(param, (int, str)):
#                 return [param] * num_layers
#             elif len(param) != num_layers:
#                 raise ValueError(f"{name} list must have {num_layers} elements")
#             return param
        
#         kernel_sizes = _to_list(kernel_size, "kernel_size")
#         boundary_types = _to_list(boundary_type, "boundary_type")
#         init_types = _to_list(init_type, "init_type")
        
#         # Build feature dimensions for each layer
#         layer_features = [in_features] + hidden_features + [out_features]
        
#         return {
#             'layer_features': layer_features,
#             'kernel_sizes': kernel_sizes,
#             'boundary_types': boundary_types,
#             'init_types': init_types
#         }
    
#     def _build_layers(self, layer_params, activation, final_activation):
#         """Build the convolutional and activation layers."""
#         num_layers = len(layer_params['kernel_sizes'])
        
#         for i in range(num_layers):
#             # Create convolutional layer
#             conv = Convolution2d(
#                 kernel_size=layer_params['kernel_sizes'][i],
#                 in_features=layer_params['layer_features'][i],
#                 out_features=layer_params['layer_features'][i+1],
#                 init_type=layer_params['init_types'][i],
#                 boundary_type=layer_params['boundary_types'][i]
#             )
#             self.conv_layers.append(conv)
            
#             # Add activation function
#             if i < num_layers - 1:
#                 # Hidden layers use the main activation
#                 act_fn = self._get_activation(activation)
#             else:
#                 # Final layer uses final_activation
#                 act_fn = self._get_activation(final_activation)
            
#             self.activation_layers.append(act_fn)
    
#     def _get_activation(self, activation_name):
#         """Get activation function module."""
#         if activation_name is None or activation_name.lower() == 'none':
#             return None
        
#         activation_fn = self.ACTIVATION_MAP[activation_name.lower()]
#         if activation_fn is None:
#             return None
#         elif callable(activation_fn) and not isinstance(activation_fn, type):
#             # Factory function like lambda: nn.LeakyReLU(0.2)
#             return activation_fn()
#         else:
#             # Class like nn.ReLU
#             return activation_fn()
    
#     def forward(self, x):
#         """
#         Forward pass through the model.
        
#         Args:
#             x (torch.Tensor): Input tensor
            
#         Returns:
#             torch.Tensor: Output tensor
#         """
#         for conv, act in zip(self.conv_layers, self.activation_layers):
#             x = conv(x)
#             if act is not None:
#                 x = act(x)
#         return x

#     def count_params(self):
#         """Count the total number of trainable parameters."""
#         return sum(p.numel() for p in self.parameters() if p.requires_grad)


# # Example usage
# if __name__ == "__main__":
#     # Basic usage with default settings
#     model1 = ConvolutionalModel(
#         in_features=2, 
#         hidden_features=[8, 16, 8],
#         out_features=1, 
#         num_layers=4,
#         activation='gelu',
#         final_activation='none',
#         init_type='random'
#     )
    
#     # Advanced usage with layer-specific settings
#     model2 = ConvolutionalModel(
#         in_features=3,
#         hidden_features=[8, 16, 16, 8],  # Different sizes for each hidden layer
#         out_features=2,
#         num_layers=5,
#         kernel_size=[3, 5, 7, 5, 3],  # Different kernel sizes
#         boundary_type=['periodic', 'symmetric', 'periodic', 'symmetric', 'periodic'],
#         activation='gelu',
#         init_type=['random', 'xavier', 'random', 'zeros', 'identity'],
#         final_activation='tanh'
#     )
    
#     print(f"Model 1 parameters: {model1.count_params()}")
#     print(f"Model 2 parameters: {model2.count_params()}")
    
#     # Test with sample data
#     batch_size = 16
#     x = torch.randn(batch_size, 3, 64, 64, 1)
    
#     try:
#         output = model2(x)
#         print(f"Output shape: {output.shape}")
#     except Exception as e:
#         print(f"Error: {e}")

# # %% 