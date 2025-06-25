# %% 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import time
count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)

import sys 
sys.path.append('..')
# Import the new OptimizedBoundaryManager instead of the original BoundaryManager
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
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Padded tensor ready for convolution
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
        
        # Fall back to original implementation for mixed boundary types
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
        """
        Get the current value of the convolution kernel.
        
        Returns:
            torch.Tensor: The kernel as a tensor
        """
        return self.kernel.detach()
    
    def count_params(self):
        """
        Count the number of parameters in the kernel.
        
        Returns:
            int: The number of parameters
        """
        return self.kernel.numel()
    
    def set_boundary_type(self, boundary_type, value=0.0):
        """
        Set the boundary condition type for all sides.
        
        Args:
            boundary_type (str): Type of boundary condition - 'dirichlet', 'neumann', 'periodic', 'symmetric'
            value (float): Value for Dirichlet boundary condition (default 0.0)
        """
        self.boundary_manager.set_all_boundaries(boundary_type, value)
    
    def set_specific_boundary(self, side, bc_type, value=0.0):
        """
        Set boundary condition for a specific side.
        
        Args:
            side (str): One of 'left', 'right', 'top', 'bottom'
            bc_type (str): Boundary condition type
            value (float): Value for Dirichlet boundary condition
        """
        self.boundary_manager.set_boundary_type(side, bc_type, value)

    def forward(self, x=None):
        """
        Forward pass to apply convolution with proper boundary conditions.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, features, Nx, Ny, 1)
                              If None, return the current kernel
        
        Returns:
            torch.Tensor: Result of convolution with shape adjusted for output features
                         (batch_size, out_features, Nx, Ny, 1)
        """
        import time
        
        # Dictionary to store profiling results
        profile_times = {}
        
        start_time = time.time()
        
        if x is None:
            return self.kernel
        
        profile_times["initial_check"] = time.time() - start_time
        start_time = time.time()
        
        # Input shape validation and save original shape
        original_shape = x.shape
        
        profile_times["save_original_shape"] = time.time() - start_time
        start_time = time.time()
        
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
        
        profile_times["reshape_input"] = time.time() - start_time
        start_time = time.time()
        
        batch_size = x.shape[0]
        x_channels = x.shape[1]
        
        profile_times["get_dimensions"] = time.time() - start_time
        start_time = time.time()
        
        # Handle the case where input channels don't match kernel in_features
        if x_channels != self.in_features:
            # Create a new tensor with the correct number of channels
            new_x = torch.zeros(batch_size, self.in_features, x.shape[2], x.shape[3], device=x.device)
            
            profile_times["create_new_tensor"] = time.time() - start_time
            start_time = time.time()
            
            # Optimize channel copying for the common case of expanding a single channel
            if x_channels == 1 and self.in_features > 1:
                # Broadcasting is much faster than a loop for this case
                new_x = x.expand(-1, self.in_features, -1, -1)
            else:
                # Copy data from the input tensor, handling both cases (more or fewer channels)
                for i in range(min(x_channels, self.in_features)):
                    new_x[:, i] = x[:, i % x_channels]
            
            x = new_x
            
            profile_times["copy_channels"] = time.time() - start_time
            start_time = time.time()
        
        # Apply padding based on boundary conditions
        torch.cuda.synchronize() if x.is_cuda else None  # Ensure accurate timing
        start_padding = time.time()
        padded_x = self.apply_boundary_padding(x)
        torch.cuda.synchronize() if padded_x.is_cuda else None
        profile_times["boundary_padding"] = time.time() - start_padding
        
        # Now perform convolution using F.conv2d
        torch.cuda.synchronize() if padded_x.is_cuda else None
        start_conv = time.time()
        output = F.conv2d(padded_x, self.kernel)
        torch.cuda.synchronize() if output.is_cuda else None
        profile_times["convolution"] = time.time() - start_conv
        
        start_time = time.time()
        
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
        
        profile_times["reshape_output"] = time.time() - start_time
        
        # Print profiling results
        print("\n--- Forward Pass Profiling Results ---")
        print(f"{'Operation':<25} {'Time (ms)':<15}")
        print("-" * 40)
        total_time = 0
        for op, t in profile_times.items():
            ms_time = t * 1000  # Convert to milliseconds
            total_time += ms_time
            print(f"{op:<25} {ms_time:<15.4f}")
        print("-" * 40)
        print(f"{'Total':<25} {total_time:<15.4f}")
        print("-----------------------------------\n")
        
        return output
# %% 
# Create the model
conv = Convolution2d(kernel_size=3, in_features=2, out_features=4)

# Run a forward pass with profiling
input_tensor = torch.randn(8, 2, 64, 64, 1)  # A 5D input tensor
output = conv(input_tensor)
# %%
