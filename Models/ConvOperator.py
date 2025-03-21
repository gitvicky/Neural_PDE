
import torch 
import torch.nn as nn
import torch.nn.functional as F
count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)

from Utils.boundary_conditions import BoundaryManager

class Convolution2d(nn.Module):
    def __init__(self, kernel_size=3, features=2, init_type='random', boundary_type='periodic'):
        """
        Create a learnable convolution with configurable kernel and boundary conditions.
        
        Args:
            kernel_size (int): The size of the convolution kernel
            features (int): Number of feature channels
            init_type (str): Initialization strategy - 'random', 'identity', 'zer   os', or 'ones'
            boundary_type (str): Type of boundary condition - 'dirichlet', 'neumann', 'periodic', 'symmetric'
        """
        super(Convolution2d, self).__init__()
        
        # Create a parameter tensor for convolutional kernel
        self.kernel = nn.Parameter(torch.empty(features, features, kernel_size, kernel_size))
        self.features = features
        self.kernel_size = kernel_size
        
        # Initialize the kernel based on the specified strategy
        if init_type == 'random':
            # Xavier/Glorot initialization
            nn.init.xavier_uniform_(self.kernel)
        elif init_type == 'identity':
            # Initialize to approximate identity operation
            # (for convolution, this is a kernel with 1 in center, 0 elsewhere)
            nn.init.zeros_(self.kernel)
            # Set center pixel to 1 for each feature map
            for i in range(features):
                self.kernel[i, i, kernel_size//2, kernel_size//2] = 1.0
        elif init_type == 'zeros':
            # Initialize with zeros
            nn.init.zeros_(self.kernel)
        elif init_type == 'ones':
            # Initialize with ones
            nn.init.ones_(self.kernel)
        else:
            raise ValueError(f"Unknown initialization type: {init_type}")
        
        # Set up boundary manager from the boundary_conditions.py
        self.boundary_manager = BoundaryManager(kernel_size)
        self.boundary_manager.set_all_boundaries(boundary_type)
 
    def apply_boundary_padding(self, x):
        """
        Apply padding to the input tensor based on boundary conditions.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Padded tensor ready for convolution
        """
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
            x (torch.Tensor): Input tensor with shape (batch_size, features, height, width)
                              If None, return the current kernel
        
        Returns:
            torch.Tensor: Result of convolution with same shape as input x
        """
        if x is None:
            return self.kernel
        
        # Input shape validation and reshaping if needed
        original_shape = x.shape
        original_ndim = len(original_shape)
        
        # Handle different input shapes
        if original_ndim == 2:  # (height, width)
            x = x.unsqueeze(0).unsqueeze(0)  # -> (1, 1, height, width)
        elif original_ndim == 3:  # (batch, height, width)
            x = x.unsqueeze(1)  # -> (batch, 1, height, width)
        
        batch_size = x.shape[0]
        x_channels = x.shape[1]
        
        # Handle the case where input channels don't match kernel features
        if x_channels != self.features:
            # Either repeat the input channels or use only first 'features' channels
            if x_channels < self.features:
                # Repeat input channels to match feature count
                x = x.repeat(1, self.features // x_channels + 1, 1, 1)[:, :self.features, :, :]
            else:
                # Use only the first 'features' channels
                x = x[:, :self.features, :, :]
        
        # Apply padding based on boundary conditions
        padded_x = self.apply_boundary_padding(x)
        
        # Now perform convolution directly using F.conv2d
        output = F.conv2d(padded_x, self.kernel)
        
        # Restore original shape dimensionality
        if original_ndim == 2:
            output = output.squeeze(0).squeeze(0)  # -> (height, width)
        elif original_ndim == 3:
            output = output.squeeze(1)  # -> (batch, height, width)
        
        return output
