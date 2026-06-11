# %% 
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class BoundaryManager:
    """
    A class to manage boundary conditions for generic convolution operations.
    """
    
    SUPPORTED_TYPES = [
        'dirichlet',    # Fixed value at boundary
        'neumann',      # Zero gradient at boundary
        'periodic',     # Periodic boundary
        'symmetric',    # Reflection at boundary (alias for symmetric)
        'free_slip',    # Zero normal, zero gradient tangential (for vector fields)
        'outflow'       # Zero gradient (alias for neumann)
    ]
    
    def __init__(self, kernel_size):
        """
        Initialize the boundary manager
        
        Args:
            kernel_size: Tuple (height, width) of kernel size or int
        """
        # Store kernel size
        if isinstance(kernel_size, int):
            self.kernel_height = kernel_size
            self.kernel_width = kernel_size
        else:
            self.kernel_height, self.kernel_width = kernel_size
            
        # Calculate padding size
        self.pad_left = self.kernel_width // 2
        self.pad_right = self.kernel_width // 2
        self.pad_top = self.kernel_height // 2
        self.pad_bottom = self.kernel_height // 2
        
        # Default boundary conditions (periodic on all sides)
        self.boundary_types = {
            'left': 'periodic',
            'right': 'periodic',
            'top': 'periodic',
            'bottom': 'periodic'
        }
        
        # Default boundary values (for Dirichlet)
        self.boundary_values = {
            'left': 0.0,
            'right': 0.0,
            'top': 0.0,
            'bottom': 0.0
        }
        
    def set_boundary_type(self, side, bc_type, value=0.0):
        """
        Set boundary condition type for a specific side
        
        Args:
            side: One of 'left', 'right', 'top', 'bottom'
            bc_type: Boundary condition type from SUPPORTED_TYPES
            value: Value for Dirichlet boundary condition (default 0.0)
        """
        if side not in ['left', 'right', 'top', 'bottom']:
            raise ValueError(f"Unknown side: {side}. Use 'left', 'right', 'top', or 'bottom'")
            
        if bc_type.lower() not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported boundary type: {bc_type}")
            
        self.boundary_types[side] = bc_type.lower()
        self.boundary_values[side] = value
        
    def set_all_boundaries(self, bc_type, value=0.0):
        """Set all boundaries to the same type and value"""
        for side in ['left', 'right', 'top', 'bottom']:
            self.set_boundary_type(side, bc_type, value)
    
    def _pad_side(self, result, bc_type, bc_value, pad):
        """Apply a single one-sided pad for a non-periodic boundary condition.

        ``free_slip`` (and any unrecognised type) is a no-op, matching the
        original behaviour where it had no padding branch.
        """
        if bc_type == 'dirichlet':
            return F.pad(result, pad, mode='constant', value=bc_value)
        elif bc_type in ('neumann', 'outflow'):
            return F.pad(result, pad, mode='replicate')
        elif bc_type == 'symmetric':
            return F.pad(result, pad, mode='reflect')
        return result

    def _pad_axis(self, result, lo_side, hi_side, lo_pad, hi_pad, dim):
        """
        Pad both sides of a single axis (``dim=3`` is the width axis, ``dim=2``
        the height axis).

        Periodic boundaries are a genuine two-sided wrap. Padding each side
        sequentially is WRONG: the second side's wrap is sliced from the
        already-padded tensor (duplicating the near edge instead of wrapping to
        the far edge). So when both sides of the axis are periodic we slice both
        wraps from the same unpadded tensor and concatenate once. Otherwise we
        pad each side independently, applying any periodic side first so a later
        non-periodic pad cannot pollute the wrap source.
        """
        lo_type = self.boundary_types[lo_side]
        hi_type = self.boundary_types[hi_side]

        # True two-sided periodic wrap for the all-periodic axis.
        if lo_pad > 0 and hi_pad > 0 and lo_type == 'periodic' and hi_type == 'periodic':
            lo_wrap = result.narrow(dim, result.size(dim) - lo_pad, lo_pad)
            hi_wrap = result.narrow(dim, 0, hi_pad)
            return torch.cat([lo_wrap, result, hi_wrap], dim=dim)

        # Per-side fallback. Apply periodic side(s) first.
        sides = [(lo_side, lo_type, lo_pad, True), (hi_side, hi_type, hi_pad, False)]
        sides.sort(key=lambda s: s[1] != 'periodic')
        for side, bc_type, pad_amt, is_lo in sides:
            if pad_amt <= 0:
                continue
            if bc_type == 'periodic':
                if is_lo:
                    wrap = result.narrow(dim, result.size(dim) - pad_amt, pad_amt)
                    result = torch.cat([wrap, result], dim=dim)
                else:
                    wrap = result.narrow(dim, 0, pad_amt)
                    result = torch.cat([result, wrap], dim=dim)
            else:
                if dim == 3:
                    pad = (pad_amt, 0, 0, 0) if is_lo else (0, pad_amt, 0, 0)
                else:
                    pad = (0, 0, pad_amt, 0) if is_lo else (0, 0, 0, pad_amt)
                result = self._pad_side(result, bc_type, self.boundary_values[side], pad)
        return result

    def pad_signal(self, signal):
        """
        Apply padding to a signal based on boundary conditions

        Args:
            signal: Input tensor of shape [batch, channel, height, width] or [height, width]

        Returns:
            Padded tensor with appropriate boundary conditions
        """
        # Store original shape to restore later
        original_ndim = signal.ndim

        # Convert to 4D if input is 2D
        if original_ndim == 2:
            signal = signal.unsqueeze(0).unsqueeze(0)

        result = signal

        # Width axis (dim=3): 'left'/'right'.  Height axis (dim=2): 'top'/'bottom'.
        result = self._pad_axis(result, 'left', 'right', self.pad_left, self.pad_right, dim=3)
        result = self._pad_axis(result, 'top', 'bottom', self.pad_top, self.pad_bottom, dim=2)

        # Restore original shape dimension
        if original_ndim == 2:
            result = result.squeeze(0).squeeze(0)

        return result
    
    def apply_convolution(self, signal, kernel):
        """
        Apply convolution with specified kernel and boundary conditions
        
        Args:
            signal: Input tensor of shape [height, width]
            kernel: Convolution kernel of shape [height, width]
            
        Returns:
            Convolution result
        """
        # Convert kernel to PyTorch format
        kernel_torch = kernel.unsqueeze(0).unsqueeze(0)
        
        # Pad signal with boundary conditions
        padded_signal = self.pad_signal(signal)
        
        # Convert to 4D if needed
        if len(padded_signal.shape) == 2:
            padded_signal = padded_signal.unsqueeze(0).unsqueeze(0)
        
        # Apply convolution
        result = F.conv2d(padded_signal, kernel_torch)
        
        # Restore original shape
        result = result.squeeze(0).squeeze(0)
        
        return result
