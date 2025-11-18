# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 6 Jan 2023
author: @vgopakum
Utilities required for training neural-pde surrogate models.
"""
import numpy as np 
import torch 
import torch.nn as nn 
import torch.functional as F 


# %%
##################################
# Normalisation Functions
##################################

def Normalisation(norm_strategy):
    if norm_strategy == 'Min-Max':
        normalizer = MinMax_Normalizer
    elif norm_strategy == 'Min-Max_variable':
        normalizer = MinMax_Normalizer_variable
    elif norm_strategy == 'Range':
        normalizer = Range_Normalizer
    elif norm_strategy == 'Gaussian':
        normalizer = Gaussian_Normalizer
    elif norm_strategy == 'Identity':
        normalizer = Identity_Normalizer
    else:
        # Fix: Raise proper error for invalid normalization type
        raise KeyError(f"Unknown normalization strategy: {norm_strategy}")
    return normalizer

# normalization, pointwise gaussian
class UnitGaussian_Normalizer(object):
    def __init__(self, x, eps=0.01):
        super(UnitGaussian_Normalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.a = torch.mean(x, 0)
        self.b = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.a) / (self.b + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.b + self.eps  # n
            mean = self.a
        else:
            if len(self.a.shape) == len(sample_idx[0].shape):
                std = self.b[sample_idx] + self.eps  # batch*n
                mean = self.a[sample_idx]
            if len(self.a.shape) > len(sample_idx[0].shape):
                std = self.b[:, sample_idx] + self.eps  # T*batch*n
                mean = self.a[:, sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()


# normalization, Gaussian
class Gaussian_Normalizer(object):
    def __init__(self, x, eps=0.01):
        super(Gaussian_Normalizer, self).__init__()

        self.a = torch.mean(x) #mean
        self.b = torch.std(x) #std
        self.eps = eps

    def encode(self, x):
        x = (x - self.a) / (self.b + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.b + self.eps)) + self.a
        return x

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()


# normalization, scaling by range
class Range_Normalizer(object):
    def __init__(self, x, low=-1.0, high=1.0):
        super(Range_Normalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        # Fix: Handle case where min == max (constant data)
        range_val = mymax - mymin
        range_val = torch.where(range_val == 0, torch.ones_like(range_val), range_val)
        
        self.a = (high - low) / range_val
        self.b = -self.a * mymax + high

    def encode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = self.a * x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = (x - self.b) / self.a
        x = x.view(s)
        return x

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()


# normalization, rangewise but each variable at a time.
class MinMax_Normalizer_variable(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(MinMax_Normalizer_variable, self).__init__()
        
        self.num_vars = x.shape[1]
        aa = []
        bb = []
        
        for ii in range(self.num_vars):
            min_u = torch.min(x[:, ii, :, :, :])
            max_u = torch.max(x[:, ii, :, :, :])
            
            # Fix: Handle case where min == max (constant data)
            range_val = max_u - min_u
            if range_val == 0:
                # For constant data, set scaling to identity with offset to target range
                aa.append(torch.tensor(1.0))
                bb.append(high - min_u)
            else:
                aa.append((high - low) / range_val)
                bb.append(-aa[ii] * max_u + high)
        
        self.a = torch.stack(aa)
        self.b = torch.stack(bb)
        self.low = low
        self.high = high

    def encode(self, x, var_idx=None):
        if var_idx is None:
            # Encode all variables
            for ii in range(self.num_vars):
                x[:, ii] = self.a[ii] * x[:, ii] + self.b[ii]
        else:
            # Encode only the specified variable
            if isinstance(var_idx, int):
                var_idx = [var_idx]  # Convert single index to list
            
            for ii, idx in enumerate(var_idx):
                if 0 <= idx < self.num_vars:
                    x[:, ii] = self.a[idx] * x[:, ii] + self.b[idx]
                else:
                    raise IndexError(f"Variable index {idx} out of range (0-{self.num_vars-1})")
        
        return x

    def decode(self, x, var_idx=None):
        if var_idx is None:
            # Decode all variables
            for ii in range(self.num_vars):
                x[:, ii] = (x[:, ii] - self.b[ii]) / self.a[ii]
        else:
            # Decode only the specified variable
            if isinstance(var_idx, int):
                var_idx = [var_idx]  # Convert single index to list
            
            for ii, idx in enumerate(var_idx):
                if 0 <= idx < self.num_vars:
                    x[:, ii] = (x[:, ii] - self.b[idx]) / self.a[idx]
                else:
                    raise IndexError(f"Variable index {idx} out of range (0-{self.num_vars-1})")
        
        return x
    
    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()
        return self

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()
        return self


class LogNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0, eps=0.01):
        super(LogNormalizer, self).__init__()

        self.num_vars = x.shape[1]
        aa = []
        bb = []
        
        for ii in range(self.num_vars):
            min_u = torch.min(x[:, ii, :, :, :])
            max_u = torch.max(x[:, ii, :, :, :])
            
            # Fix: Handle case where min == max (constant data)
            range_val = max_u - min_u
            if range_val == 0:
                aa.append(torch.tensor(1.0))
                bb.append(high - min_u)
            else:
                aa.append((high - low) / range_val)
                bb.append(-aa[ii] * max_u + high)
        
        self.a = torch.stack(aa)
        self.b = torch.stack(bb)
        self.eps = eps

    def encode(self, x):
        # First apply min-max normalization
        for ii in range(self.num_vars):
            x[:, ii] = self.a[ii] * x[:, ii] + self.b[ii] 
        
        # Then apply log transform
        x = torch.log(x + 1 + self.eps)
        return x

    def decode(self, x):
        # First reverse log transform
        x = torch.exp(x) - 1 - self.eps
        
        # Then reverse min-max normalization
        for ii in range(self.num_vars):
            x[:, ii] = (x[:, ii] - self.b[ii]) / self.a[ii] 
        
        return x

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()                                                                                                                                                                                                                         
        self.b = self.b.cpu()


#normalization, rangewise but across the full domain 
class MinMax_Normalizer(object):
    def __init__(self, x, low=-1.0, high=1.0):
        super(MinMax_Normalizer, self).__init__()
        mymin = torch.min(x)
        mymax = torch.max(x)

        # Fix: Handle case where min == max (constant data)
        range_val = mymax - mymin
        if range_val == 0:
            # For constant data, set scaling to identity with offset to target range
            self.a = torch.tensor(1.0)
            self.b = high - mymax
        else:
            self.a = (high - low) / range_val
            self.b = -self.a * mymax + high

    def encode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = self.a * x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = (x - self.b) / self.a
        x = x.view(s)
        return x

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()


#normalization, Identity_Normalizer - does nothing
class Identity_Normalizer(object):
    def __init__(self, x, low=-1.0, high=1.0):
        super(Identity_Normalizer, self).__init__()
        self.a = torch.tensor(0.0)
        self.b = torch.tensor(0.0)

    def encode(self, x):
        return x 

    def decode(self, x):
        return x

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()

# %% 
##################################
# Loss Functions
##################################

# loss function with rel/abs Lp loss
# https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/losses/data_losses.py
class LpLoss(object):
    r"""
    LpLoss provides the L-p norm between two 
    discretized d-dimensional functions. Note that 
    LpLoss always averages over the spatial dimensions.

    .. note :: 
        In function space, the Lp norm is an integral over the
        entire domain. To ensure the norm converges to the integral,
        we scale the matrix norm by quadrature weights along each spatial dimension.

        If no quadrature is passed at a call to LpLoss, we assume a regular 
        discretization and take ``1 / measure`` as the quadrature weights. 

    Parameters
    ----------
    d : int, optional
        dimension of data on which to compute, by default 1
    p : int, optional
        order of L-norm, by default 2
        L-p norm: [\sum_{i=0}^n (x_i - y_i)**p] ** (1/p)
    measure : float or list, optional
        measure of the domain, by default 1.0
        either single scalar for each dim, or one per dim

        .. note::

        To perform quadrature, ``LpLoss`` scales ``measure`` by the size
        of each spatial dimension of ``x``, and multiplies them with 
        ||x-y||, such that the final norm is a scaled average over the spatial
        dimensions of ``x``. 
    reduction : str, optional
        whether to reduce across the batch and channel dimensions
        by summing ('sum') or averaging ('mean')

        .. warning:: 

            ``LpLoss`` always reduces over the spatial dimensions according to ``self.measure``.
            `reduction` only applies to the batch and channel dimensions.
    eps : float, optional
        small number added to the denominator for numerical stability when using the relative loss

    Examples
    --------

    
    """

    def __init__(self, d=1, p=2, measure=1., reduction='sum', eps=1e-8):
        super().__init__()

        self.d = d
        self.p = p
        self.eps = eps
        
        allowed_reductions = ["sum", "mean"]
        assert reduction in allowed_reductions,\
        f"error: expected `reduction` to be one of {allowed_reductions}, got {reduction}"
        self.reduction = reduction

        if isinstance(measure, float):
            self.measure = [measure]*self.d
        else:
            self.measure = measure
    
    @property
    def name(self):
        return f"L{self.p}_{self.d}Dloss"
    
    def uniform_quadrature(self, x):
        """
        uniform_quadrature creates quadrature weights
        scaled by the spatial size of ``x`` to ensure that 
        ``LpLoss`` computes the average over spatial dims. 

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        quadrature : list
            list of quadrature weights per-dim
        """
        quadrature = [0.0]*self.d
        for j in range(self.d, 0, -1):
            quadrature[-j] = self.measure[-j]/x.size(-j)
        
        return quadrature

    def reduce_all(self, x):
        """
        reduce x across the batch according to `self.reduction`

        Params
        ------
        x: torch.Tensor
            inputs
        """
        if self.reduction == 'sum':
            x = torch.sum(x)
        else:
            x = torch.mean(x)
        
        return x

    def abs(self, x, y, quadrature=None):
        """absolute Lp-norm

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        quadrature : float or list, optional
            quadrature weights for integral
            either single scalar or one per dimension
        """
        #Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature]*self.d
        
        const = math.prod(quadrature)**(1.0/self.p)
        diff = const*torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                                              p=self.p, dim=-1, keepdim=False)

        diff = self.reduce_all(diff).squeeze()
            
        return diff

    def rel(self, x, y):
        """
        rel: relative LpLoss
        computes ||x-y||/(||y|| + eps)

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        """

        diff = torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                          p=self.p, dim=-1, keepdim=False)
        ynorm = torch.norm(torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False)

        diff = diff/(ynorm + self.eps)

        diff = self.reduce_all(diff).squeeze()
            
        return diff

    def __call__(self, y_pred, y, **kwargs):
        return self.rel(y_pred, y)

class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss


# Fully Connected Network or a Multi-Layer Perceptron
class MLP(nn.Module):
    def __init__(self, in_features, out_features, num_layers, num_neurons, activation=torch.tanh):
        super(MLP, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.num_neurons = num_neurons

        self.act_func = activation

        self.layers = nn.ModuleList()

        self.layer_input = nn.Linear(self.in_features, self.num_neurons)

        for ii in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.num_neurons, self.num_neurons))
        self.layer_output = nn.Linear(self.num_neurons, self.out_features)

    def forward(self, x):
        x_temp = self.act_func(self.layer_input(x))
        for dense in self.layers:
            x_temp = self.act_func(dense(x_temp))
        x_temp = self.layer_output(x_temp)
        return x_temp
    

#Sampling equidistantly from a 2D grid.
def sample_equidistant(grid, num_samples):
   
    #    Sample values from a 2D NumPy grid in an equidistant manner.

    # Args:
    #     grid (numpy.ndarray): A 3D NumPy array representing the num_sim, x_grid, y_grid
    #     num_samples (int): The total number of samples to retrieve equidistantly along each axis.

    # Returns:
    #     numpy.ndarray: A 3D NumPy array containing the sampled values from the equidistant points

    sims, height, width = grid.shape
    num_samples = int(np.sqrt(num_samples))
    # Generate equidistant x-coordinates
    x_coords = np.linspace(0, width - 1, num_samples, dtype=int)
    
    # Generate equidistant y-coordinates
    y_coords = np.linspace(0, height - 1, num_samples, dtype=int)
    
    # Create a meshgrid of the x and y coordinates
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    # Flatten the meshgrid to get the indices
    indices = np.vstack((yy.flatten(), xx.flatten())).T
    
    # Sample the grid using the indices
    samples = grid[:, indices[:, 0], indices[:, 1]]
    
    return samples
