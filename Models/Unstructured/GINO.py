#%%
import torch 
import torch.nn as nn 
from torch_geometric.nn import  MessagePassing, GCNConv, NNConv
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F

import operator
from functools import reduce
from functools import partial
from collections import OrderedDict

from GNNs import *
from Models.FNO_classic import *

class GINO(nn.Module):
    def __init__(self, in_channel, gno_width, out_channel, radius, fno_disc, fno_width, modes):
        super().__init__()

        #GNO params
        self.in_channel = in_channel
        self.gno_width = gno_width
        self.mid_width = gno_width*2
        self.out_channel = out_channel
        self.r = radius

        #FNO params
        self.modes = modes
        self.fno_width = fno_width
        self.num_vars = in_channel 

        self.Nx, self.Ny = fno_disc[0], fno_disc[1]
        x, y = np.linspace(0, 1, self.Nx), np.linspace(0, 1, self.Ny)#x-y discretisation
        xx, yy = np.meshgrid(x, y)
        x_fno = np.stack((xx.flatten(), yy.flatten())).T #Nodes, x-y pos. 
        x_fno = torch.tensor(x_fno, dtype=torch.float32)
        self.x_fno = x_fno.unsqueeze(0)
                
        self.gno_enc = GNO(self.in_channel, self.gno_width, self.mid_width, self.out_channel)
        self.fno = FNO_multi2d(1, 1, modes, modes, self.num_vars, self.fno_width)
        self.gno_dec = GNO(self.in_channel, self.gno_width, self.mid_width, self.out_channel)


    def get_graph(self, x_in, x_out=None):
        if x_out is None:
            x_in = x_in.squeeze()
            pwd = torch.cdist(x_in, x_in).squeeze()
            edge_index = torch.stack(torch.where(pwd <= self.r))
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=x_in.device)
            edge_attr = torch.cat([x_in[edge_index[0].T], x_in[edge_index[1].T]], dim=-1)
        else:
            x_in = x_in.squeeze()
            x_out = x_out.squeeze()
            N_in = x_in.shape[0]
            pwd = torch.cdist(x_in, x_out).squeeze()
            edge_index = torch.stack(torch.where(pwd <= self.r))
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=x_in.device)
            edge_attr = torch.cat([x_in[edge_index[0].T], x_out[edge_index[1].T]], dim=-1)
            edge_index[1, :] = edge_index[1, :] + N_in
        return edge_index.detach(), edge_attr.detach()

    def forward(self, u_in, x_in=None, x_out=None):
        """
        u_in: (N_in, C)
        x_in: (N_in, d) or None
        When both x_in and x_out are None, the first input is just vectices and there's no feature associated to the vertices.
        Synthesize features.
        x_out: (N_out, d) or None
        """
        u_enc = self.gno_enc(u_in, x_in, self.x_fno)
        u_enc = u_enc.reshape(u_enc.shape[0], u_enc.shape[-1], self.Nx, self.Ny, 1)
        u_fno = self.fno(u_enc)
        u_fno = u_fno.reshape(u_fno.shape[0], self.Nx*self.Ny, u_enc.shape[-1])
        u_dec = self.gno_dec(u_fno, self.x_fno, x_out)

        return u_dec


# #Example Usage
# #Input Grid
# x, y = np.linspace(0, 1, 32), np.linspace(0, 1, 32)#x-y discretisation
# xx, yy = np.meshgrid(x, y)
# x_in = np.stack((xx.flatten(), yy.flatten())).T #Nodes, x-y pos. 
# x_in = torch.tensor(x_in, dtype=torch.float32)
# x_in = x_in.unsqueeze(0)
# u_in = np.sin(xx) + np.cos(yy)#Arbitrary node features
# u_in = np.expand_dims(u_in, 0)#Adding an additional dimension for time. 
# u_in = u_in.reshape(u_in.shape[0], -1, 1)
# u_in = torch.tensor(u_in, dtype=torch.float32)

# #FNO Grid discretisation. 
# x, y = np.linspace(0, 1, 16), np.linspace(0, 1, 16)#x-y discretisation
# xx, yy = np.meshgrid(x, y)
# x_fno = np.stack((xx.flatten(), yy.flatten())).T #Nodes, x-y pos. 
# x_fno = torch.tensor(x_fno, dtype=torch.float32)
# x_fno = x_fno.unsqueeze(0)

# #Output Grid
# x, y = np.linspace(0, 1, 64), np.linspace(0, 1, 64)#x-y discretisation
# xx, yy = np.meshgrid(x, y)
# x_out = np.stack((xx.flatten(), yy.flatten())).T #Nodes, x-y pos. 
# x_out = torch.tensor(x_out, dtype=torch.float32)
# x_out = x_out.unsqueeze(0)

# model = GINO(in_channel=1, gno_width=32, out_channel=1, radius=0.05, fno_disc=[16,16], fno_width=32, modes=4)
# out = model(u_in, x_in, x_out)
# print(f'Input shape: {u_in.shape, x_in.shape, x_out.shape}, Output shape: {out.shape}')

# %% 