
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
from FNO import *

class GINO(nn.Module):
    def __init__(self, in_channel, gno_width, out_channel, r,  fno_width, modes):
        super().__init__()

        #GNO params
        self.in_channel = in_channel
        self.gno_width = gno_width
        self.mid_width = gno_width*2
        self.out_channel = out_channel

        #FNO params
        self.modes = modes
        self.fno_width = fno_width
        self.num_vars = in_channel 
        
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

        u_enc = self.gno_enc(u_in, x_in, x_out)
        #reshape
        u_fno = self.fno(u_enc)
        #reshape
        u_dec = self.gno_dec(u_fno, x_out, x_in)

        return u_dec
