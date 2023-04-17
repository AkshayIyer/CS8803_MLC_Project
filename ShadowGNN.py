import json
import ase
from ase import Atoms
import ase.neighborlist
import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential
import os.path as osp
import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.nn import  global_mean_pool, CGConv
from torch_geometric.data import DataLoader, Dataset, Data
from torch_geometric.loader import ShaDowKHopSampler
from torch_geometric.nn import SAGEConv, global_mean_pool, TransformerConv, GATv2Conv, ChebConv, TAGConv


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = ChebConv(in_channels, hidden_channels, 5)
        self.conv2 = TAGConv(hidden_channels, hidden_channels)
        self.conv3 = TransformerConv(hidden_channels, hidden_channels)
        self.lin1 = torch.nn.Linear(2 * hidden_channels, out_channels)
        self.softmax = torch.nn.Softmax(dim = 0)
        self.lin2 = torch.nn.Linear(out_channels, out_channels)
      

    def forward(self, x, edge_index, batch, root_n_id):
        
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=p1)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=p2, training=self.training)
        x = self.conv3(x, edge_index).relu()
        x = F.dropout(x, p=p3, training=self.training)

        # We merge both central node embeddings and subgraph embeddings:
        x = torch.cat([x[root_n_id], global_mean_pool(x, batch)], dim=-1)
        x = self.lin1(x)
        x  = self.softmax(x)
        x = self.lin2(x)
        
        return x.mean(axis = 0).unsqueeze(-1)