import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d
import torch_geometric
from torch_geometric.nn import (
    Set2Set,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    CGConv,
)
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter

from models.torchmdnet_utils import (
    NeighborEmbedding,
    CosineCutoff,
    Distance,
    GatedEquivariantBlock,
    rbf_class_mapping,
    act_class_mapping,
)

class CGCNN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_edge_features,
        pre_layer_dim: int,
        post_layer_dim: int,
        num_pre_layers: int,
        num_post_layers: int,
        num_gc_layers: int,
        output_dim: int = 1,
        num_rbf: int = 32,
        trainable_rbf: bool = True,
        max_num_neighbors: int = 12,
        hidden_channels: int = 64,
        cutoff_lower: float = 0.,
        cutoff_upper: float = 5.0,
        max_atom_type: int = 100,
        pbc: bool = False,
        rbf_type="expnorm",
        pooling="global_mean_pool",
        act_fn="relu",
        noisy_nodes: bool = False,
        **kwargs
    ):
        super(CGCNN, self).__init__()
        self.num_features = num_features
        self.num_edge_features = num_edge_features
        self.pre_layers_dim = pre_layer_dim
        self.post_layers_dim = post_layer_dim
        self.num_pre_layers = num_pre_layers
        self.num_post_layers = num_post_layers
        self.num_gc_layers = num_gc_layers
        self.output_dim = output_dim
        self.pooling = pooling
        self.act_fn = act_fn

        self.gc_dim = self.pre_layers_dim
        self.post_fc_dim = self.pre_layers_dim

        self.hidden_channels = hidden_channels
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_atom_type = max_atom_type
        self.pbc = pbc
        self.noisy_nodes = noisy_nodes

        self.type_embedding = torch.nn.Embedding(self.max_atom_type, self.num_features)
        self.distance = Distance(
            cutoff_lower,
            cutoff_upper,
            max_num_neighbors=max_num_neighbors,
            return_vecs=True,
            loop=True,
            pbc=pbc,
        )
        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )

        self.initialize_layers()

    def initialize_layers(self):
        # set up pre GC fully connected layers
        self.pre_layers = torch.nn.ModuleList()
        self.pre_layers.append(Linear(self.num_features, self.pre_layers_dim))
        for i in range(1, self.num_pre_layers):
            self.pre_layers.append(Linear(self.pre_layers_dim, self.pre_layers_dim))

        # set up post GC fully connected layers
        self.post_layers = torch.nn.ModuleList()
        self.post_layers.append(Linear(self.post_fc_dim, self.post_layers_dim))
        for i in range(1, self.num_post_layers):
            self.post_layers.append(Linear(self.post_layers_dim, self.post_layers_dim))
        
        # output layer
        self.output_layer = Linear(self.post_layers_dim, self.output_dim)

        # set up GC layers
        self.gc_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        for i in range(self.num_gc_layers):
            self.gc_layers.append(CGConv(
                self.gc_dim, self.num_edge_features, aggr="mean", batch_norm=False
            ))
            self.bn_layers.append(BatchNorm1d(
                self.gc_dim, track_running_stats=True,
            ))

        # noisy nodes
        if self.noisy_nodes:
            self.nn_layers = torch.nn.ModuleList()
            self.nn_layers.append(Linear(self.post_layers_dim, self.post_layers_dim))
            self.nn_layers.append(Linear(self.post_layers_dim, self.post_layers_dim//2))
            self.nn_layers.append(Linear(self.post_layers_dim//2, 3))

    def forward_noisy_nodes(self, x):
        # nn layers
        for i in range(len(self.nn_layers)):
            x = self.nn_layers[i](x)
            x = getattr(F, self.act_fn)(x)

        return x
        

    def forward(self, data):
        x = self.type_embedding(data.x)
        edge_index, edge_weight, edge_vec = self.distance(data.pos, data.batch, data=data)

        assert (
            edge_vec is not None
        ), "Distance module did not return directional information"

        edge_attr = self.distance_expansion(edge_weight)

        # pre GC layers
        for i in range(self.num_pre_layers):
            x = self.pre_layers[i](x)
            x = getattr(F, self.act_fn)(x)

        # GC layers
        for i in range(self.num_gc_layers):
            x = self.gc_layers[i](x, edge_index, edge_attr)
            x = self.bn_layers[i](x)
            
        return x