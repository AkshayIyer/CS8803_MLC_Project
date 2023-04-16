# Based heavily on 
# https://github.com/mengliu1998/DeeperGNN/blob/master/DeeperGNN/dagnn.py and
# https://github.com/Fung-Lab/MatDeepLearn_dev/blob/main/matdeeplearn/models/cgcnn.py

from matdeeplearn.common.registry import registry
from matdeeplearn.models.base_model import BaseModel
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from torch.nn import Linear
import torch.ones as ones
import torch.stack as stack
import torch.matmul as matmul
import torch.sigmoid as sigmoid
import torch.nn.functional as F

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    fill_value = 2. if improved else 1.
    num_nodes = int(edge_index.max()) + 1 if num_nodes is None else num_nodes
    if edge_weight is None:
        edge_weight = ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class Prop(MessagePassing):
    def __init__(self, num_classes, K, **kwargs):
        super(Prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.proj = Linear(num_classes, 1)
        
    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(edge_index, edge_weight, x.size(0), dtype=x.dtype)
        preds = []
        preds.append(x)
        for _ in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            preds.append(x)
        pps = stack(preds, dim=1)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        retain_score = sigmoid(retain_score)
        retain_score = retain_score.unsqueeze(1)
        out = matmul(retain_score, pps).squeeze()
        return out
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)
    
    def reset_parameters(self):
        self.proj.reset_parameters()

@registry.register_model("DAGNN")
class DAGNN(BaseModel):
    def __init__(
        self,
        data,
        dim1 = 64,
        dim2 = 10,
        dropout_rate = 0.0,
        gc_count = 3,
        **kwargs
    ):
        super(DAGNN, self).__init__()
        self.lin1 = Linear(data.num_features, dim1)
        self.lin2 = Linear(dim1, dim2)
        self.prop = Prop(dim2, gc_count)

        self.dropout_rate = dropout_rate

        # Determine output dimension length
        self.output_dim = 1 if data[0][self.target_attr].ndim == 0 else len(data[0][self.target_attr])

    @property
    def target_attr(self):
        return "y"

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return x