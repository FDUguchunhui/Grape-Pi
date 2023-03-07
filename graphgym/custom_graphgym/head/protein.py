import torch.nn as nn

from torch_geometric.graphgym.register import register_head
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import MLP, new_layer_config

@register_head('protein')
class ExampleNodeHead(nn.Module):
    '''Head of GNN, node prediction'''
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.layer_post_mp = MLP(
            new_layer_config(dim_in, dim_out, cfg.gnn.layers_post_mp,
                             has_act=False, has_bias=True, cfg=cfg))


    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        pred, label = batch.x, batch.y
        return pred, label
