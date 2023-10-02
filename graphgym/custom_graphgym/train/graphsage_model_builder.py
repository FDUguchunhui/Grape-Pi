import time
from typing import Any, Dict, Tuple

import torch

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.model_builder import GraphGymModule
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import network_dict, register_network, register_train


class GraphsageGraphGymModule(GraphGymModule):
    def __init__(self, dim_in, dim_out, cfg):
        super().__init__(dim_in, dim_out, cfg)


    def training_step(self, batch, *args, **kwargs):
        logits, true = self(batch)
        loss, pred_score = compute_loss(logits, true)
        step_end_time = time.time()
        return dict(loss=loss, true=true, pred_score=pred_score.detach(),
                    step_end_time=step_end_time)


    def validation_step(self, batch, *args, **kwargs):
        logits, true = self(batch)
        logits, true = logits[batch.val_mask], true[batch.val_mask]
        loss, pred_score = compute_loss(logits, true)
        step_end_time = time.time()
        return dict(loss=loss, true=true, pred_score=pred_score.detach(),
                    step_end_time=step_end_time)


    def test_step(self, batch, *args, **kwargs):
        logits, true = self(batch)
        logits, true = logits[batch.test_mask], true[batch.test_mask]
        loss, pred_score = compute_loss(logits, true)

        step_end_time = time.time()
        return dict(loss=loss, true=true, pred_score=pred_score.detach(),
                    step_end_time=step_end_time)



@register_train("graphsage_create_model")
def create_model(to_device=True, dim_in=None, dim_out=None) -> GraphGymModule:
    r"""Create model for graph machine learning.

    Args:
        to_device (bool, optional): Whether to transfer the model to the
            specified device. (default: :obj:`True`)
        dim_in (int, optional): Input dimension to the model
        dim_out (int, optional): Output dimension to the model
    """
    dim_in = cfg.share.dim_in if dim_in is None else dim_in
    dim_out = cfg.share.dim_out if dim_out is None else dim_out
    # binary classification, output dim = 1
    if 'classification' == cfg.dataset.task_type and dim_out == 2:
        dim_out = 1

    model = GraphsageGraphGymModule(dim_in, dim_out, cfg)
    if to_device:
        model.to(torch.device(cfg.accelerator))
    return model
