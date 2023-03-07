import torch

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


@register_loss('protein_semisupervised')
def loss_protein(pred, true):
    if cfg.model.loss_fun == 'protein_semisupervised':
        weight = [32.9722, 9.0297]
        criterion = torch.nn.CrossEntropyLoss(weight=weight, reduction='mean')
        loss = criterion(pred, true)
        return loss, pred


