import torch

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


@register_loss('protein_semisupervised')
def loss_protein(pred, true):
    if cfg.model.loss_fun == 'protein_semisupervised':
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        mask = torch.logical_and(data.val_mask, data.loss_mask)
        loss = criterion(out[mask], data.y[mask])  # Compute the loss
        loss = criterion(pred, true)
        return loss, pred


