import torch
from sklearn.utils import class_weight
import numpy as np
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss
import torch.nn as nn


@register_loss('protein_semisupervised')
def loss_protein(pred, true):
    if cfg.model.loss_fun == 'binary_cross_entropy_with_weight':
        # automatically calculate weight based on proportion of postivie/negative labels
        pos_weight = torch.tensor([cfg.model.loss_pos_weight], dtype=torch.float).to(cfg.accelerator)
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
        true = true.float()
        loss = bce_loss(pred, true)
        return loss, torch.sigmoid(pred)

        # if cfg.model.loss_pos_weight == -1:
        #     y_np = true.detach().cpu().numpy()
        #     pos_weight = class_weight.compute_class_weight('balanced', classes=np.unique(y_np), y=y_np)
        #     pos_weight = pos_weight[1]/pos_weight[0]
        #     pos_weight = torch.tensor([pos_weight], dtype=torch.float).to(cfg.accelerator)
        # else:
