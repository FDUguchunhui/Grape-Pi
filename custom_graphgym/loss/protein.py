import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss



@register_loss('protein_semisupervised')
def loss_protein(pred, true):
    if cfg.model.loss_fun == 'binary_cross_entropy_with_weight':
        # automatically calculate weight based on proportion of postivie/negative labels
        pos_weight = torch.tensor([cfg.train.loss_pos_weight], dtype=torch.float).to(cfg.accelerator)
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
        true = true.float()
        loss = bce_loss(pred, true)
        return loss, torch.sigmoid(pred)

