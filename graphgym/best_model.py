import logging
import os

from matplotlib import pyplot as plt
from sklearn import metrics

import custom_graphgym # noqa, register custom modules
import torch
from torch_geometric import seed_everything
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (
    cfg,
    dump_cfg,
    load_cfg,
    set_out_dir,
    set_run_dir, get_fname,
)
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym import optim, compute_loss
from custom_graphgym.utils import logger
from torch_geometric.graphgym.checkpoint import load_ckpt



if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    load_cfg(cfg, args)
    # Set model checkpoint loading path
    fname = get_fname(args.cfg_file)
    cfg.out_dir = os.path.join(cfg.out_dir, fname)
    cfg.run_dir = os.path.join(cfg.out_dir, str(cfg.seed))
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    dump_cfg(cfg)
    set_printing()
    # Set configurations for each run
    cfg.seed = cfg.seed + 1
    seed_everything(cfg.seed)
    auto_select_device()
    # Set machine learning pipeline
    datamodule = GraphGymDataModule()
    model = create_model()

    # Print model info
    logging.info(model)
    logging.info(cfg)
    cfg.params = params_count(model)
    logging.info('Num parameters: %s', cfg.params)

    # loggers = logger.create_logger()
    optimizer = optim.create_optimizer(model.model.parameters(), cfg.optim)
    scheduler = optim.create_scheduler(optimizer, cfg.optim)

    # load model from checkpoint with best performance
    current_epoch = load_ckpt(model, optimizer, scheduler,
                            cfg.train.epoch_resume)

    # evaluate model on test dataset
    model.eval()
    total_loss = 0
    total_pred_score = None
    total_true = None
    for data in datamodule.loaders[2]:
        data.to(torch.device(cfg.accelerator))
        pred, true = model(data)
        # semi-supervised learning only training on labeled data
        mask = torch.logical_and(data.test_mask, data.loss_mask)
        loss, pred_score = compute_loss(pred[mask], true[mask])
        total_loss += loss
        if total_pred_score is None:
            total_pred_score = pred_score.detach().cpu()
        else:
            total_pred_score = torch.cat([total_pred_score, pred_score.detach().cpu()])

        if total_true is None:
            total_true = true[mask].detach().cpu()
        else:
            total_true = torch.cat([total_true, true[mask].detach().cpu()])

    # calculate AUC metrics
    fpr, tpr, thresholds = metrics.roc_curve(total_true, total_pred_score)
    roc_auc = metrics.auc(fpr, tpr)
    print('ROC AUC is', roc_auc)
    #  # Plotting
    roc_display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                          estimator_name='GNN estimator at epoch {epoch}'.format(epoch=current_epoch))
    roc_display.plot()
    plt.title('Test ROC')
    plt.show()

    # how to extract prediction?