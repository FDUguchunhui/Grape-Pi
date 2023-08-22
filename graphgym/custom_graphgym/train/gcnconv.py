import logging
import time

import numpy as np
import torch
from sklearn import metrics

from torch_geometric.graphgym.checkpoint import (
    clean_ckpt,
    load_ckpt,
    save_ckpt,
)
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_ckpt_epoch, is_eval_epoch

def train_epoch(logger, loader, model, optimizer, scheduler):
    model.train()
    time_start = time.time()
    total_loss = 0
    total_pred_score = None
    total_true = None
    for data in loader:
        optimizer.zero_grad()
        data.to(torch.device(cfg.accelerator))
        pred, true = model(data)
        pred, true = pred[data.train_mask], true[data.train_mask]
        loss, pred_score = compute_loss(pred, true)
        loss.backward()
        optimizer.step()
        total_loss += loss
        if total_pred_score is None:
            total_pred_score = pred_score.detach().cpu()
        else:
            total_pred_score = torch.cat([total_pred_score, pred_score.detach().cpu()])

        if total_true is None:
            total_true = true.detach().cpu()
        else:
            total_true = torch.cat([total_true, true.detach().cpu()])


    logger.update_stats(true=total_true,
                        pred=total_pred_score,
                        loss=total_loss.item(),     # the loss for gcnconv will be too small if being averaged
                        lr=scheduler.get_last_lr()[0],
                        time_used=time.time() - time_start,
                        params=cfg.params)
    scheduler.step()


def eval_val_epoch(logger, loader, model):
    model.eval()
    time_start = time.time()

    total_loss = 0
    total_pred_score = None
    total_true = None
    for data in loader:
        data.to(torch.device(cfg.accelerator))
        pred, true = model(data)
        pred, true = pred[data.val_mask], true[data.val_mask]
        loss, pred_score = compute_loss(pred, true)
        total_loss += loss
        if total_pred_score is None:
            total_pred_score = pred_score.detach().cpu()
        else:
            total_pred_score = torch.cat([total_pred_score, pred_score.detach().cpu()])

        if total_true is None:
            total_true = true.detach().cpu()
        else:
            total_true = torch.cat([total_true, true.detach().cpu()])

        logger.update_stats(true=total_true,
                            pred=total_pred_score,
                            loss=total_loss.item(),
                            lr=0,
                            time_used=time.time() - time_start,
                            params=cfg.params)

def eval_test_epoch(logger, loader, model):
    model.eval()
    time_start = time.time()

    total_loss = 0
    total_pred_score = None
    total_true = None


    for data in loader:
        data.to(torch.device(cfg.accelerator))
        pred, true = model(data)
        pred, true = pred[data.test_mask], true[data.test_mask]
        loss, pred_score = compute_loss(pred, true)
        total_loss += loss
        if total_pred_score is None:
            total_pred_score = pred_score.detach().cpu()
        else:
            total_pred_score = torch.cat([total_pred_score, pred_score.detach().cpu()])

        if total_true is None:
            total_true = true.detach().cpu()
        else:
            total_true = torch.cat([total_true, true.detach().cpu()])


        logger.update_stats(true=total_true,
                            pred=total_pred_score,
                            loss=total_loss.item(),
                            lr=0,
                            time_used=time.time() - time_start,
                            params=cfg.params)

@register_train('gcnconv')
def train(loggers, loaders, model, optimizer, scheduler):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler,
                                cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)


    num_splits = len(loggers)
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler)
        loggers[0].write_epoch(cur_epoch)
        eval_val_epoch(loggers[1], loaders[1], model)
        loggers[1].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            eval_test_epoch(loggers[2], loaders[2], model)
            loggers[2].write_epoch(cur_epoch)
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in %s', cfg.run_dir)
