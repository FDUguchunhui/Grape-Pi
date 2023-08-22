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
        pred, true = pred[:cfg.train.batch_size], true[:cfg.train.batch_size]
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
        # only use labeled true
    logger.update_stats(true=total_true,
                        pred=total_pred_score,
                        loss=total_loss.item()/len(total_true),
                        lr=scheduler.get_last_lr()[0],
                        time_used=time.time() - time_start,
                        params=cfg.params)
    scheduler.step()


def eval_val_test_epoch(val_logger, test_logger, loader, model):
    model.eval()
    time_start = time.time()

    val_total_loss = test_total_loss = 0
    val_total_pred_score = test_total_pred_score = None
    val_total_true = test_total_true = None


    for data in loader:
        data.to(torch.device(cfg.accelerator))
        pred, true = model(data)
        val_pred, val_true = pred[data.val_mask], true[data.val_mask]
        test_pred, test_true = pred[data.test_mask], true[data.test_mask]

        val_loss, val_pred_score = compute_loss(val_pred, val_true)
        test_loss, test_pred_score = compute_loss(test_pred, test_true)

        val_total_loss += val_loss
        if val_total_pred_score is None:
            val_total_pred_score = val_pred_score.detach().cpu()
        else:
            val_total_pred_score = torch.cat([val_total_pred_score, val_pred_score.detach().cpu()])

        if val_total_true is None:
            val_total_true = val_true.detach().cpu()
        else:
            val_total_true = torch.cat([val_total_true, val_true.detach().cpu()])

        test_total_loss += test_loss
        if test_total_pred_score is None:
            test_total_pred_score = test_pred_score.detach().cpu()
        else:
            test_total_pred_score = torch.cat([test_total_pred_score, test_pred_score.detach().cpu()])

        if test_total_true is None:
            test_total_true = test_true.detach().cpu()
        else:
            test_total_true = torch.cat([test_total_true, test_true.detach().cpu()])

        val_logger.update_stats(
            true=val_total_true,
            pred=val_total_pred_score,
            loss=val_total_loss.item()/len(val_total_true),
            lr=0,
            time_used=time.time() - time_start,
            params=cfg.params)
        test_logger.update_stats(
            true=test_total_true,
            pred=test_total_pred_score,
            loss=test_total_loss.item() / len(test_total_true),
            lr=0,
            time_used=time.time() - time_start,
            params=cfg.params)

@register_train('graphsage')
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
        eval_val_test_epoch(loggers[1], loggers[2], loaders[1], model)
        # logger val and test together
        loggers[1].write_epoch(cur_epoch)
        loggers[2].write_epoch(cur_epoch)
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in %s', cfg.run_dir)
