import warnings
from typing import Optional

import torch
from torch.utils.data import DataLoader

from torch_geometric.data.lightning.datamodule import LightningDataModule
from torch_geometric.graphgym import create_loader
from torch_geometric.graphgym.checkpoint import get_ckpt_dir
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.imports import pl
from torch_geometric.graphgym.loader import load_dataset, create_dataset
from torch_geometric.graphgym.logger import LoggerCallback
from torch_geometric.graphgym.model_builder import GraphGymModule
from torch_geometric.graphgym.register import register_train
from torch_geometric.loader import NeighborLoader

import copy


@register_train("GraphsageGraphGymDataModule")
class CustomGraphGymDataModule(LightningDataModule):
    def __init__(self):
        # create_loader call create_dataset function under the hood and it initializes some model parameter such as
        # dim_in. So, we customized how dataloader is created, create_dataset need to be called.
        dataset = create_dataset()
        super().__init__(has_val=True, has_test=True)

        data = dataset[0]
        self.loaders = []
        self.loaders.append(NeighborLoader(
            data, num_neighbors=cfg.train.neighbor_sizes[:cfg.gnn.layers_mp],
            input_nodes=data.train_mask,
            batch_size=cfg.train.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True))

        # Inspired by https://github.com/pyg-team/pytorch_geometric/blob/684f17958fe8c949644c1fe1d6b2d5f70e313411/examples/reddit.py
        # subgraph_loader is used for eval validation and test performance in each epoch
        self.loaders.append(NeighborLoader(
            copy.copy(data), num_neighbors=[-1],
            input_nodes=None,
            batch_size=cfg.train.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True))
