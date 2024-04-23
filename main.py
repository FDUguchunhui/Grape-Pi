import glob
import logging
import os

from torch_geometric.data import DataLoader, NeighborSampler

import graphgym.custom_graphgym # noqa, register custom modules
import torch
from torch_geometric import seed_everything
import argparse
from torch_geometric.graphgym.config import (
    cfg,
    dump_cfg,
    load_cfg,
    set_out_dir,
    set_run_dir,
)
from torch_geometric.graphgym.register import train_dict
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device

from graphgym import logger
import pandas as pd
from src import protein_loader

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=True,
                        help='The configuration file path.')
    parser.add_argument('--data', dest='data_file', type=str, required=True,
                        help='The data file path.')
    parser.add_argument('--checkpoint', dest= 'checkpoint', type=str, default=None,
                        help='The checkpoint file path.')
    parser.add_argument('-threshold', dest='threshold', type=float, default=0.9)


    # Load cmd line args
    args = parser.parse_args()
    # Load config file
    load_cfg(cfg, args)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    # Repeat for different random seeds
    logger.set_printing()

    seed_everything(cfg.seed)
    auto_select_device()
    # Set machine learning pipeline

    model, datamodule = None, None
    # use the right customized datamodule and graphgymmodule
    if cfg.train.grape_pi == 'graphsage':
        datamodule = train_dict['graphsage_graphgym_datamodule']()
        model = train_dict['graphsage_create_model']()
    elif cfg.train.grape_pi == 'gcnconv':
        datamodule = GraphGymDataModule()
        model = train_dict['gcnconv_create_model']()

    # Load all data without splitting validation and test set
    dataset = protein_loader.ProteinBatchDataset(root=args.data_file, rebuild=True,
                                                 numeric_columns=cfg.dataset.numeric_columns,
                                                 label_column=cfg.dataset.label_column,
                                                 remove_unlabeled_data=False,
                                                 num_val=cfg.dataset.split[1], num_test=cfg.dataset.split[2])


    pw = cfg.num_workers > 0
    if cfg.train.sampler == "full_batch" or len(dataset) > 1:
        dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size,
                                  shuffle=True, num_workers=cfg.num_workers,
                                  pin_memory=True, persistent_workers=pw)

    elif cfg.train.sampler == "neighbor":
        loader_train = NeighborSampler(
            dataset[0], sizes=cfg.train.neighbor_sizes[:cfg.gnn.layers_mp],
            batch_size=cfg.train.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True)

    # Print model info
    logging.info(model)
    logging.info(cfg)
    cfg.params = params_count(model)
    logging.info('Num parameters: %s', cfg.params)

    # get the dictionary mapping from global node index to original protein accession
    # get the file in subdirectory "raw" of args.data and has "mapping" in the name
    # Construct the path to the "raw" subdirectory of args.data
    raw_dir = os.path.join(args.data, 'raw')
    # Use glob to get all files in the directory that have "mapping" in the name
    mapping = glob.glob(os.path.join(raw_dir, '*mapping*'))
    mapping = dict(zip(mapping['integer_id'], mapping['protein_id']))

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
        # ../saved_results/gastric-graphsage/epoch=199-step=4800.ckpt'

    else:
        # todo: train the model


    # prediction on the unconfident proteins
    model.eval()
    accession_indices = []
    all_pred_prob = []

    with torch.no_grad():
        for batch in dataloader:

            logits, true = model(batch)
            logits, true, index = logits[:batch.batch_size], true[:batch.batch_size], batch.n_id[:batch.batch_size]
            logits = logits.squeeze(-1)
            pred_prob = torch.nn.functional.sigmoid(logits)
            # retrieve the index of the protein in the original file
            accession_indices += index.tolist()
            all_pred_prob += pred_prob.tolist()

    # convert the node index to original protein ID
    accession = [mapping[key] for key in accession_indices]

    # create a dataframe to store the prediction probability
    # accession is the original protein ID and not necessarily UniProt accession and can be any ID
    all_proteins_df = pd.DataFrame({'accession': accession, 'pred_prob': all_pred_prob})

    # read the original protein data
    # get the file in subdirectory "raw" of args.data and has "protein" in the name
    # Construct the path to the "raw/protein" subdirectory of args.data
    raw_dir = os.path.join(args.data, 'raw/protein')
    protein = glob.glob(os.path.join(raw_dir))
    dat = pd.read_csv(protein[0])
    # get the name of first column
    protein_id_col = dat.columns[0]
    # combine the test_proteins_df with the original protein data
    all_proteins_df = all_proteins_df .merge(dat, left_on='accession', right_on='protein_id_col', how='inner')

    all_proteins_df = all_proteins_df.loc[:, ['accession', 'gene_symbol', 'pred_prob', 'protein_probability', 'protein_probability_soft_label', "hard_label", 'mRNA_TPM''']]
    all_proteins_df.columns = ['accession', 'gene_symbol', 'pred_prob', 'raw_prob', 'soft_label', 'hard_label', 'mRNA']
    # filter only keep soft_label between 0.3 and 0.7
    test_proteins_df = all_proteins_df[(all_proteins_df['soft_label'] > 0.3) & (all_proteins_df['soft_label'] < 0.7)]

    confident_protein = test_proteins_df[(test_proteins_df['raw_prob'] >= 0.9)]['accession']
    test_proteins_df = test_proteins_df[(test_proteins_df['raw_prob'] < args.threshold)]


