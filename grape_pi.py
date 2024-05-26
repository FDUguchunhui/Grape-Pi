import glob
import logging
import os
import subprocess

from torch_geometric.data import DataLoader
from torch_geometric.loader import NeighborLoader

import custom_graphgym  # noqa, register custom modules
import torch
from torch_geometric import seed_everything
import argparse
from torch_geometric.graphgym.config import (
    cfg,
    load_cfg,
)
from torch_geometric.graphgym.register import train_dict
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device

import logger
import pandas as pd
from custom_graphgym.loader.protein import ProteinDataset


def main():

    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=True,
                        help='The configuration file path.')
    parser.add_argument('--checkpoint', dest= 'checkpoint', type=str, default=None,
                        help='The checkpoint file path.')
    parser.add_argument('--threshold', dest='threshold', type=float, default=0.9,
                        help='The threshold to determine the unconfident proteins based on raw protein probability.')
    parser.add_argument('--num-promoted', dest='num_promoted', type=int, default=100,
                        help='The number, N, of proteins to be promoted '
                             'Top N Proteins will be selected based on the prediction probability from unconfident'
                             'proteins.')
    parser.add_argument('--output', dest='output', type=str, default=None,
                        help='The output file path to save the promoted proteins.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')


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
    # raise NotImplementedError
        raise NotImplementedError('Only "graphsage" is implemented yet')

    # Load all data without splitting validation and test set
    dataset = ProteinDataset(root=cfg.dataset.dir, rebuild=True,
                                                 numeric_columns=cfg.dataset.numeric_columns,
                                                 label_column=cfg.dataset.label_column,
                                                 remove_unlabeled_data=False,
                                                 num_val=0, num_test=0)


    pw = cfg.num_workers > 0
    if cfg.train.sampler == "full_batch" or len(dataset) > 1:
        dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size,
                                  shuffle=True, num_workers=cfg.num_workers,
                                  pin_memory=True, persistent_workers=pw)

    elif cfg.train.sampler == "neighbor":
        dataloader = NeighborLoader(
            dataset[0], num_neighbors=cfg.train.neighbor_sizes[:cfg.gnn.layers_mp],
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
    raw_dir = os.path.join(cfg.dataset.dir, 'raw')
    # Use glob to get all files in the directory that have "mapping" in the name
    mapping = pd.read_csv(glob.glob(os.path.join(raw_dir, '*mapping*'))[0])
    mapping = dict(zip(mapping['integer_id'], mapping['protein_id']))

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
        # ../saved_results/gastric-graphsage/epoch=199-step=4800.ckpt'
    else:
        # if no checkpoint is provided, retrain the model
        # run the graphgym.py script to train the model with provided configuration
        # create a directory to store the results that using the same as configuration file
        subprocess.run(['python', 'graphgym.py', '--cfg', args.cfg_file, 'train.ckpt_clean', 'True'])

        # load the last checkpoint
        # the path is output_path/0/ckpt/*.ckpt
        output_path = os.path.join(cfg.out_dir, args.cfg_file.split('/')[-1].replace('.yaml', ''))
        os.makedirs(output_path, exist_ok=True)
        checkpoint_dir = os.path.join(output_path,  str(cfg.seed), 'ckpt')
        checkpoint_file = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))[0]
        model.load_state_dict(torch.load(checkpoint_file)['state_dict'])

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
    raw_dir = os.path.join(cfg.dataset.dir, 'raw/protein')
    # list all files in the directory
    dat = pd.read_csv(glob.glob(f'{raw_dir}/[!.]*')[0])
    # get the name of first column
    protein_id_col = dat.columns[0]
    # combine the test_proteins_df with the original protein data
    all_proteins_df = all_proteins_df.merge(dat, left_on='accession', right_on=protein_id_col, how='inner')


    # get the unconfident proteins and their prediction probability
    all_proteins_df = all_proteins_df.loc[:, [protein_id_col, 'pred_prob', cfg.dataset.label_column]]
    all_proteins_df.columns = ['accession', 'pred_prob','label']
    # slice unlabeled protein if cfg.dataset.label_column is NaN
    unlabeled_protein_df = all_proteins_df[pd.isnull(all_proteins_df['label'])]

    unconfident_protein = unlabeled_protein_df[(unlabeled_protein_df.iloc[:, 1] < args.threshold)]

    # get the top N proteins to be promoted
    promoted_proteins = unconfident_protein.sort_values(by='pred_prob', ascending=False).head(args.num_promoted)
    if args.output is None:
        output_dir = os.path.join(cfg.dataset.dir, 'promoted_proteins.csv')

    else:
        output_dir = args.output

    promoted_proteins.to_csv(output_dir, index=False)
    # output message that the process completed and promoted proteins are saved
    print(f'Top {args.num_promoted} proteins are saved in {output_dir}')



if __name__ == '__main__':
    main()
