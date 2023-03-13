from typing import List

import numpy as np
import pandas as pd
import os.path as osp
import os

import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.graphgym.register import register_loader
import torch_geometric.transforms as T
from typing import Any, Callable, List, Optional, Tuple, Union
from torch_geometric.data.dataset import to_list

@register_loader('protein_batch')
def load_dataset_protein_batch(format, name, dataset_dir):
    if format == 'PyG':
        if name == 'protein_batch':
            dataset_raw = ProteinBatchDataset(dataset_dir)
            return dataset_raw


class ProteinBatchDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        # some naming convention here
        # dir: absolute path to a folder
        # file: absolute path to a file
        # filename: file name without any path information
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raw_protein_dir = osp.join(self.root, 'raw/protein')
        protein_dir_list = os.listdir(raw_protein_dir)
        raw_interaction_dir = osp.join(self.root, 'raw/interaction')
        interaction_dir_list = os.listdir(raw_interaction_dir)
        if len(interaction_dir_list) > 1:
            raise ValueError('more than one interaction file is detected!')
        elif len(interaction_dir_list) == 0:
            raise ValueError('no interaction file is detected!')
        elif len(protein_dir_list) == 0:
            raise ValueError('no protein file detected!')
        return protein_dir_list + interaction_dir_list

    @property
    def processed_file_names(self):
        # raw_protein_dir = osp.join(self.root, 'raw
        #
        # /protein')
        # protein_dir_list = os.listdir(raw_protein_dir)
        # data_file_list = ['processed_' + protein_file for protein_file in protein_dir_list]
        # return data_file_list
        return ['data.pt']

    def process(self):

        # read data into data list
        data_list = [self._process_single_datafile(protein_file) for protein_file in self.raw_paths[:-1]]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # combine list of data into a big data object
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _process_single_datafile(self, protein_filename: str) -> InMemoryDataset:
        '''
        Helper function for parse a single protein file to a torch.geometry.dataset using
        the given interaction file
        Args:
            protein_filename:

        Returns:
            torch.geometry.dataset
        '''
        # read protein mass spectrometry data
        protein_file = osp.join(self.raw_dir, 'protein', protein_filename)
        protein_dat = pd.read_csv(protein_file, index_col='protein.Accession')

        # generate confident protein reference set
        perc_1 = np.percentile(protein_dat['protein.falsePositiveRate'], 1)
        perc_99 = np.percentile(protein_dat['protein.falsePositiveRate'], 99)
        confident_positive_proteins = protein_dat[(protein_dat['protein.falsePositiveRate'] <= perc_1)].index.to_list()
        confident_negative_proteins = protein_dat[(protein_dat['protein.falsePositiveRate'] >= perc_99)].index.to_list()

        numeric_cols = ['protein.avgMass', 'protein.MatchedProducts', 'protein.matchedPeptides',
                        'protein.digestPeps', 'protein.seqCover(%)', 'protein.MatchedPeptideIntenSum',
                        'protein.top3MatchedPeptideIntenSum', 'protein.MatchedProductIntenSum',
                        'protein.sumNumBYCalc', 'protein.sumNumBYPepFrag1', 'protein.falsePositiveRate']

        # x is feature tensor for nodes
        # y is label tensor with y=1 if protein in the confident positive proteins, y=0 if in the confident negative proteins, -1 otherwise
        x, mapping, y = self._load_node_csv(path=protein_file, index_col='protein.Accession',
                                            numeric_cols=numeric_cols,
                                            protein_reference=[confident_positive_proteins,
                                                               confident_negative_proteins])

        # read protein-protein-interaction data (the last file from self.raw_file_names)
        interaction_dir = self.raw_paths[-1]
        edge_index, edge_attr = self._load_edge_csv(path=interaction_dir, src_index_col='protein1_acc',
                                                    dst_index_col='protein2_acc', mapping=mapping,
                                                    numeric_cols=['combined_score'])

        # the mask used when calculating loss
        loss_mask = np.where(y == -1, 0, 1)
        y = torch.where(y == -1, 0, y)
        data = Data(x=x, edge_index=edge_index, split=1, edge_attr=edge_attr, y=y,
                    loss_mask=torch.tensor(loss_mask, dtype=torch.long))

        # split data into train, val, and test set
        # rename the attributes to match name convention in GraphGym
        split_transformer = T.RandomNodeSplit(split='train_rest', num_splits=1, num_val=0.2, num_test=0.2)
        data = split_transformer(data)

        # set mask only to labeled data for using GraphGym framework
        # data.train_mask = torch.logical_and(data.train_mask, data.loss_mask)
        # data.val_mask = torch.logical_and(data.val_mask, data.loss_mask)
        # data.test_mask = torch.logical_and(data.test_mask, data.loss_mask)
        data.train = data.train_mask
        data.val = data.val_mask
        data.test = data.test_mask

        return data

    def _load_node_csv(self, path: str, index_col: str, numeric_cols: list, encoders: object = None,
                       protein_reference: '[iterable, iterable]' = None, **kwargs):
        df = pd.read_csv(path, index_col=index_col, **kwargs)
        mapping = {index: i for i, index in enumerate(df.index.unique())}

        # extract feature doesn't need encoder
        x = torch.tensor(df.loc[:, numeric_cols].values, dtype=torch.float)
        if encoders is not None:
            xs = [encoder(df[col]) for col, encoder in encoders.items()]
            x2 = torch.cat(xs, dim=-1).view(-1, 1)
            x = torch.hstack([x, x2])

        # based on protein reference set to create group-true label
        confident_positive_proteins, confident_negative_proteins = protein_reference
        y = np.where(df.index.isin(confident_positive_proteins), 1,
                     (np.where(df.index.isin(confident_negative_proteins), 0, -1)))
        y = torch.tensor(y).view(-1).to(dtype=torch.long)

        return x, mapping, y

    def _load_edge_csv(self, path: str, src_index_col: str, dst_index_col: str, mapping: dict,
                       numeric_cols: list, encoders: dict = None, undirected: bool = True, **kwargs):
        df = pd.read_csv(path, **kwargs)

        # only keep interactions related to proteins that in the protein dataset (i.e. in mapping keys)
        protein_data_acc = mapping.keys()
        df = df[df.protein1_acc.isin(protein_data_acc) & df.protein2_acc.isin(protein_data_acc)]

        src = [mapping[index] for index in df[src_index_col]]
        dst = [mapping[index] for index in df[dst_index_col]]
        edge_index = torch.tensor([src, dst])

        edge_attr = None
        if numeric_cols is not None:
            edge_attr = torch.tensor(df.loc[:, numeric_cols].values, dtype=float)

        if encoders is not None:
            edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
            edge_attr = torch.cat(edge_attrs, dim=-1)

        # add reversed edges if the graph is undirected
        if undirected:
            edge_index_reverse = torch.tensor([dst, src])
            edge_index = torch.cat([edge_index, edge_index_reverse], dim=-1)
            edge_attr = torch.vstack([edge_attr, edge_attr])

        return edge_index, edge_attr

    @property
    def raw_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        downloading."""
        files = self.raw_file_names
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        protein_files = files[:-1]
        interaction_file = files[-1]
        protein_paths = [osp.join(self.raw_dir, 'protein', f) for f in to_list(protein_files)]
        interaction_path = osp.join(self.raw_dir, 'interaction', interaction_file)
        return protein_paths + [interaction_path]

# for testing purpose
if __name__ == '__main__':
    import argparse
    from torch_geometric.loader import DataLoader

    parser = argparse.ArgumentParser(
        prog='ImportProteinData')

    parser.add_argument('-r', '--root', required=True,
                        help='the root directory to look for files')
    parser.add_argument('-p', '--protein', required=True,
                        help='Protein mass spectrometry data file')
    parser.add_argument('-i', '--interaction', required=True,
                        help='Protein-protein-interaction file')

    args = parser.parse_args()

    protein_dataset = ProteinBatchDataset(root=args.root,
                                     protein_filename=args.protein,
                                     interaction_filename=args.interaction)
    loader = DataLoader(protein_dataset)
    for data in loader:
        data
    print('Successfully run')
