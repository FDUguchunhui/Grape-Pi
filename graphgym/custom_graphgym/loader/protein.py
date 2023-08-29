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
from torch_geometric.utils import degree
from typing import Any, Callable, List, Optional, Tuple, Union
from torch_geometric.data.dataset import to_list
from torch_geometric.graphgym.config import cfg

@register_loader('protein')
def load_dataset_protein_batch(format, name, dataset_dir):
    if format == 'PyG':
        if name == 'protein':
            numeric_params = cfg.dataset.numeric_params
            dataset_raw = ProteinBatchDataset(dataset_dir, numeric_params=numeric_params)
            return dataset_raw


class ProteinBatchDataset(InMemoryDataset):
    def __init__(self, root, numeric_params, transform=None, pre_transform=None,
                 pre_filter=None):
        # some naming convention here
        # dir: absolute path to a folder
        # file: absolute path to a file
        # filename: file name without any path information
        self.numeric_params = numeric_params
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        # generate necessary file paths
        raw_protein_dir = osp.join(self.root, 'raw/protein')
        protein_dir_list = os.listdir(raw_protein_dir)
        raw_interaction_dir = osp.join(self.root, 'raw/interaction')
        interaction_dir_list = os.listdir(raw_interaction_dir)
        raw_reference_dir = osp.join(self.root, 'raw/reference')
        reference_dir_list = os.listdir(raw_reference_dir)
        if len(interaction_dir_list) > 1:
            raise ValueError('more than one interaction file is detected!')
        elif len(interaction_dir_list) == 0:
            raise ValueError('no interaction file is detected!')
        elif len(protein_dir_list) == 0:
            raise ValueError('no protein file detected!')
        elif len(reference_dir_list) != 2:
            raise ValueError('wrong number of reference file detected! (expect 2)')
        raw_paths = {'protein': protein_dir_list,
                     'interaction': interaction_dir_list[0], # only one interaction file is allowed
                     'reference': reference_dir_list}
        return raw_paths

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):

        # read data into data list
        data_list = [self._process_single_datafile(protein_file) for protein_file in self.raw_paths[:-3]]

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
        # protein_file = osp.join(self.raw_dir, 'protein', protein_filename)

        # if 'tsv' in os.path.basename(protein_filename):
        #     protein_dat = pd.read_csv(protein_filename, index_col=0, sep='\t')    # use the first column for protein ID
        # else:
        #     protein_dat = pd.read_csv(protein_filename, index_col=0)    # use the first column for protein ID


        numeric_cols = self.numeric_params

        # get positive/and negative reference
        positive_reference_file = osp.join(self.raw_dir, 'reference', 'positive.txt')
        with open(positive_reference_file) as f:
            positive_reference_list = f.read().splitlines()

        negative_reference_file = osp.join(self.raw_dir, 'reference', 'negative.txt')
        with open(negative_reference_file) as f:
            negative_reference_list = f.read().splitlines()

        # x is feature tensor for nodes
        # y is label tensor with y=1 if protein in the confident positive proteins, y=0 if in the confident negative proteins, -1 otherwise
        x, mapping, y = self._load_node_csv(path=protein_filename,
                                            numeric_cols=numeric_cols,
                                            protein_reference=[positive_reference_list,
                                                               negative_reference_list])

        # read protein-protein-interaction data (the last file from self.raw_file_names)
        interaction_dir = self.raw_paths[-3] # the third to last is interaction file path
        edge_index, edge_attr = self._load_edge_csv(path=interaction_dir, mapping=mapping, numeric_cols=None)

        # the mask used when calculating loss
        # Since the original graphgym only accept y as a dim=2 for output
        # add another variable to help distinguish those unlabelled nodes (y = 0 and loss_mask = 0)
        # loss_mask = np.where(y == -1, 0, 1)
        # y = torch.where(y == -1, 0, y)

        data = Data(x=x, edge_index=edge_index, split=1, edge_attr=edge_attr, y=y)

        # calculate node degree
        # deg = degree(data.edge_index[0], data.num_nodes).reshape(-1, 1)
        # x = torch.cat((x, deg), 1)

        # split data into train, val, and test set
        # rename the attributes to match name convention in GraphGym
        split_transformer = T.RandomNodeSplit(split='train_rest', num_splits=1,
                                              num_val=cfg.dataset.split[1],
                                              num_test=cfg.dataset.split[2])
        data = split_transformer(data)

        # store mapping information for translate back protein integer ID back to string ID
        data.mapping = mapping

        return data

    def _load_node_csv(self, path: str, numeric_cols: list = None, encoders: object = None,
                       protein_reference: '[iterable, iterable]' = None, **kwargs):
        df = pd.read_csv(path, index_col=0, **kwargs)

        # extract feature doesn't need encoder
        # based on protein reference set to create group-true label
        confident_positive_proteins, confident_negative_proteins = protein_reference
        y = np.where(df.index.isin(confident_positive_proteins), 1,
                     (np.where(df.index.isin(confident_negative_proteins), 0, -1)))

        # filter out proteins without label
        df = df[y != -1]
        y = y[y != -1]

        mapping = {index: i for i, index in enumerate(df.index.unique())}

        x = torch.tensor(df.loc[:, numeric_cols].values, dtype=torch.float)
        if encoders is not None:
            xs = [encoder(df[col]) for col, encoder in encoders.items()]
            x2 = torch.cat(xs, dim=-1).view(-1, 1)
            x = torch.hstack([x, x2])

        y = torch.tensor(y).view(-1).to(dtype=torch.long)

        return x, mapping, y

    def _load_edge_csv(self, path: str, mapping: dict,
                       numeric_cols: list, encoders: dict = None, undirected: bool = True, **kwargs):
        if 'tsv' in os.path.basename(path):
            df = pd.read_csv(path, usecols=[0, 1, 2], sep='\t', **kwargs)
        else:
            df = pd.read_csv(path, usecols=[0, 1, 2], **kwargs)
        # only keep interactions related to proteins that in the protein dataset (i.e. in mapping keys)
        protein_data_acc = mapping.keys()
        df = df[df.iloc[:, 0].isin(protein_data_acc) & df.iloc[:, 1].isin(protein_data_acc)]

        src = [mapping[index] for index in df.iloc[:, 0]]
        dst = [mapping[index] for index in df.iloc[:, 1]]
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
            if edge_attr is not None:   # only create indirect edge_attr when edge_attr is not None
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
        protein_files = files['protein']
        interaction_file = files['interaction']
        reference_files = files['reference']
        protein_paths = [osp.join(self.raw_dir, 'protein', f) for f in to_list(protein_files)]
        interaction_path = osp.join(self.raw_dir, 'interaction', interaction_file)
        reference_paths = [osp.join(self.raw_dir, 'reference', f) for f in to_list(reference_files)]
        return protein_paths + [interaction_path] + reference_paths

# for testing purpose
if __name__ == '__main__':
    import argparse
    from torch_geometric.loader import DataLoader

    parser = argparse.ArgumentParser(
        prog='ImportProteinData')

    parser.add_argument('-r', '--root', required=True,
                        help='the root directory to look for files')

    args = parser.parse_args()

    protein_dataset = ProteinBatchDataset(root=args.root)
    # loader = DataLoader(protein_dataset)
    # for data in loader:
    #     data
    print('Successfully run')
