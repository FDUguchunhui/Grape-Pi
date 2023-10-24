import sys
from typing import List

import numpy as np
import pandas as pd
import os.path as osp
import os

import torch
from overrides import overrides

import warnings
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.graphgym.register import register_loader
import torch_geometric.transforms as T
from torch_geometric.utils import degree
from typing import Any, Callable, List, Optional, Tuple, Union
from torch_geometric.data.dataset import to_list, _repr
from torch_geometric.graphgym.config import cfg
from torch_geometric.data.dataset import files_exist

from torch_geometric import  seed_everything
seed_everything(1234)

class ProteinBatchDataset(InMemoryDataset):
    '''
    Args:
        root: the root directory to look for files
        numeric_columns: the numeric columns in the protein file used as feature
        label_column: the label column in the protein file used as label. If not provided,
        a reference folder contains "positive.txt" and "negative.txt" is required in "raw" folder
        remove_unlabelled_data: whether to remove unlabelled data
        rebuild: whether to rebuild the dataset even if the processed file already exist
        transform: the transform function to apply to the dataset
        pre_transform: the pre_transform function to apply to the dataset
        pre_filter: the pre_filter function to apply to the dataset
    '''

    def __init__(self, root, undirected=True, rebuild=False,
                 transform=None, pre_transform=None, pre_filter=None):

        self.rebuild = rebuild
        self.undirected = undirected
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    @overrides
    def raw_file_names(self):
        # the files required for this dataset will be handled in raw_paths function
        return ['proteins.csv', 'interactions.csv', 'graph_labels.txt']

    @property
    @overrides
    def processed_file_names(self):
        return 'data.pt'

    @overrides
    def process(self):

        # read protein data file
        if 'tsv' in os.path.basename(self.raw_paths[0]):
            df_protein = pd.read_csv(self.raw_paths[0], index_col=0, sep='\t')
        else:
            df_protein = pd.read_csv(self.raw_paths[0], index_col=0)
        sample_ids = df_protein.columns.to_list()  # get ID of each samples
        # create mapping from protein ID to integer ID for all proteins in the dataset
        mapping = {index: i for i, index in enumerate(df_protein.index.unique())}


        # read interaction data file
        if 'tsv' in os.path.basename(self.raw_paths[1]):
            df_interaction = pd.read_csv(self.raw_paths[1], usecols=[0, 1, 2], sep='\t')
        else:
            df_interaction = pd.read_csv(self.raw_paths[1], usecols=[0, 1, 2])
        # only keep interactions related to proteins that in the protein dataset (i.e. in mapping keys)
        protein_data_acc = mapping.keys()
        df_interaction = df_interaction[df_interaction.iloc[:, 0].isin(protein_data_acc) & df_interaction.iloc[:, 1].isin(protein_data_acc)]

        # read graph label file
        with open(self.raw_paths[2]) as f:
            graph_labels = f.read().splitlines()
        graph_labels = [int(label) for label in graph_labels] # convert label to integer
        # self.num_classes = len(set(graph_labels)) # get number of classes
        self.num_graphs = len(graph_labels) # get number of graphs

        # Create a list with 60% 'train', 20% 'validation', and 20% 'mask'
        labels = ['train'] * int(0.6 * self.num_graphs) + ['validation'] * int(0.2 * self.num_graphs) + ['mask'] * int(
            0.2 * self.num_graphs)

        # Shuffle the list
        np.random.shuffle(labels)
        # Convert to one-hot encoded format
        one_hot_train = torch.tensor([1 if label == 'train' else 0 for label in labels], dtype=torch.int)
        one_hot_validation = torch.tensor([1 if label == 'val' else 0 for label in labels], dtype=torch.int)
        one_hot_mask = torch.tensor([1 if label == 'mask' else 0 for label in labels], dtype=torch.int)


        data_list = []

        for idx, sample in enumerate(sample_ids):
            series_sample = df_protein[sample]
            # series_sample.dropna(inplace=True)  # drop rows with NA values # there are bugs need to be fixed
            x = torch.tensor(series_sample.values, dtype=torch.float).unsqueeze(1)
            # implement encoder later
            pass

            #   convert protein ID to integer ID
            src = [mapping[index] for index in df_interaction.iloc[:, 0]]
            dst = [mapping[index] for index in df_interaction.iloc[:, 1]]
            edge_index = torch.tensor([src, dst])

            edge_attr = None
            #
            # if encoders is not None:
            #     edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
            #     edge_attr = torch.cat(edge_attrs, dim=-1)

            # add reversed edges if the graph is undirected
            if self.undirected:
                edge_index_reverse = torch.tensor([dst, src])
                edge_index = torch.hstack((edge_index, edge_index_reverse))
                if edge_attr is not None:  # only create indirect edge_attr when edge_attr is not None
                    edge_attr = torch.hstack((edge_attr, edge_attr))

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=graph_labels[idx],
                        train_mask=one_hot_train[idx], val_mask=one_hot_validation[idx], test_mask=one_hot_mask[idx])
            data_list.append(data)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # combine list of data into a big data object
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @overrides
    def download(self):
        raise Exception('Download is not supported for this type of dataset')


    @overrides
    def _process(self):
        f = osp.join(self.processed_dir, 'pre_transform.pt')
        if osp.exists(f) and torch.load(f) != _repr(self.pre_transform):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"delete '{self.processed_dir}' first or use rebuild=True.")

        f = osp.join(self.processed_dir, 'pre_filter.pt')
        if osp.exists(f) and torch.load(f) != _repr(self.pre_filter):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in "
                "the pre-processed version of this dataset. If you want to "
                "make use of another pre-fitering technique, make sure to "
                "delete '{self.processed_dir}' first or use rebuild=True.")



        if files_exist(self.processed_paths) and not self.rebuild:  # pragma: no cover
            return

        if self.log and 'pytest' not in sys.modules:
            if self.rebuild:
                print('Rebuilding...', file=sys.stderr)
            else:
                print('Processing...', file=sys.stderr)

        if self.rebuild:
            os.makedirs(self.processed_dir, exist_ok=True)
        else:
            os.makedirs(self.processed_dir)

        self.process()

        path = osp.join(self.processed_dir, 'pre_transform.pt')
        torch.save(_repr(self.pre_transform), path)
        path = osp.join(self.processed_dir, 'pre_filter.pt')
        torch.save(_repr(self.pre_filter), path)

        if self.log and 'pytest' not in sys.modules:
            print('Done!', file=sys.stderr)



# for testing purpose
if __name__ == '__main__':
    import argparse
    from torch_geometric.loader import DataLoader

    parser = argparse.ArgumentParser(
        prog='ImportProteinData')

    parser.add_argument('-r', '--root', required=True,
                        help='the root directory to look for files')

    args = parser.parse_args()

    protein_dataset = ProteinBatchDataset(root=args.root, rebuild=True)
    # loader = DataLoader(protein_dataset)
    # for data in loader:
    #     data
    print('Successfully run')
