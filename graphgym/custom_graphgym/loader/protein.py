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

@register_loader('protein')
def load_dataset_protein_batch(format, name, dataset_dir):
    if format == 'PyG':
        if name == 'protein':
            dataset_raw = ProteinDataset(dataset_dir, numeric_columns=cfg.dataset.numeric_columns,
                                         label_column=cfg.dataset.label_column, rebuild=cfg.dataset.rebuild)
            return dataset_raw


class ProteinDataset(InMemoryDataset):
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

    def __init__(self, root, numeric_columns, label_column, remove_unlabelled_data=True, rebuild=False,
                 transform=None, pre_transform=None, pre_filter=None):

        self.rebuild = rebuild
        self.remove_unlabelled_data = remove_unlabelled_data
        if numeric_columns is None:
            raise ValueError('numeric_params is required for ProteinDataset')
        self.numeric_columns = numeric_columns
        self.label_column = label_column
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    @overrides
    def raw_file_names(self):
        # the files required for this dataset will be handled in raw_paths function
        pass

    @property
    @overrides
    def processed_file_names(self):
        return 'data.pt'

    @overrides
    def process(self):
        # read data into data list
        data_list = [self._process_single_datafile(protein_file) for protein_file in self.raw_paths['protein']]
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

    def _process_single_datafile(self, protein_filename: str) -> InMemoryDataset:
        '''
        Helper function for parse a single protein file to a torch.geometry.dataset using
        the given interaction file
        Args:
            protein_filename: the absolute path to the protein file

        Returns:
            torch.geometry.dataset
        '''

        positive_reference_list, negative_reference_list = None, None
        if self.label_column is None:
            # get positive/and negative reference
            with open(self.raw_paths['positive_reference']) as f:
                positive_reference_list = f.read().splitlines()
            with open(self.raw_paths['negative_reference']) as f:
                negative_reference_list = f.read().splitlines()

        # x is feature tensor for nodes
        x, mapping, y = self._load_node_csv(path=protein_filename,
                                            numeric_columns=self.numeric_columns,
                                            label_column=self.label_column,
                                            positive_protein_reference=positive_reference_list,
                                            negative_protein_reference=negative_reference_list)


        # read protein-protein-interaction data (the last file from self.raw_file_names)
        edge_index, edge_attr = self._load_edge_csv(path=self.raw_paths['interaction'], mapping=mapping)

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
        # had better save mapping somewhere else
        base_name = os.path.basename(protein_filename)
        name_without_suffix = os.path.splitext(base_name)[0]
        pd.DataFrame(mapping.items(), columns=['protein_id', 'integer_id']).to_csv(
            os.path.join(self.raw_dir, name_without_suffix + '_mapping.csv'), index=False)

        return data

    def _load_node_csv(self, path: str, numeric_columns: list, label_column: str,
                       encoders: object = None,
                       positive_protein_reference: 'iterable' = None,
                       negative_protein_reference: 'iterable' = None,
                       **kwargs):
        '''
        Helper function for loading protein file
        Args:
            path:
            numeric_columns:
            label_column:
            encoders:
            positive_protein_reference:
            negative_protein_reference:
            **kwargs:

        Returns:

        '''

        # file can be either csv or tsv
        if 'tsv' in os.path.basename(path):
            df = pd.read_csv(path, index_col=0, sep='\t', **kwargs)
        else:
            df = pd.read_csv(path, index_col=0, **kwargs)

        # if label is provided, use label column to create y
        if label_column is not None:
            y = df[label_column].to_numpy()
        else:
            # based on protein reference set to create group-true label
            y = np.where(df.index.isin(positive_protein_reference), 1,
                         (np.where(df.index.isin(negative_protein_reference), 0, pd.NA)))

        # filter out proteins without label if needed
        if self.remove_unlabelled_data:
            row_filter = ~pd.isnull(y)
            df = df[row_filter]
            y = y[row_filter]
            y = y.astype(float)


        # create mapping from protein ID to integer ID
        # the mapping dict is needed to convert results back to protein ID
        mapping = {index: i for i, index in enumerate(df.index.unique())}

        # convert protein ID to integer ID
        x = torch.tensor(df.loc[:, numeric_columns].values, dtype=torch.float)
        if encoders is not None:
            xs = [encoder(df[col]) for col, encoder in encoders.items()]
            x2 = torch.cat(xs, dim=-1).view(-1, 1)


            x = torch.hstack([x, x2])

        # remove last dimension in y to make it a 1D tensor
        y = torch.tensor(y).view(-1).to(dtype=torch.float)

        return x, mapping, y

    def _load_edge_csv(self, path: str, mapping: dict,
                       numeric_cols: list = None, encoders: dict = None, undirected: bool = True, **kwargs):
        if 'tsv' in os.path.basename(path):
            df = pd.read_csv(path, usecols=[0, 1, 2], sep='\t', **kwargs)
        else:
            df = pd.read_csv(path, usecols=[0, 1, 2], **kwargs)
        # only keep interactions related to proteins that in the protein dataset (i.e. in mapping keys)
        protein_data_acc = mapping.keys()
        df = df[df.iloc[:, 0].isin(protein_data_acc) & df.iloc[:, 1].isin(protein_data_acc)]

        #   convert protein ID to integer ID
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
            edge_index = torch.hstack((edge_index, edge_index_reverse))
            if edge_attr is not None:  # only create indirect edge_attr when edge_attr is not None
                edge_attr = torch.hstack((edge_attr, edge_attr))

        return edge_index, edge_attr

    @property
    def raw_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        downloading."""
        # generate necessary file paths
        raw_protein_dir = osp.join(self.root, 'raw/protein')
        file_names = [f for f in os.listdir(raw_protein_dir) if not f.startswith('.')]
        if len(file_names) == 0:
            raise Exception('no protein file detected!')
        else:
            protein_file_paths = [os.path.abspath(os.path.join(raw_protein_dir, file_name)) for file_name in file_names]

        raw_interaction_dir = osp.join(self.root, 'raw/interaction')
        file_names = [f for f in os.listdir(raw_interaction_dir) if not f.startswith('.')]
        if len(file_names) != 1:
            raise Exception('Wrong number of interaction file detected! Expecting exactly one file.')
        else:
            interaction_file_path = os.path.abspath(os.path.join(raw_interaction_dir, file_names[0]))

        return_raw_paths = protein_file_paths + [interaction_file_path]

        if cfg.dataset.label_column is None:
            raw_reference_dir = osp.join(self.root, 'raw/reference')
            positive_reference_path = os.path.join(raw_reference_dir, 'positive.txt')
            negative_reference_path = os.path.join(raw_reference_dir, 'negative.txt')
            return_raw_paths += [positive_reference_path, negative_reference_path]

        else:
            positive_reference_path = None
            negative_reference_path = None

        # our data folder have special structure the original raw_paths will only used for check if the files exist
        raw_paths_dict = {'protein': protein_file_paths,
                               'interaction': interaction_file_path,  # only one interaction file is allowed
                               'positive_reference': positive_reference_path,
                               'negative_reference': negative_reference_path}

        return raw_paths_dict

    @overrides
    def _download(self):
        # check if the protein files exist otherwise raise exception
        if not files_exist(self.raw_paths['protein']):
            raise Exception('Protein file not found! Not supported for automatic download.')
        # check if reference file exist when label_column is None otherwise raise exception
        if self.label_column is None:
            if ((not osp.exists(self.raw_paths['positive_reference']))
                    or (not osp.exists(self.raw_paths['negative_reference']))):
                raise Exception('Reference file not found while the label column in protein file not provided!'
                                'not supported for automatic download.'
                                ' Expecting positive.txt and negative.txt in reference folder')
        # check interaction file exist otherwise download from STRING database
        # elf.raw_paths['interaction'] is a str so len is 0 and files_exist will return False
        if not osp.exists(self.raw_paths['interaction']):
            os.makedirs(os.path.dirname(self.raw_paths['interaction']), exist_ok=True)
            self.download()

    @overrides
    def _process(self):
        f = osp.join(self.processed_dir, 'pre_transform.pt')
        if osp.exists(f) and torch.load(f) != _repr(self.pre_transform):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"delete '{self.processed_dir}' first")

        f = osp.join(self.processed_dir, 'pre_filter.pt')
        if osp.exists(f) and torch.load(f) != _repr(self.pre_filter):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in "
                "the pre-processed version of this dataset. If you want to "
                "make use of another pre-fitering technique, make sure to "
                "delete '{self.processed_dir}' first")



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
    parser.add_argument('-n', '--numeric-columns', nargs='+', required=True,
                        help='the numeric columns in the protein file used as feature')
    parser.add_argument('-l', '--label-col', required=False, default=None,
                        help='the label column in the protein file used as label. If not provided, '
                             'a reference folder contains "positive.txt" and "negative.txt" is required in "raw" folder')

    args = parser.parse_args()

    protein_dataset = ProteinDataset(root=args.root, numeric_columns=args.numeric_columns, label_column=args.label_col)
    # loader = DataLoader(protein_dataset)
    # for data in loader:
    #     data
    print('Successfully run')
