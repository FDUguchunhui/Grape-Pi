import sys
from typing import List, Dict

import h5py
import numpy as np
import pandas as pd
import os.path as osp
import os

import torch
from overrides import overrides

import warnings
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import torch_geometric.transforms as T
from typing import Any, Callable, List, Optional, Tuple, Union
from torch_geometric.data.dataset import to_list, _repr
from torch_geometric.data.dataset import files_exist
from torch_geometric import  seed_everything


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

    def __init__(self, root, numeric_columns, label_column,
                 include_seq_embedding=False, remove_unlabeled_data=True, rebuild=False,
                 transform=None, pre_transform=None, pre_filter=None,
                 num_val=0.2, num_test=0.1, seed=1234):

        self.include_seq_embedding = include_seq_embedding
        self.rebuild = rebuild
        self.remove_unlabeled_data = remove_unlabeled_data
        if numeric_columns is None:
            raise ValueError('numeric_params is required for ProteinDataset')
        self.numeric_columns = numeric_columns
        self.label_column = label_column
        self.num_val = num_val
        self.num_test = num_test

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        seed_everything(seed)

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
        x, mapping, y, unlabeled_mask = self._load_node_csv(path=protein_filename,
                                            numeric_columns=self.numeric_columns,
                                            label_column=self.label_column,
                                            positive_protein_reference=positive_reference_list,
                                            negative_protein_reference=negative_reference_list)


        # read protein-protein-interaction data (the last file from self.raw_file_names)
        edge_index, edge_attr = self._load_edge_csv(path=self.raw_paths['interaction'], mapping=mapping)

        data = Data(x=x, edge_index=edge_index, split=1, edge_attr=edge_attr, y=y, unlabeled_mask=unlabeled_mask)

        # calculate node degree
        # deg = degree(data.edge_index[0], data.num_nodes).reshape(-1, 1)
        # x = torch.cat((x, deg), 1)

        # split data into train, val, and test set
        # rename the attributes to match name convention in GraphGym
        split_transformer = T.RandomNodeSplit(split='train_rest', num_splits=1,
                                              num_val=self.num_val,
                                              num_test=self.num_test)
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

        # if choose to keep unlabelled data, create a mask for unlabeled data
        # make the unlabeled data to have label 0. This is because the code in
        # package torch_geometric only accept label in range 0 to num_classes - 1
        unlabeled_mask = None
        # filter out proteins without label if needed
        if self.remove_unlabeled_data:
            row_filter = ~pd.isnull(y)
            df = df[row_filter]
            y = y[row_filter]

        else:
            # create a mask for proteins without label
            unlabeled_mask = pd.isnull(y)
            # fill the unlabeled proteins with 0
            y = np.nan_to_num(y, nan=0)
            unlabeled_mask = torch.tensor(unlabeled_mask, dtype=torch.int).view(-1)


        # after no NaN in y, convert y to integer
        y = y.astype(int)

        # create mapping from protein ID to integer ID
        # the mapping dict is needed to convert results back to protein ID
        mapping = {index: i for i, index in enumerate(df.index.unique())}

        # convert protein ID to integer ID
        x = torch.tensor(df.loc[:, numeric_columns].values, dtype=torch.float)
        if encoders is not None:
            xs = [encoder(df[col]) for col, encoder in encoders.items()]
            x2 = torch.cat(xs, dim=-1).view(-1, 1)
            x = torch.hstack([x, x2])

        # add protein sequence embedding
        if  self.include_seq_embedding:
            # create an empty tensor to store sequence embedding
            seq_embedding = torch.zeros((len(mapping), 1024))
            with h5py.File(self.raw_paths['embedding'], 'r') as file:
                # iterate through the file to create
                for accession, index  in mapping.items():
                    # if the protein is not in the embedding file, skip
                    # it will have an embedding vector of all zeros
                    if accession not in file:
                        continue
                    seq_embedding[index] = torch.from_numpy(np.array(file[accession]))
            x = torch.hstack([x, seq_embedding])

        # remove last dimension in y to make it a 1D tensor
        y = torch.tensor(y).view(-1).to(dtype=torch.int)

        return x, mapping, y, unlabeled_mask

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
    def raw_paths(self) -> Dict[str, List[str]]:
        r"""The absolute filepaths that must be present in order to skip
        downloading."""
        # our data folder have special structure the original raw_paths will only used for check if the files exist
        raw_paths_dict = {}

        # generate necessary file paths
        raw_protein_dir = osp.join(self.root, 'raw/protein')
        file_names = [f for f in os.listdir(raw_protein_dir) if not f.startswith('.')]
        if len(file_names) == 0:
            raise Exception('no protein file detected!')
        else:
            protein_file_paths = [os.path.abspath(os.path.join(raw_protein_dir, file_name)) for file_name in file_names]
        raw_paths_dict['protein'] = protein_file_paths

        raw_interaction_dir = osp.join(self.root, 'raw/interaction')
        file_names = [f for f in os.listdir(raw_interaction_dir) if not f.startswith('.')]
        if len(file_names) != 1:
            raise Exception('Wrong number of interaction file detected! Expecting exactly one file.')
        else:
            interaction_file_path = os.path.abspath(os.path.join(raw_interaction_dir, file_names[0]))
        raw_paths_dict['interaction'] = interaction_file_path

        if self.include_seq_embedding:
            raw_embedding_dir = osp.join(self.root, 'raw/embedding')
            file_names = [f for f in os.listdir(raw_embedding_dir) if not f.startswith('.')]
            if len(file_names) != 1:
                raise Exception('Wrong number of interaction file detected! Expecting exactly one file.')
            else:
                embedding_file_path = os.path.abspath(os.path.join(raw_embedding_dir, file_names[0]))
            raw_paths_dict['embedding'] = embedding_file_path

        if self.label_column is None:
            raw_reference_dir = osp.join(self.root, 'raw/reference')
            positive_reference_path = os.path.join(raw_reference_dir, 'positive.txt')
            negative_reference_path = os.path.join(raw_reference_dir, 'negative.txt')
            raw_paths_dict['positive_reference'] = positive_reference_path
            raw_paths_dict['negative_reference'] = negative_reference_path

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
    parser.add_argument('-i', '--include-seq-embedding', required=False, default=False, action='store_true',
                        help='whether to include protein sequence embedding')
    parser.add_argument('-u', '--remove-unlabeled-data', required=False, default=True, action='store_true',
                        help='whether to remove unlabelled data')
    parser.add_argument('-b', '--rebuild', required=False, default=False, action='store_true',
                        help='whether to rebuild the dataset even if the processed file already exist')

    args = parser.parse_args()

    protein_dataset = ProteinDataset(root=args.root,
                                     numeric_columns=args.numeric_columns,
                                     label_column=args.label_col,
                                     include_seq_embedding=args.include_seq_embedding,
                                     remove_unlabeled_data=args.remove_unlabeled_data,
                                     rebuild=args.rebuild)
    # loader = DataLoader(protein_dataset)
    # for data in loader:
    #     data
    print('Successfully run')
