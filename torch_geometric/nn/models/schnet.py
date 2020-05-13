import os
import os.path as osp
from math import pi as PI
import warnings

import torch
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear, ModuleList
import numpy as np

from torch_scatter import scatter
from torch_geometric.data.makedirs import makedirs
from torch_geometric.data import download_url, extract_zip
from torch_geometric.nn import radius_graph, MessagePassing

try:
    import schnetpack as spk
except ImportError:
    spk = None

qm9_target_dict = {
    0: 'dipole_moment',
    1: 'isotropic_polarizability',
    2: 'homo',
    3: 'lumo',
    4: 'gap',
    5: 'electronic_spatial_extent',
    6: 'zpve',
    7: 'energy_U0',
    8: 'energy_U',
    9: 'enthalpy_H',
    10: 'free_energy',
    11: 'heat_capacity',
}


class SchNet(torch.nn.Module):
    url = 'http://www.quantum-machine.org/datasets/trained_schnet_models.zip'

    def __init__(self, hidden_channels=128, num_filters=128,
                 num_interactions=6, num_gaussians=50, cutoff=10.0, mean=None,
                 std=None, atomref=None):
        super(SchNet, self).__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.mean = mean
        self.std = std

        self.embedding = Embedding(100, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, 1)

        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)

    @staticmethod
    def from_qm9_pretrained(root, dataset, target):
        if spk is None:
            raise ImportError(
                '`SchNet.from_qm9_pretrained` requires `schnetpack`.')

        root = osp.expanduser(osp.normpath(root))
        makedirs(root)
        folder = 'trained_schnet_models'
        if not osp.exists(osp.join(root, folder)):
            path = download_url(SchNet.url, root)
            extract_zip(path, root)
            os.unlink(path)

        name = f'qm9_{qm9_target_dict[target]}'
        path = osp.join(root, 'trained_schnet_models', name, 'split.npz')

        split = np.load(path)
        train_idx = split['train_idx']
        val_idx = split['val_idx']
        test_idx = split['test_idx']

        # Filter the splits to only contain characterized molecules.
        idx = dataset.data.idx
        assoc = idx.new_empty(idx.max().item() + 1)
        assoc[idx] = torch.arange(idx.size(0))

        train_idx = assoc[train_idx[np.isin(train_idx, idx)]]
        val_idx = assoc[val_idx[np.isin(val_idx, idx)]]
        test_idx = assoc[test_idx[np.isin(test_idx, idx)]]

        path = osp.join(root, 'trained_schnet_models', name, 'best_model')

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            state = torch.load(path, map_location='cpu')

        net = SchNet(hidden_channels=128, num_filters=128, num_interactions=6,
                     num_gaussians=50, cutoff=10.0,
                     atomref=dataset.atomref(target))

        net.embedding.weight = state.representation.embedding.weight

        for int1, int2 in zip(state.representation.interactions,
                              net.interactions):
            int2.mlp[0].weight = int1.filter_network[0].weight
            int2.mlp[0].bias = int1.filter_network[0].bias
            int2.mlp[2].weight = int1.filter_network[1].weight
            int2.mlp[2].bias = int1.filter_network[1].bias
            int2.lin.weight = int1.dense.weight
            int2.lin.bias = int1.dense.bias

            int2.conv.lin1.weight = int1.cfconv.in2f.weight
            int2.conv.lin2.weight = int1.cfconv.f2out.weight
            int2.conv.lin2.bias = int1.cfconv.f2out.bias

        net.lin1.weight = state.output_modules[0].out_net[1].out_net[0].weight
        net.lin1.bias = state.output_modules[0].out_net[1].out_net[0].bias
        net.lin2.weight = state.output_modules[0].out_net[1].out_net[1].weight
        net.lin2.bias = state.output_modules[0].out_net[1].out_net[1].bias

        net.mean = state.output_modules[0].standardize.mean.item()
        net.std = state.output_modules[0].standardize.stddev.item()
        if net.atomref is not None:
            net.atomref.weight = state.output_modules[0].atomref.weight

        return net, (dataset[train_idx], dataset[val_idx], dataset[test_idx])

    def forward(self, z, pos, batch=None):
        assert z.dim() == 1 and z.dtype == torch.long
        h = self.embedding(z)

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if self.atomref is not None:
            h = h + self.atomref(z)

        return h.sum(dim=0) if batch is None else scatter(h, batch, dim=0)

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super(InteractionBlock, self).__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff):
        super(CFConv, self).__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift
