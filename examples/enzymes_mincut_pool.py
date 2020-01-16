import os.path as osp
from math import ceil

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from mincut_pool import dense_mincut_pool

max_nodes = 620
average_nodes = 39  # only for ENZYMES


class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= max_nodes


path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'ENZYMES_d')
dataset = TUDataset(
    path,
    name='PROTEINS',
    transform=T.ToDense(max_nodes),
    use_edge_attr=True,
    pre_filter=MyFilter(),
    cleaned=True)

dataset = dataset.shuffle()
n = (len(dataset) + 9) // 10

test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
test_loader = DenseDataLoader(test_dataset, batch_size=32)
val_loader = DenseDataLoader(val_dataset, batch_size=32)
train_loader = DenseDataLoader(train_dataset, batch_size=32)


class Net(torch.nn.Module):
    def __init__(self, feat_in, n_classes, n_chan=32, mincut_mlp_chan=16):
        super(Net, self).__init__()

        num_nodes = ceil(0.5 * average_nodes)
        self.gnn1 = DenseSAGEConv(feat_in, n_chan, bias=False)
        self.hid1 = torch.nn.Linear(n_chan, mincut_mlp_chan)
        self.pool1 = torch.nn.Linear(mincut_mlp_chan, num_nodes)

        num_nodes = ceil(0.25 * average_nodes)
        self.gnn2 = DenseSAGEConv(n_chan, n_chan, bias=False)
        self.hid2 = torch.nn.Linear(n_chan, mincut_mlp_chan)
        self.pool2 = torch.nn.Linear(mincut_mlp_chan, num_nodes)

        self.gnn3 = DenseSAGEConv(n_chan, n_chan)

        self.lin1 = torch.nn.Linear(n_chan, 3*n_chan)
        self.lin2 = torch.nn.Linear(n_chan, n_classes)

    def forward(self, x, adj, mask=None):
        # adj norm
        # d = torch.einsum('ijk->ij', adj)
        # d = torch.sqrt(d)[:, None] + 1e-15
        # adj = (adj / d) / d.transpose(1, 2)
                
        # Block 1
        x = self.gnn1(x, adj, mask, add_loop=True)
        h = F.relu(self.hid1(x))
        s = self.pool1(h)
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)
        # x, adj, mc1, o1 = self.mincut1(x, adj, mask)

        # Block 2
        x = self.gnn2(x, adj, add_loop=True)
        h = F.relu(self.hid2(x))
        s = self.pool2(h)
        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)
        # x, adj, mc2, o2 = self.mincut2(x, adj)

        # Block 3
        x = self.gnn3(x, adj, add_loop=True)

        # Output block
        x = x.mean(dim=1)
        # x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), mc1 + mc2, o1 + o2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)


def train(epoch):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, link_loss, ent_loss = model(data.x, data.adj, data.mask)
        loss = F.nll_loss(output, data.y.view(-1)) + link_loss + ent_loss
        loss.backward()
        loss_all += data.y.size(0) * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred, link_loss, ent_loss = model(data.x, data.adj, data.mask)
        loss = F.nll_loss(pred, data.y.view(-1)) + link_loss + ent_loss
        correct += pred.max(dim=1)[1].eq(data.y.view(-1)).sum().item()

    return loss, correct / len(loader.dataset)


best_val_acc = test_acc = 0
best_val_loss = 999999999
patience = 50
for epoch in range(1, 15000):
    train_loss = train(epoch)
    _, train_acc = test(train_loader)
    val_loss, val_acc = test(val_loader)
    if val_loss < best_val_loss:
        test_loss, test_acc = test(test_loader)
        best_val_acc = val_acc
        patience = 50
    else:
        patience -= 1
        if patience == 0:
            break
    print('Epoch: {:03d}, Train Loss: {:.3f}, '
          'Train Acc: {:.3f}, Val Loss: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(
                  epoch, train_loss, train_acc, val_loss, val_acc, test_acc))
