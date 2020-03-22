import time
import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GNNExplainer
from torch.nn import Sequential, Linear

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin = Sequential(Linear(10, 10))
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
x, edge_index = data.x, data.edge_index

# for epoch in range(1, 201):
#     model.train()
#     optimizer.zero_grad()
#     log_logits = model(x, edge_index)
#     loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()
model.load_state_dict(torch.load('/Users/rusty1s/Desktop/model.pt'))

explainer = GNNExplainer(model, epochs=200)
t = time.perf_counter()
node_feat_mask, edge_mask = explainer.explain_node(10, x, edge_index)
print(time.perf_counter() - t)
# for edge_mask in edge_masks:
#     mask = edge_mask > 0
#     print(edge_mask[mask])
# explainer.visualize_subgraph(10, edge_index, edge_masks)
