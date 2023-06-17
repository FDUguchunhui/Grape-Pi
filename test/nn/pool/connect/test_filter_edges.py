import torch

from torch_geometric.nn.pool.connect import FilterEdges
from torch_geometric.nn.pool.select import SelectOutput
from torch_geometric.testing import is_full_test


def test_filter_edges():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 1, 3, 2, 2]])
    edge_attr = torch.tensor([1, 2, 3, 4, 5, 6])
    batch = torch.tensor([0, 0, 1, 1])

    node_index = torch.tensor([1, 2])
    cluster_index = torch.tensor([0, 1])
    select_output = SelectOutput(node_index=node_index, num_nodes=4,
                                 cluster_index=cluster_index, num_clusters=2)
    connect = FilterEdges()
    output1 = connect(select_output, edge_index, edge_attr, batch)
    assert output1.edge_index.tolist() == [[0, 1], [0, 1]]
    assert output1.edge_attr.tolist() == [3, 5]
    assert output1.batch.tolist() == [0, 1]

    if is_full_test():
        jit = torch.jit.script(connect)
        output2 = jit(select_output, edge_index, edge_attr, batch)
        torch.allclose(output1.edge_index, output2.edge_index)
        torch.allclose(output1.edge_attr, output2.edge_attr)
        torch.allclose(output1.batch, output2.batch)
