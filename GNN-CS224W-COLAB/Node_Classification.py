import networkx as nx
import torch
import matplotlib.pyplot as plt

def visualize(h, G, color, epoch = None, loss = None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap='Set2')
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item(): .4f}', fontsize=16)
    else:
        nx.draw_networkx(G, pos=nx.spring_layout(G,seed=42), with_labels=False, node_color=color, cmap="Set2")
    plt.show()

from torch_geometric.datasets import KarateClub
dataset = KarateClub()
data = dataset[0]

def basic_info():
    print(dataset)
    print("=========================")
    print(len(dataset))
    print(dataset.num_features)
    print(dataset.num_classes)

    print(data)
    print('=========================')
    print(f'nodes:{data.num_nodes}')
    print(f'edges:{data.num_edges}')
    print(f'node degree:{data.num_nodes / data.num_edges:.2f}')
    print(f'training nodes:{data.train_mask.sum()}')
    print(f'training node label rate:{int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'isolated nodes:{data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

def graph_info():
    from IPython.display import Javascript  # Restrict height of output cell.
    from IPython.display import display
    display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

    edge_index = data.edge_index
    print(edge_index.t())

    from torch_geometric.utils import to_networkx
    G = to_networkx(data, to_undirected=True)
    visualize(G, G, color=data.y)

from torch.nn import Linear
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(data.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()

        out = self.classifier(h)

        return out, h

def train_GCN():
    model = GCN()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def train(data):
        optimizer.zero_grad()
        out, h = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss, h

    import time
    for epoch in range(401):
        loss, h = train(data)
        if epoch % 10 == 0:
            visualize(h, h, color=data.y, epoch=epoch, loss=loss)
            time.sleep(0.3)

import torch.nn.functional as F
class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(MLP, self).__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(dataset.num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p =0.5, training = self.training)
        x = self.lin2(x)
        return x

model = MLP(hidden_channels=16)
print(model)