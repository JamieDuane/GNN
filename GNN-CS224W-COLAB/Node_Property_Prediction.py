import torch
#print(torch.__version__)
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset

#!!!! GraphGym !!!!!

from torch_geometric.datasets import TUDataset
root = './enzymes'
name = 'ENZYMES'

pyg_dataset = TUDataset(root, name)
#print(pyg_dataset)

def get_num_classes(pyg_dataset):
    return pyg_dataset.num_classes

def get_num_features(pyg_dataset):
    return pyg_dataset.num_features

def check_dataset_info():
    num_classes = get_num_classes(pyg_dataset)
    num_features = get_num_features(pyg_dataset)
    print("{} dataset has {} classes".format(name, num_classes))
    print("{} dataset has {} features".format(name, num_features))

def get_graph_class(pyg_dataset, idx):
    return pyg_dataset[idx].y[0]

def check_label_info():
    graph_0 = pyg_dataset[0]
    print(graph_0)
    idx = 100
    label = get_graph_class(pyg_dataset, idx)
    print('Graph with index {} has label {}'.format(idx, label))
#check_label_info()

def get_graph_num_edges(pyg_dataset, idx):
    return pyg_dataset[idx].edge_index.shape[1]

def check_edge_info():
    idx = 200
    num_edges = get_graph_num_edges(pyg_dataset, idx)
    print('Graph with index {} has {} edges'.format(idx, num_edges))
#check_edge_info()

dataset_name = 'ogbn-arxiv'
dataset = PygNodePropPredDataset(name=dataset_name, transform=T.ToSparseTensor())
#print('The {} dataset has {} graph'.format(dataset_name, len(dataset)))

data = dataset[0]
#print(data)

def graph_num_features(data):
    return data.num_features

import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from ogb.nodeproppred import Evaluator
data.adj_t = data.adj_t.to_symmetric()
print(data.x)
print(data.adj_t)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#print('Device: {}'.format(device))

data = data.to(device)
split_idx = dataset.get_idx_split()
print(split_idx)
train_idx = split_idx['train'].to(device)
print(train_idx)

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, return_embeds=False):
        super(GCN, self).__init__()
        self.convs=torch.nn.ModuleList(
            [GCNConv(in_channels=input_dim, out_channels=hidden_dim)]+\
            [GCNConv(in_channels=hidden_dim, out_channels=hidden_dim) for _ in range(num_layers-2)]+\
            [GCNConv(in_channels=hidden_dim, out_channels=output_dim)]
        )
        self.bns = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(num_features=hidden_dim) for _ in range(num_layers-1)]
        )
        self.softmax = torch.nn.LogSoftmax()
        self.dropout = dropout
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for conv, bn in zip(self.convs[:-1], self.bns):
            x1 = F.relu(bn(conv(x, adj_t)))
            if self.training:
                x1 = F.dropout(x1, p=self.dropout)
            x = x1
        x = self.convs[-1](x, adj_t)
        out = x if self.return_embeds else self.softmax(x)
        return out

def train(model, data, train_idx, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)
    loss = loss_fn(out[train_idx], data.y[train_idx].reshape(-1))

    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data, split_idx, evaluator):
     model.eval()
     out = None
     out = model(data.x, data.adj_t)

     y_pred = out.argmax(dim=-1, keepdim=True)

     train_acc = evaluator.eval({
         'y_true': data.y[split_idx['train']],
         'y_pred': y_pred[split_idx['train']],
     })['acc']
     valid_acc = evaluator.eval({
         'y_true': data.y[split_idx['valid']],
         'y_pred': y_pred[split_idx['valid']],
     })['acc']
     test_acc = evaluator.eval({
         'y_true': data.y[split_idx['test']],
         'y_pred': y_pred[split_idx['test']],
     })['acc']

     return train_acc, valid_acc, test_acc

def main():
    args = {
        'device': device,
        'num_layers': 3,
        'hidden_dim': 256,
        'dropout': 0.5,
        'lr': 0.01,
        'epochs': 100,
    }

    model = GCN(data.num_features, args['hidden_dim'], dataset.num_classes, args['num_layers'], args['dropout']).to(
        device)
    evaluator = Evaluator(name='ogbn-arxiv')

    import copy
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    loss_fn = F.nll_loss

    best_model = None
    best_valid_acc = 0

    for epoch in range(1, 1 + args["epochs"]):
        loss = train(model, data, train_idx, optimizer, loss_fn)
        result = test(model, data, split_idx, evaluator)
        train_acc, valid_acc, test_acc = result
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = copy.deepcopy(model)
        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_acc:.2f}%, '
              f'Valid: {100 * valid_acc:.2f}% '
              f'Test: {100 * test_acc:.2f}%')

