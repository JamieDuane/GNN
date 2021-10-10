import torch.cuda
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data import DataLoader
from tqdm.notebook import tqdm

dataset = PygGraphPropPredDataset(name='ogbg-molhiv')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)