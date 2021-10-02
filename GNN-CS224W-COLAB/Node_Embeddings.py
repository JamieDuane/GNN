import networkx as nx
import matplotlib.pyplot as plt
from torch.optim import SGD
import torch, random
import torch.nn as nn
from sklearn.decomposition import PCA

class GraphFeatures(object):
    def __init__(self, graph):
        self.G = graph

    def ave_degree(self):
        return round(2*len(self.G.edges) / len(self.G.nodes), 2)

    def ave_clustering_coefficient(self):
        return round(nx.average_clustering(self.G), 2)

    def one_iter_pagerank(self, beta, r0, node_id):
        r1 = 0
        for nei in self.G.neighbors(node_id):
            r1 += r0*beta/self.G.degree[nei]
        r1 += (1-beta)/self.G.number_of_nodes(0)
        return round(r1, 2)

    def closeness_centrality(self, node=5):
        closeness = nx.closeness_centrality(self.G, u=node)
        closeness /= len(nx.node_connected_component(self.G, node))-1
        return round(closeness, 2)

def basic_info():
    print(torch.__version__)

    # Generate 3 x 4 tensor with all ones
    ones = torch.ones(3, 4)
    print(ones)

    # Generate 3 x 4 tensor with all zeros
    zeros = torch.zeros(3, 4)
    print(zeros)

    # Generate 3 x 4 tensor with random values on the interval [0, 1)
    random_tensor = torch.rand(3, 4)
    print(random_tensor)

    # Get the shape of the tensor
    print(ones.shape)

    # Create a 3 x 4 tensor with all 32-bit floating point zeros
    zeros = torch.zeros(3, 4, dtype=torch.float32)
    print(zeros.dtype)

    # Change the tensor dtype to 64-bit integer
    zeros = zeros.type(torch.long)
    print(zeros.dtype)

class GenerateTrainData(object):
    def __init__(self, G):
        self.G = G
    def graph_to_edge_list(self):
        edge_list = [edge for edge in self.G.edges]
        return edge_list
    def edge_list_to_tensor(self, edge_list):
        edge_index = torch.tensor(edge_list).T
        return edge_index
    def sample_negative_edges(self, num_neg_samples):
        neg_edge_list = []
        for node1 in self.G.nodes:
            for node2 in self.G.nodes:
                if node1 >= node2 or (node1, node2) in self.G.edges:
                    continue
                neg_edge_list.append((node1, node2))
        neg_edge_list = random.sample(neg_edge_list, num_neg_samples)
        return neg_edge_list

    def check_neg_sample(self):
        pos_edge_list = self.graph_to_edge_list()
        # Sample 78 negative edges
        neg_edge_list = self.sample_negative_edges(len(pos_edge_list))

        # Transform the negative edge list to tensor
        neg_edge_index = self.edge_list_to_tensor(neg_edge_list)
        print("The neg_edge_index tensor has shape {}".format(neg_edge_index.shape))

        # Which of following edges can be negative ones?
        edge_1 = (7, 1)
        edge_2 = (1, 33)
        edge_3 = (33, 22)
        edge_4 = (0, 4)
        edge_5 = (4, 2)

        ## Note:
        ## 1: For each of the 5 edges, print whether it can be negative edge
        for edge in [edge_1, edge_2, edge_3, edge_4, edge_5]:
            edge = (edge[1], edge[0]) if edge[0] > edge[1] else edge
            if edge in pos_edge_list:
                print("No")
            else:
                print("Yes")

def emb_example():
    emb_sample = nn.Embedding(num_embeddings=4, embedding_dim=8)
    print('Sample embedding layer: {}'.format(emb_sample))

    # Select an embedding in emb_sample
    id = torch.LongTensor([1])
    print(emb_sample(id))

    # Select multiple embeddings
    ids = torch.LongTensor([1, 3])
    print(emb_sample(ids))

    # Get the shape of the embedding weight matrix
    shape = emb_sample.weight.data.shape
    print(shape)

    # Overwrite the weight to tensor with all ones
    emb_sample.weight.data = torch.ones(shape)

    # Let's check if the emb is indeed initilized
    ids = torch.LongTensor([0, 3])
    print(emb_sample(ids))

class GraphEmbeddings(object):
    def __init__(self, graph):
        self.G = graph
        self.train_data = GenerateTrainData(self.G)

    def create_node_emb(self, num_node=34, embedding_dim=16):
        torch.manual_seed(1)
        emb = nn.Embedding(num_embeddings=num_node, embedding_dim=embedding_dim)
        shape = emb.weight.data.shape
        emb.weight.data = torch.rand(shape)
        return emb

    def visualize_emb(self, emb):
        X = emb.weight.data.numpy()
        pca = PCA(n_components=2)
        components = pca.fit_transform(X)
        plt.figure(figsize=(6,6))
        club1_x=[]
        club1_y=[]
        club2_x=[]
        club2_y=[]
        for node in self.G.nodes(data=True):
            if node[1]['club'] == 'Mr. Hi':
                club1_x.append(components[node[0]][0])
                club1_y.append(components[node[0]][1])
            else:
                club2_x.append(components[node[0]][0])
                club2_y.append(components[node[0]][1])
        plt.scatter(club1_x, club1_y, color="red", label="Mr. Hi")
        plt.scatter(club2_x, club2_y, color="blue", label="Officer")
        plt.legend()
        plt.show()

    def accuracy(self, pred, label):
        accu = torch.sum(torch.round(pred) == label) / pred.shape[0]
        return accu

    def train(self, emb, loss_fn, sigmoid, train_label, train_edge):
        epochs = 500
        learning_rate = 0.1
        optimizer = SGD(emb.parameters(), lr=learning_rate, momentum=0.9)

        for i in range(epochs):
            optimizer.zero_grad()
            node_emb = emb(train_edge)
            dot_product = torch.sum(node_emb[0]*node_emb[1], -1)
            result = sigmoid(dot_product)
            loss = loss_fn(result, train_label)
            print("Epoch:", i+1, "Loss:", loss.item(), 'Acc', self.accuracy(result, train_label).item())
            loss.backward()
            optimizer.step()

    def call(self):
        emb = self.create_node_emb()
        pos_edge_list = self.train_data.graph_to_edge_list()
        pos_edge_index = self.train_data.edge_list_to_tensor(pos_edge_list)
        neg_edge_list = self.train_data.sample_negative_edges(len(pos_edge_list))
        neg_edge_index = self.train_data.edge_list_to_tensor(neg_edge_list)
        loss_fn = nn.BCELoss()
        sigmoid = nn.Sigmoid()
        pos_label = torch.ones(pos_edge_index.shape[1], )
        neg_label = torch.zeros(neg_edge_index.shape[1], )
        train_label = torch.cat([pos_label, neg_label], dim=0)
        train_edge = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        self.train(emb, loss_fn, sigmoid, train_label, train_edge)
        self.visualize_emb(emb)

if __name__ == '__main__':
    G = nx.karate_club_graph()
    # print(type(G))
    nx.draw(G, with_labels=True)
    # plt.show()
    test = GraphEmbeddings(graph=G)
    test.call()