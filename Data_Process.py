
import torch
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')
############################################ Load the datasets  ########################################################

def Graph():
    from utils import load_data
    DATA, adj, label, k, n ,beta,lam,sigma,name= load_data()
    Features = DATA
    Labels = label
    Adjacency = adj
    Labels = Labels.reshape(Labels.shape[0], 1)
    return Features, Labels, Adjacency


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

################################ Graph construction for adjacency matrix and similarty matrix ###################
class Graph_Construction:
    # Input: n * d.
    def __init__(self, X):
        self.X = X

    def Middle(self):
            Inner_product = self.X.mm(self.X.T)
            Graph_middle = torch.sigmoid(Inner_product)
            return Graph_middle

    # Construct the adjacency matrix by KNN
    def KNN(self, k=9):
        n = self.X.shape[0]
        D = L2_distance_2(self.X, self.X)
        _, idx = torch.sort(D)
        S = torch.zeros(n, n)
        for i in range(n):
            id = torch.LongTensor(idx[i][1: (k + 1)])
            S[i][id] = 1
        S = (S + S.T) / 2
        return S

################################# Adjacency matrix normalization or pollution ##########################################
class Convolution_Kernel():
    def __init__(self, adjacency):
        self.adjacency = adjacency

    def Adjacency_Convolution(self):
        adj = self.adjacency + torch.eye(self.adjacency.shape[0])
        degrees = torch.Tensor(adj.sum(1))
        degrees_matrix_inv_sqrt = torch.diag(torch.pow(degrees, -0.5))
        return torch.mm(degrees_matrix_inv_sqrt, adj).mm(degrees_matrix_inv_sqrt)

    def Laplacian_Raw(self):
        degrees = torch.diag(torch.Tensor(self.adjacency.sum(1)).flatten())
        return degrees - self.adjacency

    ######## Laplacian matrix convolution
    def Laplacian_Convolution(self):
        S = self.adjacency + torch.eye(self.adjacency.size(0)) * 0.001
        degrees = (torch.Tensor(S.sum(1)).flatten())
        D = torch.diag(degrees)
        L = D - self.adjacency
        D_sqrt = torch.diag(torch.pow(degrees, -0.5))
        return D_sqrt.mm(L).mm(D_sqrt)

############################################# plot the t-SNE ###########################################################
def plot_embeddings(embeddings, Features, Labels, name , Test_Y, name1):

    emb_list = []
    for k in range(Features.shape[0]):
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2, init="pca")
    # model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(Features.shape[0]):
        color_idx.setdefault(Labels[i][0], [])
        color_idx[Labels[i][0]].append(i)


    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c, s = 5)
    plt.axis('off')
    # plt.legend()
    plt.gca.legend_ = None
    plt.savefig(name)
    plt.show()

    color_idx1 = {}
    for i in range(Features.shape[0]):
        color_idx1.setdefault(Test_Y[i][0], [])
        color_idx1[Test_Y[i][0]].append(i)
    for c, idx in color_idx1.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c, s = 5)
    plt.axis('off')
    # plt.legend()
    plt.gca.legend_ = None
    plt.savefig(name1)
    plt.show()

def get_weight_initial(d1, d2):
    bound = torch.sqrt(torch.Tensor([6.0 / (d1 + d2)]))
    nor_W = -bound + 2*bound*torch.rand(d1, d2)
    return torch.Tensor(nor_W)

########################################################## Pre Link Prediction #######################################
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    degree_mat_inv_sqrt = sp.diags(np.power(np.array(adj_.sum(1)), -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return sparse_to_tuple(adj_normalized)

