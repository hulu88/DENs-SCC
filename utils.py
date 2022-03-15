from __future__ import division
from __future__ import print_function
import scipy.io as sio
import numpy as np
import torch
import scipy.sparse as sp
from sklearn import preprocessing

def load_data():
    DATA_ADHD = sio.loadmat('./ADHD_connectivity.mat')
    DATA = DATA_ADHD['connectivities']
    ADHD_label = sio.loadmat('./ADHD_labels.mat')
    ADHD_label = ADHD_label['VarName1']
    label = np.squeeze(ADHD_label)
    k = 29
    n = 3
    beta = 0.5
    lam = 0.5
    sigma = 0.4
    name = 'ADHD'
    adj = c_adj(DATA, k)

    return DATA, adj, label, k, n,beta,lam,sigma,name

def c_adj(data,k):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    # features = data.detach().cpu().numpy()
    features = data
    standardizer = StandardScaler()
    features_standardized = standardizer.fit_transform(features)
    nearestneighbors_euclidean = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(features_standardized)
    adj= nearestneighbors_euclidean.kneighbors_graph(features_standardized).toarray()
    from scipy import sparse
    adj = sparse.csr_matrix(adj)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def normalize(adj):
    """Symmetrically normalize adjacency matrix."""

    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
