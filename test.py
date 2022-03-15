from Models import *
from Metrics import eval
from Data_Process import *
from sklearn.cluster import KMeans
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')
from utils import load_data,sparse_mx_to_torch_sparse_tensor

DATA, adj, label, k, n,beta,lam,sigma,name = load_data()
Hidden_Layer_1 = 32
Hidden_Layer_2 = n

Features, Labels, Adjacency_Matrix_raw = Graph()
Features = torch.Tensor(Features)
Adjacency_Matrix = normalize(Adjacency_Matrix_raw)
Adjacency_Matrix = sparse_mx_to_torch_sparse_tensor(Adjacency_Matrix)
Adjacency_Convolution = Adjacency_Matrix

net = torch.load('model.{}.pkl'.format(name))
model_GAE = myGAE(Features.shape[1], Hidden_Layer_1, Hidden_Layer_2)
model_GAE.load_state_dict(net)

zf, zc, Z, x_rec, wf ,x_rec1= model_GAE(Adjacency_Convolution, Features)
Latent_Representation = Z.cpu().detach().numpy()
kmeans = KMeans(n_clusters=n)
Y_pred_OK = kmeans.fit_predict(Latent_Representation)
Labels_K = np.array(Labels).flatten()
Test_Y = np.expand_dims(Y_pred_OK, axis=1)
ACC = eval(Labels_K, Y_pred_OK, print_msg=True)
Y_pred_OK = kmeans.fit_predict(zf.cpu().detach().numpy())
Labels_K = np.array(Labels).flatten()
ACC1 = eval(Labels_K, Y_pred_OK, print_msg=True)
Y_pred_OK = kmeans.fit_predict(zc.cpu().detach().numpy())
Labels_K = np.array(Labels).flatten()
ACC2 = eval(Labels_K, Y_pred_OK, print_msg= True)

