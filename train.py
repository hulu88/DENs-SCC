from Models import *
from Metrics import eval
from Data_Process import *
from sklearn.cluster import KMeans
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

seed =4321
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

l1 = 1
l2 = 0.1

Clustering = True
t_SNE = True

Epoch_Num = 200
Learning_Rate =1e-3

from orth import Orthonorm, squared_distance
from utils import  load_data,  normalize, sparse_mx_to_torch_sparse_tensor

DATA, adj, label, k, n,beta,lam,sigma,name = load_data()

Hidden_Layer_1 = 32
Hidden_Layer_2 = n
################################### Load dataset   ######################################################################
Features, Labels, Adjacency_Matrix_raw = Graph()
Features = torch.Tensor(Features)

Adjacency_Matrix = normalize(Adjacency_Matrix_raw)
Adjacency_Matrix = sparse_mx_to_torch_sparse_tensor(Adjacency_Matrix)
Adjacency_Convolution = Adjacency_Matrix

ACC_GAE_total = []
ACC1_GAE_total = []
ACC2_GAE_total = []

LOSS_TOTAL = []
mse_loss = torch.nn.MSELoss()
model_GAE = myGAE(Features.shape[1], Hidden_Layer_1, Hidden_Layer_2)
optimzer = torch.optim.Adam(model_GAE.parameters(), lr=Learning_Rate)

for epoch in range(Epoch_Num):
    zf, zc, Z, x_rec, wf ,x_rec1= model_GAE(Adjacency_Convolution, Features)
    x = Orthonorm(Z)
    Y = squared_distance(x)
    W = torch.exp(-squared_distance(Z) / (2 * (sigma ** 2)))
    W = W.view(-1)
    B = Y.view(-1)
    # --------------------------------loss-------------------------------------
    lossc = l1 * torch.matmul(W, B) / (DATA.shape[0] * n)
    lossx = mse_loss(Features, x_rec)
    losskl = - l2 * torch.sum(wf.view(-1) * wf.view(-1).log()) / DATA.shape[0]
    loss = lossx + losskl + lossc
    LOSS_TOTAL.append(loss)
    print('{: .4f}    ,{: .4f}, {: .4f}, {: .4f}, {}'.format(loss, lossx, lossc, losskl, epoch))
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()

    Latent_Representation = Z.cpu().detach().numpy()

    if Clustering and (epoch + 1) % 5 == 0:
        ACC_H2 = []
        ACC1_H2 = []
        ACC2_H2 = []

        kmeans = KMeans(n_clusters=n)
        Y_pred_OK = kmeans.fit_predict(Latent_Representation)
        Labels_K = np.array(Labels).flatten()
        Test_Y = np.expand_dims(Y_pred_OK,axis=1)

        ACC = eval(Labels_K, Y_pred_OK, print_msg=True,epoch=epoch)

        Y_pred_OK = kmeans.fit_predict(zf.cpu().detach().numpy())
        Labels_K = np.array(Labels).flatten()

        ACC1 = eval(Labels_K, Y_pred_OK, print_msg=False)
        Y_pred_OK = kmeans.fit_predict(zc.cpu().detach().numpy())
        Labels_K = np.array(Labels).flatten()

        ACC2 = eval(Labels_K, Y_pred_OK, print_msg=False)
        if (epoch+1)==Epoch_Num:
            print('     三个精度的比较         {: .4f}  ,{: .4f},  {: .4f},  {}'.format( ACC, ACC1, ACC2, epoch))
            fh = open('{}acc_compare.txt'.format(name), 'a')
            fh.write( 'ACC=%f, ACC1=%f, ACC2=%f' % (ACC,ACC1,ACC2))
            fh.write('\r\n')
            fh.flush()
            fh.close()

        ACC_H2.append(ACC)
        ACC1_H2.append(ACC1)
        ACC2_H2.append(ACC2)

        ACC_GAE_total.append(100 * np.mean(ACC_H2))
        ACC1_GAE_total.append(100 * np.mean(ACC1_H2))
        ACC2_GAE_total.append(100 * np.mean(ACC2_H2))

state = model_GAE.state_dict()
torch.save(state,'model.{}.pkl'.format(name))

if Clustering:
    ax = range(0, int(Epoch_Num / 5))
    bx = range(0, int(Epoch_Num / 5))
    cx = range(0, int(Epoch_Num / 5))
    dx = range(0, int(Epoch_Num))

    ay = ACC_GAE_total
    by = ACC1_GAE_total
    cy = ACC2_GAE_total
    dy = LOSS_TOTAL

    plt.subplot(1, 2, 1)
    plt.plot(ax, ay)
    plt.subplot(1, 2, 1)
    plt.plot(bx, by)
    plt.subplot(1, 2, 1)
    plt.plot(cx, cy)
    plt.title('ACC')

    plt.subplot(1, 2, 2)
    plt.plot(dx, dy)
    plt.title('LOSS')
    plt.show()



