from Data_Process import *
import torch.nn as nn
from Data_Process import *
import torch
from utils import load_data
DATA, adj, label, k, n ,beta,lam,sigma,name = load_data()
from torch.nn.parameter import Parameter

class GCNencoder(nn.Module ):
    def __init__(self,d_0, d_1, d_2):
        super(GCNencoder,self).__init__()
        self.gconv1 = torch.nn.Sequential(torch.nn.Linear(d_0, d_1), torch.nn.ReLU(inplace=True))
        self.gconv1[0].weight.data = get_weight_initial(d_1, d_0)
        self.gconv2 = torch.nn.Sequential(torch.nn.Linear(d_1, d_2), )
        self.gconv2[0].weight.data = get_weight_initial(d_2, d_1)

    def forward(self, Adjacency_Modified, H_0):
        H_1 = self.gconv1(torch.matmul(Adjacency_Modified, H_0))
        H_2 = self.gconv2(torch.matmul(Adjacency_Modified, H_1))
        return H_2

class myGAE(torch.nn.Module):
    def __init__(self, d_0, d_1, d_2):
        super(myGAE, self).__init__()
        self.d0 = d_0
        self.d1 = d_1
        self.d2 = d_2
        self.preGCNencoder = GCNencoder(d_0, d_1, d_2)
        self.gconv1 = torch.nn.Sequential( torch.nn.Linear(d_0, d_1),  torch.nn.ReLU(inplace=True)  )
        self.gconv1[0].weight.data = get_weight_initial(d_1, d_0)
        self.gconv2 = torch.nn.Sequential( torch.nn.Linear(d_1, d_2), )
        self.gconv2[0].weight.data = get_weight_initial(d_2, d_1)

        self.decoderx = nn.Sequential(nn.Linear(d_2,128),nn.ReLU(True),nn.Linear(128, 512),nn.ReLU(True), nn.Linear(512, d_0),)
        self.decoderx1 = nn.Sequential(nn.Linear(d_2, 128), nn.ReLU(True), nn.Linear(128, 512), nn.ReLU(True),
                                      nn.Linear(512, d_0), )
        self.EncoderFC = nn.Sequential(
            nn.Linear(d_0,256),
            nn.ReLU(True),
            nn.Linear(256, d_2),
            nn.ReLU(True),
        )

        self.fcsoftmax = nn.Sequential(
            nn.Linear(d_2, 128),
            nn.ReLU(True),
            nn.Linear(128, n),
            nn.Softmax(dim=1)
        )
        self.zcsoftmax = nn.Sequential(
            nn.Linear(d_2, 32),
            nn.ReLU(True),
            nn.Linear(32, n),
            nn.Softmax(dim=1)
        )

        self.weight = Parameter(torch.FloatTensor(adj.shape[0], 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(0, 1)

    def Encoder(self, Adjacency_Modified, H_0):
        H_1 = self.gconv1(torch.matmul(Adjacency_Modified, H_0))
        H_2 = self.gconv2(torch.matmul(Adjacency_Modified, H_1))
        return H_2

    def Graph_Decoder(self, H_2):
        graph_re = Graph_Construction(H_2)
        Graph_Reconstruction = graph_re.Middle()
        return Graph_Reconstruction

    def forward(self, Adjacency_Modified, H_0):
        z_c = self.Encoder(Adjacency_Modified, H_0)
        z_f = self.EncoderFC(H_0)

        import torch.nn.functional as F
        Z = lam*F.softmax(z_c,dim=1)+beta*F.softmax(z_f,dim=1)
        x_Reconstruction = self.decoderx(Z)
        x_Reconstruction1 = self.decoderx1(z_f)
        wf = self.fcsoftmax(z_f)
        return z_f,z_c, Z, x_Reconstruction, wf,x_Reconstruction1





