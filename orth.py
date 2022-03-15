import torch

def Orthonorm(x, epsilon=1e-7):
    q,r = torch.qr(x)
    return q

def squared_distance(X, Y=None, W=None):
    if Y is None:
        Y = X
    # distance = squaredDistance(X, Y)
    dim =2
    X = torch.unsqueeze(X, dim=1)

    if W is not None:
        # if W provided, we normalize X and Y by W
        D_diag = torch.unsqueeze(torch.sqrt(torch.sum(W, dim=1)), dim=1)
        X /= D_diag
        Y /= D_diag
    distance = torch.sum(torch.pow(X-Y,2), dim=2)
    return distance

def orthogonal_regularization( weight,beta=1e-4):
    # beta * (||W^T.W * (1-I)||_F)^2 or
    # beta * (||W.W.T * (1-I)||_F)^2
    # 若 H < W,可以使用前者， 若 H > W, 可以使用后者，这样可以适当减少内存
    loss_orth = torch.tensor(0., dtype=torch.float32)
    weight_squared = torch.bmm(weight, weight.permute(0, 2, 1))  # (N * C) * H * H
    ones = torch.ones(N * C, H, H, dtype=torch.float32)  # (N * C) * H * H
    diag = torch.eye(H, dtype=torch.float32)  # (N * C) * H * H
    loss_orth += ((weight_squared * (ones - diag)) ** 2).sum()

    return loss_orth * beta

