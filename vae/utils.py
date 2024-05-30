import torch
import torch.nn as nn


def get_linear_layers(inp, out, hidden, dropout):
    if len(hidden) == 0:
        return [nn.Linear(inp, out)]
    layers = []
    for i in range(len(hidden)):
        if i == 0:
            layers.append(nn.Linear(inp, hidden[i]))
        else:
            layers.append(nn.Linear(hidden[i-1], hidden[i]))
        layers.append(nn.ReLU())        
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden[-1], out))            
    return layers


def get_eps():
    return 1e-6

def bidiagonal_inverse_batched(a, b):
    # a is of shape batch x n, b is of shape batch x (n-1)
    batch, n = a.size()
    # b = torch.cat([torch.ones(batch, 1), b_], dim=1)    
    theta = torch.zeros(batch, n+1)
    phi = torch.zeros(batch, n+1)    
    # compute a inverse
    a_inv = 1/a
    # embed a_inv into diagonal
    # inv = torch.diag_embed(a_inv)
    inv = torch.zeros(batch, n, n)
    # set
    ijpairs = set()
    for i in range(n):
        ijpairs.add((i, i))
        inv[:, i, i] = a_inv[:, i]
    # init
    theta[:, 0] = 1
    theta[:, 1] = a[:, 0]
    # phi[:, n] = 1
    # phi[:, n-1] = a[:, n-1]
    for i in range(2, n+1):
        theta[:, i] = a[:, i-1] * theta[:, i-1]
    # for i in range(n-1, 0, -1):
    #     phi[:, i] = a[:, i-1] * phi[:, i+1]
    # loop over i and j
    for j in range(1, n+1):
        for i in range(j-1, 0, -1):
            # i, j are 1-indexed
            # inv[:, i-1, j-1] = -b[:, i-1] * inv[:, i, j-1] * theta[:, i-1] / theta[:, i]
            inv[:, i-1, j-1] = -b[:, i-1] * (inv[:, i, j-1] * theta[:, i-1] / theta[:, i]).detach()
            if (i-1, j-1) not in ijpairs:
                ijpairs.add((i-1, j-1))
            else:
                print('Duplicate found')
    # print(inv)
    return inv


def derivative_time_series(x):    
    # x is of shape (batch, time, dim)
    # return derivative of x
    # pad x with zeros on both sides    
    zeros = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device)
    x = torch.cat([zeros, x, zeros], dim=1)
    # centred
    return (x[:, 2:, :] - x[:, :-2, :])/2    
    # forward
    # return x[:, 2:, :] - x[:, 1:-1, :]    
    # backward
    # return x[:, 1:-1, :] - x[:, :-2, :]    


def normal_cdf(x, nu, offset=0):
    return torch.clip(0.5 * (1 + torch.erf(x * nu + offset)), min=get_eps())


def rbf_kernel(time, sigma):
    # sigma is a scalar, time is number of bins
    # return kernel of shape (time, time)
    time_range = torch.arange(time).unsqueeze(0).float()
    time_diff = time_range - time_range.t()
    kernel = torch.exp(-time_diff**2 / (2 * sigma**2))    
    return kernel


# def normal_likelihood(x, mu, cov_det, inv):
#     # x, mu are of shape (batch, dim)
#     # cov is of shape (batch, dim, dim)
#     # return likelihood of shape (batch)
#     dim = x.shape[-1]    
#     # calculate exponent
#     exponent = -0.5 * torch.sum((x - mu).unsqueeze(-1) * torch.bmm(inv, (x - mu).unsqueeze(-1)), dim=(1, 2))
#     # calculate constant
#     constant = 1 / ((2 * math.pi)**(dim/2) * torch.sqrt(cov_det))
#     return constant * torch.exp(exponent)    
