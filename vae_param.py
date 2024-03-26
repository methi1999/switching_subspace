import torch
import torch.nn as nn
import math
import os
import matplotlib.pyplot as plt
import numpy as np

eps = 1e-6
# TODO: Make cholesky decomposition work

class VAEParameterised(nn.Module):
    def __init__(self, config, input_dim, xz_list, neuron_bias=None, init=''):
        super().__init__()
        # keep only non-zero values in xz_list
        xz_list = [x for x in xz_list if x > 0]
        xz_ends = torch.cumsum(torch.tensor(xz_list), dim=0)
        xz_starts = torch.tensor([0] + xz_ends.tolist()[:-1])
        xz_l = torch.stack([xz_starts, xz_ends], dim=1)
        # register as a buffer
        self.register_buffer('xz_l', xz_l)
        
        self.x_dim = sum(xz_list)
        self.z_dim = len(xz_list)
        output_dim = (self.x_dim)*(self.x_dim + 1)        

        hidden_dim, num_layers = config['rnn']['hidden_size'], config['rnn']['num_layers']
        bidirectional = config['rnn']['bidirectional']
        dropout = config['rnn']['dropout']

        # self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, hidden_dim))
        # self.posterior = nn.Linear(hidden_dim, output_dim)

        # self.encoder = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
        #                       bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        # self.posterior = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, output_dim)

        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                              bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        self.posterior = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, output_dim)
        
        bidirectional = True
        self.z_encoder = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True,
                              bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)        
        self.z_var_entropy = config['rnn']['z_var_entropy']
        if self.z_var_entropy:
            self.z_mean_var = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, self.z_dim)
        else:
            self.z_mean_var = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, 3*self.z_dim)
            self.log_num_time_bins = math.log(int(2.5/config['shape_dataset']['win_len']))
        
        # reconstruction
        self.linear_maps = nn.ModuleList([nn.Linear(i, input_dim) for i in xz_list])        
        
        # softmax temperature
        self.softmax_temp = config['rnn']['softmax_temp']

        # name model
        self.arch_name = 'vae_paramz_{}_{}_{}'.format(xz_list, hidden_dim, num_layers)
        if bidirectional:
            self.arch_name += '_bi'
        if neuron_bias is not None:
            self.arch_name += '_bias'
                
        # optmizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['rnn']['lr'], weight_decay=config['rnn']['weight_decay'])        
        
        if config['rnn']['scheduler']['which'] == 'cosine':
            restart = config['rnn']['scheduler']['cosine_restart_after']
            scheduler1 = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1, total_iters=restart+restart//2)
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=restart)
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[scheduler1, scheduler2], milestones=[restart//2])
        elif config['rnn']['scheduler']['which'] == 'decay':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.97)
        else:
            print('Scheduler not implemented for GRU')
            self.scheduler = None
    
    def split(self, encoded):
        batch, seq, _ = encoded.shape
        mu = encoded[:, :, :self.x_dim]
        A = encoded[:, :, self.x_dim:].reshape(batch, seq, self.x_dim, self.x_dim)
        return mu, A
    
    def reparameterize(self, mu, A):
        # mu is of shape (batch, seq, z+x)
        batch, seq, mu_dim = mu.shape
        A_flat = A.view(batch*seq, mu_dim, mu_dim)
        mu_flat = mu.reshape(batch*seq, mu_dim).unsqueeze(-1)
        eps = torch.randn_like(mu_flat)
        # print(mu.shape, A.shape, eps.shape)
        sample = (mu_flat + torch.bmm(A_flat, eps)).squeeze(-1)        
        return sample.view(batch, seq, mu_dim)

    def forward(self, y, n_samples):
        # y is of shape (batch_size, seq_len, input_dim)
        encoded_x, _ = self.encoder(y)
        # encoded_x = self.encoder(y)
        # if isinstance(encoded_x, tuple):
        #     encoded_x = encoded_x[0]

        encoded_x = self.posterior(encoded_x)
        mu, A = self.split(encoded_x)
        x = torch.cat([self.reparameterize(mu, A) for _ in range(n_samples)], dim=0)
        # tanh on x
        # x = nn.Tanh()(x)

        # obtain z
        encoded_z, _ = self.z_encoder(y)
        z_out = self.z_mean_var(encoded_z)
        
        # split into 3 across last dimension
        time_bin = z_out[:, :, :self.z_dim]
        # take argmax across time
        # time_bin = torch.argmax(time_bin, dim=1)
        # apply softmax across time and take weighted average
        time_bin_distribution = nn.Softmax(dim=1)(time_bin)
        # plt.plot(time_bin_distribution[30, :, 0].numpy())        
        # return
        # print(time_bin[0, :, 0])
        time_bin = torch.sum(time_bin_distribution * torch.arange(time_bin.shape[1]).unsqueeze(0).unsqueeze(-1), dim=1)

        # if self.z_var_entropy:
        #     # split z across closest point near time_bin
        #     log_var1, log_var2 = torch.zeros_like(time_bin), torch.zeros_like(time_bin)
        #     time_bin_round = time_bin.round().long()
        #     for i in range(z_out.shape[0]):
        #         for k in range(z_out.shape[2]):
        #             t = time_bin_round[i, k]
        #             distri_1 = time_bin_distribution[i, :t, k]
        #             distri_2 = time_bin_distribution[i, t:, k]
        #             # scale them so that sum is 1
        #             distri_1 = distri_1 / torch.sum(distri_1)
        #             distri_2 = distri_2 / torch.sum(distri_2)
        #             # take entropy
        #             log_var1[i, k] = -torch.sum(distri_1 * torch.log(distri_1 + eps))
        #             log_var2[i, k] = -torch.sum(distri_2 * torch.log(distri_2 + eps))
        # else:
        # log_var_timestep = z_out.shape[2]//2
        log_var_timestep = -1
        log_var1, log_var2 = z_out[:, log_var_timestep, self.z_dim:2*self.z_dim], z_out[:, log_var_timestep, 2*self.z_dim:]        
        # # log_var1, log_var2 = hidden_z[0], hidden_z[1]
        # log_var1 = nn.Sigmoid()(log_var1) * self.log_num_time_bins
        # log_var2 = nn.Sigmoid()(log_var2) * self.log_num_time_bins
        # # print(log_var1, log_var2)
        
        # construct gaussian with mean and variance
        z = construct_gaussian(time_bin, log_var1, log_var2, y.shape[1])
        # add random noise to it
        # z = torch.cat([z + torch.randn_like(z)*0.01 for _ in range(n_samples)], dim=0)
        z = torch.cat([z for _ in range(n_samples)], dim=0)
        # softmax
        # z = nn.Softmax(dim=2)(z/self.softmax_temp)        
        # z = z/(z.sum(dim=2, keepdim=True) + eps)

        # # construct z0 as 1-(z1+z2)/2
        z0 = 1 - (z[:, :, 0] + z[:, :, 1])/2
        z = torch.stack([z[:, :, 0], z[:, :, 1], z0], dim=2)
        
        # map x to observation
        Cx_list = [self.linear_maps[i](x[:, :, s: e]) for i, (s, e) in enumerate(self.xz_l)]
        # if any element is a tuple, take the first element
        # Cx_list = [self.linear2(Cx[0]) if isinstance(Cx, tuple) else Cx for Cx in Cx_list]
        # print([x.shape for x in Cx_list])
        Cx = torch.stack(Cx_list, dim=-1)
        y_recon = torch.sum(Cx * z.unsqueeze(2), dim=3)        

        # if self.neuron_bias is not None:
        #     y_recon = y_recon + self.neuron_bias
        y_recon = nn.Softplus()(y_recon)
        return y_recon, mu, A, z, x

    def loss(self, y, y_recon, mu, A, z):
        """
        y and y_recon are of shape [batch * n_samples, time, dim]
        mu and A are of shape [batch, time, z+x] and [batch, time, z+x, z+x]
        """
        batch, seq, _ = mu.shape
        num_samples = y_recon.shape[0] // batch
        # compute AAt
        flattened_A = A.reshape(batch*seq, self.x_dim, self.x_dim)        
        # flattened_A = torch.bmm(flattened_A, torch.transpose(flattened_A, 1, 2))        
        mu = mu.reshape(batch*seq, self.x_dim)
        # print(cov.shape)
        # poisson loss
        # print(y.shape, y_recon.shape)
        # repeat ground truth        
        y = torch.cat([y]*num_samples, dim=0)
        recon_loss = torch.sum(y_recon - y * torch.log(y_recon))
        # print((torch.sum(mu.pow(2), dim=1) + torch.einsum("...ii", cov) - mu.shape[1] - torch.log(det+eps)).shape)
        
        # original KL loss
        cov = torch.bmm(flattened_A, torch.transpose(flattened_A, 1, 2))
        det = torch.det(cov)
        kl_loss = 0.5 * torch.sum(torch.sum(mu.pow(2), dim=1) + torch.einsum("...ii", cov) - mu.shape[1] - torch.log(det+eps))

        # # new KL loss
        # l, d = mu.shape[0], mu.shape[1]
        # mu1 = torch.zeros(l, d)
        # sigma1 = torch.eye(d, d).repeat(l, 1, 1)
        # p = torch.distributions.MultivariateNormal(mu1, scale_tril=sigma1)
        # q = torch.distributions.MultivariateNormal(mu, scale_tril=flattened_A)        
        # # compute the kl divergence
        # kl_loss = torch.distributions.kl_divergence(p, q).sum()
        # kl_loss = 0 

        # z is of shape batch x time x z_dim  
        # l2 = torch.sum((torch.sum(z, dim=2)-1)**2)
        # print(l2.shape)
        
        # print(flattened_A[0])
        return (recon_loss + kl_loss)/(batch*num_samples)
    

def construct_gaussian(x_mean, x_log_var1, x_log_var2, time_bins):
    # x_mean, x_var is of shape (batch, z); mean is an integer, var is variance
    # output should be of shape (batch, time_bins, z)
    batch, num_z = x_mean.shape
    x_var1, x_var2 = torch.exp(x_log_var1), torch.exp(x_log_var2)
    # x_var = torch.ones_like(x_log_var)*2
    # constuct a gaussian with mean and variance and time_bins
    t = torch.arange(time_bins)
    t = t.unsqueeze(0).unsqueeze(-1).repeat(batch, 1, num_z)
    # calculate the gaussian
    gaussian_z1 = torch.exp(-0.5 * (t - x_mean.unsqueeze(1))**2 / x_var1.unsqueeze(1)) / (torch.sqrt(2 * math.pi * x_var1.unsqueeze(1)))
    gaussian_z2 = torch.exp(-0.5 * (t - x_mean.unsqueeze(1))**2 / x_var2.unsqueeze(1)) / (torch.sqrt(2 * math.pi * x_var2.unsqueeze(1)))
    # rescale so that max is 1
    gaussian_z1 = gaussian_z1 / torch.max(gaussian_z1, dim=1, keepdim=True).values
    gaussian_z2 = gaussian_z2 / torch.max(gaussian_z2, dim=1, keepdim=True).values
    # take first z till t mean and second z after t mean
    gaussian_z = torch.where(t <= x_mean.unsqueeze(1), gaussian_z1, gaussian_z2)
    return gaussian_z

if __name__ == '__main__':
    x_mean = torch.tensor([[10, 5], [10, 13]], dtype=torch.float32)
    x_log_var1 = torch.tensor([[2, 0], [0, 0]], dtype=torch.float32)
    x_log_var2 = torch.tensor([[9, 0], [0, 4]], dtype=torch.float32)
    time_bins = 25
    out = construct_gaussian(x_mean, x_log_var1, x_log_var2, time_bins)
    # plot
    plt.plot(np.arange(time_bins), out[0, :, 0].numpy())
    plt.plot(np.arange(time_bins), out[0, :, 1].numpy())
    plt.show()
