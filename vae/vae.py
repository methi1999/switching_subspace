import torch
import torch.nn as nn
# from decoder import LinearAccDecoder
from misc.priors import moving_average
import math
import os

eps = 1e-6

class VAE(nn.Module):
    def __init__(self, config, input_dim, neuron_bias=None):
        super().__init__()               
        # keep only non-zero values in dim_x_z
        dim_x_z = config['dim_x_z']
        xz_ends = torch.cumsum(torch.tensor(dim_x_z), dim=0)
        xz_starts = torch.tensor([0] + xz_ends.tolist()[:-1])
        self.xz_l = torch.stack([xz_starts, xz_ends], dim=1).tolist()        
        
        self.x_dim = sum(dim_x_z)
        self.z_dim = len(dim_x_z)
        output_dim = (self.z_dim + self.x_dim)*(self.z_dim + self.x_dim + 1)        

        hidden_dim, num_layers = config['vae']['rnn']['hidden_size'], config['vae']['rnn']['num_layers']
        bidirectional = config['vae']['rnn']['bidirectional']
        dropout = config['vae']['rnn']['dropout']

        # self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, hidden_dim))
        # self.posterior = nn.Linear(hidden_dim, output_dim)

        # self.encoder = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
        #                       bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        # self.posterior = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, output_dim)

        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                              bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        self.posterior = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, output_dim)        
        
        # reconstruction
        self.linear_maps = nn.ModuleList([nn.Linear(i, input_dim) for i in dim_x_z])        

        # reconstruction_layers = nn.Sequential(nn.Linear(1, 32), nn.Tanh(),
        #                                       nn.Linear(32, 32), nn.ReLU(),
        #                                       nn.Linear(32, input_dim))
        # self.linear_maps = nn.ModuleList([reconstruction_layers for _ in range(x_dim)])

        # gru_recon = nn.RNN(1, 16, num_layers=1, bidirectional=False, batch_first=True)
        # self.linear_maps = nn.ModuleList([gru_recon, nn.Linear(1, input_dim)])
        # # self.linear_maps = nn.ModuleList([nn.Linear(1, input_dim), gru_recon])
        # self.linear2 = nn.Linear(16, input_dim)

        # def ret(input_dim, output_dim):
        #     return nn.Sequential(nn.Linear(input_dim, 4), nn.Tanh(),
        #                          nn.Linear(4, 8), nn.Tanh(),
        #                          nn.Linear(8, output_dim))
        # self.linear_maps = nn.ModuleList([ret(1, input_dim) for _ in range(x_dim)])        
        
        # expand neuron bias in batch dimension        
        self.neuron_bias = neuron_bias.unsqueeze(0) if neuron_bias is not None else None
        # self.sigmoid_scaling_factor = nn.Parameter(torch.tensor(2.0), requires_grad=True)

        # softmax temperature
        self.softmax_temp = config['vae']['rnn']['softmax_temp']

        # name model
        self.arch_name = 'vae_{}_{}_{}'.format(dim_x_z, hidden_dim, num_layers)
        if bidirectional:
            self.arch_name += '_bi'
        if neuron_bias is not None:
            self.arch_name += '_bias'        
        # optmizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['vae']['lr'], weight_decay=config['vae']['weight_decay'])        

    def split(self, encoded):
        batch, seq, _ = encoded.shape
        mu = encoded[:, :, :self.z_dim+self.x_dim]
        A = encoded[:, :, self.z_dim+self.x_dim:].reshape(batch, seq, self.z_dim+self.x_dim, self.z_dim+self.x_dim)
        # # make A diagonal
        # A = A * torch.eye(A.shape[2]).unsqueeze(0)
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
    
    # def reparameterize(self, mu, A):
    #     # mu is of shape (batch, seq, z+x)
    #     batch, seq, mu_dim = mu.shape
    #     mu_flat = mu.reshape(batch*seq, mu_dim)
    #     A_flat = A.reshape(batch*seq, mu_dim, mu_dim)

    #     dist = torch.distributions.MultivariateNormal(mu_flat, scale_tril=A_flat)     
    #     # A_flat = torch.bmm(A_flat, A_flat.transpose(1, 2))        
    #     # dist = torch.distributions.MultivariateNormal(mu_flat, covariance_matrix=A_flat)     

    #     return dist.sample().reshape(batch, seq, mu_dim)

    def forward(self, y, n_samples, use_mean_for_decoding):
        # y is of shape (batch_size, seq_len, input_dim)
        batch, seq, input_dim = y.shape
        encoded, _ = self.encoder(y)
        # encoded = self.encoder(y)
        # if isinstance(encoded, tuple):
        #     encoded = encoded[0]

        encoded = self.posterior(encoded)
        mu, A = self.split(encoded)
        
        if use_mean_for_decoding:
            assert n_samples == 1, "n_samples should be 1 when using mean for decoding"
            sample_zx = mu
        else:
            sample_zx = torch.cat([self.reparameterize(mu, A) for _ in range(n_samples)], dim=0)
        
        # extract x and z
        z, x = sample_zx[:, :, :self.z_dim], sample_zx[:, :, self.z_dim:]        
        z = torch.nn.Softmax(dim=-1)(z/self.softmax_temp)                
        
        # map x to observation
        Cx_list = [self.linear_maps[i](x[:, :, s: e]) for i, (s, e) in enumerate(self.xz_l)]        
        Cx = torch.stack(Cx_list, dim=-1)        
        y_recon = torch.sum(Cx * z.unsqueeze(2), dim=3)        

        if self.neuron_bias is not None:
            y_recon = y_recon + self.neuron_bias
        y_recon = nn.Softplus()(y_recon)        
        return {'y_recon': y_recon, 'x_samples': x, 'z_samples': z, 'combined_mean': mu, 'combined_A': A}        

    def loss(self, y, model_output, behavior_batch):
        """
        y and y_recon are of shape [batch * n_samples, time, dim]
        mu and A are of shape [batch, time, z+x] and [batch, time, z+x, z+x]
        """
        y_recon = model_output['y_recon']
        batch, seq, _ = y.shape
        num_samples = y_recon.shape[0] // batch

        # repeat ground truth        
        y = torch.cat([y]*num_samples, dim=0)
        recon_loss = torch.sum(y_recon - y * torch.log(y_recon))
        
        A, mu = model_output['combined_A'], model_output['combined_mean']
        # compute AAt
        flattened_A = A.reshape(batch*seq, self.z_dim+self.x_dim, self.z_dim+self.x_dim)        
        # flattened_A = torch.bmm(flattened_A, torch.transpose(flattened_A, 1, 2))
        mu = mu.reshape(batch*seq, self.z_dim+self.x_dim)
        # print(cov.shape)
        # poisson loss
        # print(y.shape, y_recon.shape)
        
        # print((torch.sum(mu.pow(2), dim=1) + torch.einsum("...ii", cov) - mu.shape[1] - torch.log(det+eps)).shape)
        
        # """
        # original KL loss
        cov = torch.bmm(flattened_A, torch.transpose(flattened_A, 1, 2))
        det = torch.det(cov)
        kl_loss = 0.5 * torch.sum(torch.sum(mu.pow(2), dim=1) + torch.einsum("...ii", cov) - mu.shape[1] - torch.log(det+eps))
        # kl_loss = 0
        # print(kl_loss, 'original')
        # """
        
        """        
        # new KL loss
        l, d = mu.shape[0], mu.shape[1]
        mu1 = torch.zeros(l, d)
        sigma1 = torch.eye(d, d).repeat(l, 1, 1)
        p = torch.distributions.MultivariateNormal(mu1, scale_tril=sigma1)
        cov_mat = torch.bmm(flattened_A, torch.transpose(flattened_A, 1, 2)) + eps*torch.eye(d).unsqueeze(0)
        q = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov_mat)
        # compute the kl divergence
        kl_loss = torch.distributions.kl_divergence(q, p).sum()
        # print(kl_loss, 'new')
        # return
        """
        
        # print(flattened_A[0])
        return (recon_loss + kl_loss)/(batch*num_samples)
    
    def extract_relevant(self, vae_output):
        y_recon = vae_output['y_recon'].detach().numpy()
        mean_z = vae_output['combined_mean'][:, :, :self.z_dim].detach().numpy()
        mean_x = vae_output['combined_mean'][:, :, self.z_dim:].detach().numpy()
        A = vae_output['combined_A'].detach()
        batch, time, dim, dim = A.shape
        flat_A = A.reshape(batch*time, dim, dim)    
        cov = torch.bmm(flat_A, torch.transpose(flat_A, 1, 2)).detach().view(batch, time, dim, dim).numpy()
        return y_recon, mean_x, mean_z, cov, cov, vae_output['x_samples'].detach().numpy(), vae_output['z_samples'].detach().numpy(), None, None
    