import torch
import torch.nn as nn
import math
import os
import matplotlib.pyplot as plt
import numpy as np

eps = 1e-4

def get_linear(inp, out, hidden, dropout):
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



def rbf_kernel(time, sigma):
    # sigma is a scalar, time is number of bins
    # return kernel of shape (time, time)
    time_range = torch.arange(time).unsqueeze(0).float()
    time_diff = time_range - time_range.t()
    return torch.exp(-time_diff**2 / (2 * sigma**2))

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


def block_diag_precision(diag_elems, off_diag_elems, mean):
    """
    # fill elements
    prec = torch.diag_embed(diag_elems) + torch.diag_embed(off_diag_elems, offset=1, dim1=-2, dim2=-1)
    # take product of transpose
    prec = torch.bmm(prec.transpose(1, 2), prec)# + eps * torch.eye(prec.shape[-1], device=prec.device)
    """
    # """
    batch, time = diag_elems.shape
    device = diag_elems.device
    prec = torch.zeros(batch, time, time, device=device)
    a_2 = diag_elems**2
    b_2 = torch.cat([torch.zeros(batch, 1, device=device), off_diag_elems**2], dim=1)
    ab = diag_elems[:, :-1] * off_diag_elems            
    prec += torch.diag_embed(a_2+b_2+1, dim1=-2, dim2=-1)
    off_diagonal = torch.diag_embed(ab, offset=1, dim1=-2, dim2=-1)            
    prec += off_diagonal + off_diagonal.transpose(-2, -1) + eps * torch.eye(time, device=device).unsqueeze(0)
    # """
    return torch.distributions.MultivariateNormal(mean, precision_matrix=prec)
   

class TimeSeries(nn.Module):
    
    def __init__(self, config, input_dim, latent_dim, time_bins):
        super().__init__()
        model_config = config['vae_gp']
        # rnn
        hidden_dim = model_config['rnn_encoder']['hidden_size']
        num_layers = model_config['rnn_encoder']['num_layers']
        bidirectional = model_config['rnn_encoder']['bidirectional']
        dropout = model_config['rnn_encoder']['dropout']
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                          bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        # self.rnn = nn.Sequential(nn.Linear(input_dim, hidden_dim*(1+bidirectional)))
        
        # posterior mean and block diagonal
        hidden_layers = model_config['post_rnn_linear']['hidden_dims']
        dropout = model_config['post_rnn_linear']['dropout']

        inp_dim = hidden_dim*2 if bidirectional else hidden_dim
        self.posterior_mean = nn.Sequential(*get_linear(inp_dim, latent_dim, hidden_layers, dropout))
        
        self.full_cov = model_config['full_cov']
        if self.full_cov:
            self.block_diagonal = nn.Sequential(*get_linear(inp_dim, time_bins*latent_dim, hidden_layers, dropout))
        else:
            self.block_diagonal = nn.Sequential(*get_linear(inp_dim, 2*latent_dim, hidden_layers, dropout))
        
        self.smoothing_sigma = model_config['smoothing_sigma']
        if self.smoothing_sigma:
            cov_prior = rbf_kernel(time_bins, self.smoothing_sigma)
        else:
            cov_prior = torch.eye(time_bins)        
        self.prior_cholesky = torch.linalg.cholesky(cov_prior + eps * torch.eye(time_bins))
    
    def forward(self, y):
        encoded, _ = self.rnn(y)
        # encoded = self.rnn(y)
        # mean is of shape (batch, time, latent_dim)
        mean = self.posterior_mean(encoded)
        batch, time, dim = mean.shape
        distributions = []
        if self.full_cov:
            # bd is of shape (batch, time, time*latent_dim)
            bd = self.block_diagonal(encoded)
            # reshape
            bd = bd.view(batch, time, time, dim)
            for i in range(dim):
                a_matrix = bd[:, :, :, i]
                cov = torch.bmm(a_matrix, a_matrix.transpose(1, 2)) + eps * torch.eye(time, device=a_matrix.device).unsqueeze(0)
                distribution = torch.distributions.MultivariateNormal(mean[:, :, i], covariance_matrix=cov)            
                distributions.append(distribution)
        else:
            # bd is of shape (batch, time, 2*latent_dim)
            bd = self.block_diagonal(encoded)
            # bd = nn.Softplus()(bd)
            
            for i in range(dim):            
                diag_elems = nn.Softplus()(bd[:, :, i])
                # diag_elems = bd[:, :, i]
                off_diag_elems = bd[:, :-1, i+dim]
                # bd contains diagonal and off-diagonal elements. put them in a block diagonal matrix
                distribution = block_diag_precision(diag_elems, off_diag_elems, mean[:, :, i])                    
                distributions.append(distribution)
        
        return distributions
        
            
    def kl_divergence(self, distributions):
        # distributions is a list of MultivariateNormal distributions, one for each latent
        kl = 0
        for d in distributions:
            batch, time = d.mean.shape
            prior = torch.distributions.MultivariateNormal(torch.zeros(batch, time), scale_tril=self.prior_cholesky.repeat(batch, 1, 1))
            kl += torch.distributions.kl_divergence(d, prior).sum()
        return kl


class VAEGP(nn.Module):
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
        time_bins = int(2.5/config['shape_dataset']['win_len'])     

        model_config = config['vae_gp']        

        hidden_dim, num_layers = config['vae_gp']['rnn_encoder']['hidden_size'], config['vae_gp']['rnn_encoder']['num_layers']
        bidirectional = config['vae_gp']['rnn_encoder']['bidirectional']
        dropout = config['vae_gp']['rnn_encoder']['dropout']
        self.x_encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                              bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        self.x_posterior = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, self.x_dim)

        self.z_encoder = TimeSeries(config, input_dim, self.z_dim, time_bins)
        
        # reconstruction
        self.linear_maps = nn.ModuleList([nn.Linear(i, input_dim) for i in xz_list])                

        # beta for kl loss
        self.beta = model_config['kl_beta']

        # name model
        hidden_dim = model_config['rnn_encoder']['hidden_size']
        num_layers = model_config['rnn_encoder']['num_layers']
        self.arch_name = 'vae_unimodal_{}_{}_{}'.format(xz_list, hidden_dim, num_layers)        
        if neuron_bias is not None:
            self.arch_name += '_bias'
                            
        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['vae_gp']['lr'], weight_decay=config['vae_gp']['weight_decay'])        

        self.scheduler = None


    def forward(self, y, n_samples):
        # y is of shape (batch_size, seq_len, input_dim)
        batch, seq, _ = y.shape
        # x encoder
        encoded, _ = self.x_encoder(y)
        x = self.x_posterior(encoded)
        x_distributions = None
        z_distributions = self.z_encoder(y)
        # sample from distributions. shape is (batch*samples, seq, x/z)
        x_samples = torch.cat([x + torch.randn_like(x) for _ in range(n_samples)], dim=0)
        z_samples = torch.stack([d.sample((n_samples,)).view(n_samples*batch, seq) for d in z_distributions], dim=-1)        
        # sigmoid on z
        z_samples = torch.nn.Softmax(dim=2)(z_samples)        
        # print(z_samples.shape, x_samples.shape)
        
        # map x to observation
        Cx_list = [self.linear_maps[i](x_samples[:, :, s: e]) for i, (s, e) in enumerate(self.xz_l)]
        # if any element is a tuple, take the first element
        # Cx_list = [self.linear2(Cx[0]) if isinstance(Cx, tuple) else Cx for Cx in Cx_list]
        # print([x.shape for x in Cx_list])
        Cx = torch.stack(Cx_list, dim=-1)
        
        y_recon = torch.sum(Cx * z_samples.unsqueeze(2), dim=3)        

        # if self.neuron_bias is not None:
        #     y_recon = y_recon + self.neuron_bias
        y_recon = nn.Softplus()(y_recon)
        # # clamp
        y_recon = torch.clamp(y_recon, min=1e-10)
        
        return {'y_recon': y_recon, 'x_samples': x_samples, 'z_samples': z_samples, 'x_distributions': x_distributions, 'z_distributions': z_distributions}

    def loss(self, y, model_output):
        """
        y and y_recon are of shape [batch * n_samples, time, dim]
        mu and A are of shape (batch, seq, x/z) and (batch, x/z, seq, seq)
        """
        y_recon = model_output['y_recon']
        # x_distributions = model_output['x_distributions']
        z_distributions = model_output['z_distributions']
        
        batch, seq, _ = y.shape
        num_samples = y_recon.shape[0] // batch
        y = torch.cat([y]*num_samples, dim=0)        
        recon_loss = torch.sum(y_recon - y * torch.log(y_recon))
        
        # kl divergence
        # kl_x = self.x_encoder.kl_divergence(x_distributions)
        kl_z = self.z_encoder.kl_divergence(z_distributions)
        # kl = kl_x + kl_z        
        kl = kl_z        

        return (recon_loss + self.beta*kl)/(batch * num_samples)
    
    def extract_relevant(self, vae_output):
        y_recon = vae_output['y_recon'].detach().numpy()
        mean_x = None #torch.stack([x.mean for x in vae_output['x_distributions']], dim=-1).detach().numpy()
        mean_z = torch.stack([x.mean for x in vae_output['z_distributions']], dim=-1).detach().numpy()
        cov_x = None #torch.stack([x.covariance_matrix for x in vae_output['x_distributions']], dim=-1).detach().numpy()
        cov_z = torch.stack([x.covariance_matrix for x in vae_output['z_distributions']], dim=-1).detach().numpy()
        return y_recon, mean_x, mean_z, cov_x, cov_z, vae_output['x_samples'].detach().numpy(), vae_output['z_samples'].detach().numpy()