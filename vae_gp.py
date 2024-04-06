import torch
import torch.nn as nn
import math
import os
import matplotlib.pyplot as plt
import numpy as np
from vae_gp_separate import rbf_kernel, block_diag_precision, get_linear


eps = 1e-4


def derivative_time_series(x):    
    # x is of shape (batch, time, dim)
    # return derivative of x
    # pad x with zeros on both sides    
    zeros = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device)
    x = torch.cat([zeros, x, zeros], dim=1)
    # take difference
    dx = x[:, 2:, :] - x[:, :-2, :]
    return dx

def normal_cdf(x, nu):
    return torch.clip(0.5 * (1 + torch.erf(x * nu)), min=eps)


class CustomDistribution():
    def __init__(self, mean, A):
        assert len(mean.shape) == 2, "Mean should be of shape (batch_seq, dim)"
        self.mean = mean
        self.A = A
    def get_covariance_matrix(self):
        return torch.bmm(self.A, self.A.transpose(1, 2))
    def sample(self):
        batch_seq, dim = self.mean.shape
        eps = torch.randn(batch_seq, dim, 1, device=self.mean.device)
        # print(self.A.shape, z.shape)
        z = torch.bmm(self.A, eps).squeeze(2)        
        return z + self.mean
    def kl_divergence_normal(self):
        cov = self.get_covariance_matrix()
        det = torch.det(cov)
        return 0.5 * torch.sum(torch.sum(self.mean.pow(2), dim=1) + torch.einsum("...ii", cov) - self.mean.shape[1] - torch.log(det+(eps**2)))
   

class TimeSeriesCombined(nn.Module):
    
    def __init__(self, config, input_dim, latent_dim_z, latent_dim_x, time_bins):
        super().__init__()
        model_config = config['vae_gp']
        self.dim_z, self.dim_x = latent_dim_z, latent_dim_x
        # rnn
        hidden_dim = model_config['rnn_encoder']['hidden_size']
        num_layers = model_config['rnn_encoder']['num_layers']
        bidirectional = model_config['rnn_encoder']['bidirectional']
        dropout = model_config['rnn_encoder']['dropout']
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                          bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        # self.rnn = nn.Sequential(nn.Linear(input_dim, hidden_dim*(1+bidirectional)))
        # print number of parameters        
        self.name = 'gru_{}_{}_{}'.format(hidden_dim, num_layers, bidirectional)
        
        # posterior mean and block diagonal
        hidden_layers = model_config['post_rnn_linear']['hidden_dims']
        dropout = model_config['post_rnn_linear']['dropout']        

        inp_dim = hidden_dim*2 if bidirectional else hidden_dim
        self.posterior_mean = nn.Sequential(*get_linear(inp_dim, latent_dim_z+latent_dim_x, hidden_layers, dropout))
        
        self.cov_type = model_config['cov_type']
        if self.cov_type == 'full':
            self.post_z = nn.Sequential(*get_linear(inp_dim, time_bins*latent_dim_z, hidden_layers, dropout))            
        elif self.cov_type == 'banded':
            self.post_z = nn.Sequential(*get_linear(inp_dim, 2*latent_dim_z, hidden_layers, dropout))
        elif self.cov_type == 'diagonal':
            self.post_z = nn.Sequential(*get_linear(inp_dim, latent_dim_z, hidden_layers, dropout))
        else:
            raise ValueError('Invalid covariance type')
        
        self.cov_x = nn.Sequential(*get_linear(inp_dim, latent_dim_x*latent_dim_x, hidden_layers, dropout))
        
        self.smoothing_sigma = model_config['smoothing_sigma']
        if self.smoothing_sigma:            
            cov_prior = rbf_kernel(time_bins, self.smoothing_sigma)
            try:
                self.prior_cholesky = torch.linalg.cholesky(cov_prior)
            except:
                self.prior_cholesky = torch.linalg.cholesky(cov_prior + eps * torch.eye(time_bins))
                print('Cholesky for prior failed. Added epsilon to diagonal')        
        else:
            self.prior_cholesky = torch.eye(time_bins)
        
        self.name += '_smoothing_{}'.format(self.smoothing_sigma)

        # monotonicity constraint
        self.monotonic = model_config['monotonic']['use']
        if self.monotonic:
            self.nu_g = model_config['monotonic']['nu_g']
            self.nu_z = model_config['monotonic']['nu_z']
            self.g_net = nn.Sequential(*get_linear(inp_dim, latent_dim_z, hidden_layers, dropout))
            self.loss_coeff = model_config['monotonic']['coeff']
        
        # for each module, print number of parameters
        print('Number of trainable parameters in RNN:', sum(p.numel() for p in self.rnn.parameters()))
        print('Number of trainable parameters in Posterior Mean:', sum(p.numel() for p in self.posterior_mean.parameters()))
        print('Number of trainable parameters in Block Diagonal Z:', sum(p.numel() for p in self.post_z.parameters()))
        print('Number of trainable parameters in Cov X:', sum(p.numel() for p in self.cov_x.parameters()))

    
    def forward(self, y):
        encoded, _ = self.rnn(y)
        # encoded = self.rnn(y)
        # mean is of shape (batch, time, latent_dim)
        mean_both = self.posterior_mean(encoded)
        mean_z, mean_x = mean_both[:, :, :self.dim_z], mean_both[:, :, self.dim_z:]        
        
        # construct z distribution
        bd_z = self.post_z(encoded)
        z_distribution = []
        batch, time, dim = mean_z.shape            
        if self.cov_type == 'full':
            # bd is of shape (batch, time, time*(latent_dim_z+latent_dim_x))                
            # reshape
            bd = bd_z.view(batch, time, time, dim)
            for i in range(dim):
                a_matrix = bd[:, :, :, i]
                cov = torch.bmm(a_matrix, a_matrix.transpose(1, 2)) + eps * torch.eye(time, device=a_matrix.device).unsqueeze(0)
                distribution = torch.distributions.MultivariateNormal(mean_z[:, :, i], covariance_matrix=cov)
                z_distribution.append(distribution)                
        elif self.cov_type == 'banded':
            # bd is of shape (batch, time, 2*latent_dim)            
            for i in range(dim):                    
                # diag_elems = nn.Softplus()(bd[:, :, i])
                diag_elems = bd_z[:, :, i]
                off_diag_elems = bd_z[:, :-1, i+dim]
                # bd contains diagonal and off-diagonal elements. put them in a block diagonal matrix
                distribution = block_diag_precision(diag_elems, off_diag_elems, mean_z[:, :, i])
                z_distribution.append(distribution)                
        elif self.cov_type == 'diagonal':
            # bd is of shape (batch, time, latent_dim)            
            for i in range(dim):                    
                diag_elems = (bd_z[:, :, i])**2
                # contruct a distribution with diagonal elements
                # distribution = torch.distributions.MultivariateNormal(mean_z[:, :, i], scale_tril=torch.diag_embed(diag_elems))
                distribution = CustomDistribution(mean_z[:, :, i], torch.diag_embed(diag_elems))
                z_distribution.append(distribution)
        else:
            raise ValueError('Invalid covariance type')

        # construct x distribution
        cov_x = self.cov_x(encoded)
        batch, time, dim = mean_x.shape
        mean_x_flat = mean_x.view(batch*time, dim)        
        cov_x_flat = cov_x.view(batch*time, dim, dim)        
        x_distribution = CustomDistribution(mean_x_flat, cov_x_flat)

        to_ret_dict = {'z_distributions': z_distribution, 'x_distribution': x_distribution}
        # monotonicity constraint        
        if self.monotonic:
            # g is of shape (batch, time, latent_dim_z)
            out_g = self.g_net(encoded)
            to_ret_dict['g'] = out_g

        return to_ret_dict
        
            
    def kl_divergence_z(self, distributions):
        # distributions is a list of MultivariateNormal distributions, one for each latent
        kl = 0
        for d in distributions:
            batch, time = d.mean.shape
            prior = torch.distributions.MultivariateNormal(torch.zeros(batch, time), scale_tril=self.prior_cholesky.repeat(batch, 1, 1))
            kl += torch.distributions.kl_divergence(d, prior).sum()
        return kl    


class VAEGPCombined(nn.Module):
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

        self.zx_encoder = TimeSeriesCombined(config, input_dim, self.z_dim, self.x_dim, time_bins)                
        
        # reconstruction
        self.linear_maps = nn.ModuleList([nn.Linear(i, input_dim) for i in xz_list])                

        # beta for kl loss
        self.beta = model_config['kl_beta']

        if self.zx_encoder.smoothing_sigma:
            assert self.beta > 0, "Beta should be greater than 0 for smoothing sigma"

        # name model        
        self.arch_name = 'vae_gp_{}_'.format(xz_list)
        if neuron_bias is not None:
            self.arch_name += '_bias'
        self.arch_name += '_{}'.format(self.zx_encoder.name)
        self.arch_name += '_{}'.format(self.beta)
                            
        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['vae_gp']['lr'], weight_decay=config['vae_gp']['weight_decay'])        

        self.scheduler = None


    def forward(self, y, n_samples):
        # y is of shape (batch_size, seq_len, input_dim)
        batch, seq, _ = y.shape         
        out_dict = self.zx_encoder(y)
        z_distributions, x_distribution = out_dict['z_distributions'], out_dict['x_distribution']
        # sample from distributions. shape is (batch*samples, seq, x/z)        
        # z_samples = torch.stack([d.sample((n_samples,)).view(n_samples*batch, seq) for d in z_distributions], dim=-1)        
        z_samples = torch.stack([torch.cat([z_distribution.sample() for _ in range(n_samples)], dim=0) for z_distribution in z_distributions], dim=-1)
        # print(z_samples.shape)
        x_samples = torch.cat([x_distribution.sample().view(batch, seq, -1) for _ in range(n_samples)], dim=0)        
        # print(x_samples.shape, z_samples.shape)
        # sigmoid on z
        z_samples = torch.nn.Softmax(dim=2)(z_samples)
        # z_samples = nn.Sigmoid()(z_samples)
        
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
        
        # add keys to out_dict
        out_dict['y_recon'] = y_recon
        out_dict['x_samples'] = x_samples
        out_dict['z_samples'] = z_samples

        return out_dict

    def loss(self, y, model_output):
        """
        y and y_recon are of shape [batch * n_samples, time, dim]
        mu and A are of shape (batch, seq, x/z) and (batch, x/z, seq, seq)
        """
        y_recon = model_output['y_recon']
        x_distribution = model_output['x_distribution']
        z_distributions = model_output['z_distributions']
        z_samples = model_output['z_samples']
        
        batch, seq, _ = y.shape
        num_samples = y_recon.shape[0] // batch
        y = torch.cat([y]*num_samples, dim=0)        
        recon_loss = torch.sum(y_recon - y * torch.log(y_recon))                

        loss = recon_loss
        if self.beta:            
            # kl divergence
            kl_x = x_distribution.kl_divergence_normal()
            # kl_z = self.zx_encoder.kl_divergence_z(z_distributions)
            kl_z = 0
            for d in z_distributions:
                kl_z += d.kl_divergence_normal()
            kl = kl_x + kl_z
            loss += self.beta * kl

        # kl_x = x_distribution.kl_divergence_normal()
        # loss += kl_x
        
        # check for monotonicity constraint
        if 'g' in model_output:
            g = model_output['g']
            g = torch.cat([g]*num_samples, dim=0)
            nu_g = self.zx_encoder.nu_g
            nu_z = self.zx_encoder.nu_z
            coef = self.zx_encoder.loss_coeff
            # g is of shape (batch, time, latent_dim_z)
            # just cdf loss
            g_prime = derivative_time_series(g)
            f_prime = derivative_time_series(z_samples)
            # print(coef * torch.sum(torch.log(normal_cdf(g_prime, nu_g))))            
            l1 = -coef * torch.sum(torch.log(normal_cdf(g_prime, nu_g)))
            # term 1
            t1 = normal_cdf(f_prime, nu_z) * normal_cdf(-g, nu_g)
            # term 2
            t2 = normal_cdf(-f_prime, nu_z) * normal_cdf(g, nu_g)            
            l2 = -coef * torch.sum(torch.log(t1 + t2))
            # print(l1, l2, loss)
            loss += l1 + l2

        return loss/(batch * num_samples)
    
    def extract_relevant(self, vae_output):
        y_recon = vae_output['y_recon'].detach().numpy()
        mean_x = None #torch.stack([x.mean for x in vae_output['x_distributions']], dim=-1).detach().numpy()
        mean_z = torch.stack([x.mean for x in vae_output['z_distributions']], dim=-1).detach().numpy()        
        cov_x = vae_output['x_distribution'].get_covariance_matrix().detach().numpy()
        cov_z = None#torch.stack([x.covariance_matrix for x in vae_output['z_distributions']], dim=-1).detach().numpy()
        return y_recon, mean_x, mean_z, cov_x, cov_z, vae_output['x_samples'].detach().numpy(), vae_output['z_samples'].detach().numpy()