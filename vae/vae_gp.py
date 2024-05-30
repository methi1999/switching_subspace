import torch
import torch.nn as nn
import math
import os
import matplotlib.pyplot as plt
import numpy as np
from vae.utils import derivative_time_series, normal_cdf, get_linear_layers, rbf_kernel, get_eps


USING_TORCH_DIST = False


def block_diag_precision(diag_elems, off_diag_elems, mean, using_torch_dist):        
    B = torch.diag_embed(diag_elems) + torch.diag_embed(off_diag_elems, offset=1, dim1=-2, dim2=-1)
    return CustomDistribution(mean, cholesky_cov=B)
    

class CustomDistribution():
    def __init__(self, mean, cholesky_cov=None, cov=None, cholesky_prec=None):
        assert len(mean.shape) == 2, "Mean should be of shape (batch_seq, dim)"
        self.mean = mean
        self.cholesky_cov = cholesky_cov
        self.cov = cov
        self.cholesky_prec = cholesky_prec
        
        assert self.cholesky_cov is not None or self.cholesky_prec is not None, "Either cholesky_cov or cov should be provided"

        if cholesky_prec is not None and cholesky_cov is None:
            assert self.cholesky_cov is None, "Cannot have both cholesky_cov and cholesky_prec"            
            self.cholesky_cov = torch.linalg.solve_triangular(cholesky_prec, torch.eye(cholesky_prec.shape[-1], device=cholesky_prec.device).unsqueeze(0), upper=True)
            # self.cholesky_cov = bidiagonal_inverse_batched(torch.diagonal(cholesky_prec, dim1=-2, dim2=-1), torch.diagonal(cholesky_prec, offset=1, dim1=-2, dim2=-1))            
            
    def get_covariance_matrix(self):
        if self.cov is None:            
            self.cov = torch.bmm(self.cholesky_cov, self.cholesky_cov.transpose(1, 2))

        return self.cov
    
    def sample(self):
        batch_seq, dim = self.mean.shape
        eps = torch.randn(batch_seq, dim, 1, device=self.mean.device)        
        z = torch.bmm(self.cholesky_cov, eps).squeeze(2)
        return z + self.mean
    
    def kl_divergence_normal(self):        
        cov = self.get_covariance_matrix()
        log_det_term = -torch.logdet(cov)
        trace_term = torch.einsum("...ii", cov)
        mu_term = torch.sum(self.mean.pow(2), dim=1)        
                    
        return 0.5 * torch.sum(mu_term + trace_term - self.mean.shape[1] + log_det_term)
    
    def kl_divergence_any(self, p_mean, p_cov, p_inv, p_log_det):                
        # KL(q||p) where q is the current distribution and p is the prior

        # insert batch dimension for p_cov if not present        
        if len(p_inv.shape) == 2:
            # repeat p
            p_cov = p_cov.unsqueeze(0).repeat(self.mean.shape[0], 1, 1)
            p_inv = p_inv.unsqueeze(0).repeat(self.mean.shape[0], 1, 1)
            p_log_det = p_log_det.unsqueeze(0).repeat(self.mean.shape[0])
        q_cov = self.get_covariance_matrix()        
        # take log det directly
        # q_log_det = torch.logdet(q_cov)
        # print(torch.diagonal(self.cholesky_cov, dim1=-2, dim2=-1)[0])
        # since we have the cholesky decompostion, we can directly take log of diagonals
        if self.cholesky_prec is not None:            
            q_log_det = -2 * torch.sum(torch.log(torch.diagonal(self.cholesky_prec, dim1=-2, dim2=-1)), dim=1)
        else:
            q_log_det = torch.logdet(q_cov)

        log_det_term = p_log_det - q_log_det
        # trace_term = torch.einsum("...ii", torch.bmm(p_inv, q_cov))
        trace_term = torch.einsum("...ii", torch.linalg.solve(p_cov, q_cov))

        diff_term = (self.mean - p_mean).unsqueeze(-1)
        # mu_term = torch.sum(torch.bmm(diff_term.transpose(1, 2), torch.bmm(p_inv, diff_term)))        
        mu_term = torch.sum(diff_term * torch.linalg.solve(p_cov, diff_term), dim=(1, 2))

        return 0.5 * torch.sum(mu_term + trace_term - self.mean.shape[1] + log_det_term)
        
   

class TimeSeriesCombined(nn.Module):
    
    def __init__(self, config, input_dim, latent_dim_z, latent_dim_x, time_bins):
        super().__init__()
        model_config = config['vae_gp']
        self.dim_z, self.dim_x = latent_dim_z, latent_dim_x
        self.apply_softplus = model_config['apply_softplus']
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
        self.posterior_mean_z = nn.Sequential(*get_linear_layers(inp_dim, latent_dim_z, hidden_layers, dropout))
        self.posterior_mean_x = nn.Sequential(*get_linear_layers(inp_dim, latent_dim_x, hidden_layers, dropout))
        # self.posterior_mean = nn.Sequential(*get_linear_layers(inp_dim, latent_dim_x+latent_dim_z, hidden_layers, dropout))
        
        self.cov_type = model_config['cov_type']
        if self.cov_type == 'full':
            self.post_z = nn.Sequential(*get_linear_layers(inp_dim, time_bins*latent_dim_z, hidden_layers, dropout))            
        elif self.cov_type == 'banded':
            self.post_z = nn.Sequential(*get_linear_layers(inp_dim, 2*latent_dim_z, hidden_layers, dropout))
        elif self.cov_type == 'diagonal':
            self.post_z = nn.Sequential(*get_linear_layers(inp_dim, latent_dim_z, hidden_layers, dropout))
        else:
            raise ValueError('Invalid covariance type')
        
        ### gp on x
        self.cov_x = nn.Sequential(*get_linear_layers(inp_dim, latent_dim_x*latent_dim_x, hidden_layers, dropout))
        # self.cov_x = nn.Sequential(*get_linear_layers(inp_dim, latent_dim_x, hidden_layers, dropout))
        
        # gp parameters
        self.smoothing_sigma = model_config['smoothing_sigma']
        if self.smoothing_sigma:            
            self.cov_prior = rbf_kernel(time_bins, self.smoothing_sigma)
            self.cov_prior *= model_config['kernel_scale']            
            self.cov_prior += model_config['noise_sigma'] * torch.eye(time_bins)
            try:
                self.prior_cholesky = torch.linalg.cholesky(self.cov_prior)
            except:
                self.prior_cholesky = torch.linalg.cholesky(self.cov_prior + get_eps() * torch.eye(time_bins))
                print('Cholesky for prior failed. Added epsilon to diagonal')        
            self.prior_inv = torch.linalg.inv(self.cov_prior)            
            self.prior_log_det = torch.logdet(self.cov_prior)
            
            print('Log det:', self.prior_log_det, 'Inverse max: ', self.prior_inv.max(), 'Covariance max: ', self.cov_prior.max())
            self.name += '_noise_{}_rbfscale_{}'.format(model_config['noise_sigma'], model_config['kernel_scale'])
        else:
            self.prior_cholesky = torch.eye(time_bins)        
        
        self.name += '_smoothing_{}'.format(self.smoothing_sigma)

        # monotonicity constraint
        self.monotonic = model_config['monotonic']['use']
        if self.monotonic:
            self.nu_g = model_config['monotonic']['nu_g']
            self.nu_z = model_config['monotonic']['nu_z']
            self.g_net = nn.Sequential(*get_linear_layers(inp_dim, latent_dim_z, hidden_layers, dropout))
            self.loss_coeff = model_config['monotonic']['coeff']            
            self.monotonic_mask = model_config['monotonic']['mask']
            assert len(self.monotonic_mask) == latent_dim_z, "Mask should have same length as latent_dim_z"
            assert sum(self.monotonic_mask) > 0, "at least one element should be 1 in mask"
            self.name += '_monotonic_{}_{}_{}_{}'.format(self.nu_g, self.nu_z, model_config['monotonic']['coeff'], model_config['monotonic']['mask'])            

        # load pre-trained GP weights
        if model_config['load_stage1']:
            print('Loading weights for s1')
            weights = torch.load(model_config['load_stage1'])
            self.load_state_dict(weights, strict=False)
            print("Loaded following weights:", set(weights.keys()).intersection(set(self.state_dict().keys())))
            if model_config['freeze_encoder_meanz']:
                self.rnn.requires_grad_(False)
                self.posterior_mean_z.requires_grad_(False)
                print('Encoder and Posterior Mean Z frozen')        

        # for each module, print number of parameters
        print('Number of trainable parameters in RNN:', sum(p.numel() for p in self.rnn.parameters() if p.requires_grad))
        print('Number of trainable parameters in Posterior Mean X:', sum(p.numel() for p in self.posterior_mean_x.parameters()))
        print('Number of trainable parameters in Posterior Mean Z:', sum(p.numel() for p in self.posterior_mean_z.parameters()))
        print('Number of trainable parameters in Block Diagonal Z:', sum(p.numel() for p in self.post_z.parameters()))
        print('Number of trainable parameters in Cov X:', sum(p.numel() for p in self.cov_x.parameters()))

    
    def forward(self, y):
        encoded, _ = self.rnn(y)
        # encoded = self.rnn(y)
        # mean is of shape (batch, time, latent_dim)
        mean_z = self.posterior_mean_z(encoded)
        mean_x = self.posterior_mean_x(encoded)
        # m = self.posterior_mean(encoded)
        # mean_z, mean_x = m[:, :, :self.dim_z], m[:, :, self.dim_z:]
        
        # construct z distribution
        bd_z = self.post_z(encoded)
        if self.apply_softplus:
            bd_z = nn.Softplus()(bd_z)
        z_distribution = []
        batch, time, dim = mean_z.shape            
        if self.cov_type == 'full':
            # bd is of shape (batch, time, time*(latent_dim_z+latent_dim_x))                
            # reshape
            bd = bd_z.view(batch, time, time, dim)
            for i in range(dim):
                a_matrix = bd[:, :, :, i]                
                # distribution = torch.distributions.MultivariateNormal(mean_z[:, :, i], covariance_matrix=cov)
                distribution = CustomDistribution(mean_z[:, :, i], cholesky_cov=a_matrix)
                z_distribution.append(distribution)                
        elif self.cov_type == 'banded':
            # bd is of shape (batch, time, 2*latent_dim)            
            for i in range(dim):                    
                diag_elems = nn.Softplus()(bd_z[:, :, i])
                # diag_elems = bd_z[:, :, i]
                off_diag_elems = bd_z[:, :-1, i+dim]
                # bd contains diagonal and off-diagonal elements. put them in a block diagonal matrix                
                distribution = block_diag_precision(diag_elems, off_diag_elems, mean_z[:, :, i], USING_TORCH_DIST)
                z_distribution.append(distribution)                
        elif self.cov_type == 'diagonal':
            # bd is of shape (batch, time, latent_dim)            
            for i in range(dim):                    
                # diag_elems = torch.exp(bd_z[:, :, i])
                diag_elems = (bd_z[:, :, i])**2
                # contruct a distribution with diagonal elements
                # distribution = torch.distributions.MultivariateNormal(mean_z[:, :, i], scale_tril=torch.diag_embed(diag_elems))                
                # distribution = CustomDistribution(mean_z[:, :, i], cholesky_cov=torch.diag_embed(diag_elems))
                distribution = CustomDistribution(mean_z[:, :, i], cholesky_cov=torch.diag_embed(diag_elems), cov=torch.diag_embed(diag_elems**2))
                z_distribution.append(distribution)
        else:
            raise ValueError('Invalid covariance type')

        ### gp on x
        # construct x distribution
        cov_x = self.cov_x(encoded)
        batch, time, dim = mean_x.shape
        mean_x[:, :, :2] = mean_x[:, :, :2] - mean_x[:, 0:1, :2]
        mean_x_flat = mean_x.view(batch*time, dim)        
        cov_x_flat = cov_x.view(batch*time, dim, dim)
        x_distribution = CustomDistribution(mean_x_flat, cholesky_cov=cov_x_flat)
        
        # x_distribution = []
        # mean_x[:, :, :2] = mean_x[:, :, :2] - mean_x[:, 0:1, :2]
        # cov_x = self.cov_x(encoded)[:]
        # batch, time, dim = mean_x.shape
        # for i in range(dim):                    
        #     # diag_elems = torch.exp(bd_z[:, :, i])
        #     diag_elems = (cov_x[:, :, i])**2
        #     # contruct a distribution with diagonal elements
        #     # distribution = torch.distributions.MultivariateNormal(mean_z[:, :, i], scale_tril=torch.diag_embed(diag_elems))                
        #     distribution = CustomDistribution(mean_x[:, :, i], cholesky_cov=torch.diag_embed(diag_elems))
        #     x_distribution.append(distribution)


        to_ret_dict = {'z_distributions': z_distribution, 'x_distribution': x_distribution}
        # monotonicity constraint        
        if self.monotonic:
            # g is of shape (batch, time, latent_dim_z)
            out_g = self.g_net(encoded)
            # appluy tanh
            out_g = torch.tanh(out_g)
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


class VAEGP(nn.Module):
    def __init__(self, config, input_dim, neuron_bias=None, init=''):
        super().__init__()
        
        self.config = config
        # form start and end indices for x and z
        dim_x_z = config['dim_x_z']
        xz_ends = torch.cumsum(torch.tensor(dim_x_z), dim=0)
        xz_starts = torch.tensor([0] + xz_ends.tolist()[:-1])
        self.xz_l = torch.stack([xz_starts, xz_ends], dim=1).tolist()
            
        self.x_dim = sum(dim_x_z)
        self.z_dim = len(dim_x_z)
        time_bins = int(2.5/config['shape_dataset']['win_len'])     

        model_config = config['vae_gp']        

        self.zx_encoder = TimeSeriesCombined(config, input_dim, self.z_dim, self.x_dim, time_bins)                
        
        # reconstruction
        self.linear_maps = nn.ModuleList([nn.Linear(i, input_dim) for i in dim_x_z])                
        # self.linear_maps = nn.ModuleList([nn.Linear(i, input_dim, bias=False) for i in dim_x_z])                
        # self.linear_maps = nn.ModuleList([nn.Sequential(nn.Linear(i, 8), nn.Tanh(), nn.Linear(8, input_dim)) for i in dim_x_z])                
        self.softmax_temp = model_config['softmax_temp']

        # beta for kl loss
        self.beta = model_config['kl_beta']

        # neuron bias
        self.neuron_bias = neuron_bias

        # using GP
        self.using_gp = model_config['smoothing_sigma'] is not None
        
        # name model        
        self.arch_name = 'vae_gp_{}_{}'.format(dim_x_z, model_config['cov_type'])
        if neuron_bias is not None:
            self.arch_name += '_bias'
        self.arch_name += '_{}'.format(self.zx_encoder.name)
        self.arch_name += '_{}'.format(self.beta)        
        self.arch_name += '_seed_{}'.format(config['seed'])
                            
        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['vae_gp']['lr'], weight_decay=config['vae_gp']['weight_decay'])        
        # disentangle x and z across stimulus and choice     
        self.disentangle = config['vae_gp']['disentangle']   


    def forward(self, y, n_samples, use_mean_for_decoding):
        # y is of shape (batch_size, seq_len, input_dim)
        batch, seq, _ = y.shape         
        out_dict = self.zx_encoder(y)
        z_distributions, x_distributions = out_dict['z_distributions'], out_dict['x_distribution']        
        # sample from distributions. shape is (batch*samples, seq, x/z)        
        # z_samples = torch.stack([d.sample((n_samples,)).view(n_samples*batch, seq) for d in z_distributions], dim=-1)        
        if use_mean_for_decoding:
            assert n_samples == 1, "n_samples should be 1 if using mean for decoding"
            z_samples_pre = torch.stack([z_distribution.mean for z_distribution in z_distributions], dim=-1)
            z_samples = torch.nn.Softmax(dim=2)(z_samples_pre/self.softmax_temp)
            # print(z_samples.shape)
            ### gp on x
            x_samples = x_distributions.mean.view(batch, seq, -1)
            # x_samples = torch.stack([torch.cat([x_distribution.sample() for _ in range(n_samples)], dim=0) for x_distribution in x_distributions], dim=-1)        
        else:
            z_samples_pre = torch.stack([torch.cat([z_distribution.sample() for _ in range(n_samples)], dim=0) for z_distribution in z_distributions], dim=-1)
            z_samples = torch.nn.Softmax(dim=2)(z_samples_pre/self.softmax_temp)
            # print(z_samples.shape)
            ### gp on x
            x_samples = torch.cat([x_distributions.sample().view(batch, seq, -1) for _ in range(n_samples)], dim=0)                
            # x_samples = torch.stack([torch.cat([x_distribution.sample() for _ in range(n_samples)], dim=0) for x_distribution in x_distributions], dim=-1)        
        
        # normalise each weight in linear map to norm 1
        # for lin_map in self.linear_maps:
        #     lin_map.weight.data = nn.functional.normalize(lin_map.weight, p=2, dim=1)        

        # map from latent to y space
        Cx_list = [self.linear_maps[i](x_samples[:, :, s: e]) for i, (s, e) in enumerate(self.xz_l)]        
        Cx = torch.stack(Cx_list, dim=-1)        
        
        # sum
        y_recon = torch.sum(Cx * z_samples.unsqueeze(2), dim=3)        

        if self.neuron_bias is not None:
            y_recon = y_recon + self.neuron_bias
        y_recon = nn.Softplus()(y_recon)
        
        # add keys to out_dict
        out_dict['y_recon'] = y_recon
        out_dict['x_samples'] = x_samples
        out_dict['z_samples'] = z_samples
        out_dict['z_samples_pre'] = z_samples_pre

        return out_dict

    def loss(self, y, model_output, behavior):
        """
        y and y_recon are of shape [batch * n_samples, time, dim]
        mu and A are of shape (batch, seq, x/z) and (batch, x/z, seq, seq)
        """
        y_recon = model_output['y_recon']
        x_distribution = model_output['x_distribution']
        z_distributions = model_output['z_distributions']
        z_samples = model_output['z_samples']
        z_mean = torch.stack([x.mean for x in z_distributions], dim=-1)
        
        batch, seq, _ = y.shape
        num_samples = y_recon.shape[0] // batch
        y = torch.cat([y]*num_samples, dim=0)        
        recon_loss = torch.sum(y_recon - y * torch.log(y_recon))                
        loss = recon_loss
        # loss = recon_loss * 0.01

        if self.beta:            
            # kl divergence
            ### gp on x
            kl_x = x_distribution.kl_divergence_normal()
            if USING_TORCH_DIST:
                kl_z = self.zx_encoder.kl_divergence_z(z_distributions)            
            else:
                kl_z = 0
                for d in z_distributions:
                    if self.using_gp:
                        kld = d.kl_divergence_any(torch.zeros_like(d.mean), self.zx_encoder.cov_prior, self.zx_encoder.prior_inv, self.zx_encoder.prior_log_det)
                        kl_z += kld
                    else:
                        kl_z += d.kl_divergence_normal()
                ### gp on x
                # kl_x = 0
                # for d in x_distribution:
                #     if self.using_gp:
                #         kld = d.kl_divergence_any(torch.zeros_like(d.mean), self.zx_encoder.cov_prior, self.zx_encoder.prior_inv, self.zx_encoder.prior_log_det)
                #         kl_x += kld
                #     else:
                #         kl_x += d.kl_divergence_normal()
                     
            kl = kl_x + kl_z
            loss += self.beta * kl            

        # kl_x = x_distribution.kl_divergence_normal()
        # loss += kl_x
        
        # check for monotonicity constraint
        if 'g' in model_output:
            g = model_output['g']
            nu_g = self.zx_encoder.nu_g
            nu_z = self.zx_encoder.nu_z
            coef = self.zx_encoder.loss_coeff
            mask = self.zx_encoder.monotonic_mask
            # g is of shape (batch, time, latent_dim_z)
            # just cdf loss
            g = torch.cat([g]*num_samples, dim=0)
            g_prime = derivative_time_series(g)            
            f_prime = derivative_time_series(z_samples)
            # f_prime = derivative_time_series(z_mean)
            # print(f_prime)
            
            l1 = -coef * torch.log(normal_cdf(g_prime, nu_g))[:, :, mask].sum()
            # term 1
            t1 = normal_cdf(f_prime, nu_z) * normal_cdf(-g, nu_g)
            # term 2
            t2 = normal_cdf(-f_prime, nu_z) * normal_cdf(g, nu_g)
            # t2 = normal_pdf(f_prime, l=0.1) * normal_cdf(g, nu_g)
            # t2 = normal_cdf(-f_prime, nu=5, offset=0.75) * normal_cdf(g, nu_g)
            l2 = -coef * torch.log(t1 + t2)[:, :, mask].sum()
            # print(l1, l2, loss)
            loss += l1 + l2
            # loss += -coef * torch.log(normal_cdf(f_prime, nu_z)[:, :, mask]).sum()
            # loss for ensuring g0 reaches 0 before g1
            # loss += 0.2 * torch.clamp((g[:, :, 1] - g[:, :, 0]), max=0).sum()
            # loss += coef * torch.clamp((torch.sign(g[:, :, 1]) - torch.sign(g[:, :, 0])), min=0).sum()
            # print(torch.clamp((torch.sign(g[:, :, 1]) - torch.sign(g[:, :, 0])), min=0).sum())
            # loss += 10 * coef * torch.clamp((torch.sigmoid(5*g[:, :, 1]) - torch.sigmoid(5*g[:, :, 0])), min=0).sum()
            # ensures that g0 is ahead of g1 by at least 1 time step
            # rolled_g0 = torch.roll(torch.sign(g[:, :, 0]), shifts=1, dims=1)
            # rolled_g0 = torch.roll(torch.tanh(5*g[:, :, 0]), shifts=1, dims=1)
            # rolled_g0[:, 0] = -1
            # rolled_g0[:, 1] = -1
            # loss += coef * torch.clamp((torch.sign(g[:, :, 1]) - rolled_g0), min=0).sum()
            # loss += coef * torch.clamp((torch.tanh(5*g[:, :, 1]) - rolled_g0), min=0).sum()


        if self.disentangle and len(self.config['decoder']['which']):
            same_term = 500
            cross_terms = 1000
            # trial-averaged x3 where behavior is 1
            stim, choice = behavior[:, 0], behavior[:, 1]
            mean_reshaped = x_distribution.mean.view(batch, seq, self.x_dim)
            x0 = mean_reshaped[:, :, 0]
            x1 = mean_reshaped[:, :, 1]
            x2 = mean_reshaped[:, :, 2]            
            # group x2 across both
            x_stim_left, x_stim_right = x2[stim == 1].mean(dim=0), x2[stim == 0].mean(dim=0)
            x_choice_left, x_choice_right = x2[choice == 1].mean(dim=0), x2[choice == 0].mean(dim=0)
            loss += (x_stim_left - x_stim_right).pow(2).sum() * same_term            
            loss += (x_choice_left - x_choice_right).pow(2).sum() * same_term
            # group x0 across choice
            x0_choice_left, x0_choice_right = x0[choice == 1].mean(dim=0), x0[choice == 0].mean(dim=0)
            loss += (x0_choice_left - x0_choice_right).pow(2).sum() * cross_terms
            # group x1 across stimulus
            x1_stim_left, x1_stim_right = x1[stim == 1].mean(dim=0), x1[stim == 0].mean(dim=0)
            loss += (x1_stim_left - x1_stim_right).pow(2).sum() * cross_terms
            
            # group z
            z0, z1, z2 = z_mean[:, :, 0], z_mean[:, :, 1], z_mean[:, :, 2]
            # group z2 across null
            z2_stim_left, z2_stim_right = z2[stim == 1].mean(dim=0), z2[stim == 0].mean(dim=0)
            z2_choice_left, z2_choice_right = z2[choice == 1].mean(dim=0), z2[choice == 0].mean(dim=0)
            loss += (z2_stim_left - z2_stim_right).pow(2).sum() * same_term
            loss += (z2_choice_left - z2_choice_right).pow(2).sum() * same_term
            # group z0 across choice
            z0_choice_left, z0_choice_right = z0[choice == 1].mean(dim=0), z0[choice == 0].mean(dim=0)
            loss += (z0_choice_left - z0_choice_right).pow(2).sum() * cross_terms
            # group z1 across stimulus
            z1_stim_left, z1_stim_right = z1[stim == 1].mean(dim=0), z1[stim == 0].mean(dim=0)
            loss += (z1_stim_left - z1_stim_right).pow(2).sum() * cross_terms
            # group z0 across stimulus
            z0_stim_left, z0_stim_right = z0[stim == 1].mean(dim=0), z0[stim == 0].mean(dim=0)
            loss += (z0_stim_left - z0_stim_right).pow(2).sum() * same_term
            # group z1 across choice
            z1_choice_left, z1_choice_right = z1[choice == 1].mean(dim=0), z1[choice == 0].mean(dim=0)
            loss += (z1_choice_left - z1_choice_right).pow(2).sum() * same_term

        return loss/(batch * num_samples)
    
    def extract_relevant(self, vae_output):
        if USING_TORCH_DIST:
            y_recon = vae_output['y_recon'].detach().numpy()
            mean_x = vae_output['x_distribution'].mean.detach().numpy()
            mean_z = torch.stack([x.mean for x in vae_output['z_distributions']], dim=-1).detach().numpy()                    
            cov_z = torch.stack([x.covariance_matrix for x in vae_output['z_distributions']], dim=-1).detach().numpy()
            cov_x = vae_output['x_distribution'].get_covariance_matrix().detach().numpy()
        else:
            y_recon = vae_output['y_recon'].detach().numpy()
            batch, time, _ = y_recon.shape
            ### gp on x
            mean_x = vae_output['x_distribution'].mean.detach().numpy().reshape(batch, time, -1)
            cov_x = vae_output['x_distribution'].get_covariance_matrix().detach().numpy().reshape(batch, time, self.x_dim, self.x_dim)
            # mean_x = torch.stack([x.mean for x in vae_output['x_distribution']], dim=-1).detach().numpy()
            # cov_x = torch.stack([x.get_covariance_matrix() for x in vae_output['x_distribution']], dim=-1).detach().numpy()
            mean_z = torch.stack([x.mean for x in vae_output['z_distributions']], dim=-1).detach().numpy()
            cov_z = torch.stack([x.get_covariance_matrix() for x in vae_output['z_distributions']], dim=-1).detach().numpy()
        if self.zx_encoder.monotonic:
            g = vae_output['g'].detach().numpy()
        else:
            g = None
        return y_recon, mean_x, mean_z, cov_x, cov_z, vae_output['x_samples'].detach().numpy(), vae_output['z_samples'].detach().numpy(), vae_output['z_samples_pre'].detach().numpy(), g
    

    