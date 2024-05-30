import torch
import torch.nn as nn
import math
from supervised_decoders.utils import make_1d_conv


LOG2 = torch.log(torch.tensor(2))


class DecoderLocal(nn.Module):
    """
    This class defines the amplitude decoder using a Convolutional Neural Network (CNN).   
    """
    def __init__(self, model_config, inp_dim, loss_weight, z_dim, x_s_dim, x_dim_len):
        super().__init__()
        
        channels = model_config['channels']
        kernel_size = model_config['kernel_size']
        pad = (kernel_size - 1)//2
        dropout = model_config['dropout']        
        # make conv
        self.conv = nn.Sequential(*make_1d_conv(inp_dim, channels, kernel_size, pad, dropout))
        # name
        self.arch_name = 'ampdecoder_{}_{}'.format('-'.join([str(x) for x in channels]), kernel_size)
        # weight
        self.weight = loss_weight
        self.latent_dim = model_config['latent_dim']
        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=model_config['lr'], weight_decay=model_config['weight_decay'])          
        # fix slices
        self.z_dim, self.x_s_dim, self.x_dim_len = z_dim, x_s_dim, x_dim_len
        # whether to normalize
        self.normalize = model_config['normalize_trial_time']

    def forward(self, x, z):
        # x is of shape (batch_size*num_samples, seq_len, input_dim)                
        # z = z.detach()
        # x = x * z
        # take slices of x and z
        x = x[:, self.x_s_dim: self.x_s_dim+self.x_dim_len, :]
        z = z[:, self.z_dim: self.z_dim+1, :]        
        # if normalize, time is across dim 2 after permute
        if self.normalize_trials:
            x = x - x.mean(dim=2, keepdim=True)
            # print('here')
            # # normalise by std
            # x = x - x[:, 0:1, :]
            # x = x / x.std(dim=1, keepdim=True)  
        # forward                
        x = self.conv(x)        
        # gate with latent variable
        x = x * z
        return x

    def loss(self, predicted, ground_truth, reduction='mean'):
        """
        MSE loss
        predicted: (batch_size*num_samples, time, 1)
        ground_truth: (batch_size, time, 1)
        """
        batch_size = ground_truth.size(0)
        num_samples = predicted.size(0)//batch_size
        # repeat ground truth        
        ground_truth = torch.cat([ground_truth]*num_samples, dim=0)                
        loss_fn = nn.MSELoss(reduction=reduction)
        return loss_fn(predicted, ground_truth) * self.weight


