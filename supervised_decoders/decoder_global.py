import torch
import torch.nn as nn
import math
from supervised_decoders.utils import make_1d_conv


LOG2 = torch.log(torch.tensor(2))


class DecoderGlobal(nn.Module):
    """
    This class defines the amplitude decoder using a Convolutional Neural Network (CNN).   
    """
    def __init__(self, model_config, inp_dim, loss_weight, which_forward, z_dim, x_s_dim, x_dim_len):
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
        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=model_config['lr'], weight_decay=model_config['weight_decay'])          

        # which forward to use
        if which_forward == 'pool':
            self.cnn_forward = self.forward_pool
        elif which_forward == 'hardpeak':
            self.cnn_forward = self.forward_hardpeak
        elif which_forward == 'argmax':
            self.cnn_forward = self.forward_argmax
        elif which_forward == 'onlyx':
            self.cnn_forward = self.forward_onlyx
        else:            
            raise NotImplementedError
        
        # fix slices
        self.z_dim, self.x_s_dim, self.x_dim_len = z_dim, x_s_dim, x_dim_len

        # whether to normalize
        self.normalize_trials = model_config['normalize_trial_time']

    def forward_pool(self, x, z):
        x_part = x[:, self.x_s_dim: self.x_s_dim+self.x_dim_len, :]        
        z_part = z[:, self.z_dim: self.z_dim+1, :]     
        
        x_part = self.conv(x_part)
        x_part = x_part * z_part        
        # x_part = torch.max(x_part, dim=2).values
        x_part = torch.mean(x_part, dim=2)
        return x_part
    

    def forward_onlyx(self, x, z):
        x_part = x[:, self.x_s_dim: self.x_s_dim+self.x_dim_len, :]
        x_part = self.conv(x_part)
        x_part = torch.mean(x_part, dim=2)
        return x_part
    
    def forward_hardpeak(self, x, z):
        x_part = x[:, self.x_s_dim: self.x_s_dim+self.x_dim_len, :]        
        z_part = z[:, self.z_dim: self.z_dim+1, :]        

        # zero out all x values outside a window of 3 around the peak of z
        peak_z = torch.argmax(z_part, dim=2).squeeze(-1)                
        mask = torch.zeros_like(x[:, 0, :])                
        mask.scatter_(1, peak_z.unsqueeze(1), 1)
        mask.scatter_(1, torch.clip(peak_z-1, 0).unsqueeze(1), 1)
        mask.scatter_(1, torch.clip(peak_z+1, 0, z.shape[-1]-1).unsqueeze(1), 1)
        mask.scatter_(1, torch.clip(peak_z-2, 0).unsqueeze(1), 1)
        mask.scatter_(1, torch.clip(peak_z+2, 0, z.shape[-1]-1).unsqueeze(1), 1)
        # mask.scatter_(1, torch.clip(peak_z-3, 0).unsqueeze(1), 1)
        # mask.scatter_(1, torch.clip(peak_z+3, 0, z.shape[-1]-1).unsqueeze(1), 1)
        mask = mask.unsqueeze(1)                
        x_part = x_part * mask
        x_part = self.conv(x_part)        
        x_part = torch.mean(x_part, dim=2)
        # x_part = torch.max(x_part, dim=2).values
        return x_part
    
    def forward_argmax(self, x, z):
        x_part = x[:, self.x_s_dim: self.x_s_dim+self.x_dim_len, :]        
        # keep only those time bins of x where z argmax is z_dim
        z_argmax = torch.argmax(z, dim=1).squeeze(-1) == self.z_dim        
        # keep only those time bins of x where z is higher than null z
        # z_argmax = z[:, z_dim, :] > z[:, -1, :]
        mask = torch.zeros_like(x[:, 0, :])
        mask[z_argmax] = 1
        
        mask = mask.unsqueeze(1)
        x_part = x_part * mask
        x_part = self.conv(x_part)
        x_part = torch.mean(x_part, dim=2)
        # x_part = torch.max(x_part, dim=2).values
        return x_part

    def forward(self, x, z):
        # x is of shape (batch_size*num_samples, seq_len, input_dim) 

        # if normalize, time is across dim 2 after permute
        if self.normalize_trials:
            x = x - x.mean(dim=2, keepdim=True)
            # print('here')
            # # normalise by std
            # x = x - x[:, 0:1, :]
            # x = x / x.std(dim=1, keepdim=True)  
        
        # forward
        return self.cnn_forward(x, z)

    def loss(self, predicted, ground_truth, reduction='mean'):
        """
        Binary cross entropy loss
        predicted: (batch_size*num_samples, 4)
        ground_truth: (batch_size, 2)
        """        
        batch_size = ground_truth.size(0)
        num_samples = predicted.size(0)//batch_size
        # repeat ground truth        
        ground_truth = torch.cat([ground_truth]*num_samples, dim=0)
        
        # for bceloss
        loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)
        ground_truth = ground_truth.float()
        
        # for cel
        # loss_fn = nn.CrossEntropyLoss(reduction=reduction)

        return loss_fn(predicted, ground_truth) * self.weight


