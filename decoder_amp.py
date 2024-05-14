import torch
import torch.nn as nn
import math


LOG2 = torch.log(torch.tensor(2))




class AmpDecoderCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        channels = config['decoder_amplitude']['channels']
        kernel_size = config['decoder_amplitude']['kernel_size']
        pad = (kernel_size - 1)//2
        dropout = config['decoder_amplitude']['dropout']        

        def make_1d_conv(inp_dim):
            # 1d conv
            layers = []            
            for i in range(len(channels)):
                if i == 0:
                    layers.append(nn.Conv1d(in_channels=inp_dim, out_channels=channels[i], kernel_size=kernel_size, padding=pad))
                else:
                    layers.append(nn.Conv1d(in_channels=channels[i-1], out_channels=channels[i], kernel_size=kernel_size, padding=pad))
                # layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.LeakyReLU())
                # layers.append(nn.Tanh())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))            
            # linear layer
            layers.append(nn.Conv1d(in_channels=channels[-1], out_channels=1, kernel_size=1))
            return layers
                                     
        # name
        self.arch_name = 'ampdecoder_{}_{}'.format('-'.join([str(x) for x in channels]), kernel_size)
        # weight
        self.weight = config['decoder_amplitude']['weight']
        self.latent_dim = config['decoder_amplitude']['latent_dim']
        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['decoder']['cnn']['lr'], weight_decay=config['decoder']['cnn']['weight_decay'])
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 90, 120, 150, 180], gamma=0.5)
        if config['decoder']['scheduler']['which'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=config['decoder']['scheduler']['cosine_restart_after'])
            print('Using cosine annealing for decoder')
        elif config['decoder']['scheduler']['which'] == 'decay':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config['decoder']['scheduler']['const_factor'])
            print('Using decay annealing for decoder')
        else:
            print('Scheduler not implemented for decoder')
            self.scheduler = None        

    def forward(self, x, z):
        # x is of shape (batch_size*num_samples, seq_len, input_dim)                
        # z = z.detach()
        # x = x * z
        x = x.permute(0, 2, 1)
        z = z.permute(0, 2, 1)
        
        x_stim = x[:, self.latent_dim:self.latent_dim+1, :]
        x_stim = self.conv_stim(x_stim)        
        x_stim = x_stim * z[:, self.latent_dim:self.latent_dim+1, :]
        return x_stim

    def loss(self, predicted, ground_truth, reduction='mean'):
        """
        MSE loss
        predicted: (batch_size*num_samples, time, 1)
        ground_truth: (batch_size*num_samples, time, 1)
        """
        batch_size = ground_truth.size(0)
        num_samples = predicted.size(0)//batch_size
        # repeat ground truth        
        ground_truth = torch.cat([ground_truth]*num_samples, dim=0)                
        loss_fn = nn.MSELoss(reduction=reduction)
        return loss_fn(predicted, ground_truth) * self.weight


