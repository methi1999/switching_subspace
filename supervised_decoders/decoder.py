import torch
import torch.nn as nn
import math
import random
from supervised_decoders.decoder_global import DecoderGlobal
from supervised_decoders.decoder_local import DecoderLocal

LOG2 = torch.log(torch.tensor(2))


class CNNDecoderIndividual(nn.Module):
    def __init__(self, config):
        super().__init__()
        stim_dim = config['decoder']['stimulus_latent']        
        choice_dim = config['decoder']['choice_latent']
        amp_dim = config['decoder']['amplitude_latent']
        dim_x_z = config['dim_x_z']
        
        # check atleast 1 is not None
        assert stim_dim is not None or choice_dim is not None or amp_dim is not None, "Atleast one of stimulus, choice or amplitude should be set"
        
        # define cnn
        if stim_dim is not None:
            stimulus_weight = config['decoder']['stimulus_weight']
            stim_start = 0
            stim_xdim = dim_x_z[stim_dim]
            self.conv_stim = DecoderGlobal(config['decoder']['cnn_global'], stim_xdim, stimulus_weight, config['decoder']['cnn_global']['which_forward'], stim_dim, stim_start, stim_xdim)
            print("Using stimulus decoder")
        else:
            self.conv_stim = None
            stim_xdim = 0

        if choice_dim is not None:
            choice_weight = config['decoder']['choice_weight']
            choice_start = dim_x_z[stim_dim] if stim_dim is not None else 0
            choice_xdim = dim_x_z[choice_dim]
            self.conv_choice = DecoderGlobal(config['decoder']['cnn_global'], choice_xdim, choice_weight, config['decoder']['cnn_global']['which_forward'], choice_dim, choice_start, choice_xdim)
            print("Using choice decoder")
        else:
            self.conv_choice = None        
            choice_xdim = 0

        
        if amp_dim is not None:
            amp_weight = config['decoder']['amplitude_weight']
            amp_xdim = dim_x_z[amp_dim]            
            amp_start = stim_xdim + choice_xdim            
            self.conv_amp = DecoderLocal(config['decoder']['cnn_global'], amp_xdim, amp_weight, amp_dim, amp_start, amp_xdim)
            print("Using amplitude decoder")
        else:
            self.conv_amp = None
                        
        # name        
        self.arch_name = 'cnn_{}_{}_{}'.format(stim_dim, choice_dim, amp_dim)                  
    

    def forward(self, x, z):
        # x is of shape (batch_size*num_samples, seq_len, input_dim)
        
        # permute
        x = x.permute(0, 2, 1)
        z = z.permute(0, 2, 1)      

        if self.conv_stim:
            x_stim = self.conv_stim(x, z)            
        else:            
            x_stim = torch.zeros(x.size(0), 1, device=x.device)            
        
        if self.conv_choice:
            x_choice = self.conv_choice(x, z)
        else:
            x_choice = torch.zeros(x.size(0), 1, device=x.device)
        
        if self.conv_amp:
            x_amp = self.conv_amp(x, z)
        else:
            x_amp = torch.zeros(x.size(0), 1, x.size(2), device=x.device)
        
        return torch.cat([x_stim, x_choice], dim=1), x_amp

    def loss(self, predicted, ground_truth, amp_pred, amp_batch, reduction='mean'):                        
        loss = 0 

        if self.conv_stim:
            loss += self.conv_stim.loss(predicted[:, 0], ground_truth[:, 0], reduction)
        
        if self.conv_choice:
            loss += self.conv_choice.loss(predicted[:, 1], ground_truth[:, 1], reduction)

        if self.conv_amp:
            loss += self.conv_amp.loss(amp_pred, amp_batch, reduction)

        return loss
    
    def step(self):
        if self.conv_stim:
            self.conv_stim.optimizer.step()
        
        if self.conv_choice:
            self.conv_choice.optimizer.step()
        
        if self.conv_amp:
            self.conv_amp.optimizer.step()
    
    def zero_grad(self):
        if self.conv_stim:
            self.conv_stim.optimizer.zero_grad()
        
        if self.conv_choice:
            self.conv_choice.optimizer.zero_grad()
        
        if self.conv_amp:
            self.conv_amp.optimizer.zero_grad()



