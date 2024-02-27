import torch
import torch.nn as nn
from decoder import LinearAccDecoder, CNNDecoder, CNNDecoderIndivdual
from vae import VAE
import os
import utils
from gp import gp_ll

class Model(nn.Module):
    def __init__(self, config, input_dim, neuron_bias=None):
        super().__init__()
        self.config = config
        # dimensions
        z_dim, x_dim = config['dim_x_z'], config['dim_x_z']
        # vae        
        self.vae = VAE(config, input_dim, z_dim, x_dim, neuron_bias)        
        # print num train params in vae
        print('Number of trainable parameters in VAE:', utils.count_parameters(self.vae))
        
        # behavior decoder
        behavior_decoder = config['decoder']['which']        
        behavior_weight = config['decoder']['behavior_weight']
        self.behavior_weight = behavior_weight
            
        if behavior_decoder == 'linear':            
            self.behavior_decoder = LinearAccDecoder(config, x_dim)                        
            print('Number of trainable parameters in behavior decoder:', utils.count_parameters(self.behavior_decoder))
        elif behavior_decoder == 'cnn':
            self.behavior_decoder = CNNDecoder(config, x_dim)            
            print('Number of trainable parameters in behavior decoder:', utils.count_parameters(self.behavior_decoder))
        elif behavior_decoder == 'cnn_indi':            
            self.behavior_decoder = CNNDecoderIndivdual(config, x_dim)
            print('Number of trainable parameters in behavior decoder:', utils.count_parameters(self.behavior_decoder))
        else:
            self.behavior_decoder = None
            self.behavior_weight = 0        
            print("No behavior decoder")        
        
        # name model
        self.arch_name = self.vae.arch_name        
        if self.behavior_decoder:
            self.arch_name += self.behavior_decoder.arch_name                
        self.final_path = utils.model_store_path(self.config, self.arch_name)
        if not os.path.exists(self.final_path):
            os.makedirs(self.final_path)
    
    def forward(self, spikes):
        y_recon, (mu, A), (z, x) = self.vae(spikes)        
        if self.behavior_decoder:
            behavior = self.behavior_decoder(x, z)
            # behavior = self.behavior_decoder(mu[:, :, self.vae.z_dim:], mu[:, :, :self.vae.z_dim])
        else:
            behavior = None
        return y_recon, (mu, A), (z, x), behavior

    def loss(self, epoch, y, y_recon, mu, A, z, x, behavior_pred, behavior_truth):
        loss = self.vae.loss(y, y_recon, mu, A)
        # loss = torch.tensor(0.0)
        loss_l = [loss.item()]
        if self.behavior_decoder:
            behave_loss = self.behavior_weight * self.behavior_decoder.loss(behavior_pred, behavior_truth, z)            
            loss += behave_loss
            loss_l.append(behave_loss.item())
        
        return loss, loss_l
    
    def optim_step(self):
        self.vae.optimizer.step()
        if self.behavior_decoder:
            self.behavior_decoder.optimizer.step()

    def scheduler_step(self):
        if self.vae.scheduler:
            self.vae.scheduler.step()
        if self.behavior_decoder and self.behavior_decoder.scheduler:
            self.behavior_decoder.scheduler.step()            

    def optim_zero_grad(self):
        self.vae.optimizer.zero_grad()
        if self.behavior_decoder:
            self.behavior_decoder.optimizer.zero_grad()
        
    def save_model(self, save_prefix):
        """
        Save model for inference
        :param save_prefix: if using CV. if None, best is used as model name
        :return: nothing
        """        
        if save_prefix is not None:
            filename = os.path.join(self.final_path, str(save_prefix))
        else:
            filename = os.path.join(self.final_path, 'best')

        torch.save({
            'model_state_dict': self.state_dict(),
            # 'optimizer_state_dict': module.optimizer.state_dict(),
        }, filename)
        # print("Saved model for {}".format(suffix))
        # also dump model config
        utils.dump_config(self.config, self.final_path)

    def load_model(self, save_prefix, base_path=None, strict=True):
        """
        Loads saved model for inference
        :param strict: whether to load ALL parameters
        :param save_prefix: fold number
        :param base_path: optional base path of model
        """
        
        if base_path is None:
            base_path = utils.model_store_path(self.config, self.arch_name)
        if save_prefix is not None:
            filename = os.path.join(base_path, str(save_prefix))
        else:
            filename = os.path.join(base_path, 'best')

        # load model parameters
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        # module.optimizer.load_state_dict(checkpoint['optimizer_state_dict'], strict=strict)
        print("Loaded model")