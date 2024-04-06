import torch
import torch.nn as nn
from decoder import LinearAccDecoder, CNNDecoder, CNNDecoderIndivdual, RNNDecoderIndivdual
from vae import VAE
from vae_family import VAEParameterised
from vae_gp_separate import VAEGP
import os
import utils
from priors import GaussianPrior
from vae_gp import VAEGPCombined


class Model(nn.Module):
    def __init__(self, config, input_dim, neuron_bias=None):
        super().__init__()
        self.config = config
        # dimensions
        xz_list = config['dim_x_z']
        # vae
        which_vae = config['which_vae']
        if which_vae == 'baseline':        
            self.vae = VAE(config, input_dim, xz_list, neuron_bias)        
        elif which_vae == 'parameterised':
            self.vae = VAEParameterised(config, input_dim, xz_list, neuron_bias)        
        elif which_vae == 'vae_gp':
            self.vae = VAEGP(config, input_dim, xz_list, neuron_bias)
        elif which_vae == 'vae_gp_combined':
            self.vae = VAEGPCombined(config, input_dim, xz_list, neuron_bias)
        else:
            raise ValueError("Unknown VAE type")
        # print num train params in vae
        print('Number of trainable parameters in VAE:', utils.count_parameters(self.vae))            
            
        # behavior decoder
        behavior_decoder = config['decoder']['which']        
        behavior_weight = config['decoder']['behavior_weight']
        self.behavior_weight = behavior_weight
            
        if behavior_decoder == 'linear':            
            self.behavior_decoder = LinearAccDecoder(config, xz_list)                        
            print('Number of trainable parameters in behavior decoder:', utils.count_parameters(self.behavior_decoder))
        elif behavior_decoder == 'cnn':
            self.behavior_decoder = CNNDecoder(config, xz_list)            
            print('Number of trainable parameters in behavior decoder:', utils.count_parameters(self.behavior_decoder))
        elif behavior_decoder == 'cnn_indi':            
            self.behavior_decoder = CNNDecoderIndivdual(config, xz_list)
            print('Number of trainable parameters in behavior decoder:', utils.count_parameters(self.behavior_decoder))
        elif behavior_decoder == 'rnn':
            self.behavior_decoder = RNNDecoderIndivdual(config, xz_list)
        else:
            self.behavior_decoder = None
            self.behavior_weight = 0        
            print("No behavior decoder")
            assert sum(xz_list[:2]) == 0, "Behavior decoder not present but z dimensions are not zero"

        # prior over z
        self.z_prior = config['z_prior']['include']
        if self.z_prior:
            # keep only non-zero values in xz_list
            xz_list = [x for x in xz_list if x > 0]
            self.z_prior_wts = config['z_prior']['weights']
            assert len(self.z_prior_wts) == len(xz_list), "Number of priors should match number of z dimensions"
            self.learn_prior = config['z_prior']['learn']
            prior_on_mean = config['z_prior']['prior_on_mean']
            self.prior_modules = []            
            for m, s, w in zip(config['z_prior']['means'], config['z_prior']['stds'], self.z_prior_wts):
                self.prior_modules.append(GaussianPrior(m, s, w, 0.1, self.learn_prior, prior_on_mean))
            self.prior_modules = nn.ModuleList(self.prior_modules)

        # name model
        self.arch_name = self.vae.arch_name        
        if self.behavior_decoder:
            self.arch_name += self.behavior_decoder.arch_name
        if self.z_prior:
            self.arch_name += '_prior'
        self.final_path = utils.model_store_path(self.config, self.arch_name)
        if not os.path.exists(self.final_path):
            os.makedirs(self.final_path)
    
    def forward(self, spikes, n_samples, use_mean_for_decoding=False):
        vae_output = self.vae(spikes, n_samples)
        x, z = vae_output['x_samples'], vae_output['z_samples']
        if self.behavior_decoder:
            if use_mean_for_decoding:
                behavior = self.behavior_decoder(mu[:, :, self.vae.z_dim:], mu[:, :, :self.vae.z_dim])
            else:
                behavior = self.behavior_decoder(x, z)
        else:
            behavior = None
        return vae_output, behavior

    def loss(self, epoch, spikes_batch, behavior_batch, vae_pred, behavior_pred):
        # loss = torch.tensor(0.0)
        loss = self.vae.loss(spikes_batch, vae_pred)
        loss_l = [loss.item()]
        if self.behavior_decoder:
            behave_loss = self.behavior_weight * self.behavior_decoder.loss(behavior_pred, behavior_batch)            
            loss += behave_loss
            loss_l.append(behave_loss.item())
        
        return loss, loss_l
    
    def optim_step(self, train_decoder):
        self.vae.optimizer.step()
        if self.behavior_decoder and train_decoder:            
            self.behavior_decoder.optimizer.step()
        if self.z_prior and self.learn_prior:
            for prior in self.prior_modules:
                prior.optimizer.step()

    def scheduler_step(self, step_decoder):
        if self.vae.scheduler:
            self.vae.scheduler.step()
        if self.behavior_decoder and self.behavior_decoder.scheduler and step_decoder:
            self.behavior_decoder.scheduler.step()
            # print("LR: ", self.behavior_decoder.scheduler.get_lr())

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