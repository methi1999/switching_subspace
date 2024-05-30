import torch
import torch.nn as nn
# from decoder import LinearAccDecoder, CNNDecoder, CNNDecoderIndividual, RNNDecoderIndivdual, LogReg
from supervised_decoders.decoder import CNNDecoderIndividual
from vae.vae import VAE
from vae.vae_gp import VAEGP
import os
import utils


class Model(nn.Module):
    def __init__(self, config, input_dim, neuron_bias=None):
        super().__init__()
        self.config = config        
        # vae
        which_vae = config['which_vae']
        if which_vae == 'vae':
            self.vae = VAE(config, input_dim, neuron_bias)                
        elif which_vae == 'vae_gp':
            self.vae = VAEGP(config, input_dim, neuron_bias)
        else:
            raise ValueError("Unknown VAE type")
        # print num train params in vae
        print('Number of trainable parameters in VAE:', utils.count_parameters(self.vae))            
            
        # behavior decoder
        behavior_decoder = config['decoder']['which']        
            
        if behavior_decoder == 'cnn':            
            self.behavior_decoder = CNNDecoderIndividual(config)
            print('Number of trainable parameters in behavior decoder:', utils.count_parameters(self.behavior_decoder))        
        else:
            self.behavior_decoder = None
            print("No behavior decoder")
        
        # name model
        self.arch_name = self.vae.arch_name        
        if self.behavior_decoder:
            self.arch_name += self.behavior_decoder.arch_name
        
        self.model_store_pth = utils.model_store_path(self.config, self.arch_name)
        if not os.path.exists(self.model_store_pth):
            os.makedirs(self.model_store_pth)

        if config['vae_gp']['load_stage2']:
            print('Loading weights for s2')
            pth = os.path.join(self.model_store_pth, 'post_stage_2.pth')
            weights = torch.load(open(pth, 'rb'))            
            # intersect keys
            keys = set(weights.keys()).intersection(set(self.state_dict().keys()))            
            self.load_state_dict(weights, strict=False)
            print('Weights loaded for s2:', keys)
            assert config['vae_gp']['freeze_encoder_meanz'] is False, "Cannot freeze encoder after stage 2"
    
    def forward(self, spikes, n_samples, use_mean_for_decoding):
        vae_output = self.vae(spikes, n_samples, use_mean_for_decoding)        
        if self.behavior_decoder:
            if use_mean_for_decoding:
                ### gp on x
                mean_x, mean_z = vae_output['x_distribution'].mean, torch.stack([x.mean for x in vae_output['z_distributions']], dim=-1)                
                # mean_x, mean_z = torch.stack([x.mean for x in vae_output['x_distribution']], dim=-1), torch.stack([x.mean for x in vae_output['z_distributions']], dim=-1)                
                softmax = nn.Softmax(dim=-1)
                mean_z = softmax(mean_z)
                # reshape flattened x
                mean_x = mean_x.reshape(mean_z.shape[0], mean_z.shape[1], -1)
                behavior, amp = self.behavior_decoder(mean_x, mean_z)
            else:
                x, z = vae_output['x_samples'], vae_output['z_samples']
                behavior, amp = self.behavior_decoder(x, z)
        else:
            behavior = None
            amp = None
        return vae_output, behavior, amp

    def loss(self, epoch, spikes_batch, behavior_batch, amp_batch, vae_pred, behavior_pred, amp_pred):
        loss = self.vae.loss(spikes_batch, vae_pred, behavior_batch)
        loss_l = [loss.item()]
        if self.behavior_decoder:
            behave_loss = self.behavior_decoder.loss(behavior_pred, behavior_batch, amp_pred, amp_batch)            
            loss += behave_loss
            loss_l.append(behave_loss.item())
        
        return loss, loss_l
    
    def optim_step(self, train_decoder):
        # torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1)
        self.vae.optimizer.step()
        if self.behavior_decoder and train_decoder:            
            self.behavior_decoder.step()

    def optim_zero_grad(self):
        self.vae.optimizer.zero_grad()
        if self.behavior_decoder:
            self.behavior_decoder.zero_grad()
        
    def save_model(self, save_prefix):
        """
        Save model for inference
        :param save_prefix: if using CV. if None, best is used as model name
        :return: nothing
        """        
        if save_prefix is not None:
            filename = os.path.join(self.model_store_pth, str(save_prefix))
        else:
            filename = os.path.join(self.model_store_pth, 'best')

        torch.save({
            'model_state_dict': self.state_dict(),
            # 'optimizer_state_dict': module.optimizer.state_dict(),
        }, filename)
        # print("Saved model for {}".format(suffix))
        # also dump model config
        utils.dump_config(self.config, self.model_store_pth)

    def load_model(self, save_prefix, base_path=None, strict=True):
        """
        Loads saved model for inference
        :param strict: whether to load ALL parameters
        :param save_prefix: fold number
        :param base_path: optional base path of model
        """
        
        if base_path is None:
            base_path = self.model_store_pth
        if save_prefix is not None:
            filename = os.path.join(base_path, str(save_prefix))
        else:
            filename = os.path.join(base_path, 'best')

        # load model parameters
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)        
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        # module.optimizer.load_state_dict(checkpoint['optimizer_state_dict'], strict=strict)
        print("Loaded model")