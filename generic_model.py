import torch
import torch.nn as nn
import os
import abc
import code_torch.utils as utils


class GenericModel(nn.Module):
    """
    Contains basic functions for storing and loading a model
    Only Master modules will be derived from this class
    """
    # __metaclass__ = abc.ABCMeta

    def __init__(self, config):
        super(GenericModel, self).__init__()
        self.modules = {}
        self.config = config
        self.final_path = utils.model_store_path(self.config)
        if not os.path.exists(self.final_path):
            os.makedirs(self.final_path)

    def count_parameters(self):
        return sum([sum(p.numel() for p in model.parameters() if p.requires_grad) for model in self.modules.values()])

    def save_model(self, save_prefix):
        """
        Save model for inference
        :param save_prefix: if using CV. if None, best is used as model name
        :return: nothing
        """

        for suffix, module in self.modules.items():
            if not module.optimizer:
                continue
            if save_prefix is not None:
                filename = os.path.join(self.final_path, str(save_prefix) + '_' + suffix)
            else:
                filename = os.path.join(self.final_path, 'best_' + suffix)

            torch.save({
                'model_state_dict': module.state_dict(),
                # 'optimizer_state_dict': module.optimizer.state_dict(),
            }, filename)
            # print("Saved model for {}".format(suffix))

    def load_model(self, save_prefix, base_path=None, strict=True):
        """
        Loads saved model for inference
        :param strict: whether to load ALL parameters
        :param save_prefix: fold number
        :param base_path: optional base path of model
        """
        for suffix, module in self.modules.items():
            if not module.optimizer:
                continue
            if base_path is None:
                base_path = utils.model_store_path(self.config)
            if save_prefix is not None:
                filename = os.path.join(base_path, str(save_prefix) + '_' + suffix)
            else:
                filename = os.path.join(base_path, 'best_' + suffix)

            # load model parameters
            checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
            module.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            # module.optimizer.load_state_dict(checkpoint['optimizer_state_dict'], strict=strict)
            print("Loaded model for {}".format(suffix))

    def to(self, device):
        for name in self.modules.keys():
            self.modules[name] = self.modules[name].to(device)

    def train_mode(self):
        for name, model in self.modules.items():
            self.modules[name] = self.modules[name].train()

    def eval_mode(self):
        for name, model in self.modules.items():
            self.modules[name] = self.modules[name].eval()

    def opti_zero_grad(self):
        for name, model in self.modules.items():
            if model.optimizer:
                # model.optimizer.zero_grad()
                for param in model.parameters():
                    param.grad = None

    def opti_step(self):
        for name, model in self.modules.items():
            if model.optimizer:
                model.optimizer.step()

    @abc.abstractmethod
    def forward(self, batch):
        pass

    @abc.abstractmethod
    def calculate_loss(self, out):
        pass

