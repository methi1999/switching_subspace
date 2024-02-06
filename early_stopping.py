import numpy as np
from model import Model


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience"""

    def __init__(self, patience=100, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.slow_down = False
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, score, model: Model, save_model, save_prefix):

        if self.best_score is None:
            self.best_score = score
            if save_model:
                model.save_model(save_prefix)
        elif score > self.best_score - self.delta:
            self.counter += 1
            # self.trace_func('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
            self.slow_down = True
        else:
            self.best_score = score
            if save_model:
                model.save_model(save_prefix)
            # self.trace_func('Found best')
            self.counter = 0
            self.slow_down = False
