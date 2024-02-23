import numpy as np
from scipy.special import gammaln
import logging
import os
import random
import torch
import json
import yaml
import pickle
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_curve(model, config, train_losses, test_losses):
    # plot train and test loss
    train_epochs = [x[0] for x in train_losses]
    train_losses_only = np.array([x[1] for x in train_losses])
    test_epochs = [x[0] for x in test_losses]
    test_losses_only = np.array([x[1] for x in test_losses])
    behave_weight = config['decoder']['behavior_weight']
    # plot
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(train_epochs, train_losses_only[:, 0], label='Train Reconstruction', color='blue', linestyle='--')
    ax2.plot(train_epochs, train_losses_only[:, 1]/behave_weight, label='Train Decoding', color='red', linestyle='--')
    ax1.plot(test_epochs, test_losses_only[:, 0], label='Test Reconstruction', color='blue')
    ax2.plot(test_epochs, test_losses_only[:, 1]/behave_weight, label='Test Decoding', color='red')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Reconstruction Loss', color='blue')
    ax2.set_ylabel('Decoding Loss', color='red')
    plt.title('Train and Test Loss')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    # plt.show()
    plt.savefig(os.path.join(model_store_path(config, model.arch_name), 'train_test_loss.png'))


# read yaml files that defines hyper-parameters and the location of data
def read_config(path='config.yaml'):
    with open(path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)

def dump_config(config, folder_path):
    with open(os.path.join(folder_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

def load_dataset(config):
    base_path = config['dir']['dataset']
    session = config['shape_dataset']['id']
    file_name = 'shape_processed_behave_spike_contact_{}_ms'.format(int(config['shape_dataset']['win_len']*1000))
    with open(os.path.join(base_path, session, file_name), 'rb') as f:
        behaviour_data, spikes = pickle.load(f)    
    return behaviour_data, spikes

def model_store_path(config, arch_name):    
    data_des = 'dandi_{}/{}_ms'.format(config['shape_dataset']['id'], int(config['shape_dataset']['win_len']*1000))
    return os.path.join(config['dir']['results'], data_des, arch_name)    

def set_seeds(seed):
    # set seeds for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def neg_log_likelihood(rates, spikes, zero_warning=True):
    """Calculates Poisson negative log likelihood given rates and spikes.
    formula: -log(e^(-r) / n! * r^n)
        = r - n*log(r) + log(n!)

    Parameters
    ----------
    rates : np.ndarray
        numpy array containing rate predictions
    spikes : np.ndarray
        numpy array containing true spike counts
    zero_warning : bool, optional
        Whether to print out warning about 0 rate
        predictions or not

    Returns
    -------
    float
        Total negative log-likelihood of the data
    """
    assert (
        spikes.shape == rates.shape
    ), f"neg_log_likelihood: Rates and spikes should be of the same shape. spikes: {spikes.shape}, rates: {rates.shape}"

    if np.any(np.isnan(spikes)):
        mask = np.isnan(spikes)
        rates = rates[~mask]
        spikes = spikes[~mask]

    assert not np.any(np.isnan(rates)), "neg_log_likelihood: NaN rate predictions found"

    assert np.all(rates >= 0), "neg_log_likelihood: Negative rate predictions found"
    if np.any(rates == 0):
        if zero_warning:
            logger.warning(
                "neg_log_likelihood: Zero rate predictions found. Replacing zeros with 1e-9"
            )
        rates[rates == 0] = 1e-9

    result = rates - spikes * np.log(rates) + gammaln(spikes + 1.0)
    return np.sum(result, axis=(0, 1))

def bits_per_spike(rates, spikes):
    """Computes bits per spike of rate predictions given spikes.
    Bits per spike is equal to the difference between the log-likelihoods (in base 2)
    of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    Parameters
    ----------
    rates : np.ndarray
        3d numpy array containing rate predictions
    spikes : np.ndarray
        3d numpy array containing true spike counts

    Returns
    -------
    float
        Bits per spike of rate predictions
    """
    nll_model = neg_log_likelihood(rates, spikes)
    null_rates = np.tile(
        np.nanmean(spikes, axis=tuple(range(spikes.ndim - 1)), keepdims=True),
        spikes.shape[:-1] + (1,),
    )
    nll_null = neg_log_likelihood(null_rates, spikes, zero_warning=False)
    return (nll_null - nll_model) / np.nansum(spikes) / np.log(2)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)