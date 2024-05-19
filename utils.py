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
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)



behave_columns = ['outcome_cat_correct', 'outcome_cat_error', 'outcome_cat_spoil',
       'choice_cat_left', 'choice_cat_nogo', 'choice_cat_right',
       'prev_choice_cat_left', 'prev_choice_cat_nogo', 'prev_choice_cat_right',
       'rewarded_side_cat_left', 'rewarded_side_cat_right',
       'servo_position_cat_close', 'servo_position_cat_far',
       'servo_position_cat_medium', 'rt',
       'c0_count', 'c1_count', 'c2_count', 'c3_count', 'c0_angle', 'c1_angle', 'c2_angle', 'c3_angle',
       'time', 'amp']



def get_decoding_accuracies(model, behaviour_data, spikes):    

    with torch.no_grad():
        model.eval()
        behavior_pred = model.forward(spikes, n_samples=1, use_mean_for_decoding=True)[1]                
        pred_stim = torch.argmax(behavior_pred[:, :2], dim=1).numpy()        
        pred_choice = torch.argmax(behavior_pred[:, 2:4], dim=1).numpy()
        
        # compute accuracy        
        accuracy_stim = accuracy_score(behaviour_data[:, 0], pred_stim)        
        # do the same for choice
        accuracy_choice = accuracy_score(behaviour_data[:, 1], pred_choice)        
    
    return accuracy_stim, accuracy_choice


def extract_mean_covariance(dist):
    """
    Extract mean and covariance matrix from distribution object
    """
    if isinstance(dist, torch.distributions.Normal):
        return dist.mean, dist.variance
    elif isinstance(dist, torch.distributions.MultivariateNormal):
        return dist.mean, dist.covariance_matrix
    else:
        raise ValueError("Distribution type not supported")


def plot_loss_curve(model, config, train_losses, test_losses):
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
    if train_losses_only.shape[1] > 1:
        ax2.plot(train_epochs, train_losses_only[:, 1]/behave_weight, label='Train Decoding', color='red', linestyle='--')
    ax1.plot(test_epochs, test_losses_only[:, 0], label='Test Reconstruction', color='blue')
    if train_losses_only.shape[1] > 1:
        ax2.plot(test_epochs, test_losses_only[:, 1]/behave_weight, label='Test Decoding', color='red')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Reconstruction Loss', color='blue')
    ax2.set_ylabel('Decoding Loss', color='red')
    plt.title('Train and Test Loss')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    # ax2.set_yscale('log')
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
            return {}

def dump_config(config, folder_path):
    with open(os.path.join(folder_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

def load_dataset(config):
    base_path = config['dir']['dataset']
    session = config['shape_dataset']['id']
    file_name = 'shape_processed_behave_spike_contact_{}_ms'.format(int(config['shape_dataset']['win_len']*1000))
    with open(os.path.join(base_path, session, file_name), 'rb') as f:
        behaviour_data, spikes, trial_id = pickle.load(f)
    if config['chosen_neurons']:
        spikes = np.array(spikes)[:, :, config['chosen_neurons']]
    return behaviour_data, spikes, trial_id

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
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True    

    
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