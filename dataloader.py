import numpy as np
import utils
import os
import pickle
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset


def load_dataset(config):
    base_path = config['dir']['dataset']
    session = config['shape_dataset']['id']
    file_name = 'shape_processed_behave_spike_contact_{}_ms'.format(int(config['shape_dataset']['win_len']*1000))
    with open(os.path.join(base_path, session, file_name), 'rb') as f:
        behaviour_data, spikes, trial_id = pickle.load(f)
    if config['chosen_neurons']:
        spikes = np.array(spikes)[:, :, config['chosen_neurons']]
    return behaviour_data, spikes, trial_id

def preprocess(behaviour_data, spikes, trial_id):
    stim_idx, choice_idx, amp_idx = 9, 3, 24
    stim = [x[0, stim_idx] for x in behaviour_data]
    choice = [x[0, choice_idx] for x in behaviour_data]
    amp = np.array([x[:, amp_idx] for x in behaviour_data], dtype=float)
    # normalize amp by max value
    amp = amp / amp.max()
    num_contacts = [np.sum(x[:, 15:19], axis=1) for x in behaviour_data]
    # concat them
    behaviour_data = np.stack((stim, choice), axis=1)
    return behaviour_data, spikes, num_contacts, amp, trial_id

def get_data_splits(data, labels, n_splits, random_state):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    for train_index, test_index in skf.split(data, labels):
        splits.append((train_index, test_index))
    return splits

def make_labels_from_stim_choice(stim, choice):
    labels = 2*stim + choice
    return labels

def generate_splits():
    config = utils.read_config()
    base_path = config['dir']['dataset']
    session = config['shape_dataset']['id']
    file_name = 'shape_splits'
    pth = os.path.join(base_path, session, file_name)
    # if file exists, return
    if os.path.exists(pth):
        print('File exists')
        with open(pth, 'rb') as f:
            return pickle.load(f)        
    # otherwise, generate splits
    behaviour_data, spikes, trial_id = load_dataset(config)
    behaviour_data, spikes, num_contacts, amp, trial_id = preprocess(behaviour_data, spikes, trial_id)
    labels_for_split = make_labels_from_stim_choice(behaviour_data[:, 0], behaviour_data[:, 1])
    splits = get_data_splits(spikes, labels_for_split, config['cv']['num_folds'], config['cv']['seed'])
    # dump splits in the shape dataset directory
    with open(os.path.join(base_path, session, file_name), 'wb') as f:
        pickle.dump(splits, f)
    return splits

def get_datasets(train_idx, test_idx, behave_data, spikes_data, amp_data):
    # index into the data and create datasets
    train_data = spikes_data[train_idx]
    test_data = spikes_data[test_idx]
    train_behave = behave_data[train_idx]
    test_behave = behave_data[test_idx]
    train_amp = amp_data[train_idx]
    test_amp = amp_data[test_idx]
    train_dataset = TensorDataset(torch.Tensor(train_data), torch.Tensor(train_behave), torch.Tensor(train_amp))
    test_dataset = TensorDataset(torch.Tensor(test_data), torch.Tensor(test_behave), torch.Tensor(test_amp))
    return train_dataset, test_dataset