# %%
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from copy import deepcopy

from model import Model
import utils
from early_stopping import EarlyStopping
from misc.priors import moving_average
from sklearn.decomposition import PCA

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# %%
config = utils.read_config()
# set seeds
utils.set_seeds(config['seed'])
# utils.set_seeds(7)

# %%
behaviour_data, spikes = utils.load_dataset(config)
# consider data from only t = -1
# time_from = int(1/bin_len)
# behaviour_data, spikes = [x[time_from:, :] for x in behaviour_data], [x[time_from:, :] for x in spikes]
num_trials, time_bins, emissions_dim = np.array(spikes).shape

# %%
stim_idx, choice_idx = 6, 3
stim = [x[0, stim_idx] for x in behaviour_data]
choice = [x[0, choice_idx] for x in behaviour_data]
num_contacts = [np.sum(x[:, -9:-5], axis=1) for x in behaviour_data]
# concat them
behaviour_data = np.stack((stim, choice), axis=1)

# %%
# convert to torch tensors
behaviour_data = torch.tensor(behaviour_data, dtype=torch.long)
# behaviour_data = torch.tensor(behaviour_data, dtype=torch.float32)
spikes = torch.tensor(spikes, dtype=torch.float32)
spikes = spikes[:, -10:, :]

# %%
# create dataloader with random sampling for training and testing
# split data into training and testing
# behaviour_data_train, behaviour_data_test, spikes_train, spikes_test = train_test_split(behaviour_data, spikes, test_size=0.3, random_state=42)
behaviour_data_train, behaviour_data_test, spikes_train, spikes_test = train_test_split(behaviour_data, spikes, test_size=0.2, random_state=7)

# create dataloaders
train_dataset = TensorDataset(behaviour_data_train, spikes_train)
test_dataset = TensorDataset(behaviour_data_test, spikes_test)

batch_size = config['batch_size']
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# %%
# distribution of choice and stimulus in test
print("Train distribution of Stimulus: {}, Choice: {}".format(np.mean(behaviour_data_train[:, 0].numpy()), np.mean(behaviour_data_train[:, 1].numpy())))
print("Test distribution of Stimulus: {}, Choice: {}".format(np.mean(behaviour_data_test[:, 0].numpy()), np.mean(behaviour_data_test[:, 1].numpy())))

# %%
# mean firing rate of neurons in tran spikes
neuron_bias = torch.mean(spikes_train, dim=0)

# %%
# # check if mps is available
# device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
# print(device)
# model = model.to(device)
# spikes = spikes.to(device)


# %%
# convert to numpy
spikes_test_np = spikes_test.detach().numpy()
spikes_train_np = spikes_train.detach().numpy()

# with torch.no_grad():
#     model.eval()
#     y_recon, mu, A, z, x, _ = model.forward(spikes, n_samples=1)
#     # run on only test
#     y_recon_test, mu_test, A_test, z_test, x_test, _ = model.forward(spikes_test, n_samples=1)
#     # run only on train
#     y_recon_train, mu_train, A_train, z_train, x_train, _ = model.forward(spikes_train, n_samples=1)
# # use model output
# y_recon_np = y_recon.detach().numpy()
# y_recon_test_np = y_recon_test.detach().numpy()

# do pca on train and test data
pca = PCA(n_components=1)
trials_train, num_neurons = len(spikes_train), spikes_train[0].shape[1]
trials_test = len(spikes_test)
pca.fit(spikes_train.reshape(-1, num_neurons))
print("PCA done")
spikes_train_pca = pca.transform(spikes_train.reshape(-1, num_neurons))
spikes_test_pca = pca.transform(spikes_test.reshape(-1, num_neurons))
# reconstruct spikes
y_recon_np = pca.inverse_transform(spikes_train_pca).reshape(trials_train, -1, num_neurons)
y_recon_test_np = pca.inverse_transform(spikes_test_pca).reshape(trials_test, -1, num_neurons)
# apply softplus
# print(y_recon_np[0][0, :5], spikes_train_np[0][0, :5])
# y_recon_np = np.log(1 + np.exp(y_recon_np))
# y_recon_test_np = np.log(1 + np.exp(y_recon_test_np))
# relu
y_recon_np = np.maximum(y_recon_np, 0)
y_recon_test_np = np.maximum(y_recon_test_np, 0)
# print(y_recon_np[0][0, :5], spikes_train_np[0][0, :5])

# compute bits/spike
bits_per_spike_all = utils.bits_per_spike(y_recon_np, spikes_train_np)
bits_per_spike_test = utils.bits_per_spike(y_recon_test_np, spikes_test_np)
# show distribution of bits per spike
# plt.hist(bits_per_spike_all, bins=50)
# plt.xlabel('Bits/spike')
# plt.ylabel('Frequency')
print("Bits per spike all: {}, test: {}".format(np.sum(bits_per_spike_all), np.sum(bits_per_spike_test)))
# plt.show()
# print('Bits per spike: {}'.format(bits_per_spike))
# print explained variance
print("Explained variance: {}".format(np.cumsum(pca.explained_variance_ratio_[:10])))