# %%
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

from model import Model
import utils
from early_stopping import EarlyStopping
from priors import moving_average

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
# stim_idx, choice_idx = 6, 3
stim = [x[0, stim_idx] for x in behaviour_data]
choice = [x[0, choice_idx] for x in behaviour_data]
num_contacts = [np.sum(x[:, 15:19], axis=1) for x in behaviour_data]
# concat them
behaviour_data = np.stack((stim, choice), axis=1)

# %%
# convert to torch tensors
behaviour_data = torch.tensor(behaviour_data, dtype=torch.float32)
spikes = torch.tensor(spikes, dtype=torch.float32)

# %%
# create dataloader with random sampling for training and testing
# split data into training and testing
behaviour_data_train, behaviour_data_test, spikes_train, spikes_test = train_test_split(behaviour_data, spikes, test_size=0.3, random_state=42)

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
def test(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for _, (behavior_batch, spikes_batch) in enumerate(test_loader):
            y_recon, (mu, A), (z, x), behavior_batch_pred = model(spikes_batch)
            _, loss_l = model.loss(100, spikes_batch, y_recon, mu, A, z, x, behavior_batch_pred, behavior_batch)
            test_loss += np.array(loss_l)
    # divide loss by total number of samples in dataloader    
    return test_loss/len(test_loader)

# %%
config = utils.read_config()
# training loop
num_epochs = config['epochs']
# create model and optimizer
model = Model(config, input_dim=emissions_dim) #, neuron_bias=neuron_bias
early_stop = EarlyStopping(patience=config['early_stop']['patience'], delta=config['early_stop']['delta'], trace_func=print)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold=1, verbose=True, patience=5, factor=0.5)
# print named parameters of model
# print("Model's state_dict:")
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data.shape)

# %%
torch.autograd.set_detect_anomaly(True)
train_losses, test_losses = [], []
def train(model, val_loader):    
    test_every = config['test_every']    
    save_model = True
    for epoch in range(num_epochs):
        # forward pass
        # print(model.behavior_decoder.scheduler.get_last_lr())
        # model.vae.scheduler.get_last_lr()
        epoch_loss = 0
        model.train()
        for i, (behavior_batch, spikes_batch) in enumerate(train_loader):            
            y_recon, (mu, A), (z, x), behavior_pred = model(spikes_batch)
            loss, loss_l = model.loss(epoch, spikes_batch, y_recon, mu, A, z, x, behavior_pred, behavior_batch)            
            # backward pass
            model.optim_zero_grad()
            loss.backward()
            model.optim_step()            
            epoch_loss += np.array(loss_l)
        
        train_losses.append((epoch, epoch_loss/len(train_loader)))
        model.scheduler_step()
        # test loss
        if (epoch+1) % test_every == 0:            
            test_loss = test(model, val_loader)
            sum_test_loss = np.sum(test_loss)
            # scheduler.step(sum_test_loss)
            test_losses.append((epoch, test_loss))
            early_stop(sum_test_loss, model, save_model=save_model, save_prefix='best')
            model.save_model(save_prefix=str(epoch))
            print('Epoch [{}/{}], Train Loss: {}, Test Loss: {}, Best Loss: {}'.format(epoch+1, num_epochs, train_losses[-1][1], test_losses[-1][1], early_stop.best_score))
            if early_stop.slow_down:
                test_every = config['early_stop']['test_every_new']
            else:
                test_every = config['test_every']
            if early_stop.early_stop:
                print("Early stopping")
                break
            
    
    only_test_loss = [np.sum(x[1]) for x in test_losses]
    
    # compute min test loss and return it    
    # return np.min(only_test_loss), train_losses, test_losses
    
    # compute median of test loss in a window of 5
    meds = []
    half_window = 10
    only_test_loss = [0]*(half_window) + only_test_loss + [0]*(half_window)
    for i in range(half_window, len(only_test_loss)-half_window):
        meds.append(np.max(only_test_loss[i-half_window:i+half_window]))
    return np.min(meds), train_losses, test_losses

_ = train(model, test_loader)
# train model
# min_test_loss, train_losses, test_losses = train(model, test_loader)
