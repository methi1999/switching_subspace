# %%
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from copy import deepcopy

from model import Model
import utils
from early_stopping import EarlyStopping
from priors import moving_average
from sklearn.decomposition import PCA

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
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
behaviour_data = torch.tensor(behaviour_data, dtype=torch.long)
# behaviour_data = torch.tensor(behaviour_data, dtype=torch.float32)
spikes = torch.tensor(spikes, dtype=torch.float32)

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
def test(model, test_loader, n_samples=1):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for _, (behavior_batch, spikes_batch) in enumerate(test_loader):
            y_recon, mu, A, z, x, behavior, misc = model.forward(spikes_batch, n_samples=n_samples)
            _, loss_l = model.loss(None, spikes_batch, y_recon, mu, A, z, x, behavior, behavior_batch, misc)
            # l.append(loss_l[1])
            test_loss += np.array(loss_l)            
            # print(np.mean(l), np.std(l))
    # divide loss by total number of samples in dataloader    
    return test_loss/len(test_loader)

# %%
config = utils.read_config()
# take one float as input from argparse
for l in [5]:
    config['rnn']['var_penalty'] = l
    # training loop
    num_epochs = config['epochs']
    # create model and optimizer
    model = Model(config, input_dim=emissions_dim) #, neuron_bias=neuron_bias
    # model = torch.compile(model)
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

    def train(model: Model, train_loader, val_loader):    
        test_every = config['test_every']    
        train_decoder_after = config['decoder']['train_decoder_after']    
        num_samples_train = config['num_samples_train']
        save_model = True    
        for epoch in range(num_epochs):
            # forward pass
            # print(model.behavior_decoder.scheduler.get_last_lr())
            # model.vae.scheduler.get_last_lr()
            epoch_loss = 0
            model.train()
            for i, (behavior_batch, spikes_batch) in enumerate(train_loader):            
                # behavior_batch = behavior_batch.long()
                y_recon, mu, A, z, x, behavior_pred, misc = model(spikes_batch, n_samples=num_samples_train)
                # calculate loss
                loss, loss_l = model.loss(epoch, spikes_batch, y_recon, mu, A, z, x, behavior_pred, behavior_batch, misc)
                # backward pass
                model.optim_zero_grad()
                loss.backward()
                # print gradient of any weight
                # if epoch > 10:
                #     print(model.behavior_decoder.conv_choice[1].weight.grad)            
                model.optim_step(train_decoder = epoch >= train_decoder_after)                
                epoch_loss += np.array(loss_l)
            
            # if epoch % 100 == 0:
            #     # print lr of decoder
            #     print(model.behavior_decoder.scheduler.get_last_lr())
            train_losses.append((epoch, epoch_loss/len(train_loader)))
            model.scheduler_step(step_decoder = epoch >= train_decoder_after)
            # test loss
            if (epoch+1) % test_every == 0:            
                test_loss = test(model, val_loader, n_samples=config['num_samples_test'])
                sum_test_loss = np.sum(test_loss)
                # scheduler.step(sum_test_loss)
                test_losses.append((epoch, test_loss))
                early_stop(sum_test_loss, model, save_model=save_model, save_prefix='best')
                model.save_model(save_prefix=str(epoch))
                # print('Epoch [{}/{}], Train Loss: {}, Test Loss: {}, Best Loss: {}'.format(epoch+1, num_epochs, train_losses[-1][1], test_losses[-1][1], early_stop.best_score))
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

    _ = train(model, train_loader, test_loader)
    # train model
    # min_test_loss, train_losses, test_losses = train(model, test_loader)

    # %%
    train_losses_og, test_losses_og = train_losses[:], test_losses[:]

    # %%
    # model.prior_modules[0].plot_gaussian()
    # model.prior_modules[0].log_std

    # %%
    # utils.plot_curve(model, config, train_losses, test_losses)

    # %%
    # load best model
    model.load_model('best')
    # load model from epoch x
    # model.load_model('366')

    # %%
    # lin_maps = model.vae.linear_maps
    # # c1, c2 = lin_maps[0].weight.detach().numpy(), lin_maps[1].weight.detach().numpy()
    # # print(c1.T.dot(c2)/(np.linalg.norm(c1)*np.linalg.norm(c2)))
    # c1, c2, c3 = lin_maps[0].weight.detach().numpy(), lin_maps[1].weight.detach().numpy(), lin_maps[2].weight.detach().numpy()
    # print("Norms: {}, {}, {}".format(np.linalg.norm(c1), np.linalg.norm(c2), np.linalg.norm(c3)))

    # %%
    # convert to numpy
    spikes_np = spikes_train.detach().numpy()
    spikes_test_np = spikes_test.detach().numpy()

    with torch.no_grad():
        model.eval()
        y_recon, mu, A, z, x, _, _ = model.forward(spikes, n_samples=1)
        # run on only test
        y_recon_test, mu_test, A_test, z_test, x_test, _, _ = model.forward(spikes_test, n_samples=1)
        # run only on train
        y_recon_train, mu_train, A_train, z_train, x_train, _, _ = model.forward(spikes_train, n_samples=1)
    # use model output
    y_recon_np = y_recon_train.detach().numpy()
    y_recon_test_np = y_recon_test.detach().numpy()

    # compute bits/spike
    bits_per_spike_all = utils.bits_per_spike(y_recon_np, spikes_np)
    bits_per_spike_test = utils.bits_per_spike(y_recon_test_np, spikes_test_np)
    # show distribution of bits per spike
    # plt.hist(bits_per_spike_all, bins=50)
    # plt.xlabel('Bits/spike')
    # plt.ylabel('Frequency')
    # plt.show()
    # print('Bits per spike: {}'.format(bits_per_spike))
    print("Bits per spike all: {}, test: {}".format(np.sum(bits_per_spike_all), np.sum(bits_per_spike_test)))
