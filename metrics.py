import numpy as np
import utils
import torch
from sklearn.metrics import accuracy_score
from model import Model


def get_bits_spike_from_output(model: Model, spikes_train, spikes_test):
    # convert to numpy
    spikes_train_np = spikes_train.detach().numpy()
    spikes_test_np = spikes_test.detach().numpy()
    with torch.no_grad():
        model.eval()        
        # run on only test
        vae_output, _, amp_out_test = model.forward(spikes_test, n_samples=1, use_mean_for_decoding=True)  
        y_recon_test, x_mu_test, z_mu_test, x_A_test, z_A_test, x_test, z_test, z_test_presoftmax, g_test = model.vae.extract_relevant(vae_output)
        # run only on train
        vae_output, _, amp_out_train = model.forward(spikes_train, n_samples=1, use_mean_for_decoding=True)
        y_recon_train, x_mu_train, z_mu_train, x_A_train, z_A_train, x_train, z_train, z_train_presoftmax, g_train = model.vae.extract_relevant(vae_output)
    
    # compute bits/spike
    bits_per_spike_train = utils.bits_per_spike(y_recon_train, spikes_train_np).sum()
    bits_per_spike_test = utils.bits_per_spike(y_recon_test, spikes_test_np).sum()

    to_write = (y_recon_train, x_mu_train, z_mu_train, x_A_train, z_A_train, x_train, z_train, z_train_presoftmax, g_train,
            y_recon_test, x_mu_test, z_mu_test, x_A_test, z_A_test, x_test, z_test, z_test_presoftmax, g_test,            
            amp_out_train, amp_out_test)

    return bits_per_spike_train, bits_per_spike_test, to_write


def get_decoding_accuracies(model, behaviour_data, spikes):    
    
    with torch.no_grad():
        model.eval()
        behavior_pred = model.forward(spikes, n_samples=1, use_mean_for_decoding=True)[1]
        # print(behavior_pred[:, :2], behavior_pred[:, 2:4])
        # pred_stim = torch.argmax(behavior_pred[:, :2], dim=1).numpy()
        # pred_choice = torch.argmax(behavior_pred[:, 2:4], dim=1).numpy()
        pred_stim = (behavior_pred[:, 0] > 0).numpy()        
        pred_choice = (behavior_pred[:, 1] > 0).numpy()        
        # compute accuracy        
        accuracy_stim = accuracy_score(behaviour_data[:, 0], pred_stim)        
        # do the same for choice
        accuracy_choice = accuracy_score(behaviour_data[:, 1], pred_choice)        
    
    return accuracy_stim, accuracy_choice