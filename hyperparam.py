import numpy as np
import os
import torch
import optuna
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from model import Model
from early_stopping import EarlyStopping
import utils

colors = ['red', 'blue', 'green', 'black', 'yellow', 'pink']
only_look_at_decoder = False

is_cuda = torch.cuda.is_available()
# is_cuda = False
# if we have a GPU available, we'll set our device to GPU
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device: {}".format(device))

config_global = utils.read_config()
# set seeds
utils.set_seeds(config_global['seed'])
behaviour_data, spikes, trial_ids = utils.load_dataset(config_global)
stim_idx, choice_idx, amp_idx = 9, 3, 24
stim = [x[0, stim_idx] for x in behaviour_data]
choice = [x[0, choice_idx] for x in behaviour_data]
amp = torch.tensor([x[:, amp_idx] for x in behaviour_data], dtype=torch.float32)
# normalize amp by max value
amp = amp / amp.max()
num_contacts = [np.sum(x[:, 15:19], axis=1) for x in behaviour_data]
behaviour_data = np.stack((stim, choice), axis=1)
# convert to torch tensors
behaviour_data = torch.tensor(behaviour_data, dtype=torch.long)
# behaviour_data = torch.tensor(behaviour_data, dtype=torch.float32)
spikes = torch.tensor(spikes, dtype=torch.float32)
num_trials, time_bins, emissions_dim = np.array(spikes).shape
# create dataloader with random sampling for training and testing
# split data into training and testing
# behaviour_data_train, behaviour_data_test, spikes_train, spikes_test = train_test_split(behaviour_data, spikes, test_size=0.3, random_state=42)
behaviour_data_train, behaviour_data_test, spikes_train, spikes_test, amp_train, amp_test = train_test_split(behaviour_data, spikes, amp, test_size=0.2, random_state=7)
# create dataloaders
train_dataset = TensorDataset(behaviour_data_train, spikes_train, amp_train)
test_dataset = TensorDataset(behaviour_data_test, spikes_test, amp_test)

batch_size = config_global['batch_size']
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# convert to numpy
spikes_train_np = spikes_train.detach().numpy()
spikes_test_np = spikes_test.detach().numpy()
spikes_np = spikes.detach().numpy()

def test(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for _, (behavior_batch, spikes_batch, amp_batch) in enumerate(test_loader):
            vae_pred, behavior_pred, amp_pred = model(spikes_batch, n_samples=20, use_mean_for_decoding=True)
            # calculate loss
            loss, loss_l = model.loss(np.inf, spikes_batch, behavior_batch, amp_batch, vae_pred, behavior_pred, amp_pred)
            # l.append(loss_l[1])
            test_loss += np.array(loss_l)            
            # print(np.mean(l), np.std(l))
    # divide loss by total number of samples in dataloader    
    return test_loss/len(test_loader)


def train(config, model: Model, train_loader, val_loader, early_stop):   
    train_losses, test_losses = [], []
    test_every = config['test_every']    
    train_decoder_after = config['decoder']['train_decoder_after']    
    num_samples_train = config['num_samples_train']
    optim_size = config['optim_size']
    save_model = False    
    for epoch in range(config['epochs']):
        # forward pass
        # print(model.behavior_decoder.scheduler.get_last_lr())
        # model.vae.scheduler.get_last_lr()
        epoch_loss = 0
        model.train()
        model.optim_zero_grad()
        optim_counter = 0
        for i, (behavior_batch, spikes_batch, amp_batch) in enumerate(train_loader):            
            # behavior_batch = behavior_batch.long()
            vae_pred, behavior_pred, amp_pred = model(spikes_batch, n_samples=num_samples_train, use_mean_for_decoding=False)            
            optim_counter += len(behavior_batch)
            # calculate loss            
            loss, loss_l = model.loss(epoch, spikes_batch, behavior_batch, amp_batch, vae_pred, behavior_pred, amp_pred)
            epoch_loss += np.array(loss_l)            
            # backward pass            
            loss.backward()
            
            # print gradient of any weight
            # if epoch > 10:
            #     print(model.behavior_decoder.conv_choice[1].weight.grad)
            if optim_counter >= optim_size:
                model.optim_step(train_decoder = epoch >= train_decoder_after)
                model.optim_zero_grad()
                print("Stepping inside")
                optim_counter = 0
        # do it for the rest        
        model.optim_step(train_decoder = epoch >= train_decoder_after)
        model.optim_zero_grad()
        
        # if epoch % 100 == 0:
        #     # print lr of decoder
        #     print(model.behavior_decoder.scheduler.get_last_lr())
        train_losses.append((epoch, epoch_loss/len(train_loader)))
        model.scheduler_step(step_decoder = epoch >= train_decoder_after)
        # test loss
        if (epoch+1) % test_every == 0:
            test_loss = test(model, val_loader)
            sum_test_loss = np.sum(test_loss)
            # scheduler.step(sum_test_loss)
            test_losses.append((epoch, test_loss))
            early_stop(sum_test_loss, model, save_model=save_model, save_prefix='best')
            # model.save_model(save_prefix=str(epoch))
            print('Epoch [{}/{}], Train Loss: {}, Test Loss: {}, Best Loss: {}'.format(epoch+1, config['epochs'], train_losses[-1][1], test_losses[-1][1], early_stop.best_score))
            if early_stop.slow_down:
                test_every = config['early_stop']['test_every_new']
            else:
                test_every = config['test_every']
            if early_stop.early_stop:
                print("Early stopping")
                break
            
    utils.plot_loss_curve(model, config, train_losses, test_losses)
    with torch.no_grad():
        model.eval()        
        # run on only test
        vae_output, _, amp_out_test = model.forward(spikes_test, n_samples=1, use_mean_for_decoding=True)  
        y_recon_test, x_mu_test, z_mu_test, x_A_test, z_A_test, x_test, z_test, z_test_presoftmax, g_test = model.vae.extract_relevant(vae_output)
        # run only on train
        vae_output, _, amp_out_train = model.forward(spikes_train, n_samples=1, use_mean_for_decoding=True)
        y_recon_train, x_mu_train, z_mu_train, x_A_train, z_A_train, x_train, z_train, z_train_presoftmax, g_train = model.vae.extract_relevant(vae_output)
        # run on both
        vae_output, _, amp_out_all = model.forward(spikes, n_samples=1, use_mean_for_decoding=True)
        y_recon_all, x_mu_all, z_mu_all, x_A_all, z_A_all, x_all, z_all, z_presoftmax_all, g_all = model.vae.extract_relevant(vae_output)

    # compute bits/spike
    bits_per_spike_train = utils.bits_per_spike(y_recon_train, spikes_train_np).sum()
    bits_per_spike_test = utils.bits_per_spike(y_recon_test, spikes_test_np).sum()
    bits_per_spike = utils.bits_per_spike(y_recon_all, spikes_np).sum()

    plt.figure()
    z = z_all
    # z = z_mu_all    
    x = x_mu_all
    z_std = np.std(z, axis=0)
    z_avg = np.mean(z, axis=0)
    # make x ticks of range 0.1 from -2 to 0.5
    bin_len = config['shape_dataset']['win_len']
    t = np.arange(-2, 0.5, bin_len)
    for i in range(z.shape[2]):
        plt.plot(t, z_avg[:, i], label='z{}'.format(i), color=colors[i])    
        plt.fill_between(t, z_avg[:, i]-z_std[:, i], z_avg[:, i]+z_std[:, i], alpha=0.3, color=colors[i])
    # plt.set_title('z')
    plt.legend()
    plt.savefig(os.path.join(utils.model_store_path(config, model.arch_name), 'z_avg.png'))

    agg_pred_train, agg_pred_test = [], []
    agg_y_train, agg_y_test = [], []
    # convert to numpy
    y_train = behaviour_data_train.numpy()
    y_test = behaviour_data_test.numpy()
    # accuracy of stimulus and choice
    acc_stim_train, acc_stim_test = [], []
    acc_choice_train, acc_choice_test = [], []

    with torch.no_grad():
        model.eval()
        behavior_pred_train = model.forward(spikes_train, n_samples=1, use_mean_for_decoding=True)[1]
        behavior_pred_test = model.forward(spikes_test, n_samples=1, use_mean_for_decoding=True)[1]
        # behavior_pred_train = model.forward(spikes_train, n_samples=1, use_mean_for_decoding=False)[1]
        # behavior_pred_test = model.forward(spikes_test, n_samples=1, use_mean_for_decoding=False)[1]
        # convert to numpy
        # pred_train = behavior_pred_train.numpy() > 0
        # pred_test = behavior_pred_test.numpy() > 0        
        pred_train_stim = torch.argmax(behavior_pred_train[:, :2], dim=1).numpy()
        pred_test_stim = torch.argmax(behavior_pred_test[:, :2], dim=1).numpy()
        pred_train_choice = torch.argmax(behavior_pred_train[:, 2:4], dim=1).numpy()
        pred_test_choice = torch.argmax(behavior_pred_test[:, 2:4], dim=1).numpy()    
        # compute accuracy        
        accuracy_train_stim = accuracy_score(y_train[:, 0], pred_train_stim)
        accuracy_test_stim = accuracy_score(y_test[:, 0], pred_test_stim)        
        # do the same for choice
        accuracy_train_choice = accuracy_score(y_train[:, 1], pred_train_choice)
        accuracy_test_choice = accuracy_score(y_test[:, 1], pred_test_choice)
    
    # dump all results
    pth = utils.model_store_path(config, model.arch_name)
    with open(os.path.join(pth, 'all_results.pkl'), 'wb') as f:
        to_dump = (train_losses, test_losses, bits_per_spike_train, bits_per_spike_test, accuracy_train_stim, accuracy_test_stim, accuracy_train_choice, accuracy_test_choice)


def one_train(config, device):        
    # create model and optimizer
    model = Model(config, input_dim=emissions_dim) #, neuron_bias=neuron_bias
    # model = torch.compile(model)
    early_stop = EarlyStopping(patience=config['early_stop']['patience'], delta=config['early_stop']['delta'], trace_func=print)
    # create dataloaders
    batch_size = config['batch_size']
    utils.set_seeds(config['seed'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # train model
    train(config, model, train_loader, test_loader, early_stop)
    

def loop_fixed(idx):
    print("Exp with idx = {}".format(idx))
    config = deepcopy(config_global)    
    config['dir']['results'] = 'results/seed'

    for seed in range(idx, idx+10):
        config['seed'] = seed
        print("Seed = {}".format(seed))
        one_train(config, device)        

# create optuna function
def objective_(trial):
    config = deepcopy(config_global)

    # config['rnn']['hidden_size'] = trial.suggest_categorical('hidden_size', [24, 32, 48])
    config['vae_gp']['rnn_encoder']['hidden_size'] = trial.suggest_categorical('hidden_size', [4, 8])
    config['vae_gp']['rnn_encoder']['num_layers'] = trial.suggest_categorical('num_layers_rnn', [1, 2])        
    config['vae_gp']['lr'] = trial.suggest_float('lr', 1e-3, 0.1, log=True)
    
    config['decoder']['choice_weight'] = trial.suggest_int('choice_weight', 5, 20)
    multiplier = trial.suggest_float('multipler', 1, 4)
    config['decoder']['stimulus_weight'] = config['decoder']['choice_weight'] * multiplier
    config['decoder']['cnn']['channels'] = [trial.suggest_int('channels', 1, 4)*4]

    return one_train(config, device)


def exp():    
    study_name = 'results/vaegp_tuning'  # Unique identifier of the study    
    config_global['dir']['results'] = 'results/vaegp_tuning/'    

    if not os.path.exists(study_name + '.db'):
        study = optuna.create_study(study_name=study_name, storage='sqlite:///' + study_name + '.db', direction="minimize")
    else:
        study = optuna.load_study(study_name=study_name, storage='sqlite:///' + study_name + '.db')
    
    study.optimize(objective_, n_trials=100)
    df = study.trials_dataframe()
    df.to_csv(open(study_name + ".csv", 'w'), index=False, header=True)


if __name__ == '__main__':
    # loop_fixed(0)
    # one_train(config_global, device)
    # exp()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int)
    args = parser.parse_args()    
    loop_fixed(args.idx)