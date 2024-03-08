import numpy as np
import os
import torch
import optuna
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pickle

from model import Model
from early_stopping import EarlyStopping
import utils


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
behaviour_data, spikes = utils.load_dataset(config_global)
num_trials, time_bins, emissions_dim = np.array(spikes).shape
# load behaviour data
stim_idx, choice_idx = 6, 3
stim = [x[0, stim_idx] for x in behaviour_data]
choice = [x[0, choice_idx] for x in behaviour_data]
# concat them
behaviour_data = np.stack((stim, choice), axis=1)
spikes = np.array(spikes)
# convert to torch tensors
behaviour_data = torch.tensor(behaviour_data).float()
spikes = torch.tensor(spikes).float()
# split data
# create dataloader with random sampling for training and testing
# split data into training and testing
behaviour_data_train, behaviour_data_test, spikes_train, spikes_test = train_test_split(behaviour_data, spikes, test_size=0.3, random_state=42)
# transfer to device
behaviour_data_train, behaviour_data_test = behaviour_data_train.to(device), behaviour_data_test.to(device)
spikes_train, spikes_test = spikes_train.to(device), spikes_test.to(device)
# create dataloaders
train_dataset = TensorDataset(behaviour_data_train, spikes_train)
test_dataset = TensorDataset(behaviour_data_test, spikes_test)

# bias
# mean firing rate of neurons in tran spikes
neuron_bias = torch.mean(spikes_train, dim=0)


def test(model: Model, test_loader, n_samples=200):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for _, (behavior_batch, spikes_batch) in enumerate(test_loader):
            samples_xz, samples_behavior = model.sample(spikes_batch, n_samples=n_samples)
            # l = []
            for i in range(len(samples_xz)):
                y_recon, (mu, A), (z, x) = samples_xz[i]
                behavior_batch_pred = samples_behavior[i]
                # calculate loss
                _, loss_l = model.loss(None, spikes_batch, y_recon, mu, A, z, x, behavior_batch_pred, behavior_batch)
                # l.append(loss_l[1])
                test_loss += np.array(loss_l)            
            # print(np.mean(l), np.std(l))
    # divide loss by total number of samples in dataloader    
    return test_loss/(len(test_loader) * n_samples)


def train(config, model: Model, train_loader, val_loader):
    train_losses, test_losses = [], []
    test_every = config['test_every']
    num_epochs = config['epochs']
    early_stop = EarlyStopping(patience=config['early_stop']['patience'], delta=config['early_stop']['delta'],
                            trace_func=print)
    save_model = False
    for epoch in range(num_epochs):
        # print(model.behavior_decoder.scheduler.get_last_lr())
        # forward pass
        epoch_loss = 0
        for i, (behavior_batch, spikes_batch) in enumerate(train_loader):            
            model.train()
            y_recon, (mu, A), (z, x), behavior_pred = model(spikes_batch)
            loss, loss_l = model.loss(epoch, spikes_batch, y_recon, mu, A, z, x, behavior_pred, behavior_batch)        
            # backward pass
            model.optim_zero_grad()
            loss.backward()
            model.optim_step()
            epoch_loss += np.array(loss_l)

        train_losses.append((epoch, epoch_loss/len(train_loader)))
        model.scheduler_step(epoch)
        # test loss
        if (epoch+1) % test_every == 0:            
            test_loss = test(model, val_loader)
            sum_test_loss = np.sum(test_loss)
            # scheduler.step(sum_test_loss)
            test_losses.append((epoch, test_loss))            
            if only_look_at_decoder:
                early_stop(test_loss[-1], model, save_model=save_model, save_prefix='best')
            else:
                early_stop(sum_test_loss, model, save_model=save_model, save_prefix='best')
            print('Epoch [{}/{}], Train Loss: {}, Test Loss: {}, Best Loss: {}'.format(epoch+1, num_epochs, train_losses[-1][1], test_losses[-1][1], early_stop.best_score))
            if early_stop.slow_down:
                test_every = config['early_stop']['test_every_new']
            else:
                test_every = config['test_every']
            if early_stop.early_stop:
                print("Early stopping")
                break
    # utils.plot_curve(model, config, train_losses, test_losses)
    only_test_loss = [x[1][1] for x in test_losses]
    # if only_look_at_decoder:
    #     only_test_loss = [x[1][1] for x in test_losses]
    # else:
    #     only_test_loss = [np.sum(x[1]) for x in test_losses]
    
    # compute min test loss and return it    
    # return np.min(only_test_loss), train_losses, test_losses
    
    # compute median of test loss in a window of 15
    meds = []
    window = 5
    only_test_loss = [np.nan]*(window) + only_test_loss + [np.nan]*(window)
    for i in range(window, len(only_test_loss)-window):
        meds.append(np.nanmean(only_test_loss[i-window:i+window]))
    return np.min(meds), train_losses, test_losses


def one_train(config, device):        
    # create model and optimizer
    model = Model(config, input_dim=emissions_dim).to(device)
    # create dataloaders
    batch_size = config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # train
    best_test_loss, train_losses, test_losses = train(config, model, train_loader, test_loader)
    # utils.plot_curve(model, config, train_losses, test_losses)
    # save losses
    pth = utils.model_store_path(config, model.arch_name)
    with open(os.path.join(pth, 'losses.pkl'), 'wb') as f:
        pickle.dump((best_test_loss, train_losses, test_losses), f)
    return best_test_loss
   


# create optuna function
def objective_(trial):
    config = deepcopy(config_global)    
    # config['rnn']['hidden_size'] = trial.suggest_categorical('hidden_size', [24, 32, 48])
    config['rnn']['hidden_size'] = trial.suggest_categorical('hidden_size', [24, 32, 48])
    config['rnn']['num_layers'] = trial.suggest_categorical('num_layers', [1, 2])        
    # config['rnn']['num_layers'] = 1       
    config['rnn']['dropout'] = trial.suggest_float('dropout', 0.1, 0.4)
    # config['rnn']['dropout'] = 0.15
    config['batch_size'] = trial.suggest_categorical('batch_size', [32, 48, 96])    
    # config['batch_size'] = 48
    # config['decoder']['cnn']['lr'] = trial.suggest_float('cnn_lr', 1e-5, 0.1, log=True)
    config['decoder']['cnn']['lr'] = trial.suggest_float('cnn_lr', 1e-4, 0.1, log=True)
    config['decoder']['cnn']['kernel_size'] = trial.suggest_categorical('kernel_size', [3, 5, 9])        
    config['decoder']['cnn']['dropout'] = trial.suggest_float('dropout_cnn', 0.1, 0.5)
    # config['decoder']['cnn']['dropout'] = 0.25
    # chans = trial.suggest_categorical('channels', [6, 12])
    chans = trial.suggest_categorical('channels', [4, 8, 16])
    # chans = 8
    layers = trial.suggest_categorical('lay', [2, 4, 6])
    config['decoder']['cnn']['channels'] = [chans]*layers    

    res = []
    for _ in range(3):
        utils.set_seeds(np.random.randint(1000))
        res.append(one_train(config, device))
    return np.mean(res)    


def exp():    
    study_name = 'results/stim'  # Unique identifier of the study    
    config_global['dir']['results'] = 'results/stim/'
    if not os.path.exists(study_name + '.db'):
        study = optuna.create_study(study_name=study_name, storage='sqlite:///' + study_name + '.db', direction="minimize")
    else:
        study = optuna.load_study(study_name=study_name, storage='sqlite:///' + study_name + '.db')
    
    # study.optimize(objective_, n_trials=400)
    df = study.trials_dataframe()
    df.to_csv(open(study_name + ".csv", 'w'), index=False, header=True)


if __name__ == '__main__':
#     one_train(config_global, device)
    exp()