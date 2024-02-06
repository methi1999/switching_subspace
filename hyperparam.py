import numpy as np
import os
import torch
import optuna
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from model import Model
from early_stopping import EarlyStopping
import utils

# is_cuda = torch.cuda.is_available()
is_cuda = False
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

# create dataloaders
train_dataset = TensorDataset(behaviour_data_train, spikes_train)
test_dataset = TensorDataset(behaviour_data_test, spikes_test)

# bias
# mean firing rate of neurons in tran spikes
neuron_bias = torch.mean(spikes_train, dim=0)


def test(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for _, (behavior_batch, spikes_batch) in enumerate(test_loader):
            y_recon, (mu, A), (z, x), behavior_batch_pred = model(spikes_batch)
            _, loss_l = model.loss(spikes_batch, y_recon, mu, A, z, x, behavior_batch_pred, behavior_batch)
            test_loss += np.array(loss_l)
    # divide loss by total number of samples in dataloader    
    return test_loss/len(test_loader)


def train(config, model, optimizer, num_epochs, train_loader, val_loader):
    train_losses, test_losses = [], []
    test_every = config['test_every']
    early_stop = EarlyStopping(patience=config['early_stop']['patience'], delta=config['early_stop']['delta'],
                            trace_func=print)
    save_model = True
    for epoch in range(num_epochs):
        # forward pass
        epoch_loss = 0
        for i, (behavior_batch, spikes_batch) in enumerate(train_loader):
            model.train()
            y_recon, (mu, A), (z, x), behavior_pred = model(spikes_batch)
            loss, loss_l = model.loss(spikes_batch, y_recon, mu, A, z, x, behavior_pred, behavior_batch)        
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
            epoch_loss += np.array(loss_l)
        train_losses.append(epoch_loss/len(train_loader))
        # test loss
        if (epoch+1) % test_every == 0:
            test_loss = test(model, val_loader)
            test_losses.append(test_loss)
            early_stop(np.sum(test_loss), model, save_model=save_model, save_prefix='best')
            print('Epoch [{}/{}], Train Loss: {}, Test Loss: {}'.format(epoch+1, num_epochs, train_losses[-1], test_losses[-1]))            
            if early_stop.slow_down:
                test_every = config['early_stop']['test_every_new']
            else:
                test_every = config['test_every']
            if early_stop.early_stop:
                print("Early stopping")
                break
    # compute min test loss and return it    
    return np.min([np.sum(x) for x in test_losses])


def one_train(config, device):
    num_epochs, learning_rate = config['epochs'], config['lr']
    # create model and optimizer
    model = Model(config, input_dim=emissions_dim, z_dim=2, x_dim=2, neuron_bias=neuron_bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # create dataloaders
    batch_size = config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # train
    best_test_loss = train(config, model, optimizer, num_epochs, train_loader, test_loader)
    return best_test_loss
   


# create optuna function
def objective_mlp(trial):
    config = deepcopy(config_global)
    config['rnn']['hidden_size'] = trial.suggest_categorical('dim_z', [8, 16, 32, 64])
    config['rnn']['num_layers'] = trial.suggest_categorical('num_layers', [1, 2, 3])    
    config['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64, 128])        
    config['rnn']['dropout'] = trial.suggest_float('dropout', 0.1, 0.5)
    config['lr'] = trial.suggest_float('lr', 1e-3, 1e-1, log=True)

    return one_train(config, device)         


def exp():    
    study_name = 'results/only_decoding'  # Unique identifier of the study    
    if not os.path.exists(study_name + '.db'):
        study = optuna.create_study(study_name=study_name, storage='sqlite:///' + study_name + '.db', direction="minimize")
    else:
        study = optuna.load_study(study_name=study_name, storage='sqlite:///' + study_name + '.db')

    # func = lambda trial: objective_rnn(trial, base_seed, study_name)
    study.optimize(objective_mlp, n_trials=200)
    df = study.trials_dataframe()
    df.to_csv(open(study_name + ".csv", 'w'), index=False, header=True)


if __name__ == '__main__':
#     one_train(config_global, device)
    exp()