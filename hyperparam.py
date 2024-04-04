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
behaviour_data_train, behaviour_data_test, spikes_train, spikes_test = train_test_split(behaviour_data, spikes, test_size=0.2, random_state=7)
# transfer to device
behaviour_data_train, behaviour_data_test = behaviour_data_train.to(device), behaviour_data_test.to(device)
spikes_train, spikes_test = spikes_train.to(device), spikes_test.to(device)
# create dataloaders
train_dataset = TensorDataset(behaviour_data_train, spikes_train)
test_dataset = TensorDataset(behaviour_data_test, spikes_test)

# bias
# mean firing rate of neurons in tran spikes
neuron_bias = torch.mean(spikes_train, dim=0)


def test(model, test_loader, n_samples):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for _, (behavior_batch, spikes_batch) in enumerate(test_loader):
            vae_pred, behavior_pred = model(spikes_batch, n_samples=n_samples)
            # calculate loss
            loss, loss_l = model.loss(None, spikes_batch, behavior_batch, vae_pred, behavior_pred)
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
    save_model = True    
    for epoch in range(config['epochs']):
        # forward pass
        # print(model.behavior_decoder.scheduler.get_last_lr())
        # model.vae.scheduler.get_last_lr()
        epoch_loss = 0
        model.train()
        for i, (behavior_batch, spikes_batch) in enumerate(train_loader):            
            # behavior_batch = behavior_batch.long()
            vae_pred, behavior_pred = model(spikes_batch, n_samples=num_samples_train)
            # calculate loss
            loss, loss_l = model.loss(epoch, spikes_batch, behavior_batch, vae_pred, behavior_pred)
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
            # test_loss = test(model, val_loader, n_samples=config['num_samples_test'])
            # sum_test_loss = np.sum(test_loss)
            # scheduler.step(sum_test_loss)
            # test_losses.append((epoch, test_loss))
            # early_stop(sum_test_loss, model, save_model=save_model, save_prefix='best')
            early_stop(train_losses[-1][-1], model, save_model=False, save_prefix='best')
            # model.save_model(save_prefix=str(epoch))

            # print('Epoch [{}/{}], Train Loss: {}, Test Loss: {}, Best Loss: {}'.format(epoch+1, config['epochs'], train_losses[-1][1], test_losses[-1][1], early_stop.best_score))
            print('Epoch [{}/{}], Train Loss: {}'.format(epoch+1, config['epochs'], train_losses[-1][1]))
            if early_stop.slow_down:
                test_every = config['early_stop']['test_every_new']
            else:
                test_every = config['test_every']
            if early_stop.early_stop:
                print("Early stopping")
                break
            
    
    # to_consider = [np.sum(x[1]) for x in test_losses]
    to_consider = [x[1] for x in train_losses]
    
    # compute min test loss and return it    
    return np.min(to_consider), train_losses, test_losses


def one_train(config, device):        
    # create model and optimizer
    config['test_every'] = 1
    model = Model(config, input_dim=emissions_dim) #, neuron_bias=neuron_bias
    # model = torch.compile(model)
    early_stop = EarlyStopping(patience=config['early_stop']['patience'], delta=config['early_stop']['delta'], trace_func=print)
    # create dataloaders
    batch_size = config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # train
    best_test_loss, train_losses, test_losses = train(config, model, train_loader, test_loader, early_stop)
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
    config['vae_gp']['rnn_encoder']['hidden_size'] = trial.suggest_categorical('hidden_size', [8, 24])
    config['vae_gp']['rnn_encoder']['num_layers'] = trial.suggest_categorical('num_layers', [1, 2, 3])        
    config['vae_gp']['lr'] = trial.suggest_float('lr', 1e-4, 0.1, log=True)
    config['vae_gp']['kl_beta'] = 0.01
    config['vae_gp']['smoothing_sigma'] = None
    # config['rnn_encoder']['num_layers'] = 1       
    # config['vae_gp']['rnn_encoder']['dropout'] = trial.suggest_float('dropout', 0.1, 0.4)
    # config['rnn']['dropout'] = 0.15

    num_layers = trial.suggest_categorical('num_layers', [1, 2, 3])
    num_units = trial.suggest_categorical('num_units', [8, 16])
    config['vae_gp']['post_rnn_linear']['hidden_dims'] = [num_units]*num_layers

    config['batch_size'] = trial.suggest_categorical('batch_size', [8, 32, 96])
    config['num_samples_train'] = trial.suggest_categorical('num_samples_train', [10, 50])


    # config['batch_size'] = 48
    # config['decoder']['cnn']['lr'] = trial.suggest_float('cnn_lr', 1e-5, 0.1, log=True)
    # config['decoder']['cnn']['lr'] = trial.suggest_float('cnn_lr', 1e-4, 0.1, log=True)
    # config['decoder']['cnn']['kernel_size'] = trial.suggest_categorical('kernel_size', [3, 5, 9])        
    # config['decoder']['cnn']['dropout'] = trial.suggest_float('dropout_cnn', 0.1, 0.5)
    # config['decoder']['cnn']['dropout'] = 0.25
    # chans = trial.suggest_categorical('channels', [6, 12])
    # chans = trial.suggest_categorical('channels', [4, 8, 16])
    # chans = 8
    # layers = trial.suggest_categorical('lay', [2, 4, 6])
    # config['decoder']['cnn']['channels'] = [chans]*layers    

    return one_train(config, device)


def exp():    
    study_name = 'results/vae_gp_combined'  # Unique identifier of the study    
    config_global['dir']['results'] = 'results/vae_gp_combined/'
    # study_name = 'results/full_cov'  # Unique identifier of the study    
    # config_global['dir']['results'] = 'results/full_cov/'
    config_global['vae_gp']['full_cov'] = False

    if not os.path.exists(study_name + '.db'):
        study = optuna.create_study(study_name=study_name, storage='sqlite:///' + study_name + '.db', direction="minimize")
    else:
        study = optuna.load_study(study_name=study_name, storage='sqlite:///' + study_name + '.db')
    
    study.optimize(objective_, n_trials=100)
    df = study.trials_dataframe()
    df.to_csv(open(study_name + ".csv", 'w'), index=False, header=True)


if __name__ == '__main__':
#     one_train(config_global, device)
    exp()