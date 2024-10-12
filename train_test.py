import torch
import numpy as np
from model import Model
from torch.utils.data import DataLoader
from early_stopping import EarlyStopping

torch.autograd.set_detect_anomaly(True)


def test(model: Model, test_loader: DataLoader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for _, (spikes_batch, behavior_batch, amp_batch) in enumerate(test_loader):
            vae_pred, behavior_pred, amp_pred = model(spikes_batch, n_samples=1, use_mean_for_decoding=True)
            # vae_pred, behavior_pred, amp_pred = model(spikes_batch, n_samples=20, use_mean_for_decoding=False)
            # calculate loss
            loss, loss_l = model.loss(np.inf, spikes_batch, behavior_batch, amp_batch, vae_pred, behavior_pred, amp_pred)
            # l.append(loss_l[1])
            test_loss += np.array(loss_l)            
            
    # divide loss by total number of samples in dataloader    
    return test_loss/len(test_loader)

# train the model

def train(config: dict, model: Model, train_loader: DataLoader, val_loader: DataLoader, early_stop: EarlyStopping, save_epochs: bool, train_losses: list, test_losses: list, save_prefix: str):    
        
    test_every = config['test_every']    
    train_decoder_after = config['decoder']['train_decoder_after']    
    num_samples_train = config['num_samples_train']
    optim_size = config['optim_size']
    # train    
    for epoch in range(config['epochs']):        
        epoch_loss = 0
        model.train()
        model.optim_zero_grad()
        optim_counter = 0
        for i, (spikes_batch, behavior_batch, amp_batch) in enumerate(train_loader):            
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
        # test loss
        if (epoch+1) % test_every == 0:
            test_loss = test(model, val_loader)
            sum_test_loss = np.sum(test_loss)
            # scheduler.step(sum_test_loss)
            test_losses.append((epoch, test_loss))
            early_stop(sum_test_loss, model, save_model=True, save_prefix=f'{save_prefix}_best')
            if save_epochs and epoch % 100 == 0:
                model.save_model(save_prefix=f'{save_prefix}_epoch_{epoch}')
            print('Epoch [{}/{}], Train Loss: {}, Test Loss: {}, Best Loss: {}'.format(epoch+1, config['epochs'], train_losses[-1][1], test_losses[-1][1], early_stop.best_score))
            if early_stop.slow_down:
                test_every = config['early_stop']['test_every_new']
            else:
                test_every = config['test_every']
            if early_stop.early_stop:
                print("Early stopping")
                break
            
    
    only_test_loss = [np.sum(x[1]) for x in test_losses]
    
    # compute min test loss and return it    
    return np.min(only_test_loss), train_losses, test_losses
    
    # # compute median of test loss in a window of 5
    # meds = []
    # half_window = 10
    # only_test_loss = [0]*(half_window) + only_test_loss + [0]*(half_window)
    # for i in range(half_window, len(only_test_loss)-half_window):
    #     meds.append(np.max(only_test_loss[i-half_window:i+half_window]))
    # return np.min(meds), train_losses, test_losses