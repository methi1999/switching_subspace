import torch
import torch.nn as nn


class LinearAccDecoder(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        # generate linear layers with hidden dims 
        layers = []        
        hidden_dims = config['decoder']['linear']['hidden_dims']
        dropout = config['decoder']['linear']['dropout']
        for i in range(len(hidden_dims)):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dims[i]))
            else:
                layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dims[-1], 2))
        self.fc = nn.Sequential(*layers)        
        self.stimulus_weight = config['decoder']['stimulus_weight']
        # name
        self.arch_name = 'linear_{}'.format(hidden_dims)

        self.forward = self.forward1
        self.loss = self.loss1
    
    def forward1(self, x, z):        
        # x and z are of shape (batch_size, seq_len, input_dim)
        x = self.fc(x)
        num_heads = x.size(-1)        
        # weight by z
        element_wise = x * z[:, :, :num_heads]
        return torch.sum(element_wise, dim=1)

    def forward2(self, x, z):
        # x and z are of shape (batch_size, seq_len, input_dim)             
        ret = self.fc(x)        

    def loss1(self, predicted, ground_truth, z):
        # bce loss
        loss = nn.BCEWithLogitsLoss()
        choice_loss = loss(predicted[:, 0], ground_truth[:, 0])
        stimulus_loss = loss(predicted[:, 1], ground_truth[:, 1])
        return self.stimulus_weight * stimulus_loss + choice_loss
    
    def loss2(self, predicted, ground_truth, z):
        # bce loss
        loss = nn.BCEWithLogitsLoss(reduction='none')     
        ground_truth = ground_truth.unsqueeze(1).expand(-1, predicted.size(1), -1)        
        choice_loss = loss(predicted[:, :, 0], ground_truth[:, :, 0])*z[:, :, 0]
        stimulus_loss = loss(predicted[:, :, 1], ground_truth[:, :, 1])*z[:, :, 1]
        # sum        
        return self.stimulus_weight * stimulus_loss.mean() + choice_loss.mean()


class CNNDecoder(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.stimulus_weight = config['decoder']['stimulus_weight']

        channels = config['decoder']['cnn']['channels']
        kernel_size = config['decoder']['cnn']['kernel_size']
        pad = (kernel_size - 1)//2
        dropout = config['decoder']['cnn']['dropout']        
        # 1d conv
        # make list of 1d convs
        layers = []
        for i in range(len(channels)):
            if i == 0:
                layers.append(nn.Conv1d(in_channels=input_dim, out_channels=channels[i], kernel_size=kernel_size, padding=pad))
            else:
                layers.append(nn.Conv1d(in_channels=channels[i-1], out_channels=channels[i], kernel_size=kernel_size, padding=pad))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], 2)        

        # name
        self.arch_name = 'cnn_{}_{}'.format('-'.join([str(x) for x in channels]), kernel_size)

    def forward(self, x, z):
        # x is of shape (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)
        x = self.conv(x)       
        x = torch.max(x, dim=2).values        
        x = self.fc(x)
        # max pool across time
        return x

    def loss(self, predicted, ground_truth, z):
        # bce loss
        loss = nn.BCEWithLogitsLoss()
        choice_loss = loss(predicted[:, 0], ground_truth[:, 0])
        stimulus_loss = loss(predicted[:, 1], ground_truth[:, 1])
        return self.stimulus_weight * stimulus_loss + choice_loss