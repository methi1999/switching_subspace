import torch
import torch.nn as nn


class LinearAccDecoder(nn.Module):
    def __init__(self, config, input_dim):
        super(LinearAccDecoder, self).__init__()
        # generate linear layers with hidden dims 
        layers = []        
        hidden_dims = config['decoder']['linear']['hidden_dims']
        for i in range(len(hidden_dims)):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dims[i]))
            else:
                layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dims[-1], 2))
        self.fc = nn.Sequential(*layers)        
        self.stimulus_weight = config['decoder']['stimulus_weight']
        # name
        self.arch_name = 'linear_{}'.format(hidden_dims)
    
    def forward(self, x, z):
        # x and z are of shape (batch_size, seq_len, input_dim)
        x = self.fc(x)
        # weight by z
        element_wise = x * z
        return torch.sum(element_wise, dim=1)

    def loss(self, predicted, ground_truth):
        # bce loss
        loss = nn.BCEWithLogitsLoss()
        choice_loss = loss(predicted[:, 0], ground_truth[:, 0])
        stimulus_loss = loss(predicted[:, 1], ground_truth[:, 1])
        return self.stimulus_weight * stimulus_loss + choice_loss


class CNNDecoder(nn.Module):
    def __init__(self, input_dim, stimulus_weight=1):
        super(CNNDecoder, self).__init__()
        channels = [8, 8, 8, 8]
        # 1d conv
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=channels[1], out_channels=channels[2], kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=channels[2], out_channels=channels[3], kernel_size=3, padding=1)
        self.fc = nn.Linear(channels[-1], 2)
        self.stimulus_weight = stimulus_weight
    
    def forward(self, x, z):
        # x is of shape (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))        
        x = torch.max(x, dim=2).values
        x = self.fc(x)
        # max pool across time
        return x

    def loss(self, predicted, ground_truth):
        # bce loss
        loss = nn.BCEWithLogitsLoss()
        choice_loss = loss(predicted[:, 0], ground_truth[:, 0])
        stimulus_loss = loss(predicted[:, 1], ground_truth[:, 1])
        return self.stimulus_weight * stimulus_loss + choice_loss