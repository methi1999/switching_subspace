import torch
import torch.nn as nn


class CNNDecoderIndivdual(nn.Module):
    def __init__(self, config, xz_list):
        super().__init__()
        self.stim_dim, self.choice_dim = xz_list[0], xz_list[1]        

        channels = config['decoder']['cnn']['channels']
        kernel_size = config['decoder']['cnn']['kernel_size']
        pad = (kernel_size - 1)//2
        dropout = config['decoder']['cnn']['dropout']

        def make_1d_conv(inp_dim):
            # 1d conv        
            layers = []
            for i in range(len(channels)):
                if i == 0:
                    layers.append(nn.Conv1d(in_channels=inp_dim, out_channels=channels[i], kernel_size=kernel_size, padding=pad))
                else:
                    layers.append(nn.Conv1d(in_channels=channels[i-1], out_channels=channels[i], kernel_size=kernel_size, padding=pad))
                layers.append(nn.Tanh())
                layers.append(nn.Dropout(dropout))            
            # linear layer
            layers.append(nn.Conv1d(in_channels=channels[-1], out_channels=1, kernel_size=1))
            return layers
        
        if self.stim_dim > 0:
            self.stimulus_weight = config['decoder']['stimulus_weight']
            self.conv_stim = nn.Sequential(*make_1d_conv(self.stim_dim))
            print("Using stimulus decoder")
        else:
            self.conv_stim = None      
        if self.choice_dim > 0:
            self.choice_weight = config['decoder']['choice_weight']
            self.conv_choice = nn.Sequential(*make_1d_conv(self.choice_dim))
            print("Using choice decoder")
        else:
            self.conv_choice = None                                
        # name
        self.arch_name = 'cnn_{}_{}'.format('-'.join([str(x) for x in channels]), kernel_size)
        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['decoder']['cnn']['lr'], weight_decay=config['decoder']['cnn']['weight_decay'])
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 90, 120, 150, 180], gamma=0.5)
        if config['decoder']['scheduler']['which'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=config['decoder']['scheduler']['cosine_restart_after'])
        elif config['decoder']['scheduler']['which'] == 'decay':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        else:
            print('Scheduler not implemented for decoder')
            self.scheduler = None

    def forward(self, x, z):
        # x is of shape (batch_size, seq_len, input_dim)        
        # x = x * z
        x = x.permute(0, 2, 1)
        z = z.permute(0, 2, 1)
        if self.conv_stim:
            x_stim = x[:, :self.stim_dim, :]
            x_stim = self.conv_stim(x_stim)
            x_stim = x_stim * z[:, 0:1, :]
            x_stim = torch.max(x_stim, dim=2).values
        else:
            x_stim = torch.zeros(x.size(0), 1)
        if self.conv_choice:
            x_choice = x[:, self.stim_dim:self.stim_dim+self.choice_dim, :]
            x_choice = self.conv_choice(x_choice)
            x_choice = x_choice * z[:, 1:2, :]
            x_choice = torch.max(x_choice, dim=2).values
        else:
            x_choice = torch.zeros(x.size(0), 1)
                
        return torch.cat([x_stim, x_choice], dim=1)        

    def loss(self, predicted, ground_truth, z, reduction='mean'):
        # bce loss
        loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)
        loss = 0
        if self.conv_stim:
            loss += loss_fn(predicted[:, 0], ground_truth[:, 0]) * self.stimulus_weight
        if self.conv_choice:
            loss += loss_fn(predicted[:, 1], ground_truth[:, 1]) * self.choice_weight        
        return loss

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
        # optim
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['decoder']['linear']['lr'], weight_decay=config['decoder']['linear']['weight_decay'])
    
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
        self.choice_weight = config['decoder']['choice_weight']
        
        input_dim = 2

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
            if i < len(channels) - 1:
                # layers.append(nn.ReLU())
                layers.append(nn.Tanh())
                layers.append(nn.Dropout(dropout))
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], 2)        

        # name
        self.arch_name = 'cnn_{}_{}'.format('-'.join([str(x) for x in channels]), kernel_size)
        # optim
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['decoder']['cnn']['lr'], weight_decay=config['decoder']['cnn']['weight_decay'])

        if config['decoder']['scheduler']['which'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=config['decoder']['scheduler']['cosine_restart_after'])
        elif config['decoder']['scheduler']['which'] == 'decay':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        else:
            print('Scheduler not implemented for decoder')
            self.scheduler = None

    def forward(self, x, z):
        # x is of shape (batch_size, seq_len, input_dim)
        x = x * z
        x = x.permute(0, 2, 1)
        x = x[:, :2, :]
        x = self.conv(x)        
        x = torch.max(x, dim=2).values        
        # x = torch.mean(x, dim=2)
        x = self.fc(x)
        # max pool across time
        return x

    def forward1(self, x, z):
        # x is of shape (batch_size, seq_len, input_dim)
        # x = x * z
        x = x.permute(0, 2, 1)
        x = x[:, :2, :]
        x = self.conv(x)        
        # x = torch.mean(x, dim=2)
        x = self.fc(x.transpose(1, 2))
        # print(x.shape, z.shape)
        x = x * z
        x = torch.max(x, dim=2).values
        # max pool across time
        return x

    def loss(self, predicted, ground_truth, z, reduction='mean'):
        # bce loss
        loss = nn.BCEWithLogitsLoss(reduction=reduction)
        stimulus_loss = loss(predicted[:, 0], ground_truth[:, 0])
        choice_loss = loss(predicted[:, 1], ground_truth[:, 1])        
        return self.stimulus_weight * stimulus_loss + choice_loss * self.choice_weight