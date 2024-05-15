import torch
import torch.nn as nn
import math


LOG2 = torch.log(torch.tensor(2))




class CNNDecoderIndividual(nn.Module):
    def __init__(self, config, xz_list):
        super().__init__()
        self.stim_dim, self.choice_dim = xz_list[0], xz_list[1]        
        self.choice_idx = 0
        if self.stim_dim > 0:
            self.choice_idx += 1
        assert self.stim_dim + self.choice_dim > 0, "Either stimulus or choice must be set"

        channels = config['decoder']['cnn']['channels']
        kernel_size = config['decoder']['cnn']['kernel_size']
        pad = (kernel_size - 1)//2
        dropout = config['decoder']['cnn']['dropout']
        self.one_sided_window = (kernel_size-1)//2

        def make_1d_conv(inp_dim):
            # 1d conv
            layers = []            
            for i in range(len(channels)):
                if i == 0:
                    layers.append(nn.Conv1d(in_channels=inp_dim, out_channels=channels[i], kernel_size=kernel_size, padding=pad))
                else:
                    layers.append(nn.Conv1d(in_channels=channels[i-1], out_channels=channels[i], kernel_size=kernel_size, padding=pad))
                # layers.append(nn.BatchNorm1d(channels[i]))
                # layers.append(nn.LeakyReLU())
                layers.append(nn.Tanh())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))            
            # linear layer
            layers.append(nn.Conv1d(in_channels=channels[-1], out_channels=2, kernel_size=1))
            # layers.append(nn.Conv1d(in_channels=channels[-1], out_channels=1, kernel_size=1))
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
        # cross terms
        self.cross_terms = config['decoder']['cross_terms']                                
        # normalize
        self.normalize_trials = config['decoder']['cnn']['normalize_trial_time']
        # name
        self.arch_name = 'cnn_{}_{}'.format('-'.join([str(x) for x in channels]), kernel_size)
        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['decoder']['cnn']['lr'], weight_decay=config['decoder']['cnn']['weight_decay'])
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 90, 120, 150, 180], gamma=0.5)
        if config['decoder']['scheduler']['which'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=config['decoder']['scheduler']['cosine_restart_after'])
            print('Using cosine annealing for decoder')
        elif config['decoder']['scheduler']['which'] == 'decay':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config['decoder']['scheduler']['const_factor'])
            print('Using decay annealing for decoder')
        else:
            print('Scheduler not implemented for decoder')
            self.scheduler = None
    
    def cnn_forward(self, x, z, z_dim, x_s_dim, x_e_dim, conv):
        x = x[:, x_s_dim: x_e_dim+1, :]
        z = z[:, z_dim: z_dim+1, :]        
        x = conv(x)
        x = x * z        
        # x = torch.mean(x, dim=2)
        x = torch.max(x, dim=2).values
        return x
    
    # def cnn_forward(self, x, z, z_dim, x_s_dim, x_e_dim, conv):
    #     x = x[:, x_s_dim: x_e_dim+1, :]
    #     z = z[:, z_dim: z_dim+1, :]
    #     # original
    #     # keep only certain predictions
    #     # zero out all x values outside a window of 3 around the peak of z
    #     peak_z = torch.argmax(z, dim=2).squeeze(-1)                
    #     mask = torch.zeros_like(x[:, 0, :])                
    #     mask.scatter_(1, peak_z.unsqueeze(1), 1)
    #     mask.scatter_(1, torch.clip(peak_z-1, 0).unsqueeze(1), 1)
    #     mask.scatter_(1, torch.clip(peak_z+1, 0, z.shape[-1]-1).unsqueeze(1), 1)
    #     mask.scatter_(1, torch.clip(peak_z-2, 0).unsqueeze(1), 1)
    #     mask.scatter_(1, torch.clip(peak_z+2, 0, z.shape[-1]-1).unsqueeze(1), 1)
    #     # mask.scatter_(1, torch.clip(peak_z-3, 0).unsqueeze(1), 1)
    #     # mask.scatter_(1, torch.clip(peak_z+3, 0, z.shape[-1]-1).unsqueeze(1), 1)
    #     mask = mask.unsqueeze(1)                
    #     x = x * mask        
    #     x = conv(x)        
    #     # x = torch.mean(x, dim=2)
    #     x = torch.max(x, dim=2).values
    #     return x

    def forward(self, x, z):
        # x is of shape (batch_size*num_samples, seq_len, input_dim)                
        # z = z.detach()
        # x = x * z
        if self.normalize_trials:
            x = x - x.mean(dim=1, keepdim=True)
            # x = x / torch.abs(x).max(dim=1, keepdim=True).values
        x = x.permute(0, 2, 1)
        z = z.permute(0, 2, 1)
        if self.conv_stim:
            x_stim = self.cnn_forward(x, z, 0, 0, self.stim_dim-1, self.conv_stim)            
            if self.cross_terms:
                x_choicepred = self.cnn_forward(x, z, self.choice_idx, self.stim_dim, self.stim_dim+self.choice_dim-1, self.conv_stim)
        else:
            # x_stim = torch.zeros(x.size(0), 1, device=x.device)        
            x_stim = torch.zeros(x.size(0), 2, device=x.device)
            if self.cross_terms:
                x_choicepred = torch.zeros(x.size(0), 2, device=x.device)
        
        if self.conv_choice:
            x_choice = self.cnn_forward(x, z, self.choice_idx, self.stim_dim, self.stim_dim+self.choice_dim-1, self.conv_choice)
            if self.cross_terms:
                x_stimpred = self.cnn_forward(x, z, 0, 0, self.stim_dim-1, self.conv_choice)
        else:
            # x_choice = torch.zeros(x.size(0), 1, device=x.device)
            x_choice = torch.zeros(x.size(0), 2, device=x.device)
            if self.cross_terms:
                x_stimpred = torch.zeros(x.size(0), 2, device=x.device)
        if self.cross_terms:
            return torch.cat([x_stim, x_choice, x_stimpred, x_choicepred], dim=1)
        else:
            return torch.cat([x_stim, x_choice], dim=1)        

    def loss(self, predicted, ground_truth, reduction='mean'):
        """
        Binary cross entropy loss
        predicted: (batch_size*num_samples, 4)
        ground_truth: (batch_size, 2)
        """
        batch_size = ground_truth.size(0)
        num_samples = predicted.size(0)//batch_size
        # repeat ground truth        
        ground_truth = torch.cat([ground_truth]*num_samples, dim=0)                
        # for bceloss
        # loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)
        # ground_truth = ground_truth.float()
        # print(ground_truth.dtype, predicted.dtype)
        loss_fn = nn.CrossEntropyLoss(reduction=reduction)
        loss = 0        
        if self.conv_stim:
            # loss += loss_fn(predicted[:, 0], ground_truth[:, 0]) * self.stimulus_weight
            loss += loss_fn(predicted[:, :2], ground_truth[:, 0]) * self.stimulus_weight
            if self.cross_terms:
                loss += self.stimulus_weight * (predicted[:, 6:8] ** 2).mean() / 2
                # loss += 0.25 * self.stimulus_weight * (loss_fn(predicted[:, 6:8], ground_truth[:, 1]) - LOG2)**2                
        if self.conv_choice:
            # loss += loss_fn(predicted[:, 1], ground_truth[:, 1]) * self.choice_weight
            loss += loss_fn(predicted[:, 2:4], ground_truth[:, 1]) * self.choice_weight
            if self.cross_terms:
                loss += self.choice_weight * (predicted[:, 4:6]**2).mean() / 2
                # loss += 0.25 * self.choice_weight * (loss_fn(predicted[:, 4:6], ground_truth[:, 0]) - LOG2)**2                
        return loss




# class CNNDecoderIndividual(nn.Module):
#     def __init__(self, config, xz_list):
#         super().__init__()
#         self.stim_dim, self.choice_dim = xz_list[0], xz_list[1]        
#         self.choice_idx = 0
#         if self.stim_dim > 0:
#             self.choice_idx += 1
#         assert self.stim_dim + self.choice_dim > 0, "Either stimulus or choice must be set"

#         channels = config['decoder']['cnn']['channels']
#         kernel_size = config['decoder']['cnn']['kernel_size']
#         pad = (kernel_size - 1)//2
#         dropout = config['decoder']['cnn']['dropout']

#         def make_1d_conv(inp_dim):
#             # 1d conv
#             layers = []            
#             for i in range(len(channels)):
#                 if i == 0:
#                     layers.append(nn.Conv1d(in_channels=inp_dim, out_channels=channels[i], kernel_size=kernel_size, padding=pad))
#                 else:
#                     layers.append(nn.Conv1d(in_channels=channels[i-1], out_channels=channels[i], kernel_size=kernel_size, padding=pad))
#                 # layers.append(nn.BatchNorm1d(channels[i]))
#                 # layers.append(nn.LeakyReLU())
#                 layers.append(nn.Tanh())
#                 if dropout > 0:
#                     layers.append(nn.Dropout(dropout))            
#             # linear layer
#             layers.append(nn.Conv1d(in_channels=channels[-1], out_channels=2, kernel_size=1))
#             return layers
        
#         if self.stim_dim > 0:
#             self.stimulus_weight = config['decoder']['stimulus_weight']
#             self.conv_stim = nn.Sequential(*make_1d_conv(self.stim_dim))
#             print("Using stimulus decoder")
#         else:
#             self.conv_stim = None

#         if self.choice_dim > 0:
#             self.choice_weight = config['decoder']['choice_weight']
#             self.conv_choice = nn.Sequential(*make_1d_conv(self.choice_dim))
#             print("Using choice decoder")
#         else:
#             self.conv_choice = None
#         # cross terms
#         self.cross_terms = config['decoder']['cross_terms']                                
#         # name
#         self.arch_name = 'cnn_{}_{}'.format('-'.join([str(x) for x in channels]), kernel_size)
#         # optimizer
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=config['decoder']['cnn']['lr'], weight_decay=config['decoder']['cnn']['weight_decay'])
#         # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 90, 120, 150, 180], gamma=0.5)
#         if config['decoder']['scheduler']['which'] == 'cosine':
#             self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=config['decoder']['scheduler']['cosine_restart_after'])
#             print('Using cosine annealing for decoder')
#         elif config['decoder']['scheduler']['which'] == 'decay':
#             self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config['decoder']['scheduler']['const_factor'])
#             print('Using decay annealing for decoder')
#         else:
#             print('Scheduler not implemented for decoder')
#             self.scheduler = None        

#     def forward(self, x, z):
#         # x is of shape (batch_size*num_samples, seq_len, input_dim)                
#         # z = z.detach()
#         # x = x * z
#         x = x.permute(0, 2, 1)
#         z = z.permute(0, 2, 1)
#         if self.conv_stim:
#             x_stim = x[:, :self.stim_dim, :]
#             x_stim = self.conv_stim(x_stim)
#             x_stim = x_stim * z[:, 0:1, :]
#             x_stim = torch.mean(x_stim, dim=2)
#             # x_stim = torch.max(x_stim, dim=2).values
#             if self.cross_terms:
#                 x_choicepred = self.conv_stim(x[:, self.stim_dim:self.stim_dim+self.choice_dim, :])
#                 x_choicepred = x_choicepred * z[:, self.choice_idx:self.choice_idx+1, :]
#                 x_choicepred = torch.mean(x_choicepred, dim=2)
#                 # x_choicepred = torch.max(x_choicepred, dim=2).values
#         else:
#             # x_stim = torch.zeros(x.size(0), 1, device=x.device)        
#             x_stim = torch.zeros(x.size(0), 2, device=x.device)
#             if self.cross_terms:
#                 x_choicepred = torch.zeros(x.size(0), 2, device=x.device)
        
#         if self.conv_choice:
#             x_choice = x[:, self.stim_dim:self.stim_dim+self.choice_dim, :]
#             x_choice = self.conv_choice(x_choice)            
#             x_choice = x_choice * z[:, self.choice_idx:self.choice_idx+1, :]
#             # print(x_choice.shape, z[:, self.choice_idx:self.choice_idx+1, :].shape)
#             x_choice = torch.mean(x_choice, dim=2)
#             # x_choice = torch.max(x_choice, dim=2).values            
#             if self.cross_terms:
#                 x_stimpred = self.conv_choice(x[:, :self.stim_dim, :])
#                 x_stimpred = x_stimpred * z[:, 0:1, :]
#                 x_stimpred = torch.mean(x_stimpred, dim=2)                
#                 # x_stimpred = torch.max(x_stimpred, dim=2).values
#         else:
#             # x_choice = torch.zeros(x.size(0), 1, device=x.device)
#             x_choice = torch.zeros(x.size(0), 2, device=x.device)
#             if self.cross_terms:
#                 x_stimpred = torch.zeros(x.size(0), 2, device=x.device)
#         if self.cross_terms:
#             return torch.cat([x_stim, x_choice, x_stimpred, x_choicepred], dim=1)
#         else:
#             return torch.cat([x_stim, x_choice], dim=1)        

#     def loss(self, predicted, ground_truth, reduction='mean'):
#         """
#         Binary cross entropy loss
#         predicted: (batch_size*num_samples, 4)
#         ground_truth: (batch_size, 2)
#         """
#         batch_size = ground_truth.size(0)
#         num_samples = predicted.size(0)//batch_size
#         # repeat ground truth        
#         ground_truth = torch.cat([ground_truth]*num_samples, dim=0)                
#         # loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)
#         loss_fn = nn.CrossEntropyLoss(reduction=reduction)
#         loss = 0        
#         if self.conv_stim:
#             # loss += loss_fn(predicted[:, 0], ground_truth[:, 0]) * self.stimulus_weight
#             loss += loss_fn(predicted[:, :2], ground_truth[:, 0]) * self.stimulus_weight
#             if self.cross_terms:
#                 loss += self.stimulus_weight * (predicted[:, 6:8] ** 2).mean()
#         if self.conv_choice:
#             # loss += loss_fn(predicted[:, 1], ground_truth[:, 1]) * self.choice_weight
#             loss += loss_fn(predicted[:, 2:4], ground_truth[:, 1]) * self.choice_weight
#             if self.cross_terms:
#                 loss += self.choice_weight * (predicted[:, 4:6]**2).mean()
#         return loss


class RNNDecoderIndivdual(nn.Module):
    def __init__(self, config, xz_list):
        super().__init__()
        self.stim_dim, self.choice_dim = xz_list[0], xz_list[1]        
        self.choice_idx = 0
        if self.stim_dim > 0:
            self.choice_idx += 1

        layers = config['decoder']['rnn']['layers']
        hidden_dim = config['decoder']['rnn']['hidden_dim']        
        dropout = config['decoder']['rnn']['dropout']

        rnn_l = nn.RNN(input_size=1, hidden_size=hidden_dim, num_layers=layers, dropout=dropout, batch_first=True)
        final_linear = nn.Linear(hidden_dim, 1)
        def make_module():
            return nn.ModuleList([rnn_l, final_linear])
        
        if self.stim_dim > 0:
            self.stimulus_weight = config['decoder']['stimulus_weight']
            self.rnn_stim = make_module()
            print("Using stimulus decoder")
        else:
            self.rnn_stim = None      
        if self.choice_dim > 0:
            self.choice_weight = config['decoder']['choice_weight']
            self.rnn_choice = make_module()
            print("Using choice decoder")
        else:
            self.rnn_choice = None                                
        # name
        self.arch_name = 'rnn_{}_{}'.format(layers, hidden_dim)
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
        if self.rnn_stim:
            x_stim = x[:, :, :self.stim_dim]
            x_stim, _ = self.rnn_stim[0](x_stim)
            x_stim = x_stim * z[:, :, 0:1]
            x_stim = self.rnn_stim[1](x_stim)
            x_stim = torch.max(x_stim, dim=2)            
        else:
            x_stim = torch.zeros(x.size(0), 1)
        
        if self.rnn_choice:
            x_choice = x[:, :, :self.choice_dim]            
            x_choice, _ = self.rnn_choice[0](x_choice)
            x_choice = x_choice * z[:, :, 0:1]
            x_choice = self.rnn_choice[1](x_choice)
            x_choice = torch.max(x_choice, dim=1)            
        else:
            x_choice = torch.zeros(x.size(0), 1)
                
        return torch.cat([x_stim, x_choice], dim=1)        

    def loss(self, predicted, ground_truth, z, reduction='mean'):
        # bce loss
        loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)
        loss = 0
        if self.rnn_stim:
            loss += loss_fn(predicted[:, 0], ground_truth[:, 0]) * self.stimulus_weight            
        if self.rnn_choice:
            loss += loss_fn(predicted[:, 1], ground_truth[:, 1]) * self.choice_weight        
        return loss


class LinearAccDecoder(nn.Module):
    def __init__(self, config, xz_list):
        super().__init__()
        # generate linear layers with hidden dims 
        self.stim_dim, self.choice_dim = xz_list[0], xz_list[1]        
        self.choice_idx = 0
        if self.stim_dim > 0:
            self.choice_idx += 1

        hidden_dims = config['decoder']['linear']['hidden_dims']
        dropout = config['decoder']['linear']['dropout']
        
        def make_mlp(input_dim):
            layers = []
            for i in range(len(hidden_dims)):
                if i == 0:
                    layers.append(nn.Linear(input_dim, hidden_dims[i]))
                else:
                    layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
                layers.append(nn.BatchNorm1d(hidden_dims[i]))
                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dims[-1], 2))
            return layers
        
        # name
        self.arch_name = 'linear_{}'.format(hidden_dims)                

        if self.stim_dim > 0:
            self.stimulus_weight = config['decoder']['stimulus_weight']
            self.mlp_stim = nn.Sequential(*make_mlp(self.stim_dim))
            print("Using stimulus decoder")
        else:
            self.mlp_stim = None      
        if self.choice_dim > 0:
            self.choice_weight = config['decoder']['choice_weight']
            self.mlp_choice = nn.Sequential(*make_mlp(self.choice_dim))
            print("Using choice decoder")
        else:
            self.mlp_choice = None                                
        
        # optimizer        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['decoder']['linear']['lr'], weight_decay=config['decoder']['linear']['weight_decay'])
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 90, 120, 150, 180], gamma=0.5)
        if config['decoder']['scheduler']['which'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=config['decoder']['scheduler']['cosine_restart_after'])
            print('Using cosine annealing for decoder')
        elif config['decoder']['scheduler']['which'] == 'decay':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config['decoder']['scheduler']['const_factor'])
            print('Using decay annealing for decoder')
        else:
            print('Scheduler not implemented for decoder')
            self.scheduler = None
    
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
    def __init__(self, config, xz_list):
        super().__init__()
        self.stimulus_weight = config['decoder']['stimulus_weight']
        self.choice_weight = config['decoder']['choice_weight']
        assert xz_list[0] > 0 and xz_list[1] > 0, "Stimulus and choice dimensions should be greater than 0"
        input_dim = xz_list[0] + xz_list[1]

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
                layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.LeakyReLU())
                # layers.append(nn.Tanh())
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
        # x = x[:, :2, :]
        x = self.conv(x).permute(0, 2, 1)
        x = self.fc(x)
        x = torch.max(x, dim=1).values
        # x = torch.mean(x, dim=2)
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