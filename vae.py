import torch
import torch.nn as nn
from decoder import LinearAccDecoder
from priors import moving_average
import math
import os

eps = 1e-6
# TODO: Make cholesky decomposition work

class VAE(nn.Module):
    def __init__(self, config, input_dim, xz_list, neuron_bias=None, init='vae_[1, 1, 1]_8_2_bi_standard'):
        super().__init__()               
        # keep only non-zero values in xz_list
        xz_list = [x for x in xz_list if x > 0]
        xz_ends = torch.cumsum(torch.tensor(xz_list), dim=0)
        xz_starts = torch.tensor([0] + xz_ends.tolist()[:-1])
        xz_l = torch.stack([xz_starts, xz_ends], dim=1)
        # register as a buffer
        self.register_buffer('xz_l', xz_l)
        
        self.x_dim = sum(xz_list)
        self.z_dim = len(xz_list)
        output_dim = (self.z_dim + self.x_dim)*(self.z_dim + self.x_dim + 1)
        # cholesky mask
        # self.cholesky_mask = torch.tril(torch.ones(self.z_dim+self.x_dim, self.z_dim+self.x_dim))
        # # set diagonal to 0
        # self.cholesky_mask = self.cholesky_mask - torch.diag_embed(torch.diagonal(self.cholesky_mask))
        # self.cholesky_mask = self.cholesky_mask.unsqueeze(0)

        hidden_dim, num_layers = config['rnn']['hidden_size'], config['rnn']['num_layers']
        bidirectional = config['rnn']['bidirectional']
        dropout = config['rnn']['dropout']

        # self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, hidden_dim))
        # self.posterior = nn.Linear(hidden_dim, output_dim)

        # self.encoder = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
        #                       bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        # self.posterior = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, output_dim)

        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                              bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        self.posterior = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, output_dim)
        
        # self.encoder = TransformerModel(input_dim, z_dim, x_dim, hidden_dim, 4, hidden_dim, num_layers, dropout)
        # self.posterior = nn.Linear(hidden_dim, output_dim)
        
        # reconstruction
        self.linear_maps = nn.ModuleList([nn.Linear(i, input_dim) for i in xz_list])        

        # reconstruction_layers = nn.Sequential(nn.Linear(1, 32), nn.Tanh(),
        #                                       nn.Linear(32, 32), nn.ReLU(),
        #                                       nn.Linear(32, input_dim))
        # self.linear_maps = nn.ModuleList([reconstruction_layers for _ in range(x_dim)])

        # gru_recon = nn.RNN(1, 16, num_layers=1, bidirectional=False, batch_first=True)
        # self.linear_maps = nn.ModuleList([gru_recon, nn.Linear(1, input_dim)])
        # # self.linear_maps = nn.ModuleList([nn.Linear(1, input_dim), gru_recon])
        # self.linear2 = nn.Linear(16, input_dim)

        # def ret(input_dim, output_dim):
        #     return nn.Sequential(nn.Linear(input_dim, 4), nn.Tanh(),
        #                          nn.Linear(4, 8), nn.Tanh(),
        #                          nn.Linear(8, output_dim))
        # self.linear_maps = nn.ModuleList([ret(1, input_dim) for _ in range(x_dim)])        
        
        # expand neuron bias in batch dimension        
        self.neuron_bias = neuron_bias.unsqueeze(0) if neuron_bias is not None else None
        # self.sigmoid_scaling_factor = nn.Parameter(torch.tensor(2.0), requires_grad=True)

        # softmax temperature
        self.softmax_temp = config['rnn']['softmax_temp']

        # name model
        self.arch_name = 'vae_{}_{}_{}'.format(xz_list, hidden_dim, num_layers)
        if bidirectional:
            self.arch_name += '_bi'
        if neuron_bias is not None:
            self.arch_name += '_bias'
        
        # moving average
        self.moving_average = None
        if self.moving_average is not None:
            self.arch_name += '_average_'+str(self.moving_average)
            print('Using moving average of', self.moving_average)
        else:
            print('Not using moving average')

        # # smoothing
        # self.smoothing = config['rnn']['smoothing']
        # if self.smoothing:            
        #     time_points = int(2.5/config['shape_dataset']['win_len'])
        #     x = torch.linspace(-2, 0.5, time_points)
        #     # construct x - x' for all pairs of x and x'
        #     y = x.view(-1, 1) - x.view(1, -1)
        #     # construct the kernel
        #     self.smoothing_kernel = torch.exp(-y.pow(2)/2*self.smoothing**2)
        
        # optmizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['rnn']['lr'], weight_decay=config['rnn']['weight_decay'])        

        # # init model        
        # if init is not None:
        #     try:
        #         data_des = 'dandi_{}/{}_ms'.format(config['shape_dataset']['id'], int(config['shape_dataset']['win_len']*1000))
        #         pth = os.path.join(config['dir']['results'], data_des, init, 'best')
        #         checkpoint = torch.load(pth, map_location=lambda storage, loc: storage)
        #         # replace encoder in keys with nothing
        #         checkpoint['model_state_dict'] = {k.replace('vae.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        #         self.load_state_dict(checkpoint['model_state_dict'])
        #         print("Loading from pre-trained")
        #     except:
        #         print("Failed to load pre-trained")

        assert self.neuron_bias is None and self.moving_average is None, "Not implemented"

        if config['rnn']['scheduler']['which'] == 'cosine':
            restart = config['rnn']['scheduler']['cosine_restart_after']
            scheduler1 = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1, total_iters=restart+restart//2)
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=restart)
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[scheduler1, scheduler2], milestones=[restart//2])
        elif config['rnn']['scheduler']['which'] == 'decay':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.97)
        else:
            print('Scheduler not implemented for GRU')
            self.scheduler = None

    def split(self, encoded):
        batch, seq, _ = encoded.shape
        mu = encoded[:, :, :self.z_dim+self.x_dim]
        A = encoded[:, :, self.z_dim+self.x_dim:].reshape(batch, seq, self.z_dim+self.x_dim, self.z_dim+self.x_dim)
        return mu, A
    
    def reparameterize(self, mu, A):
        # mu is of shape (batch, seq, z+x)
        batch, seq, mu_dim = mu.shape
        A_flat = A.view(batch*seq, mu_dim, mu_dim)
        mu_flat = mu.reshape(batch*seq, mu_dim).unsqueeze(-1)
        eps = torch.randn_like(mu_flat)
        # print(mu.shape, A.shape, eps.shape)
        sample = (mu_flat + torch.bmm(A_flat, eps)).squeeze(-1)        
        return sample.view(batch, seq, mu_dim)
    
    # def reparameterize(self, mu, A):
    #     # mu is of shape (batch, seq, z+x)
    #     batch, seq, mu_dim = mu.shape
    #     mu_flat = mu.reshape(batch*seq, mu_dim)
    #     A_flat = A.reshape(batch*seq, mu_dim, mu_dim)

    #     dist = torch.distributions.MultivariateNormal(mu_flat, scale_tril=A_flat)     
    #     # A_flat = torch.bmm(A_flat, A_flat.transpose(1, 2))        
    #     # dist = torch.distributions.MultivariateNormal(mu_flat, covariance_matrix=A_flat)     

    #     return dist.sample().reshape(batch, seq, mu_dim)

    def forward(self, y, n_samples):
        # y is of shape (batch_size, seq_len, input_dim)
        batch, seq, input_dim = y.shape
        encoded, _ = self.encoder(y)
        # encoded = self.encoder(y)
        # if isinstance(encoded, tuple):
        #     encoded = encoded[0]

        encoded = self.posterior(encoded)
        mu, A = self.split(encoded)
        

        # # cholesky: make only diagonal positive
        # # first reshape A to (batch*seq, z+x, z+x)
        # A = A.reshape(batch*seq, self.z_dim+self.x_dim, self.z_dim+self.x_dim)        
        # diag = torch.diagonal(A, dim1=-2, dim2=-1)
        # diag = nn.Softplus()(diag)        
        # A = A * self.cholesky_mask + torch.diag_embed(diag)        
        # # reshape it back
        # A = A.reshape(batch, seq, self.z_dim+self.x_dim, self.z_dim+self.x_dim)
                
        # # smooth means        
        # if self.moving_average is not None:
        #     # both
        #     mu = moving_average(mu, self.moving_average)
        #     # only z
        #     # mu[:, :, :self.z_dim] = moving_average(mu[:, :, :self.z_dim], self.moving_average)
        #     # only x
        #     # mu[:, :, self.z_dim:] = moving_average(mu[:, :, self.z_dim:], self.moving_average)

        # accumulate
        # mu = mu - mu[:, 0:1, :] # first is 0
        # mu = torch.cumsum(mu, dim=1)
        
        # sample z and x
        # sample_zx = self.reparameterize(mu, A)
        # sample z and x 10 times and concatenate along batch
        sample_zx = torch.cat([self.reparameterize(mu, A) for _ in range(n_samples)], dim=0)
        
        # extract x and z
        z, x = sample_zx[:, :, :self.z_dim], sample_zx[:, :, self.z_dim:]
        # z = torch.sigmoid(z*self.sigmoid_scaling_factor)
        # z = torch.sigmoid(z)
        z = torch.nn.Softmax(dim=-1)(z/self.softmax_temp)                
        # z = torch.nn.Tanh(dim=-1)(z/self.softmax_temp)
        # x = torch.nn.Tanh()(x)
        
        # map x to observation
        Cx_list = [self.linear_maps[i](x[:, :, s: e]) for i, (s, e) in enumerate(self.xz_l)]
        # if any element is a tuple, take the first element
        # Cx_list = [self.linear2(Cx[0]) if isinstance(Cx, tuple) else Cx for Cx in Cx_list]
        # print([x.shape for x in Cx_list])
        Cx = torch.stack(Cx_list, dim=-1)        
        y_recon = torch.sum(Cx * z.unsqueeze(2), dim=3)        

        # if self.neuron_bias is not None:
        #     y_recon = y_recon + self.neuron_bias
        y_recon = nn.Softplus()(y_recon)
        return y_recon, mu, A, z, x

    def loss(self, y, y_recon, mu, A, z):
        """
        y and y_recon are of shape [batch * n_samples, time, dim]
        mu and A are of shape [batch, time, z+x] and [batch, time, z+x, z+x]
        """
        batch, seq, _ = mu.shape
        num_samples = y_recon.shape[0] // batch
        # compute AAt
        flattened_A = A.reshape(batch*seq, self.z_dim+self.x_dim, self.z_dim+self.x_dim)        
        # flattened_A = torch.bmm(flattened_A, torch.transpose(flattened_A, 1, 2))        
        mu = mu.reshape(batch*seq, self.z_dim+self.x_dim)
        # print(cov.shape)
        # poisson loss
        # print(y.shape, y_recon.shape)
        # repeat ground truth        
        y = torch.cat([y]*num_samples, dim=0)
        recon_loss = torch.sum(y_recon - y * torch.log(y_recon))
        # print((torch.sum(mu.pow(2), dim=1) + torch.einsum("...ii", cov) - mu.shape[1] - torch.log(det+eps)).shape)
        
        # original KL loss
        cov = torch.bmm(flattened_A, torch.transpose(flattened_A, 1, 2))
        det = torch.det(cov)
        kl_loss = 0.5 * torch.sum(torch.sum(mu.pow(2), dim=1) + torch.einsum("...ii", cov) - mu.shape[1] - torch.log(det+eps))

        # # new KL loss
        # l, d = mu.shape[0], mu.shape[1]
        # mu1 = torch.zeros(l, d)
        # sigma1 = torch.eye(d, d).repeat(l, 1, 1)
        # p = torch.distributions.MultivariateNormal(mu1, scale_tril=sigma1)
        # q = torch.distributions.MultivariateNormal(mu, scale_tril=flattened_A)        
        # # compute the kl divergence
        # kl_loss = torch.distributions.kl_divergence(p, q).sum()
        # kl_loss = 0        
        
        # print(flattened_A[0])
        return (recon_loss + kl_loss)/(batch*num_samples)
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class TransformerModel(nn.Module):

    def __init__(self, input_dim: int, z_dim, x_dim, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)        
        self.embedding = nn.Linear(input_dim, d_model)
        self.d_model = d_model                        

    def forward(self, src):
        """
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len, dim]``            

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """        
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = src.permute(1, 0, 2)        
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)        
        return output.permute(1, 0, 2)