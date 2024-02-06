import torch
import torch.nn as nn
from decoder import LinearAccDecoder

eps = 1e-6

class VAE(nn.Module):
    def __init__(self, input_dim, z_dim, x_dim, hidden_dim, num_layers, dropout, bidirectional, neuron_bias=None):
        super().__init__()
        self.z_dim, self.x_dim = z_dim, x_dim        
        assert x_dim == z_dim
        output_dim = (z_dim + x_dim)*(z_dim + x_dim + 1)

        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                              bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        self.posterior = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, output_dim)
        self.linear_maps = nn.ModuleList([nn.Linear(1, input_dim) for _ in range(x_dim)])
        # expand neuron bias in batch dimension        
        self.neuron_bias = neuron_bias.unsqueeze(0) if neuron_bias is not None else None
        # self.sigmoid_scaling_factor = nn.Parameter(torch.tensor(2.0), requires_grad=True)

        # name model
        self.arch_name = 'vae_{}_{}'.format(hidden_dim, num_layers)
        if bidirectional:
            self.arch_name += '_bi'
        if neuron_bias is not None:
            self.arch_name += '_bias'        

    def split(self, encoded):
        batch, seq, _ = encoded.shape
        mu = encoded[:, :, :self.z_dim+self.x_dim]
        A = encoded[:, :, self.z_dim+self.x_dim:].reshape(batch, seq, self.z_dim+self.x_dim, self.z_dim+self.x_dim)
        return mu, A
    
    def reparameterize(self, mu, A):
        # mu is of shape (batch, seq, z+x)
        batch, seq, mu_dim = mu.shape
        A_flat = A.view(batch*seq, mu_dim, mu_dim)
        mu_flat = mu.view(batch*seq, mu_dim, 1)
        eps = torch.randn_like(mu_flat)
        # print(mu.shape, A.shape, eps.shape)
        sample = (mu_flat + torch.bmm(A_flat, eps)).squeeze(-1)        
        return sample.view(batch, seq, mu_dim)

    def forward(self, y):
        # y is of shape (batch_size, seq_len, input_dim)
        # batch, seq, input_dim = y.shape
        encoded, _ = self.encoder(y)
        encoded = self.posterior(encoded)
        mu, A = self.split(encoded)
        # sample z and x
        sample_zx = self.reparameterize(mu, A)
        # extract x and z
        z, x = sample_zx[:, :, :self.z_dim], sample_zx[:, :, self.z_dim:]
        # z = torch.sigmoid(z*self.sigmoid_scaling_factor)
        # z = torch.sigmoid(z)
        z = torch.softmax(z, dim=-1)
        # map x to observation        
        Cx = torch.stack([self.linear_maps[i](x[:, :, i:i+1]) for i in range(self.z_dim)], dim=-1)        
        # element wise multiplication of Cx with z
        # print(Cx.shape, z.unsqueeze(3).shape)
        y_recon = torch.sum(Cx*z.unsqueeze(2), dim=3)        
        if self.neuron_bias is not None:
            y = y + self.neuron_bias
        y_recon = nn.Softplus()(y)
        return y_recon, (mu, A), (z, x)

    # def sample(self, num_samples):
    #     z = torch.randn(num_samples, latent_dim)
    #     x_recon = self.decoder(z)
    #     return x_recon

    def loss(self, y, y_recon, mu, A):
        batch, seq = y.shape[0], y.shape[1]        
        # compute AAt
        flattened_A = A.view(batch*seq, self.z_dim+self.x_dim, self.z_dim+self.x_dim)
        cov = torch.bmm(flattened_A, torch.transpose(flattened_A, 1, 2))        
        mu = mu.reshape(batch*seq, self.z_dim+self.x_dim)
        # print(cov.shape)
        # poisson loss
        # print(y.shape, y_recon.shape)
        recon_loss = torch.sum(y_recon - y * torch.log(y_recon))
        det = torch.det(cov)
        kl_loss = 0.5 * torch.sum(torch.sum(mu.pow(2), dim=1) + torch.einsum("...ii", cov) - mu.shape[1] - torch.log(det+eps))
        return (recon_loss + kl_loss)/batch

    # def generate(self, num_samples):
    #     return self.sample(num_samples).detach().numpy()
    