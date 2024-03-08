import torch
import math
import matplotlib.pyplot as plt

PI = torch.tensor(math.pi)

def moving_average(x, window):
    """
    Compute the moving average of a 1D tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, time, dim).
        window (int): Window size for the moving average.

    Returns:
        torch.Tensor: Moving average of the input tensor.
    """
    # return torch.nn.functional.conv1d(x.view(1, 1, -1), weight=torch.ones(1, 1, window) / window, padding=window-1).view(-1)
    batch, time, dim = x.shape
    x = x.permute(0, 2, 1)
    # apply moving average to each dimension
    outs = []
    kernel = torch.ones(1, 1, window) / window
    for i in range(dim):
        outs.append(torch.nn.functional.conv1d(x[:, i:i+1, :], weight=kernel, padding='same'))
    stacked = torch.cat(outs, dim=1).permute(0, 2, 1)
    return stacked


class GaussianPrior(torch.nn.Module):
    """
    Gaussian prior for the latent variables of a VAE.
    """
    def __init__(self, init_mean, init_variance, wt, step_size, learn_mean_std, prior_on_mean):
        """
        Args:
            mean (torch.Tensor): Mean of the Gaussian prior.
            variance (torch.Tensor): Variance of the Gaussian prior.
            scale (float): Scale of the Gaussian prior.
        """
        super().__init__()

        
        self.mean = torch.nn.Parameter(torch.tensor(init_mean, dtype=torch.float32), requires_grad=learn_mean_std)
        self.log_std = torch.nn.Parameter(torch.log(torch.tensor(init_variance, dtype=torch.float32)), requires_grad=learn_mean_std)
        self.wt = wt
        self.learn_mean_std = learn_mean_std
        # self.log_scale = torch.nn.Parameter(torch.tensor(scale, dtype=torch.float32), requires_grad=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)        
        
        # precompouted gaussian values
        t = torch.arange(-2, 0.5, step=step_size)        
        var = torch.exp(self.log_std)**2
        self.gaussian = torch.exp(-0.5 * (t - self.mean) ** 2 / var) / (torch.sqrt(2 * PI * var))
        self.gaussian = self.gaussian.view(1, -1)
        # whether to apply prior on mean or not
        self.prior_on_mean = prior_on_mean

    
    def loss_(self, zt):
        """
        Assume the samples come from a gaussian distribution

        Args:
            zt (torch.Tensor): 2D tensor of shape (batch, time).
        
        Returns:
            torch.Tensor: Correlation of zt with the gaussian pulse.
        """
        # find mean and std of zt        
        t = torch.arange(zt.shape[1], dtype=torch.float32).view(1, -1)
        zt_mean = torch.mean(zt*t, dim=1)
        # clip
        zt_mean = torch.clip(zt_mean, -2, 0.5)
        # print(t, zt[0], zt_mean[0])
        zt_std = torch.std(zt, dim=1)
        # compute kl divergence
        kl_t1 = torch.log(zt_std) - self.log_std
        kl_t2 = (torch.exp(self.log_std) ** 2 + (zt_mean - self.mean) ** 2) / (2 * zt_std ** 2) - 0.5        
        return torch.mean(kl_t1 + kl_t2) * self.wt
    
    def loss(self, zt, mean_zt):
        """
        Compute mse of 1d tensor zt with a gaussian pulse

        Args:
            zt (torch.Tensor): 2D tensor of shape (batch, time).
        
        Returns:
            torch.Tensor: Correlation of zt with the gaussian pulse.
        """
        if self.learn_mean_std:
            # values of gaussian
            # m = torch.clip(self.mean, -2, 0.5)
            m = self.mean
            # m = self.mean
            step = 2.5/zt.shape[1]
            t = torch.arange(-2, 0.5, step=step)
            # zt = torch.exp(self.log_scale) * zt
            # compute values of gaussian at t using mean, log_variance, and log_scale
            var = torch.exp(self.log_std)**2
            gaussian = torch.exp(-0.5 * (t - m) ** 2 / var) / (torch.sqrt(2 * PI * var))        
        else:
            gaussian = self.gaussian
        if self.prior_on_mean:
            inp = mean_zt
        else:
            inp = zt
        return (inp - gaussian).pow(2).mean() * self.wt
    
    def plot_gaussian(self):
        """
        Plot the gaussian prior.
        """
        step = 0.1
        t = torch.arange(-2, 0.5, step=step)
        var = torch.exp(self.log_std)**2
        gaussian = torch.exp(-0.5 * (t - self.mean) ** 2 / var) / (torch.sqrt(2 * PI * var))
        plt.plot(t.numpy(), gaussian.detach().numpy())
        plt.title("Gaussian Prior with mean: {:.2f}, std: {:.2f}".format(self.mean.item(), torch.exp(self.log_std).item()))
        plt.show()
                
    

if __name__ == '__main__':        
    # test GaussianPrior
    # prior = GaussianPrior(0, 0.2, 1)
    # zt = torch.tensor([[-1, 0, 0.2, 0.5, 0.2, 0, 0], [-1, 0, -0.2, -0.5, -0.2, 0, 0]], dtype=torch.float32)    
    # print(prior.loss(zt))
    # prior.plot_gaussian()
    mean, log_std = -0.3, torch.log(torch.tensor(0.1))


    step = 0.1
    t = torch.arange(-2, 0.5, step=step)
    var = torch.exp(log_std)**2
    gaussian = torch.exp(-0.5 * (t - mean) ** 2 / var) / (torch.sqrt(2 * PI * var))
    plt.plot(t.numpy(), gaussian.detach().numpy())
    plt.title("Gaussian Prior with mean: {:.2f}, std: {:.2f}".format(mean, torch.exp(log_std)))
    plt.show()