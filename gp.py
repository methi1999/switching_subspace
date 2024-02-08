import torch



def gp_ll(x, length_scale=0.1, noise=1e-6):
    """
    Compute the log likelihood of a Gaussian process with zero mean and squared exponential (RBF) kernel.

    Args:
        x (torch.Tensor): Input data of shape (time, dimensions).
        length_scale (float): Length scale hyperparameter for the RBF kernel.
        noise (float): Noise term added to the diagonal of the kernel matrix for numerical stability.

    Returns:
        torch.Tensor: Log likelihood of the Gaussian process.
    """    
    print(x.shape)
    batch, time, dimensions = x.shape

    # Construct the RBF kernel matrix
    x_sq = torch.sum(x**2, dim=1, keepdim=True)
    print(x_sq.shape, torch.bmm(x, x.permute(0, 2, 1)).shape, x_sq.transpose(1, 2).shape)
    distance_matrix = x_sq - 2 * torch.bmm(x, x.permute(0, 2, 1)) + x_sq.transpose(1, 2)
    kernel_matrix = torch.exp(-0.5 * distance_matrix / length_scale**2)

    # Create a copy of kernel_matrix to avoid in-place addition
    kernel_matrix = kernel_matrix + noise * torch.eye(time)

    # Compute the log determinant of the kernel matrix using Cholesky decomposition
    cholesky_factor = torch.cholesky(kernel_matrix, upper=False)
    log_det = 2.0 * torch.sum(torch.log(torch.diagonal(cholesky_factor)))

    # Compute the quadratic term in the log likelihood
    inverse_kernel_matrix = torch.inverse(kernel_matrix)
    data_term = torch.bmm(inverse_kernel_matrix, x).matmul(x.transpose(1, 2))

    # Compute the final log likelihood
    log_likelihood = -0.5 * (log_det + data_term + time * dimensions * torch.log(2 * torch.tensor(torch.pi)))        
    return torch.sum(log_likelihood)