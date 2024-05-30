import torch.nn as nn


def make_1d_conv(inp_dim, channels, kernel_size, pad, dropout):
    # 1d conv
    layers = []            
    for i in range(len(channels)):
        if i == 0:
            layers.append(nn.Conv1d(in_channels=inp_dim, out_channels=channels[i], kernel_size=kernel_size, padding=pad))
        else:
            layers.append(nn.Conv1d(in_channels=channels[i-1], out_channels=channels[i], kernel_size=kernel_size, padding=pad))
        # layers.append(nn.BatchNorm1d(channels[i]))
        layers.append(nn.LeakyReLU())
        # layers.append(nn.Tanh())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))            
    # linear layer
    layers.append(nn.Conv1d(in_channels=channels[-1], out_channels=1, kernel_size=1))
    return layers