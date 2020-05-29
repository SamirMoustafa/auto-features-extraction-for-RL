from src.args import args

import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from src.features_extraction.base import Encoder, Decoder, Bottleneck, View, LossFunction


class ModifiedVAEEncoder(Encoder, nn.Module):
    def __init__(self, z_dim=10, nc=1):
        super(ModifiedVAEEncoder, self).__init__()
        self.nc = nc
        self.z_dim = z_dim

        self.lin1 = nn.Linear(self.nc, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.rl1 = nn.ReLU(True)
        self.lin2 = nn.Linear(256, self.z_dim)
        self.lin3 = nn.Linear(256, self.z_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = self.bn1(x)
        out = self.rl1(x)
        return self.lin2(out), self.lin3(out)


class ModifiedVAEDecoder(Decoder, nn.Module):
    def __init__(self, z_dim=10, nc=1, target_size=(64, 64)):
        super(ModifiedVAEDecoder, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.target_size = target_size

        self.lin4 = nn.Linear(self.z_dim, 256)
        self.lin5 = nn.Linear(256, self.nc)
        self.lin6 = nn.Linear(256, self.nc)
        self.rl2 = nn.ReLU(True)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.lin4(x)
        x = self.bn2(x)
        out = self.rl2(x)
        return self.lin5(out), self.lin6(out)


class ModifiedVAELossFunction(LossFunction):
    def __call__(self, x, x_recon, distribution):
        batch_size = x.size(0)
        assert batch_size != 0

        if distribution == 'bernoulli':
            recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum').div(batch_size)
        elif distribution == 'gaussian':
            recon_loss = F.mse_loss(x_recon, x, reduction='sum').div(batch_size)
        else:
            recon_loss = None

        return recon_loss


def gaussian_sampler(mu, logsigma):
    std = logsigma.exp()
    eps = std.data.new(std.size()).normal_()
    return eps.mul(std).add(mu)


def KL_divergence(mu, logsigma):
    return -(0.5 + logsigma - 0.5 * mu**2 - 0.5 * torch.exp(2 * logsigma)).sum(dim=1).mean()


def log_likelihood(x, mu, logsigma):
    return (-logsigma - 0.5 * np.log(2 * np.pi) - 0.5 * (mu - x)**2 / torch.exp(2 * logsigma)).sum(dim=1).mean()


def loss_vae(x, mu_gen, logsigma_gen, mu_z, logsigma_z):
    return KL_divergence(mu_z, logsigma_z) - log_likelihood(x, mu_gen, logsigma_gen)


if __name__ == '__main__':

    batch_size = args['input']['batch_size']
    num_of_channels = args['input']['num_channels']
    z_dim = args['ModifiedVAE']['Z_dim']
    input_size = (batch_size, num_of_channels) + args['input']['input_size']

    x_input = torch.rand(input_size)

    encoder = ModifiedVAEEncoder(z_dim=z_dim, nc=num_of_channels)
    decoder = ModifiedVAEDecoder(z_dim=z_dim, nc=num_of_channels, target_size=input_size)
    loss_function = ModifiedVAELossFunction()

    latent_mu, latent_logsigma = encoder(x_input)

    reconstruction_mu, reconstruction_logsigma = decoder(gaussian_sampler(latent_mu, latent_logsigma))

    reconstruction_mu, reconstruction_logsigma = decoder(x)

    print('input_shape:', x_input.shape)
    print('output_shape:', x.shape)
    print('loss function value:', loss_function(x_input, x, distribution='gaussian'))
