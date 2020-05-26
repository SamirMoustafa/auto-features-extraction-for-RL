from src.args import args

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from src.features_extraction.base import Encoder, Decoder, Bottleneck, View, LossFunction


class BetaVAEEncoder(Encoder, nn.Module):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=10, nc=1):
        super(BetaVAEEncoder, self).__init__()
        self.nc = nc
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(self.nc, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32,  8,  8
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32,  4,  4
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            View((-1, 32 * 4 * 4)),  # B, 512
            nn.Linear(32 * 4 * 4, 256),  # B, 256
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 256),  # B, 256
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, z_dim * 2),  # B, z_dim*2
            nn.BatchNorm1d(z_dim * 2),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class BetaVAEDecoder(Decoder, nn.Module):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=10, nc=1, target_size=(64, 64)):
        super(BetaVAEDecoder, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.target_size = target_size

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),  # B, 256
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 256),  # B, 256
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 32 * 4 * 4),  # B, 512
            nn.BatchNorm1d(32 * 4 * 4),
            nn.ReLU(True),
            View((-1, 32, 4, 4)),  # B,  32,  4,  4
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32,  8,  8
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B,  nc, 64, 64
            nn.BatchNorm2d(nc),
            nn.Tanh(),
        )

    def forward(self, x):
        x_recon = self.decoder(x).view(self.target_size)
        return x_recon


class BetaVAEBottleneck(Bottleneck, nn.Module):
    def __init__(self, z_dim=10):
        super(BetaVAEBottleneck, self).__init__()

        self.z_dim = z_dim

    def forward(self, encoded):
        mu = encoded[:, :self.z_dim]
        log_var = encoded[:, self.z_dim:]
        z = self.re_parametrize(mu, log_var)
        return z

    def re_parametrize(self, mu, log_var):
        std = log_var.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps


class BetaVAELossFunction(LossFunction):
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


if __name__ == '__main__':

    batch_size = args['input']['batch_size']
    num_of_channels = args['input']['num_channels']
    z_dim = args['BetaVAE']['Z_dim']
    input_size = (batch_size, num_of_channels) + args['input']['input_size']

    x_input = torch.rand(input_size)

    encoder = BetaVAEEncoder(z_dim=z_dim, nc=num_of_channels)
    bottleneck = BetaVAEBottleneck(z_dim=z_dim)
    decoder = BetaVAEDecoder(z_dim=z_dim, nc=num_of_channels, target_size=input_size)
    loss_function = BetaVAELossFunction()

    x = encoder(x_input)
    x = bottleneck(x)
    x = decoder(x)

    print('input_shape:', x_input.shape)
    print('output_shape:', x.shape)
    print('loss function value:', loss_function(x_input, x, distribution='gaussian'))
