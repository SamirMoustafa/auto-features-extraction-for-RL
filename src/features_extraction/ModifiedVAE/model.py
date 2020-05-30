from src.args import args

import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from src.features_extraction.base import Encoder, Decoder, Bottleneck, View, LossFunction


# Source: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
class ModifiedVAEEncoder(Encoder, nn.Module):
    def __init__(self, latent_dim=10, in_channels=1):
        super(ModifiedVAEEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            View((-1, 512)),
            nn.Linear(512, self.latent_dim)
        )

    def forward(self, x):
        return self.encoder(x), nn.Linear(2048, self.latent_dim), nn.Linear(2048, self.latent_dim)


class ModifiedVAEDecoder(Decoder, nn.Module):
    def __init__(self, latent_dim=10, in_channels=1):
        super(ModifiedVAEDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            View((-1, 512, 1, 1)),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, out_channels=self.in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)


class ModifiedVAELossFunction(LossFunction):
    def __call__(self, x, x_recon, batch_size=100):
        return F.mse_loss(x_recon, x, reduction='sum').div(batch_size)


if __name__ == '__main__':

    batch_size = args['input']['batch_size']
    num_of_channels = args['input']['num_channels']
    latent_dim = args['ModifiedVAE']['Z_dim']
    input_size = (batch_size, num_of_channels) + args['input']['input_size']

    x_input = torch.rand(input_size)

    encoder, fc_mu, fc_var = ModifiedVAEEncoder(latent_dim=latent_dim, in_channels=num_of_channels)

    result = torch.flatten(encoder, start_dim=1)
    mu = fc_mu(result)
    log_var = fc_var(result)

    decoder = ModifiedVAEDecoder(latent_dim=latent_dim, in_channels=num_of_channels)
    loss_function = ModifiedVAELossFunction()

    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    x = decoder(eps * std + mu)

    print('input_shape:', x_input.shape)
    print('output_shape:', x.shape)
    print('loss function value:', loss_function(x_input, x))
