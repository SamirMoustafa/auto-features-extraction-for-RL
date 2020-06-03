#!/usr/bin/env python
# coding: utf-8

import torch
import torch.utils.data
from torch import nn

from src.args import args
from src.features_extraction.base import Encoder, Decoder, View, LossFunction, Bottleneck
from src.test_modules import TestModelMethods
from src.utils import re_parameterize, reconstruction_loss, kl_divergence, get_fixed_hyper_param, get_device

test = TestModelMethods()


class BetaVAEEncoder(Encoder, nn.Module):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim, nc):
        super(BetaVAEEncoder, self).__init__()
        self.nc = nc
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32,  8,  8
            nn.ReLU(True),
            View((-1, 32 * 8 * 8)),  # B, 2048
            nn.Linear(32 * 8 * 8, 512),  # B, 512
            nn.ReLU(True),
            nn.Linear(512, 256),  # B, 256
            nn.ReLU(True),
            nn.Linear(256, 2 * z_dim),  # B, z_dim*2
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class BetaVAEDecoder(Decoder, nn.Module):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=64, nc=1, target_size=(128, 128)):
        super(BetaVAEDecoder, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.target_size = target_size

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),  # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),  # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32 * 8 * 8),  # B, 2048
            nn.ReLU(True),
            View((-1, 32, 8, 8)),  # B,  32,  8,  8
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B,  nc, 64, 64
            nn.Tanh(),
            View(self.target_size),
        )

    def forward(self, x):
        x_recon = self.decoder(x)
        return x_recon


class BetaVAEBottleneck(Bottleneck, nn.Module):
    def __init__(self, z_dim=10):
        super(BetaVAEBottleneck, self).__init__()

        self.z_dim = z_dim

    def forward(self, encoded):
        mu = encoded[:, :self.z_dim]
        log_var = encoded[:, self.z_dim:]
        z = re_parameterize(mu, log_var)
        return z, mu, log_var


class BetaVAE(nn.Module):
    def __init__(self, z_dim, nc, target_size):
        super(BetaVAE, self).__init__()

        self.encoder = BetaVAEEncoder(z_dim, nc)
        self.bottleneck = BetaVAEBottleneck(z_dim)
        self.decoder = BetaVAEDecoder(z_dim, nc, target_size)

    def forward(self, x):
        x = self.encoder(x)
        z, mu, log_var = self.bottleneck(x)
        x = self.decoder(z)
        return x, mu, log_var


class BetaVAELossFunction(LossFunction):
    def __init__(self):
        super(BetaVAELossFunction, self).__init__()
        self.ITERATIONS = args['hyper_parameters']['num_epochs']
        self.args = args['beta_vae']
        self.C_max = self.args['C_max']
        self.GAMMA = self.args['Gamma']
        self.C_stop_iter = self.args['C_stop_iter']
        self.C_max = torch.FloatTensor([self.C_max])

    def __call__(self, x, x_recon, mu, log_var, distribution='gaussian'):
        C = torch.clamp(self.C_max / self.C_stop_iter * self.ITERATIONS, 0, self.C_max.data[0])
        C = C.to(get_device())

        recon_loss = reconstruction_loss(x, x_recon, distribution)
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, log_var)
        loss = recon_loss + self.GAMMA * (total_kld - C).abs()

        return loss
