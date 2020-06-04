#!/usr/bin/env python
# coding: utf-8


import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data

from src.args import args
from src.test_modules import TestModelMethods
from src.utils import get_fixed_hyper_param, reconstruction_loss
from src.features_extraction.base import Encoder, Decoder, View, LossFunction

test = TestModelMethods()


class WassersteinAEncoder(Encoder, nn.Module):
    def __init__(self, z_dim, nc):
        super(WassersteinAEncoder, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),         # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),         # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),         # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),         # B,  32,  8,  8
            nn.ReLU(True),
            View((-1, 32 * 8 * 8)),             # B, 2048
            nn.Linear(32 * 8 * 8, 512),         # B, 512
            nn.ReLU(True),
            nn.Linear(512, 256),                # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dim),          # B, z_dim*2
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class WassersteinADecoder(Decoder, nn.Module):
    def __init__(self, z_dim, nc, target_size):
        super(WassersteinADecoder, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.target_size = target_size

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                    # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32 * 8 * 8),             # B, 2048
            nn.ReLU(True),
            View((-1, 32, 8, 8)),                   # B,  32,  8,  8
            nn.ConvTranspose2d(32, 32, 4, 2, 1),    # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),    # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),    # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),    # B,  nc, 64, 64
            nn.Tanh(),
            View(self.target_size),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class WassersteinAE(nn.Module):
    def __init__(self, z_dim, nc, target_size):
        super(WassersteinAE, self).__init__()
        self.encoder = WassersteinAEncoder(z_dim, nc)
        self.decoder = WassersteinADecoder(z_dim, nc, target_size)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z


# loss aux functions
def calc_kernel(x_1, x_2, eps=1e-7, z_var=2.):
    x_1 = x_1.unsqueeze(-2)  # Make it into a column tensor
    x_2 = x_2.unsqueeze(-3)  # Make it into a row tensor

    z_dim = x_2.size(-1)
    C = 2 * z_dim * z_var
    kernel = C / (eps + C + (x_1 - x_2).pow(2).sum(dim=-1))

    # Exclude diagonal elements
    result = kernel.sum() - kernel.diag().sum()
    return result


def calc_mmd(z, reg_weight):
    prior_z = torch.rand_like(z)
    prior_z_kernel = calc_kernel(prior_z, prior_z)
    z_kernel = calc_kernel(z, z)
    prior_z_kernel = calc_kernel(prior_z, z)
    mmd = reg_weight * prior_z_kernel.mean() + reg_weight * z_kernel.mean() - 2 * reg_weight * prior_z_kernel.mean()
    return mmd


class WassersteinAELossFunction(LossFunction):
    def __init__(self, reg_weight):
        super().__init__()
        self.reg_weight = reg_weight

    def __call__(self, x, x_recon, z):
        batch_size = x.shape[0]
        bias_corr = batch_size * (batch_size - 1)
        self.reg_weight /= bias_corr

        recon_loss = reconstruction_loss(x_recon, x)
        mmd_loss = calc_mmd(z, self.reg_weight)

        loss = recon_loss + mmd_loss

        return loss


if __name__ == '__main__':

    batch_size, num_of_channels, input_size, z_dim = get_fixed_hyper_param(args['hyper_parameters'])
    reg_weight = args['wasserstein_ae']['reg_weight']

    # test model
    model = WassersteinAE(z_dim, num_of_channels, input_size)
    loss = WassersteinAELossFunction(reg_weight)
    test.test_model(model, loss)