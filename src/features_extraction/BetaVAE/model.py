from src.args import args

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from src.features_extraction.base import Encoder, Decoder, Bottleneck, View, LossFunction
from src.utils import mnist_NxN_loader


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
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, 32 * 4 * 4)),  # B, 512
            nn.Linear(32 * 4 * 4, 256),  # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),  # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dim * 2),  # B, z_dim*2
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
            nn.Linear(256, 32 * 4 * 4),  # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),  # B,  32,  4,  4
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
        z = self.re_parametrize(mu, log_var)
        return z, mu, log_var

    def re_parametrize(self, mu, log_var):
        std = log_var.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps


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
        return x, z, mu, log_var


def kl_divergence(mu, log_var):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if log_var.data.ndimension() == 4:
        log_var = log_var.view(log_var.size(0), log_var.size(1))

    klds = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum').div(batch_size)
    elif distribution == 'gaussian':
        recon_loss = F.mse_loss(x_recon, x, reduction='sum').div(batch_size)
    else:
        recon_loss = None

    return recon_loss


class BetaVAELossFunction(LossFunction):
    def __init__(self):
        super(BetaVAELossFunction, self).__init__()
        self.C_max = args['BetaVAE']['C_max']
        self.GAMMA = args['BetaVAE']['Gamma']
        self.C_stop_iter = args['BetaVAE']['C_stop_iter']
        self.ITERATIONS = args['BetaVAE']['num_epochs']
        self.C_max = torch.FloatTensor([self.C_max])

    def __call__(self, x, x_recon, mu, log_var, distribution='gaussian'):
        C = torch.clamp(self.C_max / self.C_stop_iter * self.ITERATIONS, 0, self.C_max.data[0])

        recon_loss = reconstruction_loss(x, x_recon, distribution)
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, log_var)

        loss = recon_loss + self.GAMMA * (total_kld - C).abs()

        return loss


if __name__ == '__main__':
    # work with MNIST Dataset
    train_loader, test_loader = mnist_NxN_loader()

    print('num_batches_train:', len(train_loader))
    print('num_batches_test:', len(test_loader))
    print('x_batch_shape:', next(iter(train_loader))[0].shape)
    print('y_batch_shape:', next(iter(train_loader))[1].shape)

    batch_size = args['hyper_parameters']['batch_size']
    num_of_channels = args['hyper_parameters']['num_channels']
    input_size = (batch_size, num_of_channels) + args['hyper_parameters']['input_size']
    z_dim = args['hyper_parameters']['z_dim']

    # input
    x = next(iter(train_loader))[0]

    # test model
    model = BetaVAE(z_dim, num_of_channels, input_size)
    x_recon, z, mu, log_var = model(x)
    loss = BetaVAELossFunction()
    loss(x, x_recon, mu, log_var)
