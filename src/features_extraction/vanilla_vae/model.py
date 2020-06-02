import torch
from torch import nn
import torch.utils.data

from src.args import args
from src.test_modules import TestModelMethods
from src.features_extraction.base import Encoder, Decoder, View, LossFunction, Bottleneck
from src.utils import reconstruction_loss, kl_divergence, reparameterize, get_fixed_hyper_param

test = TestModelMethods()


# Source: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
class VanillaVAEEncoder(Encoder, nn.Module):
    def __init__(self, z_dim=10, nc=1):
        super(VanillaVAEEncoder, self).__init__()

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
        return self.encoder(x)


class VanillaVAEDecoder(Decoder, nn.Module):
    def __init__(self, z_dim=10, nc=1, target_size=(128, 128)):
        super(VanillaVAEDecoder, self).__init__()
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
        return self.decoder(x)


class VanillaVAEBottleneck(Bottleneck, nn.Module):
    def __init__(self, latent_dim):
        super(VanillaVAEBottleneck, self).__init__()
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.var = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        mu = self.mu(x)
        log_var = self.var(x)
        z = reparameterize(mu, log_var)
        return z, mu, log_var


class VanillaVAE(nn.Module):
    def __init__(self, z_dim, nc, target_size):
        super(VanillaVAE, self).__init__()

        self.encoder = VanillaVAEEncoder(z_dim, nc)
        self.bottleneck = VanillaVAEBottleneck(z_dim)
        self.decoder = VanillaVAEDecoder(z_dim, nc, target_size)

    def forward(self, x):
        x = self.encoder(x)
        z, mu, log_var = self.bottleneck(x)
        x = self.decoder(z)
        return x, mu, log_var


class VanillaVAELossFunction(LossFunction):
    def __call__(self, x, x_recon, mu, log_var):
        recons_loss = reconstruction_loss(x_recon, x)
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, log_var)
        return recons_loss + total_kld * args['vanilla_vae']['M_N']


if __name__ == '__main__':
    batch_size, num_of_channels, input_size, z_dim = get_fixed_hyper_param(args['hyper_parameters'])

    model = VanillaVAE(z_dim, num_of_channels, input_size)
    loss = VanillaVAELossFunction()

    test.test_model(model, loss)
