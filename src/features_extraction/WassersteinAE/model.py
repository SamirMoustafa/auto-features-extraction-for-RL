#!/usr/bin/env python
# coding: utf-8


import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import transforms

from src.features_extraction.base import Encoder, Decoder, Bottleneck, View, LossFunction

# work with MNIST Dataset
transform = transforms.Compose([transforms.Pad(2, fill=0, padding_mode='constant'), transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=True, transform=transform, download=True)

test_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=4)

print('num_batches_train:', len(train_loader))
print('num_batches_test:', len(test_loader))
print('x_batch_shape:', next(iter(train_loader))[0].shape)
print('y_batch_shape:', next(iter(train_loader))[1].shape)


class WassersteinAEncoder(Encoder, nn.Module):
    def __init__(self, latent_dim, in_channels):
        super(WassersteinAEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.encoder = nn.Sequential(nn.Conv2d(self.in_channels, 32, 3, 2, 1),
                                     nn.BatchNorm2d(32),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(32, 64, 3, 2, 1),
                                     nn.BatchNorm2d(64),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(64, 128, 3, 2, 1),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(128, 256, 3, 2, 1),
                                     nn.BatchNorm2d(256),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(256, 512, 3, 2, 1),
                                     nn.BatchNorm2d(512),
                                     nn.LeakyReLU(),
                                     View((-1, 512)),
                                     nn.Linear(512, self.latent_dim))

    def forward(self, x):
        x = self.encoder(x)
        return x


class WassersteinADecoder(Decoder, nn.Module):
    def __init__(self, latent_dim, in_channels):
        super(WassersteinADecoder, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.decoder = nn.Sequential(nn.Linear(self.latent_dim, 512),
                                     View((-1, 512, 1, 1)),
                                     nn.ConvTranspose2d(512, 256, 3, 2, 1, 1),
                                     nn.BatchNorm2d(256),
                                     nn.LeakyReLU(),
                                     nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU(),
                                     nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
                                     nn.BatchNorm2d(64),
                                     nn.LeakyReLU(),
                                     nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                     nn.BatchNorm2d(32),
                                     nn.LeakyReLU(),
                                     nn.ConvTranspose2d(32, 32, 3, 2, 1, 1),
                                     nn.BatchNorm2d(32),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(32, self.in_channels, 3, padding=1),
                                     nn.Tanh())

    def forward(self, x):
        x = self.decoder(x)
        return x


# In[133]:


class WassersteinAE(nn.Module):
    def __init__(self):
        super(WassersteinAE, self).__init__()
        self.encoder = WassersteinAEncoder(12, 1)
        self.decoder = WassersteinADecoder(12, 1)

    def forward(self, x):
        z = self.encoder(x)
        return [self.decoder(z), x, z]


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
    priorz_z_kernel = calc_kernel(prior_z, z)
    mmd = reg_weight * prior_z_kernel.mean() + reg_weight * z_kernel.mean() - 2 * reg_weight * priorz_z_kernel.mean()
    return mmd


# In[135]:


class WassersteinLossFunction(LossFunction):
    def __call__(self, x, x_recon, z, reg_weight):
        batch_size = x.size(0)
        bias_corr = batch_size * (batch_size - 1)
        reg_weight /= bias_corr

        recon_loss = F.mse_loss(x_recon, x)
        mmd_loss = calc_mmd(z, reg_weight)

        loss = recon_loss + mmd_loss

        return loss


if __name__ == '__main__':
    model = WassersteinAE()
    x_recon, x, z = model(next(iter(train_loader))[0])
    loss = WassersteinLossFunction()
    loss(x, x_recon, z, 100)
