#!/usr/bin/env python
# coding: utf-8


from torch import optim

from src.args import args
from src.utils import get_fixed_hyper_param, get_device, game_data_loaders, load_model

from src.features_extraction import VarianceConstrainedAE, NormalizingFlowModel, VarianceConstrainedAELossFunction, \
    NormalizingLossFunction, train_vcae

device = get_device()
batch_size, num_of_channels, input_size, z_dim = get_fixed_hyper_param(args['hyper_parameters'])

DO_TRAIN = True

vcae = VarianceConstrainedAE(z_dim=z_dim, nc=num_of_channels, target_size=input_size).to(device)
nf = NormalizingFlowModel(z_dim).to(device)

loss_VCAE = VarianceConstrainedAELossFunction()
loss_NF = NormalizingLossFunction()

optimizer_vcae = optim.Adam(vcae.parameters(), lr=1e-3)
optimizer_nf = optim.Adam(nf.parameters(), lr=1e-3)

dataloaders = game_data_loaders()
train_loaders, val_loaders = dataloaders['train'], dataloaders['val']

if DO_TRAIN:
    num_epochs = int(3e3)
    train_vcae(num_epochs, vcae, train_loaders, val_loaders, optimizer_vcae, device, loss_VCAE)
    train_vcae(num_epochs, vcae, train_loaders, val_loaders, optimizer_nf, device, loss_NF, is_nf=True, nf=nf)
