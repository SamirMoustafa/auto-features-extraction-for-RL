#!/usr/bin/env python
# coding: utf-8


from torch import optim

from src.args import args
from src.features_extraction import WassersteinAE, WassersteinAELossFunction
from src.utils import get_fixed_hyper_param, train_ae, get_device, game_data_loaders

device = get_device()
batch_size, num_of_channels, input_size, z_dim = get_fixed_hyper_param(args['hyper_parameters'])
reg_weight = args['wasserstein_ae']['reg_weight']

DO_TRAIN = True

model = WassersteinAE(z_dim, num_of_channels, input_size).to(device)
loss = WassersteinAELossFunction(reg_weight)
optimizer = optim.Adam(model.parameters())

dataloaders = game_data_loaders()
train_loaders, val_loaders = dataloaders['train'], dataloaders['val']

if DO_TRAIN:
    num_epochs = int(3e3)
    train_ae(num_epochs, model, dataloaders['train'], dataloaders['val'], optimizer, device, loss)
