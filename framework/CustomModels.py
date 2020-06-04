import torch
import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models import GetModel

LATENT_DIM = 64
HEAD_DIM = 256


# general function to test the output and intermediate dimensionality of the models
def test_dimensionality(model, input, stage):
  test_tensor = torch.rand(input.shape[0], LATENT_DIM, 1, 1) if stage == 'feat' else torch.rand(input.shape[0], HEAD_DIM)
  print('test:', model(input).shape, test_tensor.shape, '\n')
  # assert model(input).shape == test_tensor.shape


class SimCLR_head(nn.Module):
  def __init__(self, latent_dim, head_dim):
    super(SimCLR_head, self).__init__()
    self.model = nn.Sequential(nn.Linear(latent_dim, 128),
                               nn.BatchNorm1d(128),
                               nn.ReLU(),
                               nn.Linear(128, head_dim, bias = False),
                               nn.BatchNorm1d(head_dim))
  def forward(self, x):
    x = torch.flatten(x, 1)
    x = self.model(x)
    return x

# always pretrained
class CustomSimCLR50(nn.Module):
  def __init__(self, batch_size, latent_dim, head_dim):
    super(CustomSimCLR50, self).__init__()
    self.pretrained_model = GetModel('Resnet50') 
    self.load_weights('./resnet50-1x.pth')
    self.features = torch.nn.Sequential(*list(self.pretrained_model.children())[:-1])
    self.features.add_module('Final Convolution', nn.Conv2d(2048, latent_dim, 1))
    self.head = SimCLR_head(latent_dim, head_dim)
    self.test_shape(batch_size, latent_dim)

  def test_shape(self, batch_size, latent_dim):
    test_dimensionality(self.features, torch.rand(batch_size, 3, 128, 128), 'feat')
    test_dimensionality(self.head, torch.rand(batch_size, latent_dim, 1, 1), 'head') 
  
  def load_weights(self, path):
    sd = torch.load(path, map_location = 'cpu')
    self.pretrained_model.load_state_dict(sd['state_dict'])

  def forward(self, x):
    x = self.features(x)
    return x, self.head(x)

# pretrained network from Zoo is available also
class CustomSimCLR18(nn.Module):
  def __init__(self, batch_size, latent_dim, head_dim, pretrained):
    super(CustomSimCLR18, self).__init__()
    self.model = GetModel('Resnet18', pretrained = pretrained)
    self.features = torch.nn.Sequential(*list(self.model.children())[:-1])
    self.features.add_module('Final Convolution', nn.Conv2d(512, latent_dim, 1))
    self.head = SimCLR_head(latent_dim, head_dim)
    self.test_shape(batch_size, latent_dim)

  def test_shape(self, batch_size, latent_dim):
    test_dimensionality(self.features, torch.rand(batch_size, 3, 128, 128), 'feat')
    test_dimensionality(self.head, torch.rand(batch_size, latent_dim, 1, 1), 'head') 

  def forward(self, x):
    '''model outputs : hidden representations [bs x LATENT_DIM x 1 x 1] which will be used as an input to RL model
                       nonlinear projections [bs x HEAD_DIM] which is used to compute the contrastive loss and optimization
    '''
    x = self.features(x)
    return torch.flatten(x, 1), self.head(x)

