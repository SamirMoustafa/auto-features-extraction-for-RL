from args import args

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

from src.features_extraction.base import Encoder, Decoder, LossFunction

#Encoder
class AAEEncoder(Encoder, nn.Module):
    '''This class is for the Encoder.
    The Hyperparamaters such as the number of Neuron per layer and the the dropout rate can be modified before execution
    The architecture is composed of 4 layersss
    The activation is Leaky ReLU
    X_dim is the size of the input, here 784
    N is the number of neurons in the hidden layers
    z_dim is the size of the bottleneck, here 2 so it can be visualized easily'''
    def __init__(self, X_dim, N, z_dim):
        super(AAEEncoder, self).__init__()
        self.layers = nn.Sequential(
            #input size is X_dim, here 784 (vectorized 28x28)
            nn.Linear(X_dim, N),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True),
            
            #output size is z_dim usually 2 for visualization
            nn.Linear(N, z_dim)        
            )
    def forward(self, x):
        #forward pass
        xgauss = self.layers(x)
        return xgauss
    
# Decoder
class AAEDecoder(Decoder, nn.Module):
    def __init__(self, X_dim, N, z_dim):
        '''This class is for the Decoder.
    The Hyperparamaters such as the number of Neuron per layer and the the dropout rate can be modified before execution
    The architecture is composed of 4 layers
    The activation is Leaky ReLU
    z_dim is the size of the bottleneck, here 2 so it can be visualized easily
    N is the number of neurons in the hidden layers
    X_dim is the output size for the reconstruction, here 784'''
        super(AAEDecoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(z_dim, N),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, X_dim),
            nn.Sigmoid()
            )
    def forward(self, x):
        out = self.layers(x)
        return out

class AAELoss(LossFunction):
    
    def __init__(self):
        self.loss = nn.MSELoss().cuda()
    
    def __call__(self, x, x_recon, batch_size=100, X_dim=784):
        
        recon_loss = self.loss(x_recon, x.resize(batch_size, X_dim))

        return recon_loss