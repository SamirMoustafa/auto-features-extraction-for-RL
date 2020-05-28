import torch
import torch.nn as nn
import torch.nn.init as init
from torch import functional as F
from IPython import display

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm


# Discriminator
class D_net_gauss(nn.Module):
    '''This class is for the Discriminator network0
    The goal of this network is to guess if its input comes from the Encoder or a Gaussian distribution computed with randn.
    The Hyperparamaters such as the number of Neuron per layer and the the dropout rate can be modified before execution
    The architecture is composed of 3 layers
    The activation is Leaky ReLU
    z_dim is the size of the bottleneck, here 2 so it can be visualized easily
    N is the number of neurons in the hidden layers
    The output size here is 1 since the discriminator's output is a scalar representing its confidence of its input being from
    the real or encoded distribution'''
    def __init__(self, z_dim, N):        
        super(D_net_gauss, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(z_dim, N),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, N),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(N, 1),
            nn.Sigmoid()
            )
    def forward(self, x):
        out = self.layers(x)
        return out
    
def train(encoder, decoder, loss_function, train_loader, discriminator, D_gauss, optim_dec, optim_enc, D_gauss_solver, batch_size=100, X_dim=784, z_dim=2, num_epoch=100):
    for epoch in range(num_epoch):
        for batch, label in tqdm(train_loader):
            #Vectorizing the input
            X = batch.view([batch_size,1,784])
            #GPU variable
            X = Variable(X).cuda()

            #Compressed digit
            z_sample = encoder(X)
            #Reconstruction
            X_sample = decoder(z_sample)

            #reconstruction loss using MSE
            recon_loss = loss_function(X_sample, X.resize(batch_size, X_dim))

            #Updating the weights of the Encoder and the Decoder
            optim_dec.zero_grad()
            optim_enc.zero_grad()
            recon_loss.backward()
            optim_dec.step()
            optim_enc.step()

            #Evaluation mode so dropout is off
            encoder.eval()
            #Generating samples from a Gaussian distribution
            z_real_gauss = Variable(torch.randn(batch_size, z_dim) * 5)   # Sample from N(0,5)
            z_real_gauss = z_real_gauss.cuda()
            #Latent code (compression of the image)
            z_fake_gauss = encoder(X)

            # Output of the Discriminator for gaussian and compressed z_dim dimensional vector
            D_real_gauss, D_fake_gauss = D_gauss(z_real_gauss), D_gauss(z_fake_gauss)

            #Loss of the discriminator from the template distribution
            D_loss_gauss = -torch.mean(torch.log(D_real_gauss + 1e-8) + torch.log(1 - D_fake_gauss + 1e-8))

            # Optimisation of the Discriminator
            D_gauss_solver.zero_grad()
            D_loss_gauss.backward()
            D_gauss_solver.step()

            # Updating Generator/Encoder
            encoder.train()
            z_fake_gauss = encoder(X)
            D_fake_gauss = D_gauss(z_fake_gauss)

            optim_enc.zero_grad()
            G_loss = -torch.mean(torch.log(D_fake_gauss + 1e-8))
            G_loss.backward()
            optim_enc.step()
        display.clear_output(wait=True)
        print(epoch+1,'was done')