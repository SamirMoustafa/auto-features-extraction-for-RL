import torch
import torch.nn as nn

from src.args import args
from src.features_extraction.base import Encoder, Decoder, LossFunction

# Encoder
from src.utils import mnist_NxN_loader


class AAEEncoder(Encoder, nn.Module):
    def __init__(self, X_dim, N, z_dim):
        super(AAEEncoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(X_dim, N),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(N, N),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(N, N),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(N, z_dim)
        )

    def forward(self, x):
        # forward pass
        x_gauss = self.layers(x)
        return x_gauss


# Decoder
class AAEDecoder(Decoder, nn.Module):
    def __init__(self, X_dim, N, z_dim):
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


class AdversarialAE(nn.Module):
    def __init__(self, X_dim, N, z_dim):
        super(AdversarialAE, self).__init__()
        self.encoder = AAEEncoder(X_dim, N, z_dim)
        self.decoder = AAEDecoder(X_dim, N, z_dim)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z


class AAELoss(LossFunction):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def __call__(self, x, x_recon):
        recon_loss = self.loss(x_recon, x)

        return recon_loss


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, z_dim, N):
        super(Discriminator, self).__init__()
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


def train(aae, loss_function, train_loader, D_gauss, optim_dec, optim_enc, D_gauss_solver, z_dim=2, num_epoch=100):
    from IPython import display

    from torch.autograd import Variable
    from tqdm import tqdm

    for epoch in range(num_epoch):
        for batch, label in tqdm(train_loader):
            # Vectorizing the input
            batch_size = batch.shape[0]
            # GPU variable
            X = Variable(batch.view(batch_size, -1))

            # Compressed digit
            X_sample, z_sample = aae(X)

            # reconstruction loss using MSE
            recon_loss = loss_function(X_sample, X)

            # Updating the weights of the Encoder and the Decoder
            optim_dec.zero_grad()
            optim_enc.zero_grad()
            recon_loss.backward()
            optim_dec.step()
            optim_enc.step()

            # Evaluation mode so dropout is off
            aae.encoder.eval()
            # Generating samples from a Gaussian distribution
            z_real_gauss = Variable(torch.randn(batch_size, z_dim) * 5)  # Sample from N(0,5)
            # Latent code (compression of the image)
            z_fake_gauss = aae.encoder(X)

            # Output of the Discriminator for gaussian and compressed z_dim dimensional vector
            D_real_gauss, D_fake_gauss = D_gauss(z_real_gauss), D_gauss(z_fake_gauss)

            # Loss of the discriminator from the template distribution
            D_loss_gauss = -torch.mean(torch.log(D_real_gauss + 1e-8) + torch.log(1 - D_fake_gauss + 1e-8))

            # Optimisation of the Discriminator
            D_gauss_solver.zero_grad()
            D_loss_gauss.backward()
            D_gauss_solver.step()

            # Updating Generator/Encoder
            aae.encoder.train()
            z_fake_gauss = aae.encoder(X)
            D_fake_gauss = D_gauss(z_fake_gauss)

            optim_enc.zero_grad()
            G_loss = -torch.mean(torch.log(D_fake_gauss + 1e-8))
            G_loss.backward()
            optim_enc.step()
        display.clear_output(wait=True)
        print(epoch + 1, 'was done')


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
    model = AdversarialAE(128, 100, 2)
    D_gauss = Discriminator(2, 100)
    x_recon, z = model(x)
    loss = AAELoss()
    loss(x, x_recon)
