import torch.nn as nn

from src.features_extraction.base import Encoder, Decoder, LossFunction


# Encoder
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


class AAELoss(LossFunction):
    def __init__(self):
        self.loss = nn.MSELoss()

    def __call__(self, x, x_recon, batch_size=100, X_dim=784):
        recon_loss = self.loss(x_recon, x.resize(batch_size, X_dim))

        return recon_loss
