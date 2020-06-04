from torch import optim

from src.args import args
from src.features_extraction import AdversarialAE, Discriminator, AAELoss, train_aae
from src.utils import get_fixed_hyper_param, get_device, game_data_loaders

device = get_device()
batch_size, num_of_channels, input_size, z_dim = get_fixed_hyper_param(args['hyper_parameters'])

DO_TRAIN = True

model = AdversarialAE(z_dim, num_of_channels, input_size).to(device)
discriminator = Discriminator(z_dim, 100).to(device)
loss = AAELoss()

encoder_optimizer = optim.Adam(model.parameters(), lr=0.0006)
decoder_optimizer = optim.Adam(model.parameters(), lr=0.0006)
discriminator_optimizer = optim.Adam(model.parameters(), lr=0.0008)

dataloaders = game_data_loaders()
train_loaders, val_loaders = dataloaders['train'], dataloaders['val']

if DO_TRAIN:
    num_epochs = int(3e3)
    train_aae(num_epochs, model, discriminator, train_loaders, val_loaders,
              encoder_optimizer, decoder_optimizer, discriminator_optimizer, device, loss)
