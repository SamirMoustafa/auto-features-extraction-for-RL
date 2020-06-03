import numpy as np
import torch
import torch.nn as nn
from fastprogress import progress_bar, master_bar

from src.args import args
from src.features_extraction.base import Encoder, Decoder, LossFunction, View
from src.test_modules import TestModelMethods
from src.utils import mnist_NxN_loader, reconstruction_loss, get_device, save_to_file, LOG_PATH, FILE_NAME, \
    FILE_EXCITON, save_model, setup_experiment, get_fixed_hyper_param

test = TestModelMethods()

# Encoder
class AAEEncoder(Encoder, nn.Module):
    def __init__(self, z_dim, nc):
        super(AAEEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32,  8,  8
            nn.ReLU(True),
            View((-1, 32 * 8 * 8)),  # B, 2048
            nn.Linear(32 * 8 * 8, 512),  # B, 512
            nn.ReLU(True),
            nn.Linear(512, 256),  # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dim),  # B, z_dim*2
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


# Decoder
class AAEDecoder(Decoder, nn.Module):
    def __init__(self, z_dim, nc, target_size):
        super(AAEDecoder, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.target_size = target_size

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),  # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),  # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32 * 8 * 8),  # B, 2048
            nn.ReLU(True),
            View((-1, 32, 8, 8)),  # B,  32,  8,  8
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B,  nc, 64, 64
            nn.Tanh(),
            View(self.target_size),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class AdversarialAE(nn.Module):
    def __init__(self, z_dim, nc, target_size):
        super(AdversarialAE, self).__init__()
        self.encoder = AAEEncoder(z_dim, nc)
        self.decoder = AAEDecoder(z_dim, nc, target_size)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z


class AAELoss(LossFunction):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def __call__(self, x, x_recon):
        return reconstruction_loss(x_recon, x)


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


def plot_loss_update(epoch, epochs, mb, train_loss, valid_loss,
                     train_discriminator_loss, val_discriminator_loss,
                     train_generator_loss, val_generator_loss):
    x = [i + 1 for i in range(epoch + 1)]
    y = np.concatenate((train_loss, valid_loss,
                        train_discriminator_loss, val_discriminator_loss,
                        train_generator_loss, val_generator_loss))

    graphs = [[x, train_loss], [x, valid_loss],
              [x, train_discriminator_loss], [x, val_discriminator_loss],
              [x, train_generator_loss], [x, val_generator_loss]]

    x_margin = 0.2
    y_margin = 0.05
    x_bounds = [1 - x_margin, epochs + x_margin]
    y_bounds = [np.min(y) - y_margin, np.max(y) + y_margin]
    mb.update_graph(graphs, x_bounds, y_bounds)


def train_aae(n_epochs, model, discriminator, train_iterator, val_iterator, encoder_optimizer, decoder_optimizer,
              discriminator_optimizer, device, criterion, save_best=True, verbose=True):
    writer, experiment_name, best_model_path = setup_experiment(model.__class__.__name__, log_dir="./tb")

    mb = master_bar(range(n_epochs))

    train_losses, val_losses, train_discriminator_losses, val_discriminator_losses, train_generator_losses, val_generator_losses = [], [], [], [], [], []
    best_val_loss = float('+inf')

    for epoch in mb:
        train_loss, train_discriminator_loss, train_generator_loss = run_epoch(model, discriminator, train_iterator,
                                                                               encoder_optimizer, decoder_optimizer,
                                                                               discriminator_optimizer, criterion, mb,
                                                                               phase='train', epoch=epoch,
                                                                               writer=writer,
                                                                               device=device)

        val_loss, val_discriminator_loss, val_generator_loss = run_epoch(model, discriminator, train_iterator,
                                                                         None, None,
                                                                         None, criterion, mb,
                                                                         phase='val', epoch=epoch,
                                                                         writer=writer,
                                                                         device=device)

        # save logs
        dict_saver = {}
        dict_saver.update({'train_loss_mean': train_loss})
        dict_saver.update({'test_loss_mean': val_loss})

        dict_saver.update({'train_discriminator_loss': train_discriminator_loss})
        dict_saver.update({'val_discriminator_loss': val_discriminator_loss})

        dict_saver.update({'train_generator_loss': train_generator_loss})
        dict_saver.update({'val_generator_loss': val_generator_loss})

        file_to_save_path = ''.join([LOG_PATH, FILE_NAME, experiment_name, FILE_EXCITON])
        save_to_file(file_to_save_path, dict_saver)

        # save the best model
        if save_best and (val_loss < best_val_loss):
            best_val_loss = val_loss
            save_model(model, best_model_path)

        if verbose:
            # append to a list for real-time plotting
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            train_discriminator_losses.append(train_discriminator_loss)
            val_discriminator_losses.append(val_discriminator_loss)

            train_generator_losses.append(train_generator_loss)
            val_generator_losses.append(val_generator_loss)

            # start plotting for notebook
            mb.main_bar.comment = f'EPOCHS, best_loss:{round(best_val_loss, 3)}'
            mb.child.comment = f"train_loss:{round(train_loss, 3)}, val_loss:{round(val_loss, 3)}"
            plot_loss_update(epoch, n_epochs, mb,
                             train_losses, val_losses,
                             train_discriminator_losses, val_discriminator_losses,
                             train_generator_losses, val_generator_losses)

    return best_model_path


def run_epoch(model, discriminator, iterator, encoder_optimizer, decoder_optimizer, discriminator_optimizer,
              criterion, master_bar, phase='train', epoch=0, writer=None, device=None):
    device = get_device() if not device else device

    is_train = (phase == 'train')
    if is_train:
        model.train()
    else:
        model.eval()

    epoch_reconstracted_loss = 0
    epoch_discriminator_loss = 0
    epoch_generator_loss = 0

    with torch.set_grad_enabled(is_train):
        for i, X in enumerate(progress_bar(iterator, parent=master_bar)):
            global_i = len(iterator) * epoch + i

            # data to device
            X = X.to(device)

            # make prediction
            X_reconstracted, z = model(X)

            # calculate loss
            recon_loss = criterion(X, X_reconstracted)

            if is_train:
                # make optimization step
                decoder_optimizer.zero_grad()
                encoder_optimizer.zero_grad()
                recon_loss.backward()
                decoder_optimizer.step()
                encoder_optimizer.step()

            # Evaluation mode so dropout is off
            model.encoder.eval()
            # Generating samples from a Gaussian distribution
            z_fake = (torch.randn_like(z) * 5).to(device)  # Sample from N(0,5)
            # Latent code (compression of the image)
            z_real = model.encoder(X)

            # Output of the Discriminator for gaussian and compressed z_dim dimensional vector
            discriminator_fake, discriminator_real = discriminator(z_fake), discriminator(z_real)

            # Loss of the discriminator from the template distribution
            discriminator_loss = -torch.mean(
                torch.log(discriminator_fake + 1e-8) + torch.log(1 - discriminator_real + 1e-8))

            if is_train:
                # Optimisation of the Discriminator
                discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                discriminator_optimizer.step()

            if is_train:
                # Updating Generator/Encoder
                model.encoder.train()

            z_real = model.encoder(X)
            discriminator_real = discriminator(z_real)
            generator_loss = -torch.mean(torch.log(discriminator_real + 1e-8))

            if is_train:
                encoder_optimizer.zero_grad()
                generator_loss.backward()
                encoder_optimizer.step()

            # dump train metrics to tensor-board
            if writer is not None and is_train:
                writer.add_scalar(f"loss/{phase}", recon_loss.item(), global_i)
                writer.add_scalar(f"discriminator_loss/{phase}", discriminator_loss.item(), global_i)
                writer.add_scalar(f"generator_loss/{phase}", generator_loss.item(), global_i)

            epoch_reconstracted_loss += recon_loss.item()
            epoch_discriminator_loss += discriminator_loss.item()
            epoch_generator_loss += generator_loss.item()

        # dump epoch metrics to tensor-board
        if writer is not None:
            writer.add_scalar(f"loss_epoch/{phase}", epoch_reconstracted_loss / len(iterator), epoch)
            writer.add_scalar(f"discriminator_loss_epoch/{phase}", epoch_discriminator_loss / len(iterator), epoch)
            writer.add_scalar(f"generator_loss_epoch/{phase}", epoch_generator_loss / len(iterator), epoch)

        return epoch_reconstracted_loss / len(iterator), epoch_discriminator_loss / len(
            iterator), epoch_generator_loss / len(iterator)
