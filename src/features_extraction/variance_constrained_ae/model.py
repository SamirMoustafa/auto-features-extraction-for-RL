import torch
import torch.utils.data
from fastprogress import master_bar, progress_bar
from torch import nn

from src.args import args
from src.features_extraction.base import LossFunction, Bottleneck
from src.features_extraction.vanilla_vae import VanillaVAEEncoder, VanillaVAEDecoder
from src.utils import reconstruction_loss, game_data_loaders, get_device, get_fixed_hyper_param, setup_experiment, \
    plot_loss_update, save_model, save_to_file, LOG_PATH, FILE_EXCITON, FILE_NAME

from src.features_extraction.variance_constrained_ae.normalizing_flows import MAF, NormalizingFlow

from src.test_modules import TestModelMethods

test = TestModelMethods()


class VarianceConstrainedAEBottleneck(Bottleneck, nn.Module):
    def __init__(self, mu=0, sigma=0.05, device=None):
        super(VarianceConstrainedAEBottleneck, self).__init__()
        self.mu = mu
        self.sigma = sigma

        if not device:
            device = get_device()
        self.device = device

    def forward(self, x):
        return (torch.randn(x.shape[:2]) * self.sigma + self.mu).to(self.device)


class VarianceConstrainedAE(nn.Module):
    def __init__(self, z_dim, nc, target_size, l_lambda=0.1):
        super(VarianceConstrainedAE, self).__init__()

        # hyper-parameters
        self.l_lambda = l_lambda
        self.v = z_dim

        self.encoder = VanillaVAEEncoder(z_dim, nc)
        self.bottleneck = VarianceConstrainedAEBottleneck()
        self.decoder = VanillaVAEDecoder(z_dim, nc, target_size)

    def forward(self, x, is_nf=False):
        x = self.encoder(x)
        z = self.bottleneck(x)

        if is_nf:
            return z
        else:
            x = self.decoder(z)
            return x, z, self.l_lambda, self.v


class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self, z_dim, device=None):
        super().__init__()

        if not device:
            device = get_device()

        flows = [MAF(dim=z_dim, parity=True, nh=64) for i in range(5)]

        self.prior = torch.distributions.Normal(torch.zeros(z_dim).to(device), torch.ones(z_dim).to(device))
        self.flow = NormalizingFlow(flows).to(device)

    def forward(self, x):
        zs, log_det = self.flow(x)
        prior_log_prob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_log_prob, log_det

    def backward(self, z):
        xs, log_det = self.flow.backward(z)
        return xs, log_det

    def sample(self, num_samples):
        z = self.prior.sample((num_samples,))
        xs, _ = self.flow.backward(z)
        return xs


class VarianceConstrainedAELossFunction(LossFunction):
    def __call__(self, x, x_recon, z, l_lambda, v):
        mse = reconstruction_loss(x, x_recon)
        mean = z.mean(dim=0)
        var = torch.norm((z - mean), dim=1).pow(2).mean()
        reg = torch.mul(torch.sub(var, v).abs(), l_lambda)
        return mse + reg


class NormalizingLossFunction(LossFunction):
    def __call__(self, nf, z):
        zs, prior_log_prob, log_det = nf(z)
        log_prob = prior_log_prob + log_det
        return -log_prob.sum(dim=0) / z.shape[0]


def run_epoch(model, iterator, optimizer, criterion, master_bar, phase='train', epoch=0, writer=None, is_nf=False,
              nf=None, device=None):
    is_train = (phase == 'train')
    if is_train:
        if not is_nf:
            model.train()
        else:
            model.eval()
            nf.train()
    else:
        if not is_nf:
            model.eval()
        else:
            model.eval()
            nf.eval()

    epoch_loss = 0

    for i, batch in enumerate(progress_bar(iterator, parent=master_bar)):
        global_i = len(iterator) * epoch + i

        if is_train:
            if not is_nf:
                model.zero_grad()
            else:
                nf.zero_grad()

        # unpack batch
        X_batch = batch
        X_batch = X_batch.to(device)

        if not is_nf:
            out, z, l_lambda, v = model(X_batch, is_nf)
            loss = criterion(X_batch, out, z, l_lambda, v)
        else:
            z = model(X_batch, is_nf)
            loss = criterion(nf, z)

        if is_train:
            # make optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # dump train metrics to tensorboard
        if writer is not None and is_train:
            if not is_nf:
                writer.add_scalar(f"loss/{phase}", loss.item(), global_i)
            else:
                writer.add_scalar(f"loss_nf/{phase}", loss.item(), global_i)

        epoch_loss += loss.item()

    # dump epoch metrics to tensorboard
    if writer is not None:
        if not is_nf:
            writer.add_scalar(f"loss_epoch/{phase}", epoch_loss / len(iterator), epoch)
        else:
            writer.add_scalar(f"loss_nf_epoch/{phase}", epoch_loss / len(iterator), epoch)

    return epoch_loss / len(iterator)


def train_vcae(n_epochs, model, train_iterator, val_iterator, optimizer, device, criterion, save_best=True,
               verbose=True, is_nf=False, nf=None):
    model_name = 'NormalizingFlow' + model.__class__.__name__ if is_nf else model.__class__.__name__
    writer, experiment_name, best_model_path = setup_experiment(model_name, log_dir="./tb")

    mb = master_bar(range(n_epochs))

    train_losses, val_losses = [], []
    best_val_loss = float('+inf')

    for epoch in mb:
        train_loss = run_epoch(model, train_iterator, optimizer, criterion, mb, phase='train', epoch=epoch,
                               writer=writer, is_nf=is_nf, nf=nf, device=device)

        val_loss = run_epoch(model, val_iterator, None, criterion, mb, phase='val', epoch=epoch,
                             writer=writer, is_nf=is_nf, nf=nf, device=device)

        # save logs
        dict_saver = {}
        dict_saver.update({'train_loss_mean': train_loss})
        dict_saver.update({'test_loss_mean': val_loss})
        file_to_save_path = ''.join([LOG_PATH, FILE_NAME, experiment_name, FILE_EXCITON])
        save_to_file(file_to_save_path, dict_saver)

        # save the best model
        if save_best and (val_loss < best_val_loss):
            best_val_loss = val_loss
            save_model(nf if is_nf else model, best_model_path)

        if verbose:
            # append to a list for real-time plotting
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # start plotting for notebook
            mb.main_bar.comment = f'EPOCHS, best_loss:{best_val_loss}'
            mb.child.comment = f"train_loss:{round(train_loss, 3)}, val_loss:{round(val_loss, 3)}"
            plot_loss_update(epoch, n_epochs, mb, train_losses, val_losses)

    return best_model_path
