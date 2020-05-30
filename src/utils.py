import csv
import os
from datetime import datetime

import numpy as np

import torch

# GPUs id to use them
from fastprogress import master_bar, progress_bar
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from src.args import args

GPU_ids = [0]
GPU_ids_str = ','.join([str(i) for i in GPU_ids])

MODELS_PATH = './saved_model/'

LOG_PATH = './logs/'
FILE_NAME = 'log_'
FILE_EXCITON = '.csv'


def save_to_file(path, dict_saver):
    """
    save logs without caring about overriding on a file or saving logs in memory.

        dict_saver = {}
        dict_saver.update({'train_loss_mean': train_loss_mean})
        dict_saver.update({'test_loss_mean': val_loss_mean})
        save_to_file(file_to_save_path, dict_saver)

    :param path: path to save file in
    :param dict_saver: dict. contains the new records only
    """

    header = list(dict_saver.keys())
    values = list(dict_saver.values())
    write_results_csv(path, header, values)


def write_results_csv(file_name, headers_name, row_data, operation='a'):
    if len(headers_name) != len(row_data):
        raise ValueError('Row data length must match the file header length')
    _write_data = list()

    if not os.path.exists(file_name):
        operation = 'w'
        _write_data.append(headers_name)

    _write_data.append(row_data)

    with open(file_name, operation) as f:
        writer = csv.writer(f)
        _ = [writer.writerow(i) for i in _write_data]


torch2numpy = lambda x: x.cpu().detach().numpy()


def mnist_NxN_loader(root=args['hyper_parameters']['dataset'],
                     batch_size=args['hyper_parameters']['batch_size'],
                     NxN=args['hyper_parameters']['input_size']):

    transform_list = [transforms.RandomRotation(15),
                      transforms.Resize(NxN),
                      transforms.ToTensor(), ]

    transform = transforms.Compose(transform_list)

    mnist_train = datasets.MNIST(root, train=True, transform=transform, target_transform=None,
                                 download=True)
    mnist_test = datasets.MNIST(root, train=False, transform=transform, target_transform=None,
                                download=True)

    # Set Data Loader(input pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_device():
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:' + GPU_ids_str)
    return device


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location='cuda:' + GPU_ids_str))
    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)


def plot_loss_update(epoch, epochs, mb, train_loss, valid_loss):
    x = [i + 1 for i in range(epoch + 1)]
    train_loss, valid_loss = np.log10(train_loss), np.log10(valid_loss)
    y = np.concatenate((train_loss, valid_loss))
    graphs = [[x, train_loss], [x, valid_loss]]
    x_margin = 0.2
    y_margin = 0.05
    x_bounds = [1 - x_margin, epochs + x_margin]
    y_bounds = [np.min(y) - y_margin, np.max(y) + y_margin]
    mb.update_graph(graphs, x_bounds, y_bounds)


def setup_experiment(title, log_dir="./tb"):
    experiment_name = "{}@{}".format(title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    writer = SummaryWriter(log_dir=os.path.join(log_dir, experiment_name))
    best_model_path = f"{title}.best.pth"
    return writer, experiment_name, best_model_path


def train_ae(n_epochs, model, train_iterator, val_iterator, optimizer, device, criterion, save_best=True, verbose=True):
    writer, experiment_name, best_model_path = setup_experiment(model.__class__.__name__, log_dir="./tb")

    mb = master_bar(range(n_epochs))

    train_losses, val_losses = [], []
    best_val_loss = float('+inf')

    for epoch in mb:
        train_loss = run_epoch(model, train_iterator, optimizer, criterion, phase='train', epoch=epoch,
                               writer=writer, device=device)

        val_loss = run_epoch(model, val_iterator, None, criterion, phase='val', epoch=epoch,
                             writer=writer, device=device)

        # save logs
        dict_saver = {}
        dict_saver.update({'train_loss_mean': train_loss})
        dict_saver.update({'test_loss_mean': val_loss})
        file_to_save_path = ''.join([LOG_PATH, FILE_NAME, experiment_name, FILE_EXCITON])
        save_to_file(file_to_save_path, dict_saver)

        # save the best model
        if save_best and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        if verbose:
            # append to a list for real-time plotting
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # start plotting for notebook
            mb.main_bar.comment = f'EPOCHS, best_loss:{best_val_loss}'
            mb.child.comment = f"train_loss:{round(train_loss, 3)}, val_loss:{round(val_loss, 3)}"
            plot_loss_update(epoch, n_epochs, mb, train_losses, val_losses)

    return best_model_path


def run_epoch(model, iterator, optimizer, criterion, phase='train', epoch=0, writer=None, device=None):
    device = get_device() if not device else device

    is_train = (phase == 'train')
    if is_train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0

    with torch.set_grad_enabled(is_train):
        for i, X in enumerate(iterator):
            global_i = len(iterator) * epoch + i

            # data to device
            X = X.to(device)

            # make prediction
            X_reconstructed = model(X)

            # calculate loss
            loss = criterion(X_reconstructed, X)

            if is_train:
                # make optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # dump train metrics to tensor-board
            if writer is not None and is_train:
                writer.add_scalar(f"loss/{phase}", loss.item(), global_i)

            epoch_loss += loss.item()

        # dump epoch metrics to tensor-board
        if writer is not None:
            writer.add_scalar(f"loss_epoch/{phase}", epoch_loss / len(iterator), epoch)

        return epoch_loss / len(iterator)


if __name__ == '__main__':
    pass