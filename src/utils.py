import csv
import os
from datetime import datetime

import torch

# GPUs id to use them
GPU_ids = [0, 1]
GPU_ids_str = ','.join([str(i) for i in GPU_ids])

MODELS_PATH = './saved_model/'

LOG_PATH = './logs/'
FILE_NAME = 'log_'
FILE_EXCITON = '.csv'

time_str = 'exp_' + datetime.utcnow().strftime('%Y-%m-%decoder %H:%M:%S.%f')
file_to_save_path = ''.join([LOG_PATH, FILE_NAME, time_str, FILE_EXCITON])


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
