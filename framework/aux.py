import torch
import shutil


# some util functions
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    f_path = checkpoint_dir + 'checkpoint.pth'
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + 'best_model.pth'
        shutil.copyfile(f_path, best_fpath)


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
