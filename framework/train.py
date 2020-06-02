import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from IPython import display
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
from aux import *


def train(model, device, export_name, lr, weight_decay, gamma, step_size, n_epochs, cloud, train_iterator, val_iterator, criterion):

    if not cloud:
        writer = SummaryWriter(f'log/{export_name}')
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma = gamma, step_size = step_size)

    train_iters = 0

    best_val_loss = np.inf
    train_loss_history = []
    val_loss_history = []  

    for epoch in trange(n_epochs):
        model.train()
        for batch_no, (xi_batch, xj_batch) in enumerate(train_iterator):
            optimizer.zero_grad()
            xi_batch_gpu = xi_batch.to(device)
            xj_batch_gpu = xj_batch.to(device)
            repr_i, proj_i = model(xi_batch_gpu)
            repr_j, proj_j = model(xj_batch_gpu)
            loss = criterion(proj_i, proj_j)
            train_loss_history.append(loss.detach().cpu().item())
            loss.backward()
            optimizer.step()

            if not cloud:
                if train_iters % 10 == 0:
                    writer.add_scalar('train loss', loss, global_step = len(train_loss_history))

            train_iters += 1

        is_best = False
        with torch.no_grad():
            model.eval()
            val_loss = 0
            for batch_no, (xi_batch, xj_batch) in enumerate(val_iterator): 
                xi_batch_gpu = xi_batch.to(device)
                xj_batch_gpu = xj_batch.to(device)
                repr_i, proj_i = model(xi_batch_gpu)
                repr_j, proj_j = model(xj_batch_gpu)
                loss = criterion(proj_i, proj_j)
                val_loss += loss.item()
        total_val_loss = val_loss / (batch_no + 1)
        val_loss_history.append(total_val_loss)

        if total_val_loss < best_val_loss:
            is_best = True
            best_val_loss = total_val_loss
            checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()}
            save_ckp(checkpoint, is_best, export_name, export_name)
        if not cloud:           
            writer.add_scalar('val loss', total_val_loss, global_step = epoch + 1)

        scheduler.step()
        
        if not cloud:
            writer.add_scalar('lr_decay', scheduler.get_lr()[0], global_step = epoch + 1)

        display.clear_output(wait = True) 
        
        if cloud:
            
            fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 5))
            ax1.plot(train_loss_history)
            ax2.plot(val_loss_history)
            plt.show()
    
        print('\n#Epoch: ', epoch + 1)
        print('Current learning rate: ', get_lr(optimizer))
        print('Current train loss: ', train_loss_history[-1])
        print('Current val loss: ', val_loss_history[-1])
        print('best val loss: ', best_val_loss)
