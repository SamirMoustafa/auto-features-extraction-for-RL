import random
import numpy as np
from skimage.util.shape import view_as_blocks

import torch
import torch.utils.data
from torch import nn

from src.features_extraction.vanilla_vae import VanillaVAE, VanillaVAELossFunction, VanillaVAEEncoder, VanillaVAEBottleneck, VanillaVAEDecoder
from src.utils import torch2numpy, numpy2torch


class JigsawVAE(VanillaVAE, nn.Module):
    def __init__(self, z_dim, nc, target_size):
        super(VanillaVAE, self).__init__()

        self.encoder = VanillaVAEEncoder(z_dim, nc)
        self.bottleneck = VanillaVAEBottleneck(z_dim)
        self.decoder = VanillaVAEDecoder(z_dim, nc, target_size)

    def forward(self, x):
        for i, _ in enumerate(x):
            x[i] = shuffle(x[i], 4, rotate=True).permute(2, 0, 1)
        x = numpy2torch(x)

        x = self.encoder(x)
        z, mu, log_var = self.bottleneck(x)
        x = self.decoder(z)
        return x, mu, log_var


class JigsawVAELossFunction(VanillaVAELossFunction):
    def __init__(self):
        super().__init__()


def shuffle(im, num, rotate=False):
    im = torch2numpy(im.permute(1, 2, 0))

    rows = cols = num
    blk_size = im.shape[0] // rows

    img_blocks = view_as_blocks(im, block_shape=(blk_size, blk_size, 3)).reshape((-1, blk_size, blk_size, 3))
    img_shuffle = np.zeros((im.shape[0], im.shape[1], 3))

    a = np.arange(rows * rows, dtype=np.uint8)
    b = np.random.permutation(a)

    map = {k: v for k, v in zip(a, b)}

    for i in range(0, rows):
        for j in range(0, cols):
            x, y = i * blk_size, j * blk_size
            if rotate:
                rot_val = random.randrange(0, 4)
                img_shuffle[x:x + blk_size, y:y + blk_size] = np.rot90(img_blocks[map[i * rows + j]], rot_val)
            else:
                img_shuffle[x:x + blk_size, y:y + blk_size] = img_blocks[map[i * rows + j]]
    img_shuffle = torch.FloatTensor(img_shuffle).permute(0, 1, 2)
    return img_shuffle
