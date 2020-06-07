import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, normalize, t, device, batch_size):
        # args: normalize: use normalization of hidden vector or not; t: temperature parameter
        super(ContrastiveLoss, self).__init__()
        self.normalize = normalize
        self.t = t
        self.batch_size = batch_size
        self.device = device
        self.sim_func = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def get_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye(2 * self.batch_size, 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye(2 * self.batch_size, 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool).to(self.device)
        return mask

    def forward(self, proj_i, proj_j):
        if self.normalize:
            proj_i = F.normalize(proj_i, dim=1)
            proj_j = F.normalize(proj_j, dim=1)

        # input: [bs x HEAD_DIM]
        projections = torch.cat([proj_i, proj_j], dim=0)
        sim_matrix = self.sim_func(projections.unsqueeze(1), projections.unsqueeze(0))

        pos = torch.cat([torch.diag(sim_matrix, self.batch_size), torch.diag(sim_matrix, -self.batch_size)]).view(
            2 * self.batch_size, 1)
        neg = sim_matrix[self.get_mask()].view(2 * self.batch_size, -1)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.t
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels) / (2 * self.batch_size)

        return loss
