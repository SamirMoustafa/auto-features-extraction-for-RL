import torch
import torch.nn as nn

from reinforcement_learning.utils.nn_layers import Flatten


class DDQNNetwork2D(nn.Module):
    def __init__(self, action_dim):
        super(DDQNNetwork2D, self).__init__()

        self.conv_ = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            Flatten(),
            nn.Linear(5408, 512),
            nn.ReLU()
        )

        self.linear_ = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        features = self.conv_.forward(state)
        q_values = self.linear_.forward(features)
        return q_values
