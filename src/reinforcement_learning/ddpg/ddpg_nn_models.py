import torch
import torch.nn as nn

from reinforcement_learning.utils.nn_layers import Flatten


class CriticNetwork2D(nn.Module):
    def __init__(self, action_dim):
        super(CriticNetwork2D, self).__init__()

        self.conv_ = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            Flatten(),
            nn.Linear(5408, 1024),
            nn.ReLU()
        )
        self.linear_ = nn.Sequential(
            nn.Linear(1024 + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, state, action):
        state = self.conv_.forward(state)
        x = torch.cat([state, action], dim=1)
        return self.linear_.forward(x)


class ActorNetwork2D(nn.Module):
    def __init__(self, action_dim):
        super(ActorNetwork2D, self).__init__()

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
        state = self.conv_.forward(state)
        return self.linear_.forward(state)
