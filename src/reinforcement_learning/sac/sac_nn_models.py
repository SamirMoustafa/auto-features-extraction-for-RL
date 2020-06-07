import torch
import torch.nn as nn
import torch.distributions as dist

import numpy as np


class SoftQNetwork2D(nn.Module):
    def __init__(self, action_dim, hidden_dim=500):
        super(SoftQNetwork2D, self).__init__()
        self.conv_ = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.linear1_ = nn.Linear(288 + action_dim, hidden_dim)
        self.linear2_ = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        state = self.conv_.forward(state)
        state = state.view((state.shape[0], -1))
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.linear1_.forward(x))
        x = self.linear2_.forward(x)
        return x


class GaussianPolicyNetwork2D(nn.Module):
    def __init__(self, action_dim, hidden_dim=500, log_std_min=-20, log_std_max=2):
        super(GaussianPolicyNetwork2D, self).__init__()

        self.log_std_min_ = log_std_min
        self.log_std_max_ = log_std_max

        self.conv_ = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.linear_common_ = nn.Sequential(
            nn.Linear(288, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.linear_mean_ = nn.Linear(hidden_dim, action_dim)
        self.linear_log_std_ = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.conv_.forward(x)
        x = x.view((x.shape[0], -1))
        x = self.linear_common_.forward(x)
        mean = self.linear_mean_.forward(x)
        log_std = self.linear_log_std_.forward(x)
        log_std = torch.clamp(log_std, self.log_std_min_, self.log_std_max_)
        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        distribution = dist.Normal(mean, std)

        z = distribution.rsample()
        action = torch.tanh(z)

        log_pi = distribution.log_prob(z) - torch.log(1.0 - action.pow(2) + epsilon)
        log_pi = log_pi.sum(1, keepdim=True)

        return action, log_pi


class SoftQNetwork1D(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=500):
        super(SoftQNetwork1D, self).__init__()
        self.linear_ = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.linear_.forward(x)


class GaussianPolicyNetwork1D(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=500, log_std_min=-20, log_std_max=2):
        super(GaussianPolicyNetwork1D, self).__init__()

        self.log_std_min_ = log_std_min
        self.log_std_max_ = log_std_max

        self.linear_common_ = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.linear_mean_ = nn.Linear(hidden_dim, action_dim)
        self.linear_log_std_ = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.linear_common_.forward(x)
        mean = self.linear_mean_.forward(x)
        log_std = self.linear_log_std_.forward(x)
        log_std = torch.clamp(log_std, self.log_std_min_, self.log_std_max_)
        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        distribution = dist.Normal(mean, std)

        z = distribution.rsample()
        action = torch.tanh(z)

        log_pi = distribution.log_prob(z) - torch.log(1.0 - action.pow(2) + epsilon)
        log_pi = log_pi.sum(1, keepdim=True)

        return action, log_pi
