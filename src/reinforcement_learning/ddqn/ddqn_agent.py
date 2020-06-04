import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F
import numpy as np

from reinforcement_learning.base.rl_agent import RLAgent
from reinforcement_learning.ddqn.ddqn_nn_models import DDQNNetwork2D
from reinforcement_learning.utils.nn_utils import copy_params
from reinforcement_learning.utils.replay_buffer import ReplayBuffer


class DDQNAgent(RLAgent):
    def __init__(self, state_dim, action_dim, action_range, gamma, tau, buffer_size,
                 lr=1e-3, force_cpu=False):
        super().__init__()

        self.state_dim_ = state_dim
        self.action_dim_ = action_dim
        self.gamma_ = gamma
        self.tau_ = tau

        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        if force_cpu:
            self.device_ = "cpu"
        self.training_mode_ = True

        self.model_ = DDQNNetwork2D(action_dim).to(self.device_)
        self.model_target_ = DDQNNetwork2D(action_dim).to(self.device_)

        self.optimizer_ = torch.optim.Adam(self.model_.parameters(), lr=lr)
        self.optimizer_target_ = torch.optim.Adam(self.model_target_.parameters(), lr=lr)

        copy_params(self.model_, self.model_target_)

        self.replay_buffer_ = ReplayBuffer(buffer_size)
        self.mse_criterion_ = nn.MSELoss()

    def get_action(self, state, eps=0.20):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device_)
        action = self.model_.forward(state)

        return action.squeeze(0).cpu().detach().numpy()

    def train_step(self, batch_size):
        # states, actions, rewards, next_states, _ = self.replay_buffer_.sample_random_batch(batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer_.sample_random_batch(
            batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device_)
        action_batch = torch.FloatTensor(action_batch).to(self.device_)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device_)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device_)
        done_batch = torch.FloatTensor(done_batch).to(self.device_)
        done_batch = done_batch.view(done_batch.size(0), 1)
        # masks = torch.FloatTensor(masks).to(self.device_)

        # print(state_batch.shape, action_batch.shape)

        # compute loss
        current_q = self.model_.forward(state_batch)  # .gather(1, action_batch)
        current_q_target = self.model_target_.forward(state_batch)  # .gather(1, action_batch)

        next_q = self.model_.forward(next_state_batch)
        next_q_target = self.model_target_.forward(next_state_batch)

        next_Q = torch.min(
            torch.max(next_q, 1)[0],
            torch.max(next_q_target, 1)[0]
        )

        next_Q = next_Q.view(next_Q.size(0), 1)
        expected_Q = reward_batch + (1 - done_batch) * self.gamma_ * next_Q

        loss = self.mse_criterion_.forward(current_q, expected_Q.detach())
        loss_target = self.mse_criterion_.forward(current_q_target, expected_Q.detach())

        self.update(loss, loss_target)

    def update(self, loss, loss_target):
        self.optimizer_.zero_grad()
        loss.backward()
        self.optimizer_.step()

        self.optimizer_target_.zero_grad()
        loss_target.backward()
        self.optimizer_target_.step()

        # update target networks
        for target_param, param in zip(self.model_target_.parameters(), self.model_.parameters()):
            target_param.data.copy_(param.data * self.tau_ + target_param.data * (1.0 - self.tau_))

    def save_checkpoint(self, checkpoint_path):
        torch.save({'q_model': self.model_.state_dict(),
                    'target_model': self.model_target_.state_dict()},
                   checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        loaded = torch.load(checkpoint_path)
        self.model_.load_state_dict(loaded['q_model'])
        self.model_target_.load_state_dict(loaded['target_model'])
        # copy_params(self.model_, self.model_target_)

    def set_eval_mode(self, mode):
        self.training_mode_ = not mode
        if mode is False:
            self.model_.eval()
            self.model_target_.eval()
        else:
            self.model_.train()
            self.model_target_.train()


