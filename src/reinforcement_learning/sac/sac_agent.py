import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np

from reinforcement_learning.base.rl_agent import RLAgent
from reinforcement_learning.sac.sac_nn_models import SoftQNetwork2D, GaussianPolicyNetwork2D
from reinforcement_learning.utils.nn_utils import copy_params
from reinforcement_learning.utils.replay_buffer import ReplayBuffer


class SACAgent(RLAgent):
    def __init__(self, state_dim, action_dim, action_range, gamma, tau, alpha, buffer_size,
                 critic_lr=1e-3, actor_lr=1e-3, alpha_lr=1e-3, force_cpu=False):
        super().__init__()
        self.state_dim_ = state_dim
        self.action_dim_ = action_dim
        self.action_range_ = action_range

        self.gamma_ = gamma
        self.tau_ = tau
        self.update_step_ = 0
        self.delay_step_ = 2

        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        if force_cpu:
            self.device_ = "cpu"
        self.training_mode_ = True

        # TODO: Add 1D-state support

        # Critic networks
        self.q_net_1_ = SoftQNetwork2D(action_dim).to(self.device_)
        self.q_net_2_ = SoftQNetwork2D(action_dim).to(self.device_)
        self.target_q_net_1_ = SoftQNetwork2D(action_dim).to(self.device_)
        self.target_q_net_2_ = SoftQNetwork2D(action_dim).to(self.device_)
        for target_param, param in zip(self.target_q_net_1_.parameters(), self.q_net_1_.parameters()):
            target_param.data.copy_(param)
        for target_param, param in zip(self.target_q_net_2_.parameters(), self.q_net_2_.parameters()):
            target_param.data.copy_(param)
        copy_params(self.q_net_1_, self.target_q_net_1_)
        copy_params(self.q_net_2_, self.target_q_net_2_)
        self.q_1_optimizer_ = optim.Adam(self.q_net_1_.parameters(), lr=critic_lr)
        self.q_2_optimizer_ = optim.Adam(self.q_net_2_.parameters(), lr=critic_lr)

        # Policy network
        self.policy_net_ = GaussianPolicyNetwork2D(action_dim).to(self.device_)
        self.policy_optimizer_ = optim.Adam(self.policy_net_.parameters(), lr=actor_lr)

        # Entropy optimization
        self.alpha_ = alpha
        self.target_entropy_ = -torch.prod(torch.Tensor((action_dim,)).to(self.device_)).item() # TODO: Check
        self.log_alpha_ = torch.zeros(1, requires_grad=True, device=self.device_)
        self.alpha_optimizer_ = optim.Adam([self.log_alpha_], lr=alpha_lr)

        self.mse_criterion_ = nn.MSELoss()

        self.replay_buffer_ = ReplayBuffer(buffer_size)

    def rescale_action_(self, action):
        return action * (self.action_range_[1] - self.action_range_[0]) / 2.0 + \
               (self.action_range_[1] + self.action_range_[0]) / 2.0

    def set_eval_mode(self, mode):
        self.training_mode_ = not mode
        if mode is False:
            self.q_net_1_.eval()
            self.q_net_2_.eval()
            self.target_q_net_1_.eval()
            self.target_q_net_2_.eval()
            self.policy_net_.eval()
        else:
            self.q_net_1_.train()
            self.q_net_2_.train()
            self.target_q_net_1_.train()
            self.target_q_net_2_.train()
            self.policy_net_.train()

    def get_action(self, state):
        with torch.set_grad_enabled(self.training_mode_):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device_)
            mean, log_std = self.policy_net_.forward(state)
            std = log_std.exp()

            normal = dist.Normal(mean, std)
            z = normal.sample()
            action = torch.tanh(z)
            action = action.cpu().detach().squeeze(0).numpy()

            return self.rescale_action_(action)

    def train_step(self, batch_size):
        self.set_eval_mode(False)

        states, actions, rewards, next_states, dones = self.replay_buffer_.sample_random_batch(batch_size)
        states = torch.FloatTensor(states).to(self.device_)
        actions = torch.FloatTensor(actions).to(self.device_)
        rewards = torch.FloatTensor(rewards).to(self.device_)
        next_states = torch.FloatTensor(next_states).to(self.device_)
        dones = torch.FloatTensor(dones).to(self.device_)
        dones = dones.view(dones.size(0), -1)

        next_actions, next_log_pi = self.policy_net_.sample(next_states)
        next_q1 = self.target_q_net_1_(next_states, next_actions)
        next_q2 = self.target_q_net_2_(next_states, next_actions)
        next_q_target = torch.min(next_q1, next_q2) - self.alpha_ * next_log_pi
        expected_q = rewards + (1 - dones) * self.gamma_ * next_q_target

        # q loss
        curr_q1 = self.q_net_1_.forward(states, actions)
        curr_q2 = self.q_net_2_.forward(states, actions)
        q1_loss = self.mse_criterion_.forward(curr_q1, expected_q.detach())
        q2_loss = self.mse_criterion_.forward(curr_q2, expected_q.detach())

        # update q networks
        self.q_1_optimizer_.zero_grad()
        q1_loss.backward()
        self.q_1_optimizer_.step()

        self.q_2_optimizer_.zero_grad()
        q2_loss.backward()
        self.q_2_optimizer_.step()

        # delayed update for policy network and target q networks
        new_actions, log_pi = self.policy_net_.sample(states)
        if self.update_step_ % self.delay_step_ == 0:
            min_q = torch.min(
                self.q_net_1_.forward(states, new_actions),
                self.q_net_2_.forward(states, new_actions)
            )
            policy_loss = (self.alpha_ * log_pi - min_q).mean()

            self.policy_optimizer_.zero_grad()
            policy_loss.backward()
            self.policy_optimizer_.step()

            # target networks
            for target_param, param in zip(self.target_q_net_1_.parameters(), self.q_net_1_.parameters()):
                target_param.data.copy_(self.tau_ * param + (1 - self.tau_) * target_param)

            for target_param, param in zip(self.target_q_net_2_.parameters(), self.q_net_2_.parameters()):
                target_param.data.copy_(self.tau_ * param + (1 - self.tau_) * target_param)

        # update temperature
        alpha_loss = (self.log_alpha_ * (-log_pi - self.target_entropy_).detach()).mean()

        self.alpha_optimizer_.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer_.step()
        self.alpha = self.log_alpha_.exp()

        self.update_step_ += 1

