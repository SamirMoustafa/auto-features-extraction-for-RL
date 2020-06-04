import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F

from reinforcement_learning.base.rl_agent import RLAgent
from reinforcement_learning.ddqn.ddqn_nn_models import DDQNNetwork2D
from reinforcement_learning.utils.nn_utils import copy_params
from reinforcement_learning.utils.replay_buffer import ReplayBuffer


class DDQNAgent(RLAgent):
    def __init__(self, state_dim, action_dim, action_low, action_high, gamma, tau, buffer_size,
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










        self.critic_ = CriticNetwork2D(action_dim).to(self.device_)
        self.critic_target_ = CriticNetwork2D(action_dim).to(self.device_)

        self.actor_ = ActorNetwork2D(action_dim).to(self.device_)
        self.actor_target_ = ActorNetwork2D(action_dim).to(self.device_)

        copy_params(self.critic_, self.critic_target_)
        copy_params(self.actor_, self.actor_target_)

        self.critic_optimizer_ = optim.Adam(self.critic_.parameters(), lr=critic_lr)
        self.actor_optimizer_ = optim.Adam(self.actor_.parameters(), lr=actor_lr)

        # self.noise_gen_ = OUNoiseGenerator(action_dim, action_low, action_high)

        self.replay_buffer_ = ReplayBuffer(buffer_size)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device_)
        action = self.actor_.forward(state)
        return action.squeeze(0).cpu().detach().numpy()

    def train_step(self, batch_size):
        # states, actions, rewards, next_states, _ = self.replay_buffer_.sample_random_batch(batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, _ = self.replay_buffer_.sample_random_batch(
            batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device_)
        action_batch = torch.FloatTensor(action_batch).to(self.device_)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device_)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device_)
        # masks = torch.FloatTensor(masks).to(self.device_)

        current_q = self.critic_.forward(state_batch, action_batch)
        next_actions = self.actor_target_.forward(next_state_batch)
        next_q = self.critic_target_.forward(next_state_batch, next_actions.detach())
        expected_q = reward_batch + self.gamma_ * next_q

        # update critic
        q_loss = F.mse_loss(current_q, expected_q.detach())

        self.critic_optimizer_.zero_grad()
        q_loss.backward()
        self.critic_optimizer_.step()

        # update actor
        policy_loss = -self.critic_.forward(state_batch, self.actor_.forward(state_batch)).mean()

        self.actor_optimizer_.zero_grad()
        policy_loss.backward()
        self.actor_optimizer_.step()

        # update target networks
        for target_param, param in zip(self.actor_target_.parameters(), self.actor_.parameters()):
            target_param.data.copy_(param.data * self.tau_ + target_param.data * (1.0 - self.tau_))

        for target_param, param in zip(self.critic_target_.parameters(), self.critic_.parameters()):
            target_param.data.copy_(param.data * self.tau_ + target_param.data * (1.0 - self.tau_))

    def save_checkpoint(self, checkpoint_path):
        torch.save({'critic': self.critic_.state_dict(),
                    'actor': self.actor_.state_dict()},
                   checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        loaded = torch.load(checkpoint_path)
        self.critic_.load_state_dict(loaded['critic'])
        self.actor_.load_state_dict(loaded['actor'])
        copy_params(self.critic_, self.critic_target_)
        copy_params(self.actor_, self.actor_target_)

    def set_eval_mode(self, mode):
        self.training_mode_ = not mode
        if mode is False:
            self.critic_.eval()
            self.critic_target_.eval()
            self.actor_.eval()
            self.actor_target_.eval()
        else:
            self.critic_.train()
            self.critic_target_.train()
            self.actor_.train()
            self.actor_target_.train()
