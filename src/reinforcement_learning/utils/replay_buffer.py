import numpy as np
from collections import deque
import random

import torch

class ReplayBuffer:
    def __init__(self, max_buffer_size):
        self.max_buffer_size_ = max_buffer_size
        self.buffer_ = deque(maxlen=max_buffer_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer_.append((state, action, reward, next_state, done))

    def sample_random_batch(self, batch_size, as_tensor=False):
        states = []
        actions = []
        rewards = []
        next_states = []
        done = []
        random_batch = random.sample(self.buffer_, batch_size)
        for experience in random_batch:
            states.append(experience[0])
            actions.append(experience[1])
            rewards.append(experience[2])
            next_states.append(experience[3])
            done.append(experience[4])
        if not as_tensor:
            return states, actions, rewards, next_states, done
        else:
            return torch.as_tensor(states, dtype=torch.float32),\
                   torch.as_tensor(actions, dtype=torch.float32), \
                   torch.as_tensor(rewards, dtype=torch.float32), \
                   torch.as_tensor(next_states, dtype=torch.float32), \
                   torch.as_tensor(done, dtype=torch.float32)

    def sample_sequence_batch(self, batch_size):
        states = []
        actions = []
        rewards = []
        next_states = []
        done = []
        start = np.random.randint(0, len(self.buffer_) - batch_size)
        for idx in range(start, start + batch_size):
            states.append(self.buffer_[idx][0])
            actions.append(self.buffer_[idx][1])
            rewards.append(self.buffer_[idx][2])
            next_states.append(self.buffer_[idx][3])
            done.append(self.buffer_[idx][4])
        return states, actions, rewards, next_states, done

    def __len__(self):
        return len(self.buffer_)
