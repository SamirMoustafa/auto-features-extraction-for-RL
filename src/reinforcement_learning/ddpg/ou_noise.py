import numpy as np


class OUNoiseGenerator(object):
    def __init__(self, action_dim, action_low, action_high,
                 mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu_ = mu
        self.theta_ = theta
        self.sigma_ = max_sigma
        self.max_sigma_ = max_sigma
        self.min_sigma_ = min_sigma
        self.decay_period_ = decay_period
        self.action_dim_ = action_dim
        self.low_ = action_low
        self.high_ = action_high
        self.state_ = None
        self.reset()

    def reset(self):
        self.state_ = np.ones(self.action_dim_) * self.mu_

    def evolve_state(self):
        x = self.state_
        dx = self.theta_ * (self.mu_ - x) + self.sigma_ * np.random.randn(self.action_dim_)
        self.state_ = x + dx
        return self.state_

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma_ = self.max_sigma_ - (self.max_sigma_ - self.min_sigma_) * min(1.0, t / self.decay_period_)
        return np.clip(action + ou_state, self.low_, self.high_)
