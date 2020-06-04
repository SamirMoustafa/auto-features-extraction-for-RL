import numpy as np
import gym_donkeycar
import gym
import os
import torch


class DonkeyCarEnvironment:
    # EXEC_REL_PATH = "/gym-donkeycar/apps/donkey_sim_custom_build.x86_64"
    EXEC_REL_PATH = "/gym-donkeycar/apps/new_releases/donkey_sim_custom_build.x86_64"
    PORT = 9090
    ENV_NAME = "donkey-warehouse-v0"

    def __init__(self, third_party_envs_path, encoder=None):
        self.env_ = gym.make(self.ENV_NAME, exe_path=third_party_envs_path+self.EXEC_REL_PATH, port=self.PORT, max_cte=20.0)
        self.encoder_ = None
        print(type(self.env_.observation_space))

        self.action_space = self.env_.action_space
        if self.encoder_ is None:
            self.observation_space = self.env_.observation_space
        else:
            self.observation_space = gym.spaces.box.Box(np.inf, -np.inf, shape=(64,))
            self.encoder_.eval()

    def reset(self):
        observation = np.transpose(self.env_.reset(), (2, 0, 1))
        if self.encoder_ is None:
            return observation
        with torch.no_grad():
            return self.encoder_.forward(torch.tensor(observation).unsqueeze(dim=0)).squeeze().numpy()

    def step(self, action):
        observation, reward, done, info = self.env_.step(action)
        observation = np.transpose(observation, (2, 0, 1))
        return observation, reward, done, info

    def close(self):
        self.env_.close()

