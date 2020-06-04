import numpy as np
import gym_donkeycar
import gym
import os
import torch

import matplotlib.pyplot as plt

class DonkeyCarEnvironment:
    EXEC_REL_PATH = "/gym-donkeycar/apps/new_releases/donkey_sim_custom_build.x86_64"
    PORT = 9090
    ENV_NAME = "donkey-warehouse-v0"

    def __init__(self, third_party_envs_path, encoder_fn=None):
        self.env_ = gym.make(self.ENV_NAME, exe_path=third_party_envs_path+self.EXEC_REL_PATH, port=self.PORT)
        self.encoder_fn_ = encoder_fn

        self.action_space = self.env_.action_space
        if self.encoder_fn_ is None:
            self.observation_space = self.env_.observation_space
        else:
            self.observation_space = gym.spaces.box.Box(np.inf, -np.inf, shape=(64,))

    def reset(self):
        observation = np.copy(np.transpose(self.env_.reset(), (2, 0, 1)))
        if self.encoder_fn_ is None:
            return observation
        with torch.no_grad():
            # observation = self.encoder_.forward(torch.FloatTensor(observation).unsqueeze(dim=0)).squeeze().numpy()
            # return observation
            return self.encoder_fn_(observation)

    def step(self, action):
        observation, reward, done, _ = self.env_.step(action)
        observation = np.copy(np.transpose(observation, (2, 0, 1)))
        if self.encoder_fn_ is not None:
            with torch.no_grad():
                #observation = self.encoder_.forward(torch.FloatTensor(observation).unsqueeze(dim=0)).squeeze().numpy()
                observation = self.encoder_fn_(observation)
        return observation, reward, done

    def close(self):
        self.env_.close()

