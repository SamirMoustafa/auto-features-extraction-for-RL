import numpy as np
import gym_donkeycar
import gym
import os


class DonkeyCarEnvironment:
    EXEC_REL_PATH = "/gym-donkeycar/apps/donkey_sim_custom_build.x86_64"
    PORT = 9090
    ENV_NAME = "donkey-warehouse-v0"

    def __init__(self, third_party_envs_path):
        self.env_ = gym.make(self.ENV_NAME, exe_path=third_party_envs_path+self.EXEC_REL_PATH, port=self.PORT)

    def reset(self):
        observation = self.env_.reset()
        return np.transpose(observation, (2, 0, 1))

    def step(self, action):
        observation, reward, done, _ = self.env_.step(action)
        observation = np.transpose(observation, (2, 0, 1))
        return observation, reward, done

    def close(self):
        self.env_.close()

    def get_action_space(self):
        return self.env_.action_space
