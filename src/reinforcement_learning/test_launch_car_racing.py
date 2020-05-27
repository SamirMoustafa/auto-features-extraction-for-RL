import gym
import random
import src.reinforcement_learning.env
from scipy import misc
from PIL import Image
import numpy as np

random.seed(0)  # make results reproducible

env = gym.make('CarRacingCustom-v0')
observation = env.reset()

N = 50
EPISODES = 1
TIMESTAMP = 500

for ep in range(EPISODES):
    observation = env.reset()
    reward = 0
    sum_reward = 0
    data_batch = {}
    reset = True
    for t in range(TIMESTAMP):
        env.render()

