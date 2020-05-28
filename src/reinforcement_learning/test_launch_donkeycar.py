import os
import gym
import gym_donkeycar
import numpy as np
import matplotlib.pyplot as plt
from pyglet.window import key

#%% SET UP ENVIRONMENT
exe_path = f"env/third_party_environments/gym-donkeycar/apps/donkey_sim.x86_64"
# exe_path = ""
port = 9091

env_list = [
            "donkey-warehouse-v0",
            "donkey-generated-roads-v0",
            "donkey-avc-sparkfun-v0",
            "donkey-generated-track-v0",
            "donkey-mountain-track-v0"
            ]

env = gym.make(env_list[-1], exe_path=exe_path, port=port)

key_handler = key.KeyStateHandler()
# env.unwrapped.window.push_handlers(key_handler)

#%% PLAY
obv = env.reset()
for t in range(100):
    # action = np.array([0.0, 0.5])  # drive straight with small speed

    action = np.array([0.0, 0.0])

    # execute the action
    obv, reward, done, info = env.step(action)
    print(type(obv))
    print(obv.shape)
    # plt.imshow(obv)
    # plt.show()
