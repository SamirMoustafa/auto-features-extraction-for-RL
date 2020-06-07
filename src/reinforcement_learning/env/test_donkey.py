import os
import gym
import gym_donkeycar
import numpy as np
import matplotlib.pyplot as plt
import sys
import threading
import os
from PIL import Image


action = np.array([0.0, 0.0])

rewards = []

NUM_EPISODES = 3
MAX_TIME_STEPS = 1000000


def select_action(env):
    return env.action_space.sample()


def simulate(env):
    for episode in range(NUM_EPISODES):

        # Reset the environment
        obv = env.reset()

        for t in range(MAX_TIME_STEPS):
            # Select an action
            action = select_action(env)
            # action = np.array([0.0, 0.0])

            # execute the action
            observation, reward, done, info = env.step(action)
            print(observation.shape)
            # observation, reward, done, info = env.viewer.observe()

            wrapped_image = observation.transpose(2, 0, 1)
            # print(type(wrapped_image))
            # print(wrapped_image.shape)
            print(reward)
            rewards.append(reward)

            if t % 100 == 0:
                plt.plot(rewards)
                plt.show()
                plt.imshow(observation)
                plt.show()

            if done:
                break

            '''
            if t % 25 == 0:
                plt.imshow(observation)
                plt.show()
                im = Image.fromarray(observation)
                PATH = os.path.abspath('env/') + '/data_samples/custom_environments/donkeycar/'
                drive_file = drive.CreateFile({'title': ENV_NAME[0] + '_fisheye' + '_time_step_' + str(t) + '.jpg',
                                                'parents': [{'id': '1iBxcuGkYQlJmNTRbdBzPcUjjh2liy_Rt'}]})
                file_name = PATH + ENV_NAME[0] + '_fisheye' + '_time_step_' + str(t) + '.jpg'
                im.save(file_name)
                drive_file.SetContentFile(file_name)
                drive_file.Upload()
                os.remove(file_name)'''

            '''
            if done:
                break'''


# SET UP ENVIRONMENT
exe_path = os.path.abspath('./') + "/third_party_environments/gym-donkeycar/apps/donkey_sim_custom_build.x86_64"
# exe_path = ""  # If you have running game in Unity
port = 9090

env_list = [
    "donkey-warehouse-v0",
    "donkey-generated-roads-v0",
    "donkey-avc-sparkfun-v0",
    "donkey-generated-track-v0",
    "donkey-mountain-track-v0"
]

env = gym.make(env_list[0], exe_path=exe_path, port=port)

simulate(env)
env.close()
print("test finished")
