import os
import gym
import gym_donkeycar
import numpy as np
import matplotlib.pyplot as plt
import sys
import threading
import os
from PIL import Image
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Creates local webserver and auto handles authentication

# Make GoogleDrive instance with Authenticated GoogleAuth instance
drive = GoogleDrive(gauth)

# IDs check
'''
def ListFolder(parent):
    filelist = []
    file_list = drive.ListFile({'q': "'%s' in parents and trashed=false" % parent}).GetList()
    for f in file_list:
        if f['mimeType'] == 'application/vnd.google-apps.folder':  # if folder
            filelist.append({"id": f['id'], "title": f['title'], "list": ListFolder(f['id'])})
        else:
            filelist.append({"title": f['title'], "title1": f['alternateLink']})
    return filelist

print(ListFolder('root'))
'''

action = np.array([0.0, 0.0])

rewards = []

NUM_EPISODES = 3
MAX_TIME_STEPS = 1000000

ENV_NAME = ['warehouse',
            'generated_roads',
            'generated_track',
            'mountain_track',
            'avc_sparkfun',
            'custom',
            'small_loop'
            ]


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
            # observation, reward, done, info = env.step(action)
            observation, reward, done, info = env.viewer.observe()

            wrapped_image = observation.transpose(2, 0, 1)
            # print(type(wrapped_image))
            # print(wrapped_image.shape)
            print(reward)
            rewards.append(reward)

            if (t+1) % 400 == 0:
                plt.plot(rewards)
                # plt.show()

            # if t % 25 == 0:
                # plt.imshow(observation)
                # plt.show()
            '''im = Image.fromarray(observation)
            PATH = os.path.abspath('env/') + '/data_samples/custom_environments/donkeycar/'
            drive_file = drive.CreateFile({'title': ENV_NAME[0] + '_special_cases' + '_time_step_' + str(t) + '.jpg',
                                            'parents': [{'id': '1iBxcuGkYQlJmNTRbdBzPcUjjh2liy_Rt'}]})
            file_name = PATH + ENV_NAME[0] + '_autopilot' + '_time_step_' + str(t) + '.jpg'
            im.save(file_name)
            drive_file.SetContentFile(file_name)
            drive_file.Upload()
            os.remove(file_name)'''

            '''
            if done:
                break
                '''


# SET UP ENVIRONMENT
# exe_path = os.path.abspath('env/') + "/third_party_environments/gym-donkeycar/apps/donkey_sim.x86_64"
exe_path = ""
port = 9090

env_list = [
    "donkey-warehouse-v0",
    "donkey-generated-roads-v0",
    "donkey-avc-sparkfun-v0",
    "donkey-generated-track-v0",
    "donkey-mountain-track-v0"
]

env = gym.make(env_list[-1], exe_path=exe_path, port=port)

simulate(env)
env.close()
print("test finished")
