import gym
import random
import numpy as np
from pyglet.window import key
import src.reinforcement_learning.env
from src.reinforcement_learning.env.gym_core_environments.car_racing_advanced.car_racing import play

random.seed(0)  # make results reproducible
advanced_mode = False

if advanced_mode:
    env = gym.make('CarRacingCustomAdvanced-v0',
                   allow_reverse=False,
                   grayscale=1,
                   show_info_panel=1,
                   discretize_actions=None,
                   num_tracks=2,
                   num_lanes=2,
                   num_lanes_changes=4,
                   max_time_out=0,
                   frames_per_state=4,
                   num_obstacles=10)
    play(env)

else:
    env = gym.make('CarRacingCustom-v0')
    a = np.array([0.0, 0.0, 0.0])
    def key_press(k, mod):
        global restart
        if k == 0xff0d: restart = True
        if k == key.LEFT:  a[0] = -1.0
        if k == key.RIGHT: a[0] = +1.0
        if k == key.UP:    a[1] = +1.0
        if k == key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT  and a[0] == -1.0: a[0] = 0
        if k == key.RIGHT and a[0] == +1.0: a[0] = 0
        if k == key.UP:    a[1] = 0
        if k == key.DOWN:  a[2] = 0

    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor
        env = Monitor(env, '/tmp/video-test', force=True)
    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()
