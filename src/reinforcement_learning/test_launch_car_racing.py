import gym
import random
import numpy as np
from pyglet.window import key
from gym.wrappers.monitor import Monitor
import src.reinforcement_learning.env

random.seed(0)  # make results reproducible
advanced_mode = False
record_video = False

if advanced_mode:
    env = gym.make('CarRacingCustomAdvanced-v0',
                   allow_reverse=True,
                   grayscale=1,
                   show_info_panel=1,
                   discretize_actions=None,
                   num_tracks=2,
                   num_lanes=2,
                   num_lanes_changes=6,
                   max_time_out=0,
                   frames_per_state=4,
                   num_obstacles=10)

    discretize = env.discretize_actions
    if discretize is None:
        a = np.array([0.0, 0.0, 0.0])
    else:
        a = np.array([0])


    def key_press(k, mod):
        global restart
        if discretize is None:
            if k == 0xff0d: restart = True
            if k == key.LEFT:  a[0] = -1.0
            if k == key.RIGHT: a[0] = +1.0
            if k == key.UP:    a[1] = +1.0
            if k == key.DOWN:  a[1] = -1.0
            if k == key.SPACE: a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
        elif discretize == "hard":
            if k == 0xff0d: restart = True
            if k == key.LEFT:  a[0] = 1
            if k == key.RIGHT: a[0] = 2
            if k == key.UP:    a[0] = 3
            if k == key.SPACE: a[0] = 4


    def key_release(k, mod):
        if discretize is None:
            if k == key.LEFT and a[0] == -1.0: a[0] = 0
            if k == key.RIGHT and a[0] == +1.0: a[0] = 0
            if k == key.UP:    a[1] = 0
            if k == key.DOWN:  a[1] = 0
            if k == key.SPACE: a[2] = 0
        else:
            a[0] = 0
        if k == key.D:     set_trace()
        if k == key.R:     env.reset()
        if k == key.Z:     env.change_zoom()
        if k == key.G:     env.switch_intersection_groups()
        if k == key.I:     env.switch_intersection_points()
        if k == key.X:     env.switch_xt_intersections()
        if k == key.E:     env.switch_end_of_track()
        if k == key.S:     env.switch_start_of_track()
        if k == key.T:     env.screenshot('./')
        if k == key.Q:     sys.exit()


    env.render()
    if record_video:
        env = Monitor(env, '/home/pavel/Skoltech/DL/final_project/auto-features-extraction-for-RL/src/reinforcement_learning/env/video-test', force=True)
    # env.key_press_fn = key_press
    # env.key_release_fn = key_release

    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    while True:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False

        while True:
            if discretize is not None:
                a_tmp = a[0]
            else:
                a_tmp = a
            s, r, done, info = env.step(a_tmp)
            total_reward += r
            if steps % 200 == 0 or done:
                # print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                steps += 1
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            if not record_video:  # Faster, but you can as well call env.render() every time to play full window.
                env.render()
            if done or restart: break

    env.close()

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
    if record_video:
        env = Monitor(env, '/home/pavel/Skoltech/DL/final_project/auto-features-extraction-for-RL/src/reinforcement_learning/env/video-test-default', force=True)
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
