import numpy as np
import gym
from skimage.transform import resize
from reinforcement_learning.ddqn.ddqn_agent import DDQNAgent
from reinforcement_learning.scenario.progress.neptune_progress_reporter import NeptuneProgressReporter
import matplotlib.pyplot as plt
import src.reinforcement_learning.env


def check_early_stop(reward, totalreward, fie, neg_reward_counter):
    max_neg_rewards = 100
    if reward < 0 and fie > 10:
        neg_reward_counter += 1
        done = (neg_reward_counter > max_neg_rewards)

        if done and totalreward <= 500:
            punishment = -20.0
        else:
            punishment = 0.0
        if done:
            neg_reward_counter = 0

        return done, punishment, neg_reward_counter
    else:
        neg_reward_counter = 0
        return False, 0.0, neg_reward_counter


def render_to_img(env):
    img = env.render(mode='rgb_array')
    img = np.ascontiguousarray(img, dtype=np.float32) / 255
    img = img[:400, 40:440, :]
    return resize(img, (128, 128)).transpose((2, 0, 1))


def run_training(env, agent, max_episodes, max_steps, batch_size, progress_reporter):
    episode_rewards = []
    max_reward = -100000

    for episode in range(max_episodes):
        _ = env.reset()
        state = render_to_img(env)
        episode_reward = 0

        counter = 0

        plt.imshow(np.transpose(state, (1, 2, 0)))
        plt.show()

        neg_reward_counter = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            _, reward, done, _ = env.step(action)
            next_state = render_to_img(env)
            agent.replay_buffer_.push(state, action, reward, next_state, done)

            early_done, punishment, neg_reward_counter = check_early_stop(reward, episode_reward, max_steps, neg_reward_counter)
            if early_done:
                reward += punishment

            done = done or early_done

            episode_reward += reward
            counter += 1

            if counter == 30:
                plt.imshow(np.transpose(state, (1, 2, 0)))
                plt.show()

            '''
            if episode_reward > max_reward:
                agent.save_checkpoint(checkpoint_path="/content/drive/My Drive/ddqn/checkpoint.pth")
                max_reward = episode_reward'''

            if len(agent.replay_buffer_) > batch_size:
                agent.train_step(batch_size)

            # progress_reporter.report_episode_reward(episode, episode_reward)

            if done or step == max_steps - 1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                progress_reporter.report_episode_reward(episode, episode_reward)
                break

            state = next_state

def main():
    env = gym.make("MountainCarContinuous-v0")
    # env = gym.make('CarRacing-v0')
    # env = gym.make('CarRacingCustom-v0')
    agent = DDQNAgent((128, 128),
                      env.action_space.shape[0],
                      [env.action_space.low,
                       env.action_space.high],
                      gamma=0.99,
                      tau=1e-2,
                      buffer_size=100000,
                      lr=1e-3)

    progress_reporter = NeptuneProgressReporter("kopanevpavel/rl-ddqn",
                                                "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiOTVhNDU4ZDctNGU1ZC00MjhlLTg3MmUtYTM5NDVlYTcyNjI4In0=")
    shared_dict = {"action": [0, 0], "exploration_mode": True}

    _ = env.reset()
    state = render_to_img(env)

    plt.imshow(np.transpose(state, (1, 2, 0)))
    plt.show()

    counter = 0

    print("Exploration mode started")
    while shared_dict["exploration_mode"]:
        action = env.action_space.sample()
        _, reward, done, info = env.step(action)
        next_state = render_to_img(env)
        # print(action)
        agent.replay_buffer_.push(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()
            shared_dict["exploration_mode"] = False

    print("Exploration finished, starting training")
    run_training(env, agent, 5000, 500, 20, progress_reporter)


if __name__ == "__main__":
    main()