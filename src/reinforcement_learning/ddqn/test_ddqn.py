import numpy as np
import gym
from skimage.transform import resize
from reinforcement_learning.ddqn.ddqn_agent import DDQNAgent
from reinforcement_learning.scenario.progress.neptune_progress_reporter import NeptuneProgressReporter


def render_to_img(env):
    img = env.render(mode='rgb_array')
    img = np.ascontiguousarray(img, dtype=np.float32) / 255
    img = img[120:380, 120:380, :]
    return resize(img, (128, 128)).transpose((2, 0, 1))


def run_training(env, agent, max_episodes, max_steps, batch_size, progress_reporter):
    episode_rewards = []

    for episode in range(max_episodes):
        _ = env.reset()
        state = render_to_img(env)
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            _, reward, done, _ = env.step(action)
            next_state = render_to_img(env)
            agent.replay_buffer_.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer_) > batch_size:
                agent.train_step(batch_size)

            if done or step == max_steps - 1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                progress_reporter.report_episode_reward(episode, episode_reward)
                break

            state = next_state

    return episode_rewards


def main():
    env = gym.make("CartPole-v0")
    agent = DDQNAgent((128, 128),
                      env.action_space.shape[0],
                      env.action_space.low,
                      env.action_space.high,
                      gamma=0.99,
                      tau=1e-2,
                      buffer_size=100000,
                      lr=1e-3)

    progress_reporter = NeptuneProgressReporter("kopanevpavel/rl-ddqn",
                                                "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiOTVhNDU4ZDctNGU1ZC00MjhlLTg3MmUtYTM5NDVlYTcyNjI4In0="),
    episode_rewards = run_training(env, agent, 50, 500, 8, progress_reporter)


if __name__ == "__main__":
    main()