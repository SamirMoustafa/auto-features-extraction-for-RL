import numpy as np
import gym
from reinforcement_learning.sac.sac_agent import SACAgent

import matplotlib.pyplot as plt


def run_training(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []

    for episode in range(max_episodes):
        _ = env.reset()
        state = env.render(mode='rgb_array').transpose((2, 0, 1))
        state = np.ascontiguousarray(state, dtype=np.float32) / 255
        print(state.shape)
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            _, reward, done, _ = env.step(action)
            print("Step: " + str(step) + ", reward: " + str(reward))
            next_state = env.render(mode='rgb_array').transpose((2, 0, 1))
            next_state = np.ascontiguousarray(next_state, dtype=np.float32) / 255
            agent.replay_buffer_.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer_) > batch_size:
                agent.train_step(batch_size)

            if done or step == max_steps - 1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

    return episode_rewards


def main():
    env = gym.make("MountainCarContinuous-v0")
    agent = SACAgent(env.observation_space.shape[0],
                     env.action_space.shape[0],
                     [env.action_space.low, env.action_space.high],
                     gamma=0.99,
                     tau=0.01,
                     alpha=0.2,
                     critic_lr=3e-4,
                     actor_lr=3e-4,
                     alpha_lr=3e-4,
                     buffer_size=1000000,
                     force_cpu=True)

    episode_rewards = run_training(env, agent, 50, 500, 8)


if __name__ == "__main__":
    main()