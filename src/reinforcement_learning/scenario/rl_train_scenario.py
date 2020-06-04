from reinforcement_learning.scenario.teleop.teleop import Teleoperator
from queue import Queue

import numpy as np

class RLTrainScenario:
    def __init__(self, env, agent, progress_reporter, n_episodes, max_steps, batch_size):
        self.env_ = env
        self.agent_ = agent
        self.reporter_ = progress_reporter
        self.n_episodes_ = n_episodes
        self.max_steps_ = max_steps
        self.batch_size_ = batch_size
        self.action_queue_ = Queue()

    def run(self):
        shared_dict ={"action": [0, 0], "manual_mode": False, "exploration_mode": True}

        teleop = Teleoperator(self.env_, shared_dict, self.action_queue_)
        teleop.start()

        # Stage 1: Exploration
        state = self.env_.reset()
        while shared_dict["exploration_mode"]:
            if not shared_dict["manual_mode"]:
                action = self.env_.env_.action_space.sample()
            else:
                action = shared_dict["action"]
            next_state, reward, done = self.env_.step(action)
            self.agent_.replay_buffer_.push(state, action, reward, next_state, done)
            state = next_state
            if done:
                state = self.env_.reset()

        print("Exploration finished, start training")
        episode_rewards = []
        total_steps = 0
        for episode in range(self.n_episodes_):
            state = self.env_.reset()
            episode_reward = 0

            for step in range(self.max_steps_):
                if not shared_dict["manual_mode"]:
                    #print("Actioning")
                    action = self.agent_.get_action(state)
                else:
                    action = shared_dict["action"]
                # print("State: " + str(np.max(state)) + ", " + str(np.min(state)))
                print("Action: " + str(action))

                next_state, reward, done = self.env_.step(action)
                self.agent_.replay_buffer_.push(state, action, reward, next_state, done)
                episode_reward += reward
                total_steps += 1

                if total_steps % 1000 == 0:
                    print("Train, steps: " + str(step))
                    self.agent_.train_step(self.batch_size_)

                if done or step == self.max_steps_ - 1:
                    print("Train, steps: " + str(step))
                    self.agent_.train_step(self.batch_size_)
                    episode_rewards.append(episode_reward)
                    self.reporter_.report_episode_reward(episode, episode_reward)
                    break

                state = next_state
