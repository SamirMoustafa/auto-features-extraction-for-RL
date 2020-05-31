from reinforcement_learning.scenario.teleop.teleop import Teleoperator
from queue import Queue


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
        shared_dict ={"action": [0, 0], "manual_mode": False}

        teleop = Teleoperator(self.env_, self.agent_.replay_buffer_, shared_dict, self.action_queue_)
        teleop.start()

        episode_rewards = []
        total_steps = 0

        exploration_mode = True

        for episode in range(self.n_episodes_):
            state = self.env_.reset()
            episode_reward = 0
            shared_dict["action"] = [0.0, 0.0]

            for step in range(self.max_steps_):
                action = None
                if not shared_dict["manual_mode"]:
                    if exploration_mode:
                        action = self.env_.env_.action_space.sample()
                    else:
                        action = self.agent_.get_action(state)
                else:
                    state, _, _ = self.env_.step([0.0, 0.0])
                    # print("Wait")
                    # action = self.action_queue_.get(block=True)
                    action = shared_dict["action"]
                    print(action)

                next_state, reward, done = self.env_.step(action)
                self.agent_.replay_buffer_.push(state, action, reward, next_state, done)
                episode_reward += reward
                total_steps += 1

                if len(self.agent_.replay_buffer_) > self.batch_size_ and total_steps > 300:
                    print("Train step")
                    self.agent_.train_step(self.batch_size_)

                if done or step == self.max_steps_ - 1:
                    episode_rewards.append(episode_reward)
                    self.reporter_.report_episode_reward(episode, episode_reward)
                    break

                state = next_state
