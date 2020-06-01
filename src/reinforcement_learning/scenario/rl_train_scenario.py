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
        shared_dict ={"action": [0, 0], "manual_mode": False, "exploration_mode": True}

        teleop = Teleoperator(self.env_, shared_dict, self.action_queue_)
        teleop.start()

        episode_rewards = []
        total_steps = 0

        exploration_mode = True
        state = self.env_.reset()

        # Stage 1: Exploration
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
        state = self.env_.reset()
        for episode in range(self.n_episodes_):
            state = self.env_.reset()
            episode_reward = 0

            for step in range(self.max_steps_):
                if not shared_dict["manual_mode"]:
                    #print("Actioning")
                    action = self.agent_.get_action(state)
                else:
                    action = shared_dict["action"]

                next_state, reward, done = self.env_.step(action)
                self.agent_.replay_buffer_.push(state, action, reward, next_state, done)
                episode_reward += reward
                total_steps += 1

                if done or step == self.max_steps_ - 1:
                    if len(self.agent_.replay_buffer_) > self.batch_size_ and not shared_dict["manual_mode"]:
                        print("Train, steps: " + str(step))
                        self.agent_.train_step(self.batch_size_)
                    episode_rewards.append(episode_reward)
                    self.reporter_.report_episode_reward(episode, episode_reward)
                    break

                state = next_state
