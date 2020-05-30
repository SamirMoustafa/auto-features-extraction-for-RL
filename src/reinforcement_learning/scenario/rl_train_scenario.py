class RLTrainScenario:
    def __init__(self, env, agent, progress_reporter, n_episodes, max_steps, batch_size):
        self.env_ = env
        self.agent_ = agent
        self.reporter_ = progress_reporter
        self.n_episodes_ = n_episodes
        self.max_steps_ = max_steps
        self.batch_size_ = batch_size

    def run(self):
        episode_rewards = []
        total_steps = 0

        for episode in range(self.n_episodes_):
            state = self.env_.reset()
            episode_reward = 0

            for step in range(self.max_steps_):
                action = self.agent_.get_action(state)
                next_state, reward, done = self.env_.step(action)
                self.agent_.replay_buffer_.push(state, action, reward, next_state, done)
                episode_reward += reward
                total_steps += 1

                if len(self.agent_.replay_buffer_) > self.batch_size_ and total_steps > 300:
                    self.agent_.train_step(self.batch_size_)

                if done or step == self.max_steps_ - 1:
                    episode_rewards.append(episode_reward)
                    self.reporter_.report_episode_reward(episode, episode_reward)
                    break

                state = next_state
