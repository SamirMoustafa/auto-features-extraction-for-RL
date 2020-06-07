from reinforcement_learning.scenario.progress.progress_reporter import TrainProgressReporter


class ConsoleProgressReporter(TrainProgressReporter):
    def report_episode_reward(self, episode, reward):
        print("Episode: " + str(episode) + ", Reward: " + str(reward))
