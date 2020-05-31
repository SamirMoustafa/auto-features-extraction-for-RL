from abc import ABC


class TrainProgressReporter(ABC):
    def report_episode_reward(self, episode, reward):
        pass
