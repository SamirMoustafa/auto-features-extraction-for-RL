import neptune

from reinforcement_learning.scenario.progress.progress_reporter import TrainProgressReporter


class NeptuneProgressReporter(TrainProgressReporter):
    REWARD_METRIC = "episode_reward"

    def __init__(self, neptune_project_name, neptune_api_token="ANONYMOUS"):
        neptune.init(neptune_project_name, api_token=neptune_api_token)
        neptune.create_experiment()

    def report_episode_reward(self, episode, reward):
        print("Episode: " + str(episode) + ", Reward: " + str(reward))
        neptune.log_metric(self.REWARD_METRIC, reward)
