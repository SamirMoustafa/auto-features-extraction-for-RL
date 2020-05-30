import os

from reinforcement_learning.scenario.rl_train_scenario import RLTrainScenario
from reinforcement_learning.ddpg.ddpg_agent import DDPGAgent
from reinforcement_learning.env.wrappers.donkey_wrapper import DonkeyCarEnvironment
from reinforcement_learning.scenario.progress.neptune_progress_reporter import NeptuneProgressReporter
from reinforcement_learning.scenario.progress.console_progress_reporter import ConsoleProgressReporter


def main():
    env = DonkeyCarEnvironment(os.path.abspath("./env/third_party_environments/"))

    agent = DDPGAgent((128, 128),
                      env.get_action_space().shape[0],
                      env.get_action_space().low,
                      env.get_action_space().high,
                      gamma=0.99,
                      tau=1e-2,
                      buffer_size=100000,
                      critic_lr=1e-3,
                      actor_lr=1e-3,
                      force_cpu=True)

    scenario = RLTrainScenario(env, agent,
                               progress_reporter=ConsoleProgressReporter(),
                               n_episodes=10000,
                               max_steps=1000,
                               batch_size=32)
    scenario.run()


if __name__ == "__main__":
    main()
