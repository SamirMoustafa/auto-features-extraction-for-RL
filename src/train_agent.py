import os

from reinforcement_learning.scenario.rl_train_scenario import RLTrainScenario
from reinforcement_learning.ddpg.ddpg_agent import DDPGAgent
from reinforcement_learning.sac.sac_agent import SACAgent
from reinforcement_learning.ddqn.ddqn_agent import DDQNAgent
from reinforcement_learning.env.wrappers.donkey_wrapper import DonkeyCarEnvironment
from reinforcement_learning.scenario.progress.neptune_progress_reporter import NeptuneProgressReporter
from reinforcement_learning.scenario.progress.console_progress_reporter import ConsoleProgressReporter


def main(agent_type):
    env = DonkeyCarEnvironment(os.path.abspath("./reinforcement_learning/env/third_party_environments/"))

    if agent_type == 'DDPGAgent':
        agent = DDPGAgent((128, 128),
                          env.get_action_space().shape[0],
                          env.get_action_space().low,
                          env.get_action_space().high,
                          gamma=0.99,
                          tau=1e-2,
                          buffer_size=100000,
                          critic_lr=1e-4,
                          actor_lr=1e-4,
                          force_cpu=True)

    elif agent_type == 'SACAgent':
        agent = SACAgent(env.env_.observation_space.shape,
                         env.action_space.shape[0],
                         [env.action_space.low, env.action_space.high],
                         gamma=0.99,
                         tau=0.01,
                         alpha=0.2,
                         critic_lr=3e-4,
                         actor_lr=3e-4,
                         alpha_lr=3e-4,
                         buffer_size=30000,
                         force_cpu=True)

    elif agent_type == 'DDQNAgent':
        agent = DDQNAgent(env.env_.observation_space.shape,
                          env.action_space.shape[0],
                          [env.action_space.low, env.action_space.high],
                          gamma=0.99,
                          tau=1e-2,
                          buffer_size=100000,
                          lr=1e-3)

    else:
        assert False, 'You can pick DDPGAgent, SACAgent or DDQNAgent'

    scenario = RLTrainScenario(env, agent,
                               progress_reporter=NeptuneProgressReporter("timeescaper/rl-full-image", "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMmE1NDY1MjQtOTJhZS00NjJmLTlkMmYtNTIyOWJlNjEyMWJhIn0="),
                               n_episodes=10000,
                               max_steps=5000,
                               batch_size=64)
    scenario.run()


if __name__ == "__main__":
    agent_type = 'SACAgent'
    main(agent_type)
