from gym.envs.registration import registry, register, make, spec

# Box2d
# ----------------------------------------

register(
    id='CarRacingCustom-v0',
    entry_point='src.reinforcement_learning.env.gym_core_environments:CarRacing',
    max_episode_steps=1000,
    reward_threshold=900,
)
