from gym.envs.registration import register

register(
    id='Gridworld-v0',
    entry_point='gym_gridworld.envs:GridworldEnv',
)
