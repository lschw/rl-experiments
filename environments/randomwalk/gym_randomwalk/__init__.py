from gym.envs.registration import register

register(
    id='RandomWalk-v0',
    entry_point='gym_randomwalk.envs:RandomWalkEnv',
)
