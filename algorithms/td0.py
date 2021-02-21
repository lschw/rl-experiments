import numpy as np
from collections import defaultdict
from .utils import *


def td0(env, pi, alpha=1, gamma=1, N_episodes=1000, ep_max_length=1000,
        alpha_decay=decay_none):
    """Evaluates state-value function with TD(0)

    Based on Sutton/Barto, Reinforcement Learning, 2nd ed. p. 120

    Args:
        env: Environment
        pi: Policy
        alpha: Step size
        gamma: Discount factor
        N_episodes: Run this many episodes
        ep_max_length: Force termination of episode after this number of steps
        alpha_decay: Decay function for alpha, default no decay

    Returns:
        v: State-value function
    """
    v = defaultdict(lambda: 0)
    for i_episode in range(N_episodes):
        print("\r> TD(0): Episode {}/{}".format(
            i_episode+1, N_episodes), end="")

        alpha_i = alpha_decay(alpha, i_episode, N_episodes)

        state = env.reset()
        done = False
        steps = 0
        while not done and steps < ep_max_length:
            action = select_action_policy(pi, state)
            state_new, reward, done, info = env.step(action)
            v[state] += alpha_i*(reward + gamma*v[state_new] - v[state])
            state = state_new
            steps +=1
    print()
    return v

