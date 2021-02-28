import numpy as np
from collections import defaultdict
from .utils import *


def gradient_mc_evaluation(env, pi, vfunc, vfunc_deriv, w,
        gamma=1, alpha=0.1, N_episodes=1000,
        ep_max_length=1000, alpha_decay=decay_none):
    """Evaluates state-value function with gradient Monte Carlo

    Based on Sutton/Barto, Reinforcement Learning, 2nd ed. p. 202

    Args:
        env: Environment
        pi: Policy
        vfunc: Value function
        vfunc_deriv: Derivative of value function
        w: Weights
        gamma: Discount factor
        alpha: Step size
        N_episodes: Run this many episodes
        ep_max_length: Force termination of episode after this number of steps
        alpha_decay: Decay function for alpha, default no decay

    Returns:
        w: Weights for state-value function
    """
    for i_episode in range(N_episodes):
        print("\r> Gradient MC evaluation: Episode {}/{}".format(
            i_episode+1, N_episodes), end="")

        alpha_i = alpha_decay(alpha, i_episode, N_episodes)

        # Generate episode and reverse order
        episode = generate_episode(env, pi, ep_max_length)
        episode = episode[::-1]

        # Run through episode in reversed order
        # and perform stochastic gradient update of weights
        G = 0
        for i,(state,action,reward) in enumerate(episode):
            G = gamma*G + reward
            w += alpha_i*(G - vfunc(state, w))*vfunc_deriv(state, w)
    print()
    return w

