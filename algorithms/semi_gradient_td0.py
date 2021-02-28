import numpy as np
from collections import defaultdict
from .utils import *


def semi_gradient_td0(env, pi, vfunc, vfunc_deriv, w,
        gamma=1, alpha=0.1, N_episodes=1000,
        ep_max_length=1000, alpha_decay=decay_none):
    """Evaluates state-value function with semi-gradient TD(0)

    Based on Sutton/Barto, Reinforcement Learning, 2nd ed. p. 203

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
        print("\r> Semi-gradient TD(0): Episode {}/{}".format(
            i_episode+1, N_episodes), end="")

        alpha_i = alpha_decay(alpha, i_episode, N_episodes)

        state = env.reset()
        done = False
        steps = 0
        while not done and steps < ep_max_length:
            action = select_action_policy(pi, state)
            state_new, reward, done, info = env.step(action)
            v = vfunc(state, w)
            dv = vfunc_deriv(state, w)
            v_new = 0 if done else vfunc(state_new, w)
            w += alpha_i*(reward + gamma*v_new - v) * dv
            state = state_new
            steps +=1
    print()
    return w

