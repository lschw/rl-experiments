import numpy as np
from collections import defaultdict
from .utils import *


def nstep_td(env, pi, alpha=1, gamma=1, n=1, N_episodes=1000,
        ep_max_length=1000):
    """Evaluates state-value function with n-step TD

    Based on Sutton/Barto, Reinforcement Learning, 2nd ed. p. 144

    Args:
        env: Environment
        pi: Policy
        alpha: Step size
        gamma: Discount factor
        n: Number of steps
        N_episodes: Run this many episodes
        ep_max_length: Force termination of episode after this number of steps

    Returns:
        v: State-value function
    """
    v = defaultdict(lambda: 0)
    for i_episode in range(N_episodes):
        print("\r> N-step TD: Episode {}/{}".format(
            i_episode+1, N_episodes), end="")

        state = env.reset()
        rewards = [0]
        states = []
        t = 0
        T = np.inf
        done = False
        while t < T and t < ep_max_length:
            if not done:
                action = select_action_policy(pi, state)
                state_new, reward, done, info = env.step(action)
                rewards.append(reward)
                states.append(state)
                state = state_new
                if done:
                    T = t+n+1

            if t-n >= 0:
                G = 0
                for i in range(min(n,T-t)):
                    G += gamma**i * rewards[t-n+1+i]
                if t < T-n:
                    G += gamma**n * v[states[t]]
                v[states[t-n]] += alpha*(G - v[states[t-n]])
            t += 1

    print()
    return v
