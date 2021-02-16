import numpy as np
from collections import defaultdict
from .utils import *


def mc_on_policy_evaluation(env, pi, gamma=1, N_episodes=1000, first_visit=True,
        ep_max_length=1000):
    """Evaluates state-value function with on-policy Monte Carlo

    Based on Sutton/Barto, Reinforcement Learning, 2nd ed. p. 92

    Args:
        env: Environment
        pi: Policy
        gamma: Discount factor
        N_episodes: Run this many episodes
        first_visit: Whether to use first-visit MC or every-visit MC
        ep_max_length: Force termination of episode after this number of steps

    Returns:
        v: State-value function
    """
    v = defaultdict(lambda: 0)
    visits = defaultdict(lambda: 0)
    for i_episode in range(N_episodes):
        print("\r> MC On-policy evaluation: Episode {}/{}".format(
            i_episode+1, N_episodes), end="")

        # Generate episode and reverse order
        episode = generate_episode(env, pi, ep_max_length)
        episode = episode[::-1]
        ep_state = [e[0] for e in episode]

        # Run through episode in reversed order and perform MC update
        G = 0
        for i,(state,action,reward) in enumerate(episode):
            G = gamma*G + reward
            if not first_visit or state not in ep_state[i+1:]:
                visits[state] += 1
                v[state] += 1./visits[state]*(G - v[state])
    print()
    return v

