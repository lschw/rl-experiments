import numpy as np
from collections import defaultdict
from .utils import *


def mc_off_policy_evaluation(env, pi, gamma=1, N_episodes=1000, b=None,
        ep_max_length=1000):
    """Evaluates action-value function with off-policy Monte Carlo

    Uses weighted importance sampling

    Based on Sutton/Barto, Reinforcement Learning, 2nd ed. p. 110

    Args:
        env: Environment
        pi: Policy
        gamma: Discount factor
        N_episodes: Run this many episodes
        b: Behavior policy (must have coverage of pi)
            or None (equiprobable policy is chosen)
        ep_max_length: Force termination of episode after this number of steps

    Returns:
        q: Action-value function
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    if b == None:
        b = defaultdict(lambda: np.ones(env.action_space.n)/env.action_space.n)

    for i_episode in range(N_episodes):
        print("\r> MC Off-policy evaluation: Episode {}/{}".format(
            i_episode+1, N_episodes), end="")

        # Generate episode and reverse order
        episode = generate_episode(env, b, ep_max_length)
        episode = episode[::-1]

        # Run through episode in reversed order and perform MC update
        G = 0
        W = 1
        for i,(state,action,reward) in enumerate(episode):
            G = gamma*G + reward
            C[state][action] += W
            Q[state][action] += W/C[state][action] * (G - Q[state][action])
            W = W * pi[state][action]/b[state][action]
            if W == 0:
                break
    print()
    return Q

