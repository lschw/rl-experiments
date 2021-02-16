import numpy as np
from collections import defaultdict
from .utils import *


def mc_off_policy_control(env, gamma=1, N_episodes=10000, b=None,
        ep_max_length=1000):
    """Determines optimal policy with off-policy Monte Carlo

    Uses weighted importance sampling

    Based on Sutton/Barto, Reinforcement Learning, 2nd ed. p. 111

    Args:
        env: Environment
        gamma: Discount factor
        N_episodes: Run this many episodes
        b: Behavior policy (must have coverage of pi)
            or None (equiprobable policy is chosen)
        ep_max_length: Force termination of episode after this number of steps

    Returns:
        pi: Policy
        Q: Action-value function
        history: List of episodes
    """
    history = [[] for i in range(N_episodes)]
    pi = defaultdict(lambda: np.ones(env.action_space.n)/env.action_space.n)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    if b == None:
        b = defaultdict(lambda: np.ones(env.action_space.n)/env.action_space.n)

    for i_episode in range(N_episodes):
        print("\r> MC off-policy control: Episode {}/{}".format(
            i_episode+1, N_episodes), end="")

        # Generate episode and reverse order
        episode = generate_episode(env, b)
        history[i_episode] = episode
        episode = episode[::-1]

        # Run through episode in reversed order and perform MC update
        G = 0
        W = 1
        for i,(state,action,reward) in enumerate(episode):
            G = gamma*G + reward
            C[state][action] += W
            Q[state][action] += W/C[state][action] * (G - Q[state][action])
            pi[state] = create_greedy_policy_state(Q, state)
            if action == np.argmax(Q[state]):
                break
            W = W * 1/b[state][action]

    print()
    return pi,Q,history
