import numpy as np
from collections import defaultdict
from .utils import *


def mc_on_policy_control(env, gamma=1, N_episodes=1000, first_visit=True,
        exploring_starts=True, epsilon=0, ep_max_length=1000,
        epsilon_decay=decay_none):
    """Determines optimal policy with on-policy Monte Carlo

    Based on Sutton/Barto, Reinforcement Learning, 2nd ed. p. 99,101

    Args:
        env: Environment
        gamma: Discount factor
        N_episodes: Run this many episodes
        first_visit: Whether to use first-visit MC or every-visit MC
        exploring_starts: Whether to use exploring starts
        epsilon: Parameter for epsilon-greedy policy
        ep_max_length: Force termination of episode after this number of steps
        epsilon_decay: Decay function for epsilon, default no decay

    Returns:
        pi: Policy
        Q: Action-value function
        history: List of episodes
    """
    history = [[] for i in range(N_episodes)]
    pi = defaultdict(lambda: np.ones(env.action_space.n)/env.action_space.n)
    Q = defaultdict(lambda: np.ones(env.action_space.n))
    visits = defaultdict(lambda: np.zeros(env.action_space.n))
    for i_episode in range(N_episodes):
        print("\r> MC On-policy control: Episode {}/{}".format(
            i_episode+1, N_episodes), end="")

        epsilon_i = epsilon_decay(epsilon, i_episode, N_episodes)

        # Generate episode and reverse order
        episode = generate_episode(env, pi, exploring_starts, ep_max_length)
        history[i_episode] = episode
        episode = episode[::-1]
        ep_state_action = [(e[0],e[1]) for e in episode]

        # Run through episode in reversed order and perform MC update
        G = 0
        for i,(state,action,reward) in enumerate(episode):
            G = gamma*G + reward
            if not first_visit or (state,action) not in ep_state_action[i+1:]:
                visits[state][action] += 1
                Q[state][action] += 1./visits[state][action]*(
                    G - Q[state][action])
                pi[state] = utils.create_epsilon_soft_policy_state(
                    Q, state, epsilon_i)
    print()
    return pi,Q,history


