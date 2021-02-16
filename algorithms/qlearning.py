import numpy as np
from collections import defaultdict
from .utils import *


def qlearning(env, alpha=1, gamma=1, epsilon=0.1, N_episodes=1000,
        epsilon_decay=decay_none, alpha_decay=decay_none):
    """Determines action-value function for optimal policy with Q-Learning

    Based on Sutton/Barto, Reinforcement Learning, 2nd ed. p. 131

    Args:
        env: Environment
        alpha: Step size
        gamma: Discount factor
        epsilon: Parameter for epsilon-greedy policy
        N_episodes: Run this many episodes
        epsilon_decay: Decay function for epsilon, default no decay
        alpha_decay: Decay function for alpha, default no decay

    Returns:
        Q: Action-value function
        history: List of episodes
    """
    history = [[] for i in range(N_episodes)]
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    for i_episode in range(N_episodes):
        print("\r> Q-Learning: Episode {}/{}".format(
            i_episode+1, N_episodes), end="")

        epsilon_i = epsilon_decay(epsilon, i_episode, N_episodes)
        alpha_i = alpha_decay(alpha, i_episode, N_episodes)

        state = env.reset()
        done = False
        while not done:
            action = select_action_epsilon_greedy(Q, state, epsilon_i)
            state_new, reward, done, info = env.step(action)
            history[i_episode].append((state,action,reward))
            Q[state][action] += alpha_i*(reward
                + gamma*np.max(Q[state_new]) - Q[state][action])
            state = state_new
    print()
    return Q,history

