import numpy as np
from collections import defaultdict
from .utils import *


def semi_gradient_sarsa(env, qfunc, qfunc_deriv, w,
        gamma=1, alpha=0.1, epsilon=0.1, N_episodes=1000,
        epsilon_decay=decay_none, alpha_decay=decay_none):
    """Determines optimal action-value function with semi-gradient SARSA

    Based on Sutton/Barto, Reinforcement Learning, 2nd ed. p. 244

    Args:
        env: Environment
        qfunc: Action-value function
        qfunc_deriv: Derivative of action-value function
        w: Weights
        gamma: Discount factor
        alpha: Step size
        epsilon: Parameter for epsilon-greedy policy
        N_episodes: Run this many episodes
        epsilon_decay: Decay function for epsilon, default no decay
        alpha_decay: Decay function for alpha, default no decay

    Returns:
        w: Weights for state-value function
        history: List of episodes
    """
    history = [[] for i in range(N_episodes)]
    for i_episode in range(N_episodes):
        print("\r> Semi-gradient SARSA: Episode {}/{}".format(
            i_episode+1, N_episodes), end="")

        epsilon_i = epsilon_decay(epsilon, i_episode, N_episodes)
        alpha_i = alpha_decay(alpha, i_episode, N_episodes)

        state = env.reset()
        Q_s = [qfunc(state, a, w) for a in range(env.action_space.n)]
        action = select_action_epsilon_greedy(Q_s, epsilon_i)
        done = False
        while not done:
            state_new, reward, done, info = env.step(action)
            history[i_episode].append((state,action,reward))
            Q_sa = qfunc(state, action, w)
            dQ_sa = qfunc_deriv(state, action, w)
            if done:
                w += alpha_i*(reward - Q_sa)*dQ_sa
            else:
                Q_s = [qfunc(state, a, w) for a in range(env.action_space.n)]
                action_new = select_action_epsilon_greedy(Q_s, epsilon_i)
                Q_sa_new = qfunc(state_new, action_new, w)
                w += alpha_i*(reward + gamma*Q_sa_new - Q_sa)*dQ_sa
            state = state_new
            action = action_new
    print()
    return w,history
