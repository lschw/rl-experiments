import numpy as np
from collections import defaultdict
from .utils import *


def semi_gradient_qlearning(env, qfunc, qfunc_deriv, w,
        gamma=1, alpha=0.1, epsilon=0.1, N_episodes=1000,
        epsilon_decay=decay_none, alpha_decay=decay_none):
    """Determines optimal action-value function with semi-gradient Q-Learning

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
        print("\r> Semi-gradient Q-Learning: Episode {}/{}".format(
            i_episode+1, N_episodes), end="")
        print(", reward: {}".format(
            np.sum([x[2] for x in history[i_episode-1]])
        ), end="")

        epsilon_i = epsilon_decay(epsilon, i_episode, N_episodes)
        alpha_i = alpha_decay(alpha, i_episode, N_episodes)

        state = env.reset()
        Q_s = [qfunc(state, a, w) for a in range(env.action_space.n)]
        done = False
        while not done:
            action = select_action_epsilon_greedy(Q_s, epsilon_i)
            state_new, reward, done, info = env.step(action)
            history[i_episode].append((state,action,reward))

            Q_s_new = [
                qfunc(state_new, a, w) for a in range(env.action_space.n)]
            dQ_sa = qfunc_deriv(state, action, w)
            w += alpha_i*(reward + gamma*np.max(Q_s_new) - Q_s[action])*dQ_sa

            state = state_new
            Q_s = Q_s_new
    print()
    return w,history
