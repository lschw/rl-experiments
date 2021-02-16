import sys
sys.path.insert(0, "../../")
import numpy as np
import gym
import algorithms as alg
from evaluate import *

env = gym.make("FrozenLake8x8-v0")

print("\nSARSA")
alg.utils.random_seed(env, 1)
Q,history_sarsa = alg.sarsa(
    env, alpha=0.1, gamma=1, epsilon=1, N_episodes=10000,
    epsilon_decay=alg.utils.decay_sigmoid)
pi = alg.utils.create_greedy_policy(Q)
print(np.array(
    [np.argmax(pi[s]) for s in range(env.nS)]).reshape(env.nrow,env.ncol))
evaluate_policy(env, pi, 10000, env.nS-1)

print("\nQ-Learning")
alg.utils.random_seed(env, 1)
Q,history_qlearning = alg.qlearning(
    env, alpha=0.1, gamma=1, epsilon=1, N_episodes=10000,
    epsilon_decay=alg.utils.decay_sigmoid)
pi = alg.utils.create_greedy_policy(Q)
print(np.array(
    [np.argmax(pi[s]) for s in range(env.nS)]).reshape(env.nrow,env.ncol))
evaluate_policy(env, pi, 10000, env.nS-1)

print("\nExpected SARSA")
alg.utils.random_seed(env, 1)
Q,history_expected_sarsa= alg.expected_sarsa(
    env, alpha=0.1, gamma=1, epsilon=1, N_episodes=10000,
    epsilon_decay=alg.utils.decay_sigmoid)
pi = alg.utils.create_greedy_policy(Q)
print(np.array(
    [np.argmax(pi[s]) for s in range(env.nS)]).reshape(env.nrow,env.ncol))
evaluate_policy(env, pi, 10000, env.nS-1)

alg.utils.plot_learning_curves(
    [history_sarsa, history_qlearning, history_expected_sarsa],
    ["SARSA", "Q-Learning", "Expected SARSA"],
    "frozenlake8x8_td_learning_curves.pdf",
    avg=100
)
