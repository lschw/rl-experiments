import sys
sys.path.insert(0, "../../")
import numpy as np
import gym
import algorithms as alg
from evaluate import *

env = gym.make("Taxi-v3")

print("\nSARSA")
alg.utils.random_seed(env, 1)
Q,history_sarsa = alg.sarsa(
    env, alpha=0.5, gamma=0.99, epsilon=0.5, N_episodes=10000,
    epsilon_decay=alg.utils.decay_linear,
    alpha_decay=alg.utils.decay_linear)
pi = alg.utils.create_greedy_policy(Q)
evaluate_policy(env, pi, 1000)

print("\nQ-Learning")
alg.utils.random_seed(env, 1)
Q,history_qlearning = alg.qlearning(
    env, alpha=0.5, gamma=0.99, epsilon=0.5, N_episodes=10000,
    epsilon_decay=alg.utils.decay_linear,
    alpha_decay=alg.utils.decay_linear)
pi = alg.utils.create_greedy_policy(Q)
evaluate_policy(env, pi, 1000)

print("\nExpected SARSA")
alg.utils.random_seed(env, 1)
Q,history_expected_sarsa= alg.expected_sarsa(
    env, alpha=0.5, gamma=0.99, epsilon=0.5, N_episodes=10000,
    epsilon_decay=alg.utils.decay_linear,
    alpha_decay=alg.utils.decay_linear)
pi = alg.utils.create_greedy_policy(Q)
evaluate_policy(env, pi, 1000)

alg.utils.plot_learning_curves(
    [history_sarsa, history_qlearning, history_expected_sarsa],
    ["SARSA", "Q-Learning", "Expected SARSA"],
    "taxi_td.pdf"
)
