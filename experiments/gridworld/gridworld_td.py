import sys
sys.path.insert(0, "../../")
import numpy as np
from collections import defaultdict
import gym
from environments import gym_gridworld
import algorithms as alg
from render import *


env = gym.make("Gridworld-v0", grid=np.array([
    [-1,-1,-1,-1,],
    [-1,-1,-1,-1,],
    [-1,-1,-1,-1,],
    [-1,-1,-1,-1,],
    [-1,-1,-1,-1,],
]), terminal_states=[0, 15])

print("TD(0) for equiprobable policy")
alg.utils.random_seed(env, 1)
pi = defaultdict(lambda: np.ones(env.action_space.n)/env.action_space.n)
v = alg.td0(env, pi, alpha=0.1, gamma=1, N_episodes=10000)
render_policy_and_value_function(env, pi, v)

print("N-step (n=3) TD for equiprobable policy")
alg.utils.random_seed(env, 1)
pi = defaultdict(lambda: np.ones(env.action_space.n)/env.action_space.n)
v = alg.nstep_td(env, pi, alpha=0.1, gamma=1, n=3, N_episodes=10000)
render_policy_and_value_function(env, pi, v)

print("SARSA")
alg.utils.random_seed(env, 1)
Q,history_sarsa = alg.sarsa(
    env, alpha=0.1, gamma=1, epsilon=0.3, N_episodes=2000)
pi = alg.utils.create_greedy_policy(Q)
v = alg.td0(env, pi, alpha=0.1, gamma=1, N_episodes=2000)
render_policy_and_value_function(env, pi, v)

print("Q-Learning")
alg.utils.random_seed(env, 1)
Q,history_qlearning = alg.qlearning(
    env, alpha=0.1, gamma=1, epsilon=0.1, N_episodes=2000)
pi = alg.utils.create_greedy_policy(Q)
v = alg.td0(env, pi, alpha=0.1, gamma=1, N_episodes=2000)
render_policy_and_value_function(env, pi, v)

print("Expected SARSA")
alg.utils.random_seed(env, 1)
Q,history_expected_sarsa= alg.expected_sarsa(
    env, alpha=0.1, gamma=1, epsilon=0.1, N_episodes=2000)
pi = alg.utils.create_greedy_policy(Q)
v = alg.td0(env, pi, alpha=0.1, gamma=1, N_episodes=2000)
render_policy_and_value_function(env, pi, v)

print("Double Q-Learning")
alg.utils.random_seed(env, 1)
Q,history_double_qlearning = alg.double_qlearning(
    env, alpha=0.1, gamma=1, epsilon=0.3, N_episodes=2000)
pi = alg.utils.create_greedy_policy(Q)
v = alg.td0(env, pi, alpha=0.1, gamma=1, N_episodes=2000)
render_policy_and_value_function(env, pi, v)

alg.utils.plot_learning_curves(
    [history_sarsa, history_qlearning, history_expected_sarsa,
    history_double_qlearning],
    ["SARSA", "Q-Learning", "Expected SARSA", "Double Q-Learning"],
    "gridworld_td_learning.pdf"
)
