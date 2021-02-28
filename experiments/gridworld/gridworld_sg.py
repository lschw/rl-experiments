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

sagg_v = alg.encoding.StateAggregation(N_states=env.nS, group_size=1)
sagg_q = alg.encoding.StateAggregation(
    N_states=env.nS, N_actions=env.action_space.n, group_size=1)

pi_eq = defaultdict(lambda: np.ones(env.action_space.n)/env.action_space.n)

print("Gradient MC for equiprobable policy")
alg.utils.random_seed(env, 1)
w = sagg_v.generate_weights()
w = alg.gradient_mc_evaluation(
    env, pi_eq, sagg_v.v, sagg_v.v_deriv, w, gamma=1, alpha=0.1,
    N_episodes=10000, ep_max_length=1000, alpha_decay=alg.utils.decay_linear)
v = [sagg_v.v(s,w) for s in range(env.nS)]
render_policy_and_value_function(env, pi_eq, v)

print("Semi-gradient TD(0) for equiprobable policy")
alg.utils.random_seed(env, 1)
w = sagg_v.generate_weights()
w = alg.semi_gradient_td0(
    env, pi_eq, sagg_v.v, sagg_v.v_deriv, w, gamma=1, alpha=0.1,
    N_episodes=10000, ep_max_length=1000, alpha_decay=alg.utils.decay_linear)
v = [sagg_v.v(s,w) for s in range(env.nS)]
render_policy_and_value_function(env, pi_eq, v)

print("Semi-gradient SARSA")
alg.utils.random_seed(env, 1)
w = sagg_q.generate_weights()
w,history_sg_sarsa = alg.semi_gradient_sarsa(
    env, sagg_q.q, sagg_q.q_deriv, w, alpha=0.1, gamma=1, epsilon=0.3,
    N_episodes=2000)
Q = {s: [sagg_q.q(s, a, w) for a in range(env.action_space.n)]
    for s in range(env.nS)}
pi = alg.utils.create_greedy_policy(Q)
v = alg.td0(env, pi, alpha=0.1, gamma=1, N_episodes=2000)
render_policy_and_value_function(env, pi, v)

print("Semi-gradient Q-Learning")
alg.utils.random_seed(env, 1)
w = sagg_q.generate_weights()
w,history_sg_qlearning = alg.semi_gradient_qlearning(
    env, sagg_q.q, sagg_q.q_deriv, w, alpha=0.1, gamma=1, epsilon=0.3,
    N_episodes=2000)
Q = {s: [sagg_q.q(s, a, w) for a in range(env.action_space.n)]
    for s in range(env.nS)}
pi = alg.utils.create_greedy_policy(Q)
v = alg.td0(env, pi, alpha=0.1, gamma=1, N_episodes=2000)
render_policy_and_value_function(env, pi, v)

alg.utils.plot_learning_curves(
    [history_sg_sarsa, history_sg_qlearning],
    ["Semi-gradient SARSA", "Semi-gradient Q-Learning"],
    "gridworld_sg_learning.pdf"
)
