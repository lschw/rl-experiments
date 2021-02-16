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

print("MC On-policy evaluation for equiprobable policy")
alg.utils.random_seed(env, 1)
pi = defaultdict(lambda: np.ones(env.action_space.n)/env.action_space.n)
v = alg.mc_on_policy_evaluation(
    env, pi, gamma=1, N_episodes=10000, first_visit=True)
render_policy_and_value_function(env, pi, v)

print("MC Off-policy evaluation for equiprobable policy")
alg.utils.random_seed(env, 1)
b = defaultdict(lambda: np.random.dirichlet(np.ones(env.action_space.n)))
pi = defaultdict(lambda: np.ones(env.action_space.n)/env.action_space.n)
Q = alg.mc_off_policy_evaluation(env, pi, gamma=1, N_episodes=10000, b=b)
render_policy_and_value_function(env, pi,
    alg.utils.create_state_value_function(Q, pi))

print("MC On-policy control (exploring-starts)")
alg.utils.random_seed(env, 1)
pi,Q,history_on_pol_es = alg.mc_on_policy_control(
    env, gamma=0.9, N_episodes=10000,
    first_visit=True, exploring_starts=True, epsilon=0, ep_max_length=1000)
pi = alg.utils.create_greedy_policy(Q)
v = alg.mc_on_policy_evaluation(
    env, pi, gamma=1, N_episodes=10000, first_visit=True)
render_policy_and_value_function(env, pi, v)

print("MC On-policy control (epsilon-soft)")
alg.utils.random_seed(env, 1)
pi,Q,history_on_pol_eps = alg.mc_on_policy_control(
    env, gamma=1, N_episodes=10000,
    first_visit=False, exploring_starts=False, epsilon=0.5, ep_max_length=1000,
    epsilon_decay=alg.utils.decay_linear)
pi = alg.utils.create_greedy_policy(Q)
v = alg.mc_on_policy_evaluation(
    env, pi, gamma=1, N_episodes=10000, first_visit=True)
render_policy_and_value_function(env, pi, v)

print("MC Off-policy control")
alg.utils.random_seed(env, 1)
pi,Q,history_off_pol = alg.mc_off_policy_control(env, gamma=1, N_episodes=10000)
render_policy_and_value_function(env, pi,
    alg.utils.create_state_value_function(Q, pi))

alg.utils.plot_learning_curves(
    [history_on_pol_es, history_on_pol_eps, history_off_pol],
    ["MC On-policy (exploring starts)", "MC On-policy (epsilon-soft)",
    "MC Off-policy"],
    "gridworld_mc_learning.pdf",
    ylim_ret=(-30,0),
    ylim_epl=(0,30)
)
