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

print("DP policy evaluation for equiprobable policy")
pi = np.ones((env.observation_space.n,env.action_space.n))/env.action_space.n
v = alg.dp_policy_evaluation(env, pi, gamma=1, tol=1e-5, iter_max=500)
render_policy_and_value_function(env, pi, v)

print("DP policy iteration")
pi,v = alg.dp_policy_iteration(env, gamma=1, tol=1e-5, iter_max=100)
render_policy_and_value_function(env, pi, v)

print("DP value iteration")
pi,v = alg.dp_value_iteration(env, gamma=1, tol=1e-5, iter_max=100)
render_policy_and_value_function(env, pi, v)
