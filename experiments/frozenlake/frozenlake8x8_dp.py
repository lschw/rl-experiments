import sys
sys.path.insert(0, "../../")
import numpy as np
import gym
import algorithms as alg
from evaluate import *

env = gym.make("FrozenLake8x8-v0")

print()
print("DP policy iteration")
pi,v = alg.dp_policy_iteration(env, gamma=1, tol=1e-5, iter_max=1000)
print(np.array(
    [np.argmax(pi[s]) for s in range(env.nS)]).reshape(env.nrow,env.ncol))
evaluate_policy(env, pi, 10000, env.nS-1)

print()
print("DP value iteration")
pi,v = alg.dp_value_iteration(env, gamma=1, tol=1e-5, iter_max=1000)
print(np.array(
    [np.argmax(pi[s]) for s in range(env.nS)]).reshape(env.nrow,env.ncol))
evaluate_policy(env, pi, 10000, env.nS-1)
