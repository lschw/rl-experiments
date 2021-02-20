import sys
sys.path.insert(0, "../../")
import numpy as np
import gym
import algorithms as alg
from evaluate import *

env = gym.make("Taxi-v3")

print("DP policy iteration")
pi,v = alg.dp_policy_iteration(env, gamma=0.9, tol=1e-3, iter_max=5000)
evaluate_policy(env, pi, 1000)

print("DP value iteration")
pi,v = alg.dp_value_iteration(env, gamma=0.9, tol=1e-3, iter_max=5000)
evaluate_policy(env, pi, 1000)

