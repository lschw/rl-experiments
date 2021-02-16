import sys
sys.path.insert(0, "../../")
import numpy as np
import gym
import algorithms as alg
from evaluate import *

env = gym.make("FrozenLake-v0")

print("On-policy MC (epsilon-soft)")
alg.utils.random_seed(env, 1)
pi,Q,history_on_pol_eps = alg.mc_on_policy_control(env, gamma=1,
    N_episodes=50000, first_visit=False, exploring_starts=False, epsilon=0.5,
    ep_max_length=1000)
print(np.array(
    [np.argmax(pi[s]) for s in range(env.nS)]).reshape(env.nrow,env.ncol))
evaluate_policy(env, pi, 10000, env.nS-1)

alg.utils.plot_learning_curves(
    [history_on_pol_eps],
    ["MC On-policy (epsilon-soft)"],
    "frozenlake4x4_mc_learning_curves.pdf",
    avg=100
)
