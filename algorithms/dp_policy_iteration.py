import numpy as np
from .utils import *
from .dp_policy_evaluation import *

def dp_policy_iteration(env, gamma, tol=1e-3, iter_max=100):
    """Determines optimal policy by performing policy iteration
    via Bellman optimality equation

    Based on Sutton/Barto, Reinforcement Learning, 2nd ed. p. 80

    Args:
        env: Environment
        gamma: Discount factor
        tol: Tolerance to stop iteration
        iter_max: Maximum iteration count

    Returns
        pi: Optimal policy
        v: Value function of policy
    """
    v = np.zeros(env.observation_space.n)
    pi = np.ones(
        (env.observation_space.n,env.action_space.n))/env.action_space.n
    for i_iter in range(iter_max):
        print("\r> DP Policy iteration: Iteration {}/{}".format(
            i_iter+1, iter_max), end="")

        # Policy evaluation
        v = dp_policy_evaluation(
                env, pi, v, gamma, tol, iter_max, verbose=False)

        # Policy improvement
        policy_stable = True
        for state in range(env.observation_space.n):
            pi_old = pi[state].copy()
            pi[state] = utils.dp_greedy_policy(env, v, state, gamma)
            if not np.array_equal(pi_old, pi[state]):
                policy_stable = False
        if policy_stable:
            break
    print()
    return pi,v
